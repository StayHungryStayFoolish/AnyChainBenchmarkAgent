"""Secret-free observation receipts emitted by the analysis domain owner."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Mapping, Sequence

from agent.utils.redaction import redact

from ..state import AgentGraphState


ANALYSIS_RECEIPT_VERSION = 1
_HASH_FIELDS = frozenset(
    {
        "block_content_hash",
        "block_id",
        "evidence_hash",
        "input_hash",
        "question_hash",
        "question_id_hash",
        "question_kind_hash",
        "requested_job_hash",
        "resolved_job_hash",
        "resolved_status_hash",
        "visible_result_hash",
    }
)
_RECEIPT_FIELDS = {
    "analysis_evidence_block": frozenset(
        {
            "receipt_type",
            "receipt_version",
            "turn_index",
            "owner",
            "operation",
            "block_id",
            "question_id_hash",
            "question_kind_hash",
            "line_count",
            "block_content_hash",
            "input_disposition",
            "input_non_empty_line_count",
            "input_hash",
            "status",
            "visible_result_hash",
            "receipt_id",
        }
    ),
    "analysis_invocation": frozenset(
        {
            "receipt_type",
            "receipt_version",
            "turn_index",
            "owner",
            "source_kind",
            "block_id",
            "evidence_hash",
            "question_hash",
            "analysis_engine",
            "invoked",
            "visible_result_hash",
            "receipt_id",
        }
    ),
    "analysis_report": frozenset(
        {
            "receipt_type",
            "receipt_version",
            "turn_index",
            "owner",
            "requested_job_hash",
            "resolved_job_hash",
            "resolved_status_hash",
            "job_read_receipt_id",
            "evidence_verified",
            "analysis_engine",
            "invoked",
            "visible_result_hash",
            "receipt_id",
        }
    ),
}
_BLOCK_OPERATIONS = frozenset(
    {"start", "append", "ignore", "finish_requested", "finish", "pause", "resume", "cancel"}
)
_BLOCK_DISPOSITIONS = frozenset(
    {"accepted", "blank_ignored", "completion", "pause", "resume", "cancel"}
)
_BLOCK_STATUSES = frozenset(
    {"active", "paused", "saved", "pending_answer", "empty", "cancelled"}
)
_ANALYSIS_SOURCES = frozenset(
    {"action_argument", "active_block", "inline", "missing", "saved_buffer"}
)


def analysis_hash(value: Any) -> str:
    """Return a stable digest without retaining the underlying sensitive value."""

    return hashlib.sha256(
        json.dumps(
            redact(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def evidence_block_id(
    question: Mapping[str, Any],
    lines: Sequence[str],
) -> str:
    """Return a stable block identity from its owner contract and first content."""

    first_line = next((str(line) for line in lines if str(line).strip()), "")
    return analysis_hash(
        {
            "question_id_hash": analysis_hash(str(question.get("id") or "")),
            "question_kind_hash": analysis_hash(str(question.get("kind") or "")),
            "first_line_hash": analysis_hash(first_line),
        }
    )


def emit_evidence_block_receipt(
    state: AgentGraphState,
    *,
    operation: str,
    question: Mapping[str, Any],
    lines: Sequence[str],
    input_text: str = "",
    input_disposition: str,
    status: str,
    visible_result: str = "",
    block_id: str = "",
) -> None:
    normalized_lines = [str(line) for line in lines if str(line).strip()]
    _append_analysis_receipt(
        state,
        "analysis_evidence_block",
        {
            "owner": "analysis",
            "operation": operation,
            "block_id": block_id or evidence_block_id(question, normalized_lines),
            "question_id_hash": analysis_hash(str(question.get("id") or "")),
            "question_kind_hash": analysis_hash(str(question.get("kind") or "")),
            "line_count": len(normalized_lines),
            "block_content_hash": analysis_hash("\n".join(normalized_lines)),
            "input_disposition": input_disposition,
            "input_non_empty_line_count": len(
                [line for line in str(input_text or "").splitlines() if line.strip()]
            ),
            "input_hash": analysis_hash(str(input_text or "")),
            "status": status,
            "visible_result_hash": analysis_hash(str(visible_result or "")),
        },
    )


def emit_analysis_invocation_receipt(
    state: AgentGraphState,
    *,
    source_kind: str,
    evidence: str,
    question: str,
    visible_result: str,
    invoked: bool,
    block_id: str = "",
) -> None:
    _append_analysis_receipt(
        state,
        "analysis_invocation",
        {
            "owner": "analysis",
            "source_kind": source_kind,
            "block_id": block_id,
            "evidence_hash": analysis_hash(str(evidence or "")),
            "question_hash": analysis_hash(str(question or "")),
            "analysis_engine": "configured_llm",
            "invoked": bool(invoked),
            "visible_result_hash": analysis_hash(str(visible_result or "")),
        },
    )


def emit_report_analysis_receipt(
    state: AgentGraphState,
    *,
    requested_job_id: str,
    resolved_job_id: str,
    resolved_status: str,
    visible_result: str,
    invoked: bool,
    job_read_receipt_id: str = "",
    evidence_verified: bool = False,
) -> None:
    _append_analysis_receipt(
        state,
        "analysis_report",
        {
            "owner": "analysis",
            "requested_job_hash": analysis_hash(str(requested_job_id or "")),
            "resolved_job_hash": analysis_hash(str(resolved_job_id or "")),
            "resolved_status_hash": analysis_hash(str(resolved_status or "")),
            "job_read_receipt_id": str(job_read_receipt_id or ""),
            "evidence_verified": bool(evidence_verified),
            "analysis_engine": "persisted_job_analyzer",
            "invoked": bool(invoked),
            "visible_result_hash": analysis_hash(str(visible_result or "")),
        },
    )


def validate_analysis_receipt(receipt: Mapping[str, Any]) -> tuple[bool, str]:
    """Validate shape, vocabulary, hashes, and content-addressed identity."""

    receipt_type = str(receipt.get("receipt_type") or "")
    expected_fields = _RECEIPT_FIELDS.get(receipt_type)
    if expected_fields is None:
        return False, "unknown receipt type"
    if set(receipt) != set(expected_fields):
        return False, "unexpected receipt fields"
    if receipt.get("receipt_version") != ANALYSIS_RECEIPT_VERSION:
        return False, "unsupported receipt version"
    if receipt.get("owner") != "analysis":
        return False, "invalid receipt owner"
    if not _is_nonnegative_int(receipt.get("turn_index")):
        return False, "invalid turn index"
    for field in expected_fields & _HASH_FIELDS:
        value = receipt.get(field)
        if value and (not isinstance(value, str) or len(value) != 64):
            return False, f"invalid {field}"
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_id"}
    if receipt.get("receipt_id") != analysis_hash(unsigned):
        return False, "receipt hash mismatch"
    if receipt_type == "analysis_evidence_block":
        if receipt.get("operation") not in _BLOCK_OPERATIONS:
            return False, "invalid block operation"
        if receipt.get("input_disposition") not in _BLOCK_DISPOSITIONS:
            return False, "invalid input disposition"
        if receipt.get("status") not in _BLOCK_STATUSES:
            return False, "invalid block status"
        if not _is_nonnegative_int(receipt.get("line_count")):
            return False, "invalid line count"
        if not _is_nonnegative_int(receipt.get("input_non_empty_line_count")):
            return False, "invalid input line count"
        expected = {
            "start": ("accepted", "active"),
            "append": ("accepted", "active"),
            "ignore": ("blank_ignored", "active"),
            "finish_requested": ("completion", "active"),
            "pause": ("pause", "paused"),
            "resume": ("resume", "active"),
            "cancel": ("cancel", "cancelled"),
        }.get(str(receipt.get("operation") or ""))
        if expected and (
            receipt.get("input_disposition"),
            receipt.get("status"),
        ) != expected:
            return False, "inconsistent block lifecycle"
        if receipt.get("operation") == "finish" and (
            receipt.get("input_disposition") != "completion"
            or receipt.get("status") not in {"saved", "pending_answer", "empty"}
        ):
            return False, "inconsistent block completion"
    elif receipt_type == "analysis_invocation":
        if receipt.get("source_kind") not in _ANALYSIS_SOURCES:
            return False, "invalid analysis source"
        if receipt.get("analysis_engine") != "configured_llm":
            return False, "invalid analysis engine"
        if not isinstance(receipt.get("invoked"), bool):
            return False, "invalid invocation flag"
        missing = receipt.get("source_kind") == "missing"
        if missing == bool(receipt.get("invoked")):
            return False, "inconsistent invocation state"
        if receipt.get("source_kind") == "active_block" and not receipt.get("block_id"):
            return False, "active block analysis requires block identity"
        if receipt.get("source_kind") != "active_block" and receipt.get("block_id"):
            return False, "unexpected block identity"
    else:
        if receipt.get("analysis_engine") != "persisted_job_analyzer":
            return False, "invalid analysis engine"
        if not isinstance(receipt.get("invoked"), bool):
            return False, "invalid invocation flag"
        if not isinstance(receipt.get("evidence_verified"), bool):
            return False, "invalid report evidence flag"
        read_receipt_id = str(receipt.get("job_read_receipt_id") or "")
        if read_receipt_id and (
            len(read_receipt_id) != 64
            or any(character not in "0123456789abcdef" for character in read_receipt_id)
        ):
            return False, "invalid job read receipt identity"
        if bool(read_receipt_id) != bool(receipt.get("evidence_verified")):
            return False, "inconsistent report evidence identity"
        if receipt.get("invoked") and not receipt.get("evidence_verified"):
            return False, "report analysis requires verified persisted evidence"
        empty_hash = analysis_hash("")
        if receipt.get("invoked") and receipt.get("resolved_job_hash") == empty_hash:
            return False, "inconsistent report invocation"
        if receipt.get("evidence_verified") and receipt.get("resolved_job_hash") == empty_hash:
            return False, "verified report has no resolved job"
    return True, ""


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _append_analysis_receipt(
    state: AgentGraphState,
    receipt_type: str,
    payload: Mapping[str, Any],
) -> None:
    turn_context = state.get("turn_context")
    if not isinstance(turn_context, Mapping) or not turn_context:
        return
    body = {
        "receipt_type": receipt_type,
        "receipt_version": ANALYSIS_RECEIPT_VERSION,
        "turn_index": int(state.get("turn_index") or 0),
        **deepcopy(dict(payload)),
    }
    body["receipt_id"] = analysis_hash(body)
    context = dict(turn_context)
    receipts = [
        dict(item)
        for item in context.get("control_receipts") or ()
        if isinstance(item, Mapping)
    ]
    if not any(item.get("receipt_id") == body["receipt_id"] for item in receipts):
        receipts.append(body)
    context["control_receipts"] = receipts
    state["turn_context"] = context

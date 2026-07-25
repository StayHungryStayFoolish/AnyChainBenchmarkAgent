"""Pure next-action oracle for the AnyChain Agent Harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .routing import next_group_and_reason

from agent.runners.job_manager import get_job, verify_job_receipt


@dataclass(frozen=True)
class NextAction:
    config_status: str
    next_blocking_group: str
    next_blocking_question_id: str
    next_blocking_reason: str
    execution_status: str
    blockers: tuple[str, ...] = field(default_factory=tuple)
    latest_job_id: str = ""
    job_read_receipt: dict[str, Any] = field(default_factory=dict)


def compute_next_action(state: dict[str, Any]) -> NextAction:
    """Compute the product-facing next action without mutating state."""

    pending = state.get("pending_question") or {}
    pending_id = str(pending.get("id") or "").strip()
    pending_group = str(pending.get("group") or state.get("active_group") or "").strip()
    latest_job_id = str((state.get("job") or {}).get("job_id") or "").strip()
    execution_status, job_read_receipt = _execution_status(state)

    if pending_id:
        return NextAction(
            config_status="in_progress",
            next_blocking_group=pending_group,
            next_blocking_question_id=pending_id,
            next_blocking_reason=_pending_reason(pending),
            execution_status=execution_status,
            blockers=(pending_id,),
            latest_job_id=latest_job_id,
            job_read_receipt=job_read_receipt,
        )

    group, reason = _next_group_and_reason(state)
    # `group in {"job_monitoring", ""}` (from `routing.next_group_and_reason`,
    # the single source of truth for "is this workflow's config complete")
    # is the only correct test here. A prior version also forced
    # config_status to "complete" whenever `execution_status` showed any job
    # status at all -- but the workflow-owned `job` receipt deliberately survives a full
    # reset (`RESET_PRESERVED_KEYS`) so `analyze_report`/`status` keep
    # working for the last completed job, so that shortcut falsely reported
    # a brand-new, still-in-progress workflow as "complete" whenever any
    # unrelated past job happened to exist in state (reproduced live: a
    # fresh sync-observe setup reported `config_status: complete` with
    # `execution_status: job_failed` from an unrelated earlier job, while
    # still correctly naming an unmet next blocking question in the same
    # response).
    config_status = "complete" if group in {"job_monitoring", ""} else "incomplete"
    blockers = () if config_status == "complete" else (reason or group,)
    return NextAction(
        config_status=config_status,
        next_blocking_group=group,
        next_blocking_question_id="",
        next_blocking_reason=reason,
        execution_status=execution_status,
        blockers=tuple(item for item in blockers if item),
        latest_job_id=latest_job_id,
        job_read_receipt=job_read_receipt,
    )


def _next_group_and_reason(state: dict[str, Any]) -> tuple[str, str]:
    """Delegate to `routing.next_group_and_reason`.

    This used to be an independent reimplementation of the same
    precondition chain `routing.next_group_and_reason` uses to drive live turn
    routing. Two hand-maintained copies could (and did, in at least one
    case) disagree; see architecture audit Finding B1.
    """

    return next_group_and_reason(state)


def _execution_status(state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    job = state.get("job") or {}
    smoke = state.get("smoke") or {}
    preflight = state.get("preflight") or {}
    if job.get("status"):
        # `state["job"]` is a one-time snapshot written at submission time
        # (`agent/harness/domains/execution_runtime.py`) and never refreshed -- a
        # `current_config`/status-dump response could claim `job_running` for
        # a job that has actually long since finished, contradicting the
        # deterministic `status`/`jobs`/`logs` commands (which already read
        # correctly from disk via `job_manager`). Prefer the live on-disk
        # status by `job_id` when available; fall back to the snapshot only
        # A checkpoint snapshot is only last-known state. It must not become a
        # verified live status when the persisted job cannot be read.
        job_id = str(job.get("job_id") or "").strip()
        status = str(job.get("status"))
        read_receipt: dict[str, Any] = {}
        if job_id:
            try:
                persisted = get_job(job_id)
                receipts = persisted.get("execution_receipts") or {}
                read_receipt = (
                    dict(receipts.get("last_read") or {})
                    if isinstance(receipts, dict)
                    else {}
                )
                if (
                    verify_job_receipt(read_receipt)
                    and read_receipt.get("job_id") == job_id
                    and read_receipt.get("observed_status") == persisted.get("status")
                ):
                    status = str(read_receipt.get("observed_status") or status)
                else:
                    read_receipt = {}
            except Exception:
                read_receipt = {}
        if read_receipt and status in {
            "running",
            "submitted",
            "completed",
            "failed",
            "partial",
        }:
            return (
                f"job_{status}" if status != "submitted" else "job_submitted",
                read_receipt,
            )
        return ("job_unverified" if job_id else status), {}
    if smoke.get("status"):
        return f"smoke_{smoke.get('status')}", {}
    if preflight.get("status"):
        return f"preflight_{preflight.get('status')}", {}
    if preflight.get("approved"):
        return "approval_recorded", {}
    pending = state.get("pending_question") or {}
    if pending.get("id") == "preflight_smoke_confirm":
        return "approval_pending", {}
    return "not_requested", {}


def _pending_reason(pending: dict[str, Any]) -> str:
    # The full prompt is presentation, not state-summary metadata. Embedding it
    # here causes current-state consultations to repeat the blocking question
    # in the blocker line, the recommendation, and the canonical renderer.
    field = str(pending.get("field") or "").strip()
    question_id = str(pending.get("id") or "").strip()
    subject = field or question_id
    return f"confirm {subject}" if subject else "answer current pending question"

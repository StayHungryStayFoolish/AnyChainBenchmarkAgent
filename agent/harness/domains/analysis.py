"""Evidence and report analysis domain.

This module owns analysis entry points. It reads persisted evidence and job
artifacts, but never mutates benchmark configuration or chooses workflow
groups outside its own domain.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Mapping

from agent.analyzers.result_analyzer import analyze_job
from agent.harness.failures import failure_record_from_job, render_failure_summary
from agent.runners.job_manager import (
    get_job,
    list_jobs,
    resume_job,
    tail_job_log,
    verify_job_receipt,
)
from ..contracts import ActionProposal, HandlerResult, StateDelta
from ..intent import analyze_evidence_with_model
from ..localization import localized
from ..state import AgentGraphState, PendingQuestion
from .analysis_receipts import (
    emit_analysis_invocation_receipt,
    emit_evidence_block_receipt,
    emit_report_analysis_receipt,
    evidence_block_id,
)


ANALYSIS_GROUPS = {"error_evidence_analysis", "report_artifact_analysis"}
JOB_ID_RE = re.compile(r"\bjob_\d{8,}_[0-9a-f]{4,}\b", re.IGNORECASE)

__all__ = [
    "ANALYSIS_GROUPS",
    "JOB_ID_RE",
    "EvidenceCollectionDisposition",
    "EvidenceCollectionOutcome",
    "apply_analysis_action",
    "analyze_saved_evidence_result",
    "analyze_inline_evidence_result",
    "cancel_evidence_collection",
    "continue_evidence_collection",
    "evidence_collection_complete",
    "evidence_help_response",
    "finish_evidence_collection",
    "is_evidence_completion_command",
    "non_empty_evidence_lines",
    "prompt_evidence_collection_waiting",
    "pause_evidence_collection",
    "report_artifact_entry_result",
    "resume_evidence_collection",
    "should_start_evidence_collection",
    "start_evidence_collection",
]

EvidenceCollectionDisposition = Literal["collecting", "saved", "pending_answer", "empty"]


@dataclass(frozen=True)
class EvidenceCollectionOutcome:
    """A collection handler result plus any RPC answer owned by the coordinator.

    Freeform log evidence is fully handled in this domain. RPC/schema evidence
    still belongs to the pending question's workflow domain, so completion
    returns its original question and collected text without applying either.
    """

    result: HandlerResult
    disposition: EvidenceCollectionDisposition
    collected_text: str = ""
    pending_question: Mapping[str, Any] = field(default_factory=dict)


def apply_analysis_action(state: AgentGraphState, action: ActionProposal) -> HandlerResult:
    """Return analysis receipts through the typed domain commit boundary."""

    previous_ids = _control_receipt_ids(state)
    result = _apply_analysis_action(state, action)
    receipts = _control_receipts_since(state, previous_ids)
    if not receipts:
        return result
    return replace(
        result,
        control_receipts=(*result.control_receipts, *receipts),
    )


def _apply_analysis_action(state: AgentGraphState, action: ActionProposal) -> HandlerResult:
    """Apply one evidence/report action without choosing another workflow group."""

    if action.action_type == "start_evidence_collection":
        pending = dict(state.get("pending_question") or {})
        if not pending or str(pending.get("kind") or "") != "evidence":
            return HandlerResult(blocker="start evidence requires an active evidence question")
        return _collection_result(
            start_evidence_collection(
                state,
                str(action.arguments.get("evidence") or ""),
                pending,
            )
        )
    if action.action_type == "append_evidence_collection":
        if not state.get("evidence_collection"):
            return HandlerResult(blocker="append evidence requires an active evidence collection")
        return _collection_result(
            continue_evidence_collection(
                state,
                str(action.arguments.get("evidence") or ""),
                state.get("evidence_collection") or {},
            )
        )
    if action.action_type == "finish_evidence_collection":
        if not state.get("evidence_collection"):
            return HandlerResult(blocker="finish evidence requires an active evidence collection")
        outcome = finish_evidence_collection(
            state,
            state.get("evidence_collection") or {},
        )
        return _collection_result(outcome)
    if action.action_type == "pause_evidence_collection":
        return pause_evidence_collection(state)
    if action.action_type == "resume_evidence_collection":
        return resume_evidence_collection(state)
    if action.action_type == "cancel_evidence_collection":
        return cancel_evidence_collection(state)

    if action.action_type == "analyze_report":
        report_context = dict(state.get("report_context") or {})
        job_id = str(action.arguments.get("job_id") or "").strip()
        subject = str(action.arguments.get("subject") or "").strip()
        if job_id:
            report_context["requested_job_id"] = job_id
        if subject:
            report_context["subject"] = subject
        state_view: AgentGraphState = dict(state)
        state_view["report_context"] = report_context
        result = report_artifact_entry_result(state_view, receipt_state=state)
        return HandlerResult(
            delta=result.delta,
            consumed_action_ids=(action.action_id,),
            evidence=result.evidence,
            visible_result=result.visible_result,
            visible_results=result.visible_results,
            clear_pending=result.clear_pending,
            next_group=result.next_group,
            completion=result.completion,
            stop_after_response=result.stop_after_response,
        )
    if action.action_type == "analyze_evidence":
        evidence = str(action.arguments.get("evidence") or "").strip()
        collecting = state.get("evidence_collection") or {}
        if collecting:
            evidence = "\n".join(str(item) for item in collecting.get("lines") or [] if str(item).strip())
        if not evidence:
            response = evidence_help_response(state)
            emit_analysis_invocation_receipt(
                state,
                source_kind="missing",
                evidence="",
                question=str(action.arguments.get("question") or state.get("last_user_input") or ""),
                visible_result=response,
                invoked=False,
            )
            return HandlerResult(
                consumed_action_ids=(action.action_id,),
                visible_result=response,
                completion="in_progress",
                stop_after_response=True,
            )
        evidence_buffer = [dict(item) for item in list(state.get("evidence_buffer") or [])]
        if not collecting:
            evidence_buffer.append({"text": evidence})
        question = str(action.arguments.get("question") or state.get("last_user_input") or evidence)
        response = analyze_evidence_with_model(state, evidence, question)
        emit_analysis_invocation_receipt(
            state,
            source_kind="active_block" if collecting else "action_argument",
            evidence=evidence,
            question=question,
            visible_result=response,
            invoked=True,
            block_id=(
                evidence_block_id(
                    collecting.get("question") if isinstance(collecting.get("question"), Mapping) else {},
                    [str(item) for item in collecting.get("lines") or ()],
                )
                if collecting
                else ""
            ),
        )
        return HandlerResult(
            delta=StateDelta.set_values({"evidence_buffer": evidence_buffer}) if not collecting else StateDelta(),
            consumed_action_ids=(action.action_id,),
            visible_result=response,
            completion="completed",
            stop_after_response=True,
        )
    return HandlerResult(blocker=f"unsupported analysis action: {action.action_type}")


def _collection_result(outcome: EvidenceCollectionOutcome) -> HandlerResult:
    """Convert collection transport output into the normal graph lifecycle."""

    if outcome.disposition != "pending_answer":
        return outcome.result
    return HandlerResult(
        delta=outcome.result.delta,
        control_receipts=outcome.result.control_receipts,
        clear_pending=False,
        pending_question=dict(outcome.pending_question),
        followup_actions=({
            "type": "answer_pending",
            "answer": outcome.collected_text,
            "source_evidence": outcome.collected_text,
            "confidence": "high",
            "selection_contract_verified": True,
        },),
        completion="completed",
    )


def should_start_evidence_collection(text: str) -> bool:
    """Return whether one partial RPC-evidence line starts collection."""

    raw = str(text or "").strip()
    if "\n" in raw or "\r" in raw:
        return False
    lowered = raw.lower()
    if _parse_rpc_params_or_request(raw)[1] is not None:
        return False
    return (
        raw.endswith("\\")
        or (lowered.startswith("curl ") and "--data" not in lowered)
        or lowered.startswith(("--header ", "--data ", "-h ", "-d "))
        or lowered in {"request:", "response:"}
        or lowered.startswith(("request:", "response:"))
    )


def start_evidence_collection(
    state: AgentGraphState,
    text: str,
    pending: PendingQuestion | Mapping[str, Any],
) -> EvidenceCollectionOutcome:
    """Start collecting RPC/schema evidence without applying the pending answer."""

    lines = non_empty_evidence_lines(text)
    collecting = {
        "question": dict(pending),
        "lines": lines,
        "language": str(state.get("language") or "en"),
        "status": "active",
    }
    block_id = evidence_block_id(dict(pending), lines)
    collecting["block_id"] = block_id
    emit_evidence_block_receipt(
        state,
        operation="start",
        question=dict(pending),
        lines=lines,
        input_text=text,
        input_disposition="accepted",
        status="active",
        block_id=block_id,
    )
    if evidence_collection_complete(lines):
        return finish_evidence_collection(state, collecting)
    return EvidenceCollectionOutcome(
        result=HandlerResult(
            delta=StateDelta.set_values({"evidence_collection": collecting}),
            clear_pending=True,
            visible_result=localized(
                collecting["language"],
                "已开始接收多行 RPC 证据。请继续粘贴 request/response/docs；完成后输入 `END`。我会在完整证据后再解析，不会用半截内容改变协议。",
                "Started collecting multi-line RPC evidence. Continue pasting request/response/docs; type `END` when done. I will parse only complete evidence and will not change protocol from a partial line.",
            ),
            completion="in_progress",
            stop_after_response=True,
        ),
        disposition="collecting",
    )


def continue_evidence_collection(
    state: AgentGraphState,
    text: str,
    collecting: Mapping[str, Any] | None = None,
) -> EvidenceCollectionOutcome:
    """Record one continuation turn or finish the active evidence block."""

    active = collecting if collecting is not None else (state.get("evidence_collection") or {})
    lines = [str(item) for item in list(active.get("lines") or []) if str(item).strip()]
    raw = str(text or "").rstrip("\n")
    question = active.get("question") if isinstance(active.get("question"), dict) else {}
    language = str(active.get("language") or state.get("language") or "en")
    block_id = str(active.get("block_id") or "") or evidence_block_id(question, lines)
    normalized = {
        "question": dict(question),
        "lines": lines,
        "language": language,
        "status": "active",
        "block_id": block_id,
    }

    if is_evidence_completion_command(raw):
        emit_evidence_block_receipt(
            state,
            operation="finish_requested",
            question=question,
            lines=lines,
            input_text=raw,
            input_disposition="completion",
            status="active",
            block_id=block_id,
        )
        return finish_evidence_collection(state, normalized)

    if not raw.strip():
        response = _evidence_waiting_response(language, question)
        emit_evidence_block_receipt(
            state,
            operation="ignore",
            question=question,
            lines=lines,
            input_text=raw,
            input_disposition="blank_ignored",
            status="active",
            visible_result=response,
            block_id=block_id,
        )
        return EvidenceCollectionOutcome(
            result=HandlerResult(
                delta=StateDelta.set_values({"evidence_collection": normalized}),
                visible_result=response,
                completion="in_progress",
                stop_after_response=True,
            ),
            disposition="collecting",
        )

    lines.append(raw)
    normalized["lines"] = lines
    emit_evidence_block_receipt(
        state,
        operation="append",
        question=question,
        lines=lines,
        input_text=raw,
        input_disposition="accepted",
        status="active",
        block_id=block_id,
    )
    if evidence_collection_complete(lines):
        return finish_evidence_collection(state, normalized)
    return EvidenceCollectionOutcome(
        result=HandlerResult(
            delta=StateDelta.set_values({"evidence_collection": normalized}),
            visible_result=localized(
                language,
                f"已记录第 {len(lines)} 行证据。请继续粘贴，完成后输入 `END`。",
                f"Recorded evidence line {len(lines)}. Continue pasting, or type `END` when done.",
            ),
            completion="in_progress",
            stop_after_response=True,
        ),
        disposition="collecting",
    )


def prompt_evidence_collection_waiting(
    state: AgentGraphState,
    collecting: Mapping[str, Any] | None = None,
) -> HandlerResult:
    """Keep an empty continuation turn inside the active collection."""

    active = collecting if collecting is not None else (state.get("evidence_collection") or {})
    question = active.get("question") if isinstance(active.get("question"), dict) else {}
    lines = [str(item) for item in list(active.get("lines") or []) if str(item).strip()]
    language = str(active.get("language") or state.get("language") or "en")
    normalized = {"question": dict(question), "lines": lines, "language": language, "status": "active"}
    block_id = str(active.get("block_id") or "") or evidence_block_id(question, lines)
    normalized["block_id"] = block_id
    response = _evidence_waiting_response(language, question)
    previous_ids = _control_receipt_ids(state)
    emit_evidence_block_receipt(
        state,
        operation="ignore",
        question=question,
        lines=lines,
        input_disposition="blank_ignored",
        status="active",
        visible_result=response,
        block_id=block_id,
    )
    return HandlerResult(
        delta=StateDelta.set_values({"evidence_collection": normalized}),
        control_receipts=_control_receipts_since(state, previous_ids),
        visible_result=response,
        completion="in_progress",
        stop_after_response=True,
    )


def finish_evidence_collection(
    state: AgentGraphState,
    collecting: Mapping[str, Any] | None = None,
    *,
    analysis_requested: bool = False,
) -> EvidenceCollectionOutcome:
    """Finish collection, saving logs or returning an RPC pending-answer value."""

    active = collecting if collecting is not None else (state.get("evidence_collection") or {})
    question = active.get("question") if isinstance(active.get("question"), dict) else {}
    lines = [str(item) for item in list(active.get("lines") or []) if str(item).strip()]
    language = str(active.get("language") or state.get("language") or "en")
    block_id = str(active.get("block_id") or "") or evidence_block_id(question, lines)
    values: dict[str, Any] = {
        "evidence_collection": {},
    }
    if not question or not lines:
        response = localized(
            language,
            "没有可解析的多行证据。请重新提供 request/response/docs，或直接输入 params JSON。",
            "No parseable multi-line evidence was collected. Provide request/response/docs again, or enter params JSON directly.",
        )
        emit_evidence_block_receipt(
            state,
            operation="finish",
            question=question,
            lines=lines,
            input_disposition="completion",
            status="empty",
            visible_result=response,
            block_id=block_id,
        )
        return EvidenceCollectionOutcome(
            result=HandlerResult(
                delta=StateDelta.set_values(values),
                clear_pending=True,
                visible_result=response,
                completion="blocked",
                stop_after_response=True,
            ),
            disposition="empty",
        )

    collected_text = "\n".join(lines)
    if str(question.get("id") or "") == "freeform_evidence":
        evidence_buffer = [dict(item) for item in list(state.get("evidence_buffer") or [])]
        evidence_buffer.append({"text": collected_text})
        values["evidence_buffer"] = evidence_buffer
        response = localized(
            language,
            f"已保存 {len(lines)} 行错误/日志证据。你可以继续问我分析原因，或让我基于这些证据生成修复/重试建议。",
            f"Saved {len(lines)} lines of error/log evidence. You can ask me to analyze the cause or generate a fix/retry plan from it.",
        )
        if analysis_requested:
            response += "\n" + localized(
                language,
                "这类内容会作为错误/日志证据分析；如果你希望继续配置 benchmark，也可以直接说明要回到哪个配置项。",
                "I will analyze this as error/log evidence. If you want to continue benchmark configuration instead, name the configuration area.",
            )
        emit_evidence_block_receipt(
            state,
            operation="finish",
            question=question,
            lines=lines,
            input_disposition="completion",
            status="saved",
            visible_result=response,
            block_id=block_id,
        )
        return EvidenceCollectionOutcome(
            result=HandlerResult(
                delta=StateDelta.set_values(values),
                clear_pending=True,
                visible_result=response,
                completion="completed",
                stop_after_response=True,
            ),
            disposition="saved",
            collected_text=collected_text,
        )

    emit_evidence_block_receipt(
        state,
        operation="finish",
        question=question,
        lines=lines,
        input_disposition="completion",
        status="pending_answer",
        block_id=block_id,
    )
    return EvidenceCollectionOutcome(
        result=HandlerResult(
            delta=StateDelta.set_values(values),
            clear_pending=True,
            completion="completed",
        ),
        disposition="pending_answer",
        collected_text=collected_text,
        pending_question=dict(question),
    )


def cancel_evidence_collection(state: AgentGraphState) -> HandlerResult:
    """Clear only evidence collection state before coordinator-owned routing."""

    collecting = state.get("evidence_collection") or {}
    question = collecting.get("question") if isinstance(collecting.get("question"), Mapping) else {}
    lines = [str(item) for item in collecting.get("lines") or () if str(item).strip()]
    emit_evidence_block_receipt(
        state,
        operation="cancel",
        question=question,
        lines=lines,
        input_disposition="cancel",
        status="cancelled",
        block_id=str(collecting.get("block_id") or "") or evidence_block_id(question, lines),
    )
    return HandlerResult(
        delta=StateDelta.set_values({"evidence_collection": {}}),
        completion="unchanged",
    )


def pause_evidence_collection(state: AgentGraphState) -> HandlerResult:
    """Suspend collection transport while preserving its buffered evidence."""

    collecting = dict(state.get("evidence_collection") or {})
    if not collecting:
        return HandlerResult(blocker="pause evidence requires an active evidence collection")
    collecting["status"] = "paused"
    question = collecting.get("question") if isinstance(collecting.get("question"), Mapping) else {}
    lines = [str(item) for item in collecting.get("lines") or () if str(item).strip()]
    block_id = str(collecting.get("block_id") or "") or evidence_block_id(question, lines)
    collecting["block_id"] = block_id
    emit_evidence_block_receipt(
        state,
        operation="pause",
        question=question,
        lines=lines,
        input_disposition="pause",
        status="paused",
        block_id=block_id,
    )
    return HandlerResult(
        delta=StateDelta.set_values({"evidence_collection": collecting}),
        completion="completed",
    )


def resume_evidence_collection(state: AgentGraphState) -> HandlerResult:
    """Reactivate a preserved collection without changing its original question."""

    collecting = dict(state.get("evidence_collection") or {})
    if not collecting or str(collecting.get("status") or "active") != "paused":
        return HandlerResult(blocker="resume evidence requires a paused evidence collection")
    collecting["status"] = "active"
    language = str(collecting.get("language") or state.get("language") or "en")
    lines = [str(item) for item in collecting.get("lines") or []]
    question = collecting.get("question") if isinstance(collecting.get("question"), Mapping) else {}
    block_id = str(collecting.get("block_id") or "") or evidence_block_id(question, lines)
    collecting["block_id"] = block_id
    response = localized(
        language,
        f"已恢复证据收集，当前保留 {len(lines)} 行。请继续粘贴，完成后输入 `END`。",
        f"Resumed evidence collection with {len(lines)} saved line(s). Continue pasting, or type `END` when done.",
    )
    emit_evidence_block_receipt(
        state,
        operation="resume",
        question=question,
        lines=lines,
        input_disposition="resume",
        status="active",
        visible_result=response,
        block_id=block_id,
    )
    return HandlerResult(
        delta=StateDelta.set_values({"evidence_collection": collecting}),
        visible_result=response,
        completion="in_progress",
        stop_after_response=True,
    )


def non_empty_evidence_lines(text: str) -> list[str]:
    """Split an initial paste into non-empty evidence lines."""

    lines = [line.rstrip("\n") for line in str(text or "").splitlines() if line.strip()]
    return lines or [str(text or "").strip()]


def evidence_collection_complete(lines: list[str]) -> bool:
    """Return whether RPC request and response evidence is self-completing."""

    text = "\n".join(lines)
    lowered = text.lower()
    if "response" in lowered and ("--data" in lowered or '"method"' in lowered or "'method'" in lowered):
        return True
    if "jsonrpc" in lowered and '"result"' in lowered and ('"method"' in lowered or "'method'" in lowered):
        return True
    return False


def is_evidence_completion_command(text: str) -> bool:
    return str(text or "").strip().upper() in {"END", "DONE", "结束"}


def analyze_inline_evidence_result(state: AgentGraphState, text: str) -> HandlerResult:
    """Analyze one complete evidence-bearing question without opening paste mode."""

    evidence = str(text or "").strip()
    evidence_buffer = [dict(item) for item in list(state.get("evidence_buffer") or [])]
    evidence_buffer.append({"text": evidence})
    response = analyze_evidence_with_model(state, evidence, evidence)
    emit_analysis_invocation_receipt(
        state,
        source_kind="inline",
        evidence=evidence,
        question=evidence,
        visible_result=response,
        invoked=True,
    )
    return HandlerResult(
        delta=StateDelta.set_values({"evidence_buffer": evidence_buffer}),
        visible_result=response,
        next_group="" if state.get("pending_question") else "error_evidence_analysis",
        completion="completed",
        stop_after_response=True,
    )


def evidence_help_response(state: AgentGraphState) -> str:
    return localized(
        state.get("language", "en"),
        "可以。请粘贴真实日志、错误栈或命令输出；多行内容会作为一个证据块接收，完成后输入 `END`。也可以指定 `job_id`，让我读取已有日志和产物。",
        "Yes. Paste real logs, a stack trace, or command output; multiline content is collected as one evidence block and ends with `END`. You may also name a `job_id` so I can read existing logs and artifacts.",
    )


def analyze_saved_evidence_result(state: AgentGraphState, user_question: str) -> HandlerResult:
    """Analyze the newest saved evidence block without mutating graph state."""

    evidence_items = list(state.get("evidence_buffer") or [])
    evidence = str((evidence_items[-1] if evidence_items else {}).get("text") or "").strip()
    response = analyze_evidence_with_model(state, evidence, user_question)
    emit_analysis_invocation_receipt(
        state,
        source_kind="saved_buffer",
        evidence=evidence,
        question=user_question,
        visible_result=response,
        invoked=True,
    )
    return HandlerResult(
        clear_pending=True,
        next_group="error_evidence_analysis",
        visible_result=response,
        completion="completed",
        stop_after_response=True,
    )


def report_artifact_entry_response(state: AgentGraphState) -> str:
    """Return a factual report response while preserving the public string API."""

    return _report_artifact_entry(state)[0]


def _report_artifact_entry(state: AgentGraphState) -> tuple[str, dict[str, Any]]:
    language = state.get("language", "en")
    report_context = state.get("report_context") or {}
    job_id = str(report_context.get("requested_job_id") or "").strip()
    if not job_id:
        explicit_reference = JOB_ID_RE.search(str(state.get("last_user_input") or ""))
        job_id = explicit_reference.group(0) if explicit_reference else ""
    try:
        jobs = list_jobs(limit=1)
        if not job_id:
            job_id = str(jobs[0].get("job_id") or "") if jobs else ""
    except Exception:
        job_id = ""
    if not job_id:
        job_id = str((state.get("job") or {}).get("job_id") or "").strip()
    if not job_id:
        return localized(
            language,
            "没有找到历史 job。请先运行 fake-node smoke、real-node benchmark 或 sync-observe，再分析报告。",
            "No historical job was found. Run fake-node smoke, real-node benchmark, or sync-observe before analyzing a report.",
        ), {}
    try:
        summary = resume_job(job_id)
    except Exception as exc:
        return localized(
            language,
            f"无法读取 job `{job_id}`：{type(exc).__name__}。请用 `jobs` 查看可用任务，或提供正确的 job_id。",
            f"Could not read job `{job_id}`: {type(exc).__name__}. Use `jobs` to list available jobs, or provide the correct job_id.",
        ), {}
    try:
        job = get_job(job_id)
    except (FileNotFoundError, OSError, ValueError):
        job = {
            "job_id": job_id,
            "status": summary.get("status", "unknown"),
            "run_dir": summary.get("run_dir", ""),
            "plan_file": summary.get("plan_file", ""),
            "artifacts": {},
            "artifact_index": summary.get("artifact_index", ""),
            "runtime_env_file": summary.get("runtime_env_file", ""),
        }
    facts = _persisted_job_facts(job, summary)
    response = _render_persisted_job_facts(facts, str(language))
    return response, job


def report_artifact_entry_result(
    state: AgentGraphState,
    *,
    receipt_state: AgentGraphState | None = None,
) -> HandlerResult:
    """Read a requested/latest job without claiming execution-state authority."""

    report_context = dict(state.get("report_context") or {})
    state_view: AgentGraphState = dict(state)
    state_view["report_context"] = report_context
    response, persisted_job = _report_artifact_entry(state_view)
    execution_receipts = (
        persisted_job.get("execution_receipts")
        if isinstance(persisted_job.get("execution_receipts"), Mapping)
        else {}
    )
    read_receipt = (
        dict(execution_receipts.get("last_read") or {})
        if isinstance(execution_receipts, Mapping)
        and isinstance(execution_receipts.get("last_read"), Mapping)
        else {}
    )
    evidence_verified = bool(
        verify_job_receipt(read_receipt)
        and read_receipt.get("job_id") == persisted_job.get("job_id")
        and read_receipt.get("observed_status") == persisted_job.get("status")
    )
    if persisted_job:
        report_context["analyzed_job_id"] = str(persisted_job.get("job_id") or "")
        report_context["analyzed_job_status"] = str(persisted_job.get("status") or "unknown")
    emit_report_analysis_receipt(
        receipt_state if receipt_state is not None else state,
        requested_job_id=str(report_context.get("requested_job_id") or ""),
        resolved_job_id=str(persisted_job.get("job_id") or ""),
        resolved_status=str(persisted_job.get("status") or ""),
        visible_result=response,
        invoked=bool(persisted_job) and evidence_verified,
        job_read_receipt_id=(
            str(read_receipt.get("receipt_id") or "")
            if evidence_verified
            else ""
        ),
        evidence_verified=evidence_verified,
    )
    return HandlerResult(
        delta=StateDelta.set_values({"report_context": report_context}),
        clear_pending=True,
        next_group="report_artifact_analysis",
        visible_result=response,
        completion="completed",
        stop_after_response=True,
    )


def _evidence_waiting_response(language: str, question: Mapping[str, Any]) -> str:
    if str(question.get("id") or "") == "freeform_evidence":
        return localized(
            language,
            "还没有收到新的日志内容。请继续粘贴真实日志/错误栈，完成后输入 `END`；如果要退出日志分析，可以直接说明要回到哪个配置项。",
            "No new log content was received. Paste the actual log/stack trace and type `END` when done; to leave log analysis, name the configuration area to return to.",
        )
    return localized(
        language,
        "还没有收到新的 RPC 证据。请继续粘贴 request/response/docs，完成后输入 `END`；如果要退出，请直接说明要回到哪个配置项。",
        "No new RPC evidence was received. Continue pasting request/response/docs and type `END` when done; to leave this flow, name the configuration area to return to.",
    )


def _control_receipt_ids(state: Mapping[str, Any]) -> set[str]:
    return {
        str(item.get("receipt_id") or "")
        for item in (state.get("turn_context") or {}).get("control_receipts") or ()
        if isinstance(item, Mapping) and str(item.get("receipt_id") or "")
    }


def _control_receipts_since(
    state: Mapping[str, Any],
    previous_ids: set[str],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        deepcopy(dict(item))
        for item in (state.get("turn_context") or {}).get("control_receipts") or ()
        if isinstance(item, Mapping)
        and str(item.get("receipt_id") or "")
        and str(item.get("receipt_id") or "") not in previous_ids
    )


def _persisted_job_facts(job: dict[str, Any], resume_summary: dict[str, Any]) -> dict[str, Any]:
    """Build report facts exclusively from persisted job artifacts and plan."""

    artifacts = dict(job.get("artifacts") or {})
    plan = _read_json_file(job.get("plan_file"))
    archive_summary = _read_json_file(artifacts.get("summary_json"))
    analysis = analyze_job(job)
    failure_record = (
        failure_record_from_job(job)
        if str(job.get("status") or "") in {"failed", "partial"}
        else {}
    )
    execution = plan.get("execution") if isinstance(plan.get("execution"), dict) else {}
    execution_env = execution.get("environment") if isinstance(execution.get("environment"), dict) else {}
    workload = plan.get("workload") if isinstance(plan.get("workload"), dict) else {}
    methods = list(workload.get("methods") or [])
    if not methods:
        override = plan.get("chain_config_override") if isinstance(plan.get("chain_config_override"), dict) else {}
        rpc_methods = override.get("rpc_methods") if isinstance(override.get("rpc_methods"), dict) else {}
        rpc_mode = str(plan.get("rpc_mode") or execution_env.get("RPC_MODE") or "")
        if rpc_mode == "single" and rpc_methods.get("single"):
            methods = [str(rpc_methods["single"])]
        elif rpc_methods.get("mixed_weighted"):
            methods = [str(item.get("method")) for item in rpc_methods["mixed_weighted"] if item.get("method")]
    return {
        "job_id": str(job.get("job_id") or ""),
        "status": str(job.get("status") or resume_summary.get("status") or "unknown"),
        "chain": str(plan.get("chain") or execution_env.get("BLOCKCHAIN_NODE") or "<unknown>"),
        "target_mode": "fake-node" if plan.get("use_fake_node") else ("sync-observe" if plan.get("workflow_type") == "sync_observe" else "real-node"),
        "workflow_type": str(plan.get("workflow_type") or plan.get("run_mode") or ""),
        "rpc_mode": str(plan.get("rpc_mode") or execution_env.get("RPC_MODE") or ""),
        "methods": methods,
        "benchmark_mode": str(plan.get("benchmark_mode") or archive_summary.get("benchmark_mode") or ""),
        "initial_qps": archive_summary.get("test_parameters", {}).get("initial_qps", execution_env.get("QUICK_INITIAL_QPS", "")),
        "max_qps": archive_summary.get("max_successful_qps", execution_env.get("QUICK_MAX_QPS", "")),
        "duration": archive_summary.get("test_parameters", {}).get("duration_per_level", execution_env.get("QUICK_DURATION", "")),
        "grade": str(analysis.get("grade") or "INCONCLUSIVE"),
        "bottleneck_detected": archive_summary.get("bottleneck_detected"),
        "artifacts": artifacts,
        "artifact_index": str(job.get("artifact_index") or resume_summary.get("artifact_index") or ""),
        "run_dir": str(job.get("run_dir") or ""),
        "error": str(job.get("error") or ""),
        "failure_record": failure_record,
        "log_highlights": _report_log_highlights(str(job.get("job_id") or ""), str(job.get("status") or "")),
    }


def _render_persisted_job_facts(facts: dict[str, Any], language: str) -> str:
    methods = ", ".join(str(item) for item in facts.get("methods") or []) or "<unknown>"
    artifacts = facts.get("artifacts") if isinstance(facts.get("artifacts"), dict) else {}
    status = str(facts.get("status") or "unknown")
    completed = status == "completed" and bool(artifacts.get("html_report") and artifacts.get("performance_csv"))
    terminal = status in {"completed", "failed", "partial", "cancelled"}
    target_mode = str(facts.get("target_mode") or "")
    if language.startswith("zh"):
        lines = [
            f"任务 `{facts['job_id']}`：{status}（{facts['grade']}）。",
            f"- 实际执行：{facts['chain']} / {target_mode} / {facts['rpc_mode'] or '<unknown>'} / {methods}",
            f"- 测试配置：{facts['benchmark_mode'] or '<unknown>'}，{facts['initial_qps']} 到 {facts['max_qps']} QPS，每档 {facts['duration']} 秒",
        ]
        if completed:
            lines.append("- 闭环结果：执行、日志、归档和 HTML 报告均已生成。")
        if facts.get("error"):
            lines.append(f"- 错误：{facts['error']}")
        if facts.get("failure_record"):
            lines.append(render_failure_summary(facts["failure_record"], language))
        if facts.get("log_highlights"):
            lines.append(f"- 日志证据：\n{facts['log_highlights']}")
        if target_mode == "fake-node":
            lines.append("- 结论边界：这次只证明 Agent、fixture、请求生成、Vegeta、采集、归档和报告链路可以闭环；不能代表真实节点的 QPS 上限、真实网络延迟、同步速度或硬件瓶颈。")
        if terminal:
            lines.extend([
                f"- HTML 报告：{artifacts.get('html_report') or '<not found>'}",
                f"- 测试摘要：{artifacts.get('summary_json') or '<not found>'}",
                f"- 性能数据：{artifacts.get('performance_csv') or '<not found>'}",
                f"- artifact_index：{facts.get('artifact_index') or '<not found>'}",
            ])
        else:
            lines.append("- 报告产物：任务完成后生成，当前不能据此判定通过或失败。")
        lines.append(f"- 运行日志：{facts.get('run_dir') or '<unknown>'}/benchmark.log")
        return "\n".join(lines)
    lines = [
        f"Job `{facts['job_id']}`: {status} ({facts['grade']}).",
        f"- Actual run: {facts['chain']} / {target_mode} / {facts['rpc_mode'] or '<unknown>'} / {methods}",
        f"- Profile: {facts['benchmark_mode'] or '<unknown>'}, {facts['initial_qps']} to {facts['max_qps']} QPS, {facts['duration']}s per level",
    ]
    if completed:
        lines.append("- Closed-loop result: execution, logs, archive, and HTML report were produced.")
    if facts.get("error"):
        lines.append(f"- Error: {facts['error']}")
    if facts.get("failure_record"):
        lines.append(render_failure_summary(facts["failure_record"], language))
    if facts.get("log_highlights"):
        lines.append(f"- Log evidence:\n{facts['log_highlights']}")
    if target_mode == "fake-node":
        lines.append("- Limitation: this proves only the Agent, fixture, request generation, Vegeta, collection, archive, and report loop. It does not establish real-node QPS capacity, network latency, sync speed, or hardware bottlenecks.")
    if terminal:
        lines.extend([
            f"- HTML report: {artifacts.get('html_report') or '<not found>'}",
            f"- summary: {artifacts.get('summary_json') or '<not found>'}",
            f"- performance data: {artifacts.get('performance_csv') or '<not found>'}",
            f"- artifact_index: {facts.get('artifact_index') or '<not found>'}",
        ])
    else:
        lines.append("- Report artifacts: generated after completion; no pass/fail conclusion is available yet.")
    lines.append(f"- run log: {facts.get('run_dir') or '<unknown>'}/benchmark.log")
    return "\n".join(lines)


def _read_json_file(path_value: Any) -> dict[str, Any]:
    path = Path(str(path_value or ""))
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_rpc_params_or_request(value: str) -> tuple[str, Any | None]:
    """Parse only enough request structure to avoid starting partial collection."""

    text = str(value or "").strip()
    if not text:
        return "", None
    if _declares_no_rpc_params(text):
        return "", []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
        for candidate in _json_objects_or_arrays_in_text(text):
            if isinstance(candidate, dict) and "params" in candidate and ("method" in candidate or "jsonrpc" in candidate):
                parsed = candidate
                break
            if isinstance(candidate, list):
                parsed = candidate
                break
        if parsed is None:
            return "", None
    if isinstance(parsed, dict) and "params" in parsed and ("method" in parsed or "jsonrpc" in parsed):
        params = parsed.get("params")
        return str(parsed.get("method") or "").strip(), params if isinstance(params, (list, dict)) else None
    if isinstance(parsed, list):
        return "", parsed
    if isinstance(parsed, dict) and set(parsed).issubset({"params", "arguments", "args"}):
        params = parsed.get("params", parsed.get("arguments", parsed.get("args")))
        return "", params if isinstance(params, (list, dict)) else None
    return "", None


def _declares_no_rpc_params(value: str) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    return any(
        marker in text
        for marker in (
            "no parameters",
            "no params",
            "without parameters",
            "without params",
            "params: none",
            "params none",
            "没有参数",
            "无参数",
            "不需要参数",
            "参数为空",
        )
    )


def _json_objects_or_arrays_in_text(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    output: list[Any] = []
    raw = str(text or "")
    for index, char in enumerate(raw):
        if char not in "{[":
            continue
        try:
            parsed, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, (dict, list)):
            output.append(parsed)
    return output


def _report_log_highlights(job_id: str, status: str, *, max_lines: int = 6) -> str:
    try:
        log = tail_job_log(job_id, lines=200)
    except Exception:
        return ""
    lines = [str(line).strip() for line in (log.get("lines") or []) if str(line).strip()]
    if not lines:
        return ""
    error_markers = ("error", "failed", "traceback", "exception", "exit status", "exit code", "not found", "❌")
    success_markers = ("success", "completed", "pass", "✅")
    markers = error_markers if status == "failed" else error_markers + success_markers
    relevant = [line for line in lines if any(marker in line.casefold() for marker in markers)]
    selected = relevant[-max_lines:] if relevant else lines[-max_lines:]
    return "\n".join(f"  {line}" for line in selected)

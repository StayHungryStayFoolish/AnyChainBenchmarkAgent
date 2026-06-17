"""Deterministic context compaction for terminal chat sessions."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from memory.token_estimator import (
    DEFAULT_COMPACT_KEEP_RECENT_TURNS,
    DEFAULT_COMPACT_TRIGGER_RATIO,
    DEFAULT_COMPACT_TURN_THRESHOLD,
    DEFAULT_CONTEXT_WINDOW_TOKENS,
    compact_token_threshold,
    estimate_tokens,
)


def compact_session_state(
    *,
    turns: list[dict[str, str]],
    request: dict[str, Any] | None,
    plan: dict[str, Any] | None,
    job: dict[str, Any] | None,
    discovery: dict[str, Any] | None,
    previous_summary: dict[str, Any] | None = None,
    keep_recent: int = DEFAULT_COMPACT_KEEP_RECENT_TURNS,
    context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS,
    trigger_ratio: float = DEFAULT_COMPACT_TRIGGER_RATIO,
    turn_threshold: int = DEFAULT_COMPACT_TURN_THRESHOLD,
    reason: str = "manual",
) -> dict[str, Any]:
    """Create a compact, structured session memory snapshot."""
    preserved = {
        "chain": _first_non_empty(_get(plan, "chain"), _get(request, "chain")),
        "goal": _first_non_empty(_get(plan, "goal"), _get(request, "goal")),
        "strategy": _get(plan, "strategy"),
        "rpc_mode": _first_non_empty(_get(plan, "rpc_mode"), _get(request, "rpc_mode")),
        "use_fake_node": _first_non_empty(_get(plan, "use_fake_node"), _get(request, "use_fake_node")),
        "plan_id": _get(plan, "plan_id"),
        "job_id": _get(job, "job_id"),
        "job_status": _get(job, "status"),
        "artifact_index": _get(job, "artifact_index"),
        "runtime_env_file": _first_non_empty(_get(job, "runtime_env_file"), _get(job, "artifacts.runtime_env_file")),
        "deployment_type": _get(discovery, "deployment.type"),
        "cloud_provider": _get(discovery, "cloud.provider"),
    }
    important_facts = [f"{key}: {value}" for key, value in preserved.items() if value not in (None, "", [], {})]
    open_questions = _open_questions(plan)
    recent_turns = deepcopy(turns[-keep_recent:])
    compacted_turn_count = max(0, len(turns) - len(recent_turns))
    previous_notes = previous_summary.get("summary") if isinstance(previous_summary, dict) else ""
    summary_text = _summary_text(preserved, compacted_turn_count, reason, previous_notes)
    token_estimate = estimate_tokens(str(turns)) + estimate_tokens(str(request)) + estimate_tokens(str(plan)) + estimate_tokens(str(job))
    return {
        "version": 1,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reason": reason,
        "summary": summary_text,
        "previous_summary": previous_notes,
        "preserved_state": preserved,
        "important_facts": important_facts,
        "open_questions": open_questions,
        "recent_turns": recent_turns,
        "compacted_turn_count": compacted_turn_count,
        "token_estimate_before": token_estimate,
        "thresholds": {
            "context_window_tokens": context_window_tokens,
            "trigger_ratio": trigger_ratio,
            "token_threshold": compact_token_threshold(context_window_tokens, trigger_ratio),
            "turn_threshold": turn_threshold,
            "keep_recent_turns": keep_recent,
        },
    }


def should_auto_compact(
    *,
    turns: list[dict[str, str]],
    context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS,
    trigger_ratio: float = DEFAULT_COMPACT_TRIGGER_RATIO,
    turn_threshold: int = DEFAULT_COMPACT_TURN_THRESHOLD,
) -> bool:
    if len(turns) >= turn_threshold:
        return True
    return estimate_tokens(str(turns)) >= compact_token_threshold(context_window_tokens, trigger_ratio)


def _summary_text(preserved: dict[str, Any], compacted_turn_count: int, reason: str, previous_notes: str) -> str:
    parts = [f"Compacted {compacted_turn_count} older chat turns ({reason})."]
    chain = preserved.get("chain")
    strategy = preserved.get("strategy")
    rpc_mode = preserved.get("rpc_mode")
    if chain or strategy or rpc_mode:
        parts.append(f"Active plan: chain={chain or 'unknown'}, strategy={strategy or 'unknown'}, rpc_mode={rpc_mode or 'unknown'}.")
    if preserved.get("job_id"):
        parts.append(f"Latest job: {preserved['job_id']} ({preserved.get('job_status') or 'unknown'}).")
    if previous_notes:
        parts.append(f"Previous summary: {previous_notes}")
    return " ".join(parts)


def _open_questions(plan: dict[str, Any] | None) -> list[str]:
    if not plan:
        return []
    questions = []
    for question in plan.get("required_questions", []):
        qid = question.get("id")
        prompt = question.get("prompt")
        severity = question.get("severity")
        if qid:
            questions.append(f"{qid} ({severity or 'unknown'}): {prompt or ''}".strip())
    return questions


def _get(payload: dict[str, Any] | None, path: str) -> Any:
    if not payload:
        return None
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None

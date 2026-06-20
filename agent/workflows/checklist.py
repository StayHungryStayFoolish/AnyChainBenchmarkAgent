"""Interactive checklist rendering and answer handling."""

from __future__ import annotations

from typing import Any


def next_blocker(plan: dict[str, Any] | None) -> dict[str, Any] | None:
    if not plan:
        return None
    for question in plan.get("required_questions", []):
        if question.get("severity") == "blocker":
            return question
    return None


def checklist_summary(plan: dict[str, Any] | None) -> dict[str, Any]:
    """Return a stable checklist view for chat and external callers."""
    if not plan:
        return {
            "has_plan": False,
            "blockers": [],
            "warnings": [],
            "next_question": None,
            "passed": False,
        }
    required_questions = plan.get("required_questions", [])
    blockers = [question for question in required_questions if question.get("severity") == "blocker"]
    warnings = [question for question in required_questions if question.get("severity") != "blocker"]
    return {
        "has_plan": True,
        "blockers": blockers,
        "warnings": warnings,
        "next_question": blockers[0] if blockers else None,
        "passed": not blockers,
    }


def format_checklist(plan: dict[str, Any] | None) -> str:
    summary = checklist_summary(plan)
    if not summary["has_plan"]:
        return "No active checklist yet. Describe the benchmark you want to run."
    if summary["passed"]:
        return "Checklist passed. No blocking configuration item is waiting for an answer."
    lines = [
        "Benchmark checklist needs input before a trusted run.",
        f"- blocking_items: {len(summary['blockers'])}",
        f"- warning_items: {len(summary['warnings'])}",
    ]
    next_question = summary["next_question"] or {}
    qid = next_question.get("id", "<unknown>")
    prompt = next_question.get("prompt") or "Provide the missing value."
    lines.append(f"- next_question: `{qid}`")
    lines.append(f"- prompt: {prompt}")
    candidates = next_question.get("candidates") or next_question.get("options") or []
    if candidates:
        lines.append("- candidates: " + ", ".join(str(item) for item in candidates[:8]))
    lines.append("Reply with the value directly, or type `answer <value>`.")
    return "\n".join(lines)


def apply_checklist_answer(request: dict[str, Any], question: dict[str, Any], answer: str) -> tuple[dict[str, Any], list[str]]:
    updated = dict(request)
    changes: list[str] = []
    qid = question.get("id", "")
    value = answer.strip()
    if not value:
        return updated, changes
    if qid == "chain":
        updated["chain"] = value.lower()
        changes.append(f"chain -> {updated['chain']}")
    elif qid == "local_rpc_url":
        if value.lower() in {"fake-node", "fake node", "mock"}:
            updated["use_fake_node"] = True
            changes.append("use_fake_node -> true")
        else:
            updated["local_rpc_url"] = value
            changes.append(f"local_rpc_url -> {value}")
    elif qid in {"ledger_device", "ledger_device_confirmation"}:
        updated["ledger_device"] = value
        _confirm(updated, "ledger_device_confirmation")
        changes.append(f"ledger_device -> {value}")
    elif qid == "blockchain_process_names":
        names = [item.strip() for item in value.replace(",", " ").split() if item.strip()]
        updated["blockchain_process_names"] = names
        changes.append("blockchain_process_names -> " + " ".join(names))
    elif qid == "data_vol_max_iops":
        updated["data_vol_max_iops"] = value
        changes.append(f"data_vol_max_iops -> {value}")
    elif qid == "data_vol_max_throughput":
        updated["data_vol_max_throughput"] = value
        changes.append(f"data_vol_max_throughput -> {value}")
    elif qid == "network_max_bandwidth_gbps":
        updated["network_max_bandwidth_gbps"] = value
        changes.append(f"network_max_bandwidth_gbps -> {value}")
    elif qid == "rpc_mode":
        lowered = value.lower()
        if lowered in {"single", "mixed"}:
            updated["rpc_mode"] = lowered
            changes.append(f"rpc_mode -> {lowered}")
    elif qid == "mixed_weights_confirmation":
        if value.lower() in {"y", "yes", "confirm", "confirmed", "true"}:
            _confirm(updated, qid)
            changes.append("mixed_weights_confirmation -> confirmed")
    elif qid == "stress_execution_confirmation":
        if value.lower() in {"y", "yes", "confirm", "confirmed", "true"}:
            _confirm(updated, qid)
            changes.append("stress_execution_confirmation -> confirmed")
    return updated, changes


def _confirm(request: dict[str, Any], confirmation: str) -> None:
    confirmations = set(request.get("confirmations", []))
    confirmations.add(confirmation)
    request["confirmations"] = sorted(confirmations)

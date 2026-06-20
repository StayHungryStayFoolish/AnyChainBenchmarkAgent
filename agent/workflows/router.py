"""Workflow selection from intent and session state."""

from __future__ import annotations

from typing import Any

from workflows.checklist import next_blocker


COMMAND_WORKFLOWS = {
    "doctor": "doctor",
    "diagnose": "doctor",
    "readiness": "doctor",
    "check readiness": "doctor",
    "discover": "discover",
    "detect environment": "discover",
    "plan": "show_plan",
    "show plan": "show_plan",
    "current plan": "show_plan",
    "checklist": "checklist",
    "show checklist": "checklist",
    "next": "checklist",
    "next question": "checklist",
    "blockers": "checklist",
    "preflight": "preflight",
    "check plan": "preflight",
    "risk": "risk",
    "risk score": "risk",
    "score risk": "risk",
    "runbook": "runbook",
    "show runbook": "runbook",
    "run mock": "submit_mock",
    "mock": "submit_mock",
    "submit mock": "submit_mock",
    "execute mock": "submit_mock",
    "run": "submit_real",
    "execute": "submit_real",
    "submit": "submit_real",
    "run real": "submit_real",
    "execute real": "submit_real",
    "status": "status",
    "job status": "status",
    "logs": "logs",
    "log": "logs",
    "tail logs": "logs",
    "benchmark logs": "logs",
    "analyze": "analyze",
    "analyse": "analyze",
    "analyze result": "analyze",
    "analysis": "analyze",
    "trace": "trace",
    "workflow trace": "trace",
}


def select_workflow(
    text: str,
    *,
    command: str,
    route: dict[str, Any],
    has_plan: bool,
    has_job: bool,
    plan: dict[str, Any] | None,
    is_plan_edit: bool,
) -> str:
    if command in COMMAND_WORKFLOWS:
        return COMMAND_WORKFLOWS[command]
    if command.startswith("yes run") or command.startswith("confirm run"):
        return "submit_real_confirmed"
    if command.startswith("ask "):
        return "framework_question"
    if command.startswith("qa "):
        return "artifact_analysis"
    if command.startswith("answer "):
        return "checklist_answer"
    if (
        has_plan
        and next_blocker(plan)
        and not _is_high_confidence_non_checklist_route(route)
        and not _looks_like_new_request(text)
        and not is_plan_edit
    ):
        return "checklist_answer"
    if has_plan and is_plan_edit:
        return "plan_edit"
    intent = route.get("intent", "")
    if intent == "out_of_scope":
        return "out_of_scope"
    if intent == "artifact_question" and has_job:
        return "artifact_analysis"
    if intent == "plan_edit" and has_plan:
        return "plan_edit"
    if intent == "benchmark_request":
        return "benchmark_request"
    if intent == "onboarding_request":
        return "onboarding_request"
    if intent == "framework_question":
        return "framework_question"
    return "framework_question"


def prompt_bundle_for_workflow(workflow: str) -> str:
    return {
        "benchmark_request": "request",
        "framework_question": "kb_answer",
        "artifact_analysis": "artifact_analysis",
        "onboarding_request": "onboarding",
        "plan_edit": "request",
        "checklist_answer": "request",
        "out_of_scope": "",
    }.get(workflow, "")


def _looks_like_new_request(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("benchmark", "test ", "run ", "create ", "压测", "测试"))


def _is_high_confidence_non_checklist_route(route: dict[str, Any]) -> bool:
    intent = route.get("intent")
    confidence = float(route.get("confidence", 0) or 0)
    if intent in {"artifact_question", "out_of_scope"}:
        return True
    if intent == "framework_question" and confidence >= 0.65:
        return True
    return False

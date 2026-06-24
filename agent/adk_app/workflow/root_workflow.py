"""Offline-safe root workflow contract for the ADK workflow migration.

This module defines the deterministic node order and event contract that the
product CLI and tests can exercise without live model credentials. The same
event payloads are the boundary between the product terminal and ADK workflow
agents.
"""

from __future__ import annotations

from typing import Any

from adk_app.agents.router import route_user_intent
from adk_app.workflow.schemas import WorkflowEvent
from diagnostics.doctor import run_doctor
from knowledge.framework_context import load_framework_context
from knowledge.gap_analyzer import analyze_capability_gap
from runners.job_manager import list_jobs


def root_workflow_dry_run(user_text: str, language: str = "en") -> dict[str, Any]:
    """Return the root workflow events for one user utterance."""
    events: list[WorkflowEvent] = []
    doctor = run_doctor()
    events.append(WorkflowEvent(
        event_type="startup_doctor",
        message="startup doctor completed",
        data={
            "status": doctor.get("status"),
            "missing_required": doctor.get("environment", {}).get("dependencies", {}).get("missing_required", []),
        },
    ))
    context = load_framework_context(language=language)
    events.append(WorkflowEvent(
        event_type="framework_context_loaded",
        message="compact framework context loaded",
        data={
            "capability_summary": context.get("capability_summary", {}),
            "authoritative_docs": context.get("authoritative_docs", []),
            "context_policy": context.get("context_policy", {}),
        },
    ))
    jobs = list_jobs(limit=1)
    events.append(WorkflowEvent(
        event_type="session_resume",
        message="latest job checked",
        data={"latest_job": jobs[0] if jobs else {}},
    ))
    route = route_user_intent(user_text, default_language=language)
    events.append(WorkflowEvent(
        event_type="intent_route",
        message="intent routed",
        data=route,
    ))
    intent = route["intent"]
    if intent == "START_BENCHMARK":
        events.append(WorkflowEvent(
            event_type="benchmark_workflow_selected",
            message="benchmark workflow selected",
            data={"entities": route["entities"], "missing": route["missing_clarifications"]},
            requires_user_input=bool(route["missing_clarifications"]),
        ))
    elif intent == "ONBOARD_CHAIN_RPC":
        chain = route["entities"].get("chain", "")
        methods = route["entities"].get("rpc_methods", [])
        events.append(WorkflowEvent(
            event_type="onboarding_workflow_selected",
            message="onboarding workflow selected",
            data=analyze_capability_gap(chain, methods),
        ))
    elif intent == "RESUME_JOB":
        events.append(WorkflowEvent(
            event_type="job_workflow_selected",
            message="job workflow selected",
            data={"entities": route["entities"], "latest_job": jobs[0] if jobs else {}},
            requires_user_input=not bool(route["entities"].get("job_id") or jobs),
        ))
    elif intent == "ANALYZE_ARTIFACTS":
        events.append(WorkflowEvent(
            event_type="analysis_workflow_selected",
            message="analysis workflow selected",
            data={"entities": route["entities"], "latest_job": jobs[0] if jobs else {}},
            requires_user_input=not bool(route["entities"].get("job_id") or jobs),
        ))
    else:
        events.append(WorkflowEvent(
            event_type="general_or_config_workflow_selected",
            message="non-benchmark workflow selected",
            data={"intent": intent, "entities": route["entities"]},
            requires_user_input=False,
        ))
    return {
        "status": "ok",
        "workflow": "root",
        "events": [event.as_dict() for event in events],
        "next_event": events[-1].event_type,
    }

"""Native ADK workflow smoke helpers.

These helpers are intentionally credential-free. They verify that the installed
google-adk runtime can build and execute a minimal workflow with deterministic
FunctionNode steps, which is the lowest-risk contract for the AnyChain workflow
migration.
"""

from __future__ import annotations

import asyncio
from typing import Any


def run_native_workflow_smoke() -> dict[str, Any]:
    """Run a minimal ADK Workflow if google-adk is installed."""
    try:
        return asyncio.run(_run_native_workflow_smoke_async())
    except ImportError as exc:
        return {
            "status": "not_installed",
            "error": str(exc),
            "recommendation": "Run scripts/install_agent_deps.sh --yes in an isolated Python 3.10+ environment.",
        }
    except Exception as exc:
        return {"status": "failed", "error": str(exc)}


async def _run_native_workflow_smoke_async() -> dict[str, Any]:
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.adk.workflow import START, FunctionNode, Workflow
    from google.genai import types

    def startup_doctor_marker() -> dict[str, str]:
        return {"node": "startup_doctor"}

    def route_marker() -> dict[str, str]:
        return {"node": "intent_route"}

    startup_node = FunctionNode(func=startup_doctor_marker, name="startup_doctor")
    route_node = FunctionNode(func=route_marker, name="intent_route")
    workflow = Workflow(
        name="anychain_native_workflow_smoke",
        edges=[
            (START, startup_node),
            (startup_node, route_node),
        ],
    )
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name="anychain-native-smoke",
        user_id="smoke-user",
        session_id="smoke-session",
    )
    runner = Runner(
        app_name="anychain-native-smoke",
        node=workflow,
        session_service=session_service,
    )
    new_message = types.Content(role="user", parts=[types.Part(text="run smoke")])
    events = []
    async for event in runner.run_async(
        user_id="smoke-user",
        session_id="smoke-session",
        new_message=new_message,
    ):
        events.append({
            "author": getattr(event, "author", ""),
            "event_id": getattr(event, "id", ""),
        })
    return {
        "status": "passed",
        "workflow": workflow.name,
        "nodes": [startup_node.name, route_node.name],
        "event_count": len(events),
        "events": events,
    }

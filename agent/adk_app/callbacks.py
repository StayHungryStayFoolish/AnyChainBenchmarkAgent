"""ADK callbacks for AnyChain benchmark safety boundaries."""

from __future__ import annotations

from typing import Any


CONFIRMATION_GATED_TOOLS = {
    "run_smoke",
    "run_fake_node_smoke_benchmark",
    "install_dependencies",
    "submit_benchmark_job",
    "stop_job",
}


def before_tool_callback(tool: Any, args: dict[str, Any], tool_context: Any) -> dict[str, Any] | None:
    """Block confirmation-gated benchmark actions until approval is explicit.

    ADK owns tool orchestration, but benchmark execution is still a high-impact
    domain action. This callback is an ADK-native guardrail: it lets the model
    see and select action tools while preventing accidental execution when the
    tool call does not include ``approved=true``.
    """
    _ = tool_context
    tool_name = _tool_name(tool)
    if tool_name not in CONFIRMATION_GATED_TOOLS:
        return None
    if bool(args.get("approved")):
        return None
    return {
        "status": "needs_confirmation",
        "data": {
            "action": tool_name,
            "summary": _confirmation_summary(tool_name),
        },
        "evidence_paths": [],
        "warnings": [],
        "next_actions": ["ask user for explicit yes/no confirmation before retrying with approved=true"],
        "requires_user_confirmation": True,
    }


def _tool_name(tool: Any) -> str:
    return str(getattr(tool, "name", "") or getattr(tool, "__name__", "") or tool)


def _confirmation_summary(tool_name: str) -> str:
    if tool_name == "run_smoke":
        return "Run a lifecycle-only mock smoke job to validate plan, runtime.env, and artifact generation."
    if tool_name == "run_fake_node_smoke_benchmark":
        return "Run the real benchmark engine in quick fake-node mode with isolated job-local output."
    if tool_name == "submit_benchmark_job":
        return "Submit a real benchmark job that can generate load against the target blockchain node."
    if tool_name == "install_dependencies":
        return "Install benchmark dependencies. This may modify the host, so explicit approval is required."
    if tool_name == "stop_job":
        return "Stop or interrupt a running benchmark job."
    return f"Run confirmation-gated tool: {tool_name}"

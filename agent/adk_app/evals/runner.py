"""Offline ADK contract evaluations.

These checks intentionally do not simulate natural-language understanding.
Intent recognition belongs to the installed ADK runtime and configured model.
Without model credentials, the reliable offline contract is that the ADK agent
loads, exposes the expected tool set, and keeps confirmation-gated actions
behind approval callbacks.
"""

from __future__ import annotations

from typing import Any

from adk_app.callbacks import before_tool_callback
from adk_app.workflow.schemas import validate_intent_route
from adk_app.instructions import ROOT_INSTRUCTION
from adk_app.root_agent import resolve_adk_model
from adk_app.agents.domain import build_domain_agents
from adk_app.tools.registry import get_adk_tools


def run_offline_evals() -> dict[str, Any]:
    """Run no-credential ADK contract checks.

    Real prompt routing must be tested through ADK with a configured model
    provider.
    """
    tool_names = {tool.__name__ for tool in get_adk_tools(include_actions=True)}
    required_tools = {
        "discover_environment",
        "run_doctor",
        "audit_dependencies",
        "load_framework_context",
        "load_framework_index",
        "load_framework_capabilities",
        "prepare_benchmark_run",
        "draft_benchmark_request",
        "generate_benchmark_plan",
        "run_preflight",
        "render_runbook",
        "run_smoke",
        "run_fake_node_smoke_benchmark",
        "submit_benchmark_job",
        "install_dependencies",
        "latest_job",
        "analyze_artifacts",
        "diagnose_artifacts",
        "draft_chain_template",
        "knowledge_search",
        "validate_required_config",
        "build_missing_config_questions",
        "validate_rpc_workload",
        "load_default_workload",
        "validate_chain_template",
        "validate_execution_gate",
        "build_onboarding_handoff",
    }

    class _FakeAgent:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    domain_agents = build_domain_agents(_FakeAgent, "eval-model")
    domain_agent_names = {agent.name for agent in domain_agents}
    required_domain_agents = {
        "intent_router_agent",
        "environment_discovery_agent",
        "dependency_agent",
        "benchmark_configuration_agent",
        "rpc_workload_agent",
        "chain_rpc_onboarding_agent",
        "execution_agent",
        "resume_analyze_agent",
        "knowledge_agent",
    }

    class _Tool:
        name = "submit_benchmark_job"

    blocked = before_tool_callback(_Tool(), {"plan_file": "/tmp/plan.json"}, tool_context=None)
    results = [
        {
            "name": "root_instruction_present",
            "passed": bool(ROOT_INSTRUCTION and "Use a structured router only for intent classification" in ROOT_INSTRUCTION),
        },
        {
            "name": "required_tools_registered",
            "passed": required_tools.issubset(tool_names),
            "missing": sorted(required_tools - tool_names),
        },
        {
            "name": "adk_domain_agents_registered",
            "passed": required_domain_agents.issubset(domain_agent_names),
            "missing": sorted(required_domain_agents - domain_agent_names),
            "agent_count": len(domain_agents),
        },
        {
            "name": "domain_agents_use_narrow_tool_surfaces",
            "passed": all(getattr(agent, "tools", None) for agent in domain_agents)
            and len(next(agent for agent in domain_agents if agent.name == "rpc_workload_agent").tools) < len(tool_names),
        },
        {
            "name": "action_callback_blocks_unapproved_submit",
            "passed": bool(blocked and blocked.get("requires_user_confirmation")),
        },
        {
            "name": "model_resolution_uses_real_default",
            "passed": resolve_adk_model() != "fake",
            "model": resolve_adk_model(),
        },
        {
            "name": "typed_intent_schema_contract",
            "passed": validate_intent_route({
                "intent": "START_BENCHMARK",
                "confidence": 0.82,
                "language": "en",
                "entities": {
                    "chain": "solana",
                    "rpc_methods": [],
                    "rpc_mode": "single",
                    "target": "fake-node",
                    "job_id": "",
                },
                "missing_clarifications": [],
            }) == [],
        },
    ]
    return {
        "status": "passed" if all(item["passed"] for item in results) else "failed",
        "case_count": len(results),
        "passed_count": sum(1 for item in results if item["passed"]),
        "results": results,
        "note": "This is an ADK contract eval. Natural-language routing requires a configured ADK model provider.",
    }

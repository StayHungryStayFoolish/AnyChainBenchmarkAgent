"""Offline ADK compatibility evaluations.

Product workflow behavior is tested through the LangGraph Harness tests and
CLI matrix. These checks only verify that the optional ADK package surface can
load without exposing retired workflow-state tools.
"""

from __future__ import annotations

from typing import Any

from adk_app.instructions import ADK_BRIDGE_BOUNDARY, ADK_COMPATIBILITY_INSTRUCTION
from adk_app.root_agent import ADK_MODEL_BRIDGE_INSTRUCTION, resolve_adk_model
from adk_app.tools.registry import get_adk_tools
from adk_app.workflow.schemas import validate_intent_route


def run_offline_evals() -> dict[str, Any]:
    """Run no-credential ADK compatibility checks."""
    tool_names = {tool.__name__ for tool in get_adk_tools(include_actions=True)}
    retired_tools = {"load_workflow_state", "update_workflow_state", "answer_pending_question"}
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
        "validate_rpc_endpoint",
        "build_onboarding_handoff",
    }

    results = [
        {
            "name": "compat_instruction_present",
            "passed": "LangGraph Harness" in ADK_MODEL_BRIDGE_INSTRUCTION
            and "LangGraph Harness" in ADK_COMPATIBILITY_INSTRUCTION
            and "retired" in ADK_BRIDGE_BOUNDARY,
        },
        {
            "name": "required_tools_registered",
            "passed": required_tools.issubset(tool_names),
            "missing": sorted(required_tools - tool_names),
        },
        {
            "name": "retired_workflow_tools_not_registered",
            "passed": not (retired_tools & tool_names),
            "unexpected": sorted(retired_tools & tool_names),
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
        "note": "Product workflow is validated through LangGraph Harness tests, not ADK workflow prompts.",
    }

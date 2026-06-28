"""Specialized ADK agents for the AnyChain benchmark domain."""

from __future__ import annotations

from adk_app.instructions import (
    BENCHMARK_CONFIG_INSTRUCTION,
    DEPENDENCY_INSTRUCTION,
    ENVIRONMENT_DISCOVERY_INSTRUCTION,
    EXECUTION_INSTRUCTION,
    INTENT_ROUTER_INSTRUCTION,
    KNOWLEDGE_INSTRUCTION,
    ONBOARDING_INSTRUCTION,
    RESUME_ANALYZE_INSTRUCTION,
    RPC_WORKLOAD_INSTRUCTION,
)
from adk_app.tools.actions import get_action_tools
from adk_app.tools.enterprise import get_enterprise_tools
from adk_app.tools.planning import get_planning_tools
from adk_app.tools.read_only import (
    audit_dependencies,
    discover_environment,
    get_read_only_tools,
    knowledge_search,
    latest_job,
    load_framework_context,
    load_framework_index,
    run_doctor,
)
from adk_app.tools.validators import (
    build_missing_config_questions,
    build_onboarding_handoff,
    load_default_workload,
    validate_chain_template,
    validate_execution_gate,
    validate_required_config,
    validate_rpc_workload,
)
from adk_app.tools.web_research import get_google_search_tools
from adk_app.tools.workflow_state import load_workflow_state, reset_workflow_state, update_workflow_state


def build_domain_agents(agent_cls, model: str) -> list:
    """Build specialized ADK sub-agents.

    The root coordinator delegates to these agents. Each sub-agent receives a
    narrow tool surface so ADK owns orchestration while AnyChain validators keep
    domain gates deterministic.
    """
    return [
        agent_cls(
            name="intent_router_agent",
            model=model,
            description="Classifies user intent and extracts entities without executing benchmark tools.",
            instruction=INTENT_ROUTER_INSTRUCTION,
            tools=[load_workflow_state, update_workflow_state, load_framework_context, load_framework_index],
        ),
        agent_cls(
            name="environment_discovery_agent",
            model=model,
            description="Discovers local cloud, VM/Kubernetes, disk, network, and dependency context.",
            instruction=ENVIRONMENT_DISCOVERY_INSTRUCTION,
            tools=[load_workflow_state, update_workflow_state, discover_environment, run_doctor, build_missing_config_questions],
        ),
        agent_cls(
            name="dependency_agent",
            model=model,
            description="Audits and installs dependencies only after explicit approval.",
            instruction=DEPENDENCY_INSTRUCTION,
            tools=[load_workflow_state, update_workflow_state, audit_dependencies, *get_action_tools()],
        ),
        agent_cls(
            name="benchmark_configuration_agent",
            model=model,
            description="Collects and validates fake-node or real-node benchmark configuration.",
            instruction=BENCHMARK_CONFIG_INSTRUCTION,
            tools=[
                load_workflow_state,
                update_workflow_state,
                validate_required_config,
                build_missing_config_questions,
                validate_chain_template,
                *get_planning_tools(),
            ],
        ),
        agent_cls(
            name="rpc_workload_agent",
            model=model,
            description="Configures single, mixed, and custom RPC workloads.",
            instruction=RPC_WORKLOAD_INSTRUCTION,
            tools=[load_workflow_state, update_workflow_state, load_default_workload, validate_rpc_workload, validate_chain_template],
        ),
        agent_cls(
            name="chain_rpc_onboarding_agent",
            model=model,
            description="Produces evidence-backed handoffs for unsupported chains, new families, and custom RPC methods.",
            instruction=ONBOARDING_INSTRUCTION,
            tools=[
                load_workflow_state,
                update_workflow_state,
                load_framework_index,
                validate_chain_template,
                build_onboarding_handoff,
                *get_google_search_tools(),
                *get_planning_tools(),
            ],
        ),
        agent_cls(
            name="execution_agent",
            model=model,
            description="Runs preflight, smoke, fake-node smoke, and real benchmark jobs through approval-gated tools.",
            instruction=EXECUTION_INSTRUCTION,
            tools=[load_workflow_state, update_workflow_state, validate_execution_gate, *get_planning_tools(), *get_action_tools()],
        ),
        agent_cls(
            name="resume_analyze_agent",
            model=model,
            description="Resumes jobs, tails logs, analyzes artifacts, and explains report evidence.",
            instruction=RESUME_ANALYZE_INSTRUCTION,
            tools=[load_workflow_state, update_workflow_state, latest_job, *get_read_only_tools(), *get_action_tools()],
        ),
        agent_cls(
            name="knowledge_agent",
            model=model,
            description="Answers framework capability and enterprise KB questions from local repo facts first.",
            instruction=KNOWLEDGE_INSTRUCTION,
            tools=[load_workflow_state, update_workflow_state, reset_workflow_state, load_framework_context, load_framework_index, knowledge_search, *get_enterprise_tools()],
        ),
    ]

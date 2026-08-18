"""Execute named Agent tools for enterprise platform integrations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from agent.analyzers.artifact_qa import answer_artifact_question
from agent.analyzers.bottleneck_rules import diagnose_artifacts
from agent.analyzers.result_analyzer import analyze_job
from agent.diagnostics.doctor import run_doctor
from agent.discovery.environment import discover_environment
from agent.knowledge.execution_contract import load_execution_contract
from agent.knowledge.framework_capabilities import load_framework_capabilities
from agent.knowledge.framework_context import load_framework_context
from agent.knowledge.framework_index import load_or_build_framework_index
from agent.knowledge.gap_analyzer import analyze_capability_gap
from agent.knowledge.loader import load_knowledge_provider, provider_status
from agent.llm.auth_status import inspect_llm_auth
from agent.harness.sync_observe_contract import SyncObserveRequest
from agent.onboarding.template_drafter import draft_chain_template
from agent.planners.strategy_planner import generate_plan
from agent.runners.application_service import (
    ExecutionOperation,
    ExecutionRequest,
    execution_service,
)
from agent.runners.dependency_installer import audit_dependencies, install_dependencies
from agent.runners.job_manager import DEFAULT_JOBS_DIR, get_job, tail_job_log
from agent.runners.tool_result import tool_result
from agent.validators.chain_template import validate_chain_template
from agent.validators.execution_gate import validate_execution_gate
from agent.validators.fixture_checks import validate_fake_node_fixture_authenticity, validate_fake_node_fixture_coverage
from agent.validators.onboarding_gate import build_onboarding_handoff
from agent.validators.rpc_workload import default_workload, validate_rpc_workload
from agent.tools.schema import TOOL_SPECS, ToolSpec


def execute_tool(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    spec = TOOL_OPERATION_BY_NAME.get(name)
    if spec is None:
        raise ValueError(f"unsupported tool: {name}")
    return spec.execute(dict(arguments or {}))


def _execute_discover_environment(args: dict[str, Any]) -> dict[str, Any]:
    return discover_environment()


def _execute_audit_dependencies(args: dict[str, Any]) -> dict[str, Any]:
    return audit_dependencies()


def _execute_run_doctor(args: dict[str, Any]) -> dict[str, Any]:
    return run_doctor()


def _execute_load_capabilities(args: dict[str, Any]) -> dict[str, Any]:
    return load_framework_capabilities()


def _execute_load_framework_context(args: dict[str, Any]) -> dict[str, Any]:
    return load_framework_context(language=args.get("language", "en"))


def _execute_load_framework_index(args: dict[str, Any]) -> dict[str, Any]:
    return load_or_build_framework_index(index_path=args.get("index_path"))


def _execute_load_execution_contract(args: dict[str, Any]) -> dict[str, Any]:
    raw = args.get("use_fake_node")
    return load_execution_contract(use_fake_node=raw if isinstance(raw, bool) else None)


def _execute_prepare_benchmark_run(args: dict[str, Any]) -> dict[str, Any]:
    return execution_service.execute(
        ExecutionRequest(operation=ExecutionOperation.PREPARE, prepare_kwargs=args)
    ).to_dict()


def _execute_draft_request(args: dict[str, Any]) -> dict[str, Any]:
    return _structured_request(args)


def _execute_generate_plan(args: dict[str, Any]) -> dict[str, Any]:
    return generate_plan(args["request"])


def _execute_run_preflight(args: dict[str, Any]) -> dict[str, Any]:
    return execution_service.execute(
        ExecutionRequest(operation=ExecutionOperation.PREFLIGHT, plan=args["plan"])
    ).to_dict()


def _execute_submit_job(args: dict[str, Any]) -> dict[str, Any]:
    request_kwargs = {
        "operation": ExecutionOperation.FINAL_BENCHMARK,
        "plan_file": args["plan_file"],
        "approved": args.get("approved") is True,
    }
    if args.get("jobs_dir"):
        request_kwargs["jobs_dir"] = args["jobs_dir"]
    return execution_service.execute(ExecutionRequest(**request_kwargs)).to_dict()


def _execute_run_fake_node_smoke_benchmark(args: dict[str, Any]) -> dict[str, Any]:
    return execution_service.execute(
        ExecutionRequest(
            operation=ExecutionOperation.FAKE_NODE_SMOKE,
            plan_file=args["plan_file"],
            jobs_dir=args.get("jobs_dir", DEFAULT_JOBS_DIR),
            approved=args.get("approved") is True,
        )
    ).to_dict()


def _execute_install_dependencies(args: dict[str, Any]) -> dict[str, Any]:
    if args.get("approved") is not True:
        return _confirmation_required(
            "install_dependencies",
            "Install or update local benchmark and Agent runtime dependencies.",
        )
    return install_dependencies(
        no_sudo=args.get("no_sudo", True),
        include_vegeta=args.get("include_vegeta", True),
        include_agent_runtime=args.get("include_agent_runtime", False),
        include_gcloud=args.get("include_gcloud", False),
        adk_venv=args.get("adk_venv", ".venv-adk"),
        allow_system_python=args.get("allow_system_python", False),
    )


def _execute_get_job_status(args: dict[str, Any]) -> dict[str, Any]:
    return _get_job(args)


def _execute_tail_job_log(args: dict[str, Any]) -> dict[str, Any]:
    if args.get("jobs_dir"):
        return tail_job_log(args["job_id"], jobs_dir=args["jobs_dir"], lines=int(args.get("lines", 80)))
    return tail_job_log(args["job_id"], lines=int(args.get("lines", 80)))


def _execute_analyze_artifacts(args: dict[str, Any]) -> dict[str, Any]:
    return analyze_job(_get_job(args))


def _execute_answer_artifact_question(args: dict[str, Any]) -> dict[str, Any]:
    job = _get_job(args) if args.get("job_id") else None
    return answer_artifact_question(args["question"], job=job, artifact_index=args.get("artifact_index"))


def _execute_diagnose_artifacts(args: dict[str, Any]) -> dict[str, Any]:
    job = _get_job(args) if args.get("job_id") else None
    return diagnose_artifacts(job=job, artifact_index=args.get("artifact_index"))


def _execute_draft_chain_template(args: dict[str, Any]) -> dict[str, Any]:
    return draft_chain_template(
        chain=args["chain"],
        adapter_family=args["adapter_family"],
        methods=args.get("methods") or [],
    )


def _execute_gap_analysis(args: dict[str, Any]) -> dict[str, Any]:
    return analyze_capability_gap(args["chain"], args.get("methods") or [])


def _execute_validate_rpc_workload(args: dict[str, Any]) -> dict[str, Any]:
    return validate_rpc_workload(
        args["chain"],
        args["rpc_mode"],
        args.get("methods") or [],
        args.get("mixed_weights") or {},
    )


def _execute_load_default_workload(args: dict[str, Any]) -> dict[str, Any]:
    return default_workload(args["chain"])


def _execute_validate_chain_template(args: dict[str, Any]) -> dict[str, Any]:
    return validate_chain_template(args["chain"])


def _execute_validate_execution_gate(args: dict[str, Any]) -> dict[str, Any]:
    return validate_execution_gate(
        args.get("plan"),
        args.get("preflight"),
        args.get("smoke"),
        bool(args.get("approved", False)),
        bool(args.get("real_execution", False)),
    )


def _execute_build_onboarding_handoff(args: dict[str, Any]) -> dict[str, Any]:
    return build_onboarding_handoff(
        args["chain"],
        args["family"],
        args.get("methods") or [],
        args.get("evidence") or {},
    )


def _execute_knowledge_search(args: dict[str, Any]) -> dict[str, Any]:
    status = provider_status()
    if not status["enabled"] or status["error"]:
        return {"status": status, "results": []}
    provider = load_knowledge_provider()
    payload: dict[str, Any] = {"status": status, "results": provider.search(args["query"])}
    if args.get("chain"):
        payload["rpc_methods"] = provider.get_rpc_methods(args["chain"])
    return payload


def _execute_validate_fake_node_fixture_coverage(args: dict[str, Any]) -> dict[str, Any]:
    return validate_fake_node_fixture_coverage(
        chains=args.get("chains", "all"),
        modes=args.get("modes", "single,mixed"),
        strict=args.get("strict", True),
    )


def _execute_validate_fake_node_fixture_authenticity(args: dict[str, Any]) -> dict[str, Any]:
    return validate_fake_node_fixture_authenticity(
        modes=args.get("modes", "single,mixed"),
        allow_incomplete=args.get("allow_incomplete", False),
    )


def _execute_inspect_llm_auth(args: dict[str, Any]) -> dict[str, Any]:
    return inspect_llm_auth()


_TOOL_HANDLER_FUNCTIONS: tuple[Callable[[dict[str, Any]], dict[str, Any]], ...] = tuple(
    handler
    for handler in (
        _execute_discover_environment,
        _execute_audit_dependencies,
        _execute_run_doctor,
        _execute_load_capabilities,
        _execute_load_framework_context,
        _execute_load_framework_index,
        _execute_load_execution_contract,
        _execute_prepare_benchmark_run,
        _execute_draft_request,
        _execute_generate_plan,
        _execute_run_preflight,
        _execute_submit_job,
        _execute_run_fake_node_smoke_benchmark,
        _execute_install_dependencies,
        _execute_get_job_status,
        _execute_tail_job_log,
        _execute_analyze_artifacts,
        _execute_answer_artifact_question,
        _execute_diagnose_artifacts,
        _execute_draft_chain_template,
        _execute_gap_analysis,
        _execute_validate_rpc_workload,
        _execute_load_default_workload,
        _execute_validate_chain_template,
        _execute_validate_execution_gate,
        _execute_build_onboarding_handoff,
        _execute_knowledge_search,
        _execute_validate_fake_node_fixture_coverage,
        _execute_validate_fake_node_fixture_authenticity,
        _execute_inspect_llm_auth,
    )
)


def _bind_tool_operations(
    specs: tuple[ToolSpec, ...],
    handlers: tuple[Callable[[dict[str, Any]], dict[str, Any]], ...],
) -> tuple[ToolSpec, ...]:
    handler_by_name = {
        handler.__name__.removeprefix("_execute_"): handler for handler in handlers
    }
    spec_names = {spec.name for spec in specs}
    if spec_names != set(handler_by_name):
        missing = sorted(spec_names - set(handler_by_name))
        extra = sorted(set(handler_by_name) - spec_names)
        raise RuntimeError(f"tool operation registry drift: missing={missing}, extra={extra}")
    return tuple(spec.bind(handler_by_name[spec.name]) for spec in specs)


TOOL_OPERATIONS = _bind_tool_operations(TOOL_SPECS, _TOOL_HANDLER_FUNCTIONS)
TOOL_OPERATION_BY_NAME = {spec.name: spec for spec in TOOL_OPERATIONS}


def _confirmation_required(action: str, summary: str) -> dict[str, Any]:
    return tool_result(
        status="needs_confirmation",
        data={"action": action, "summary": summary},
        next_actions=["ask user for explicit yes/no confirmation"],
        requires_user_confirmation=True,
    )


def load_arguments(value: str) -> dict[str, Any]:
    import json

    if not value:
        return {}
    stripped = value.strip()
    if stripped.startswith("{"):
        return json.loads(stripped)
    path = Path(value)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(stripped)


def _required(args: dict[str, Any], key: str) -> Any:
    value = args.get(key)
    if value in (None, ""):
        raise ValueError(f"missing required argument: {key}")
    return value


def _get_job(args: dict[str, Any]) -> dict[str, Any]:
    if args.get("jobs_dir"):
        return get_job(_required(args, "job_id"), jobs_dir=args["jobs_dir"])
    return get_job(_required(args, "job_id"))


def _structured_request(args: dict[str, Any]) -> dict[str, Any]:
    workflow_type = str(args.get("workflow_type") or "rpc_benchmark")
    request: dict[str, Any] = {
        "chain": args.get("chain", ""),
        "goal": args.get("goal", "baseline"),
        "rpc_mode": args.get("rpc_mode", "single"),
        "deployment": {
            "type": args.get("deployment_type", "unknown"),
            "provider": args.get("cloud_provider", ""),
        },
        "observability": {"enabled": False, "mode": "local"},
        "dependency_mode": "audit",
        "runner_mode": "detached",
        "bottleneck_focus": ["cpu", "memory", "disk", "network", "rpc_errors"],
        "source_prompt": args.get("source_prompt", ""),
        "workflow_type": workflow_type,
    }
    if workflow_type == "sync_observe":
        request.update(SyncObserveRequest.from_request_values(args).request_values())
    if "observability_enabled" in args:
        request["observability"]["enabled"] = bool(args["observability_enabled"])
    if args.get("observability_mode"):
        request["observability"]["mode"] = args["observability_mode"]
    if "observability_auto_stop" in args:
        request["observability"]["auto_stop"] = bool(args["observability_auto_stop"])
    for port in ("exporter_port", "prometheus_port", "grafana_port"):
        if args.get(port):
            request[port] = args[port]
    if isinstance(args.get("use_fake_node"), bool):
        request["use_fake_node"] = bool(args["use_fake_node"])
    if args.get("confirmations"):
        request["confirmations"] = list(args["confirmations"])
    if args.get("target_rpc_url"):
        request["local_rpc_url"] = args["target_rpc_url"]
        request["target_rpc_url"] = args["target_rpc_url"]
    for key in ("mainnet_rpc_url", "ledger_device", "accounts_device"):
        if args.get(key):
            request[key] = args[key]
    for key in (
        "cloud_region",
        "cloud_zone",
        "machine_type",
        "data_vol_type",
        "data_vol_size",
        "data_vol_max_iops",
        "data_vol_max_throughput",
        "accounts_vol_type",
        "accounts_vol_size",
        "accounts_vol_max_iops",
        "accounts_vol_max_throughput",
        "network_interface",
        "network_max_bandwidth_gbps",
    ):
        if args.get(key):
            request[key] = args[key]
    if args.get("blockchain_process_names"):
        request["blockchain_process_names"] = list(args["blockchain_process_names"])
    qps: dict[str, int] = {}
    for arg_key, qps_key in (
        ("qps_initial", "initial"),
        ("qps_max", "max"),
        ("qps_step", "step"),
        ("duration_seconds", "duration_seconds"),
    ):
        if args.get(arg_key) is not None:
            qps[qps_key] = int(args[arg_key])
    if qps:
        request["qps"] = qps
    if args.get("rpc_methods"):
        request["rpc_methods"] = list(args["rpc_methods"])
    if args.get("mixed_weights"):
        request["mixed_weighted"] = [
            {"method": method, "weight": int(weight)}
            for method, weight in args["mixed_weights"].items()
        ]
    return request

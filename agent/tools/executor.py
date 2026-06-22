"""Execute named Agent tools for enterprise platform integrations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from analyzers.artifact_qa import answer_artifact_question
from analyzers.bottleneck_rules import diagnose_artifacts
from analyzers.result_analyzer import analyze_job
from discovery.environment import discover_environment
from knowledge.framework_capabilities import load_framework_capabilities
from knowledge.gap_analyzer import analyze_capability_gap
from knowledge.loader import load_knowledge_provider, provider_status
from onboarding.template_drafter import draft_chain_template
from planners.preflight import run_preflight
from planners.strategy_planner import generate_plan
from runners.job_manager import get_job, submit_job, tail_job_log
from adk_app.tools.actions import run_fake_node_smoke_benchmark
from adk_app.tools.actions import install_dependencies
from adk_app.tools.planning import prepare_benchmark_run
from adk_app.tools.read_only import audit_dependencies


def execute_tool(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    args = arguments or {}
    if name == "discover_environment":
        return discover_environment()
    if name == "audit_dependencies":
        return audit_dependencies()
    if name == "load_capabilities":
        return load_framework_capabilities()
    if name == "prepare_benchmark_run":
        return prepare_benchmark_run(**args)
    if name == "draft_request":
        return _structured_request(args)
    if name == "generate_plan":
        return generate_plan(_required(args, "request"), discovery=args.get("discovery"))
    if name == "run_preflight":
        return run_preflight(_required(args, "plan"))
    if name == "submit_job":
        kwargs = {"mock": bool(args.get("mock", False)), "approved": bool(args.get("approved", False))}
        if args.get("jobs_dir"):
            kwargs["jobs_dir"] = args["jobs_dir"]
        return submit_job(_required(args, "plan_file"), **kwargs)
    if name == "run_fake_node_smoke_benchmark":
        return run_fake_node_smoke_benchmark(
            _required(args, "plan_file"),
            jobs_dir=args.get("jobs_dir", ".agent/jobs"),
            approved=bool(args.get("approved", False)),
        )
    if name == "install_dependencies":
        return install_dependencies(
            approved=bool(args.get("approved", False)),
            no_sudo=bool(args.get("no_sudo", True)),
            include_vegeta=bool(args.get("include_vegeta", True)),
            include_agent_runtime=bool(args.get("include_agent_runtime", True)),
            include_gcloud=bool(args.get("include_gcloud", False)),
            adk_venv=args.get("adk_venv", ".venv-adk"),
            allow_system_python=bool(args.get("allow_system_python", False)),
        )
    if name == "get_job_status":
        return _get_job(args)
    if name == "tail_job_log":
        if args.get("jobs_dir"):
            return tail_job_log(_required(args, "job_id"), jobs_dir=args["jobs_dir"], lines=int(args.get("lines", 80)))
        return tail_job_log(_required(args, "job_id"), lines=int(args.get("lines", 80)))
    if name == "analyze_artifacts":
        job = _get_job(args)
        return analyze_job(job)
    if name == "answer_artifact_question":
        job = None
        if args.get("job_id"):
            job = _get_job(args)
        return answer_artifact_question(_required(args, "question"), job=job, artifact_index=args.get("artifact_index"))
    if name == "diagnose_artifacts":
        job = None
        if args.get("job_id"):
            job = _get_job(args)
        return diagnose_artifacts(job=job, artifact_index=args.get("artifact_index"))
    if name == "draft_chain_template":
        return draft_chain_template(
            chain=_required(args, "chain"),
            adapter_family=_required(args, "adapter_family"),
            methods=args.get("methods") or [],
            output=args.get("output"),
        )
    if name == "gap_analysis":
        return analyze_capability_gap(_required(args, "chain"), args.get("methods") or [])
    if name == "knowledge_search":
        status = provider_status()
        if not status["enabled"] or status["error"]:
            return {"status": status, "results": []}
        provider = load_knowledge_provider()
        payload: dict[str, Any] = {"status": status, "results": provider.search(_required(args, "query"))}
        if args.get("chain"):
            payload["rpc_methods"] = provider.get_rpc_methods(args["chain"])
        return payload
    raise ValueError(f"unsupported tool: {name}")


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
    request: dict[str, Any] = {
        "chain": args.get("chain", ""),
        "goal": args.get("goal", "baseline"),
        "rpc_mode": args.get("rpc_mode", "single"),
        "use_fake_node": bool(args.get("use_fake_node", False)),
        "deployment": {
            "type": args.get("deployment_type", "unknown"),
            "provider": args.get("cloud_provider", ""),
        },
        "observability": {"enabled": False, "mode": "local"},
        "dependency_mode": "audit",
        "runner_mode": "detached",
        "bottleneck_focus": ["cpu", "memory", "disk", "network", "rpc_errors"],
        "source_prompt": args.get("source_prompt", ""),
    }
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

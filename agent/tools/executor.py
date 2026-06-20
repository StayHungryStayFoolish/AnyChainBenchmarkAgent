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
from qa.request_drafter import draft_request
from runners.job_manager import get_job, submit_job, tail_job_log


def execute_tool(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    args = arguments or {}
    if name == "discover_environment":
        return discover_environment()
    if name == "load_capabilities":
        return load_framework_capabilities()
    if name == "draft_request":
        return draft_request(_required(args, "prompt"))
    if name == "generate_plan":
        return generate_plan(_required(args, "request"), discovery=args.get("discovery"))
    if name == "run_preflight":
        return run_preflight(_required(args, "plan"))
    if name == "submit_job":
        kwargs = {"mock": bool(args.get("mock", False)), "approved": bool(args.get("approved", False))}
        if args.get("jobs_dir"):
            kwargs["jobs_dir"] = args["jobs_dir"]
        return submit_job(_required(args, "plan_file"), **kwargs)
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

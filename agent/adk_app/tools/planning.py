"""Planning ADK tool wrappers for AnyChain benchmark requests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from diagnostics.doctor import run_doctor as _run_doctor
from discovery.environment import discover_environment as _discover_environment
from onboarding.template_drafter import draft_chain_template as _draft_chain_template
from planners.diff import diff_plans
from planners.preflight import run_preflight as _run_preflight
from planners.strategy_planner import generate_plan as _generate_plan
from planners.strategy_planner import write_json
from runners.runbook import render_runbook as _render_runbook

from .read_only import _tool_result


def prepare_benchmark_run(
    source_prompt: str = "",
    chain: str = "",
    goal: str = "",
    rpc_mode: str = "",
    use_fake_node: bool | None = None,
    deployment_type: str = "",
    cloud_provider: str = "",
    target_rpc_url: str = "",
    mainnet_rpc_url: str = "",
    ledger_device: str = "",
    accounts_device: str = "",
    blockchain_process_names: list[str] | None = None,
    cloud_region: str = "",
    cloud_zone: str = "",
    machine_type: str = "",
    data_vol_type: str = "",
    data_vol_size: str = "",
    data_vol_max_iops: str = "",
    data_vol_max_throughput: str = "",
    accounts_vol_type: str = "",
    accounts_vol_size: str = "",
    accounts_vol_max_iops: str = "",
    accounts_vol_max_throughput: str = "",
    network_interface: str = "",
    network_max_bandwidth_gbps: str = "",
    qps_initial: int | None = None,
    qps_max: int | None = None,
    qps_step: int | None = None,
    duration_seconds: int | None = None,
    rpc_methods: list[str] | None = None,
    mixed_weights: dict[str, int] | None = None,
    output_dir: str = ".agent/prepared",
) -> dict[str, Any]:
    """Prepare a benchmark run without launching benchmark traffic.

    Use this as the default setup tool after understanding the user's goal. It
    performs read-only discovery, readiness diagnostics, structured request
    drafting, plan generation, preflight, and runbook rendering. It returns the
    inferred values, missing values, confirmation questions, and concrete plan
    path so the Agent can ask the user for only unresolved values before any
    smoke or real benchmark action.
    """
    discovery = _discover_environment()
    doctor = _run_doctor()
    request = _structured_request(
        source_prompt=source_prompt,
        chain=chain,
        goal=goal,
        rpc_mode=rpc_mode,
        use_fake_node=use_fake_node,
        deployment_type=deployment_type or discovery.get("deployment", {}).get("type", ""),
        cloud_provider=cloud_provider or discovery.get("cloud", {}).get("provider", ""),
        target_rpc_url=target_rpc_url,
        mainnet_rpc_url=mainnet_rpc_url,
        ledger_device=ledger_device or discovery.get("disks", {}).get("proposed_ledger_device", ""),
        accounts_device=accounts_device or discovery.get("disks", {}).get("proposed_accounts_device", ""),
        blockchain_process_names=blockchain_process_names,
        cloud_region=cloud_region,
        cloud_zone=cloud_zone,
        machine_type=machine_type,
        data_vol_type=data_vol_type,
        data_vol_size=data_vol_size,
        data_vol_max_iops=data_vol_max_iops,
        data_vol_max_throughput=data_vol_max_throughput,
        accounts_vol_type=accounts_vol_type,
        accounts_vol_size=accounts_vol_size,
        accounts_vol_max_iops=accounts_vol_max_iops,
        accounts_vol_max_throughput=accounts_vol_max_throughput,
        network_interface=network_interface or discovery.get("network", {}).get("default_interface", ""),
        network_max_bandwidth_gbps=network_max_bandwidth_gbps,
        qps_initial=qps_initial,
        qps_max=qps_max,
        qps_step=qps_step,
        duration_seconds=duration_seconds,
        rpc_methods=rpc_methods,
        mixed_weights=mixed_weights,
    )
    request["discovery"] = discovery
    plan = _generate_plan(request, discovery=discovery)
    preflight = _run_preflight(plan)
    runbook = _render_runbook(plan)

    prepared_dir = Path(output_dir)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    plan_file = prepared_dir / f"{plan['plan_id']}.json"
    runbook_file = prepared_dir / f"{plan['plan_id']}_runbook.md"
    write_json(plan_file, plan)
    runbook_file.write_text(runbook, encoding="utf-8")

    data = {
        "request": request,
        "plan": plan,
        "plan_file": str(plan_file),
        "runbook_file": str(runbook_file),
        "preflight": preflight,
        "doctor": doctor,
        "inferred_values": _inferred_values(plan),
        "missing_required": plan.get("required_inputs", []),
        "questions": plan.get("required_questions", []),
        "requires_confirmation": plan.get("requires_confirmation", []),
        "approval_checkpoints": plan.get("approval_checkpoints", []),
    }
    warnings = []
    warnings.extend(discovery.get("warnings", []))
    warnings.extend(doctor.get("warnings", []))
    warnings.extend(preflight.get("blockers", []))
    return _tool_result(
        status="ok" if preflight.get("passed") else "blocked",
        data=data,
        evidence_paths=[str(plan_file), str(runbook_file)],
        warnings=warnings,
        next_actions=_prepare_next_actions(plan, preflight),
    )


def draft_benchmark_request(
    source_prompt: str = "",
    discovered_context: dict | None = None,
    chain: str = "",
    goal: str = "",
    rpc_mode: str = "",
    use_fake_node: bool | None = None,
    deployment_type: str = "",
    cloud_provider: str = "",
    target_rpc_url: str = "",
    mainnet_rpc_url: str = "",
    ledger_device: str = "",
    accounts_device: str = "",
    blockchain_process_names: list[str] | None = None,
    cloud_region: str = "",
    cloud_zone: str = "",
    machine_type: str = "",
    data_vol_type: str = "",
    data_vol_size: str = "",
    data_vol_max_iops: str = "",
    data_vol_max_throughput: str = "",
    accounts_vol_type: str = "",
    accounts_vol_size: str = "",
    accounts_vol_max_iops: str = "",
    accounts_vol_max_throughput: str = "",
    network_interface: str = "",
    network_max_bandwidth_gbps: str = "",
    qps_initial: int | None = None,
    qps_max: int | None = None,
    qps_step: int | None = None,
    duration_seconds: int | None = None,
    rpc_methods: list[str] | None = None,
    mixed_weights: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Draft a normalized AnyChain request from ADK-inferred structured fields.

    ADK should infer benchmark intent from the conversation and pass explicit
    fields here. This tool intentionally does not parse free-form text or act
    as an intent router. Use ``source_prompt`` only as evidence text.
    """
    request = _structured_request(
        source_prompt=source_prompt,
        chain=chain,
        goal=goal,
        rpc_mode=rpc_mode,
        use_fake_node=use_fake_node,
        deployment_type=deployment_type,
        cloud_provider=cloud_provider,
        target_rpc_url=target_rpc_url,
        mainnet_rpc_url=mainnet_rpc_url,
        ledger_device=ledger_device,
        accounts_device=accounts_device,
        blockchain_process_names=blockchain_process_names,
        cloud_region=cloud_region,
        cloud_zone=cloud_zone,
        machine_type=machine_type,
        data_vol_type=data_vol_type,
        data_vol_size=data_vol_size,
        data_vol_max_iops=data_vol_max_iops,
        data_vol_max_throughput=data_vol_max_throughput,
        accounts_vol_type=accounts_vol_type,
        accounts_vol_size=accounts_vol_size,
        accounts_vol_max_iops=accounts_vol_max_iops,
        accounts_vol_max_throughput=accounts_vol_max_throughput,
        network_interface=network_interface,
        network_max_bandwidth_gbps=network_max_bandwidth_gbps,
        qps_initial=qps_initial,
        qps_max=qps_max,
        qps_step=qps_step,
        duration_seconds=duration_seconds,
        rpc_methods=rpc_methods,
        mixed_weights=mixed_weights,
    )
    if discovered_context:
        request["discovery"] = discovered_context
    return _tool_result(data=request, next_actions=["generate_benchmark_plan", "ask_missing_required_values"])


def generate_benchmark_plan(request: dict, discovery: dict | None = None) -> dict[str, Any]:
    """Generate an executable benchmark plan from a confirmed request.

    Use this after drafting a request and collecting enough environment context.
    The returned plan contains runtime settings, command intent, approval
    checkpoints, configuration checklist, and required follow-up questions.
    """
    plan = _generate_plan(request, discovery=discovery)
    return _tool_result(
        data=plan,
        warnings=plan.get("warnings", []),
        next_actions=["validate_benchmark_plan", "run_preflight", "render_runbook"],
    )


def validate_benchmark_plan(plan: dict) -> dict[str, Any]:
    """Return plan validation status without executing benchmark workloads."""
    errors = []
    for key in ("plan_id", "chain", "strategy", "rpc_mode", "execution"):
        if key not in plan:
            errors.append(f"missing required plan key: {key}")
    return _tool_result(
        status="ok" if not errors else "invalid",
        data={"valid": not errors, "errors": errors},
        warnings=errors,
        next_actions=["run_preflight"] if not errors else ["repair plan"],
    )


def run_preflight(plan: dict) -> dict[str, Any]:
    """Validate a generated benchmark plan before any smoke or real benchmark.

    Always call this before run_smoke or submit_benchmark_job. If blockers are
    returned, explain them and ask the user for missing configuration instead of
    launching work.
    """
    preflight = _run_preflight(plan)
    return _tool_result(
        status="ok" if preflight.get("passed") else "blocked",
        data=preflight,
        warnings=preflight.get("warnings", []) + preflight.get("blockers", []),
        next_actions=["run_smoke", "ask_missing_required_values"] if preflight.get("passed") else ["fix blockers"],
    )


def render_runbook(plan: dict, output: str = "") -> dict[str, Any]:
    """Render a human-readable runbook so the user can review the plan.

    Use this before asking for smoke or real-run confirmation. Include the
    output path as evidence when a file is written.
    """
    text = _render_runbook(plan)
    evidence_paths: list[str] = []
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        evidence_paths.append(str(path))
    return _tool_result(data={"runbook": text}, evidence_paths=evidence_paths, next_actions=["review runbook"])


def diff_plan(old_plan: dict, new_plan: dict) -> dict[str, Any]:
    """Compare two benchmark plans and summarize changes."""
    return _tool_result(data=diff_plans(old_plan, new_plan), next_actions=["review changes", "run_preflight"])


def draft_chain_template(chain: str, adapter_family: str, methods: list[str] | None = None, output: str = "") -> dict[str, Any]:
    """Draft a chain template for unsupported-chain or custom-RPC onboarding.

    Use this when the user asks to add a new chain or RPC method. The result is
    not production support; it must stay needs_review until fixtures, request
    samples, response samples, and smoke validation are complete.
    """
    payload = _draft_chain_template(
        chain=chain,
        adapter_family=adapter_family,
        methods=methods or [],
        output=output or None,
    )
    evidence_paths = [output] if output else []
    return _tool_result(
        status=payload.get("status", "draft"),
        data=payload,
        evidence_paths=evidence_paths,
        warnings=["draft template requires human review and fixture validation"],
        next_actions=payload.get("validation_commands", []),
    )


def get_planning_tools() -> list:
    """Return planning ADK tool callables."""
    return [
        prepare_benchmark_run,
        draft_benchmark_request,
        generate_benchmark_plan,
        validate_benchmark_plan,
        run_preflight,
        render_runbook,
        diff_plan,
        draft_chain_template,
    ]


def _structured_request(
    *,
    source_prompt: str,
    chain: str,
    goal: str,
    rpc_mode: str,
    use_fake_node: bool | None,
    deployment_type: str,
    cloud_provider: str,
    target_rpc_url: str,
    mainnet_rpc_url: str,
    ledger_device: str,
    accounts_device: str,
    blockchain_process_names: list[str] | None,
    cloud_region: str,
    cloud_zone: str,
    machine_type: str,
    data_vol_type: str,
    data_vol_size: str,
    data_vol_max_iops: str,
    data_vol_max_throughput: str,
    accounts_vol_type: str,
    accounts_vol_size: str,
    accounts_vol_max_iops: str,
    accounts_vol_max_throughput: str,
    network_interface: str,
    network_max_bandwidth_gbps: str,
    qps_initial: int | None,
    qps_max: int | None,
    qps_step: int | None,
    duration_seconds: int | None,
    rpc_methods: list[str] | None,
    mixed_weights: dict[str, int] | None,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "chain": chain,
        "goal": goal or "baseline",
        "rpc_mode": rpc_mode or "single",
        "use_fake_node": bool(use_fake_node) if use_fake_node is not None else False,
        "deployment": {
            "type": deployment_type or "unknown",
            "provider": cloud_provider,
        },
        "observability": {"enabled": False, "mode": "local"},
        "dependency_mode": "audit",
        "runner_mode": "detached",
        "bottleneck_focus": ["cpu", "memory", "disk", "network", "rpc_errors"],
        "source_prompt": source_prompt,
    }
    for key, value in {
        "local_rpc_url": target_rpc_url,
        "target_rpc_url": target_rpc_url,
        "mainnet_rpc_url": mainnet_rpc_url,
        "ledger_device": ledger_device,
        "accounts_device": accounts_device,
        "cloud_region": cloud_region,
        "cloud_zone": cloud_zone,
        "machine_type": machine_type,
        "data_vol_type": data_vol_type,
        "data_vol_size": data_vol_size,
        "data_vol_max_iops": data_vol_max_iops,
        "data_vol_max_throughput": data_vol_max_throughput,
        "accounts_vol_type": accounts_vol_type,
        "accounts_vol_size": accounts_vol_size,
        "accounts_vol_max_iops": accounts_vol_max_iops,
        "accounts_vol_max_throughput": accounts_vol_max_throughput,
        "network_interface": network_interface,
        "network_max_bandwidth_gbps": network_max_bandwidth_gbps,
    }.items():
        if value:
            request[key] = value
    if blockchain_process_names:
        request["blockchain_process_names"] = list(blockchain_process_names)
    qps: dict[str, int] = {}
    for key, value in {
        "initial": qps_initial,
        "max": qps_max,
        "step": qps_step,
        "duration_seconds": duration_seconds,
    }.items():
        if value is not None:
            qps[key] = int(value)
    if qps:
        request["qps"] = qps
    if rpc_methods:
        request["rpc_methods"] = list(rpc_methods)
    if mixed_weights:
        request["mixed_weighted"] = [
            {"method": method, "weight": int(weight)}
            for method, weight in mixed_weights.items()
        ]
    return request


def _inferred_values(plan: dict[str, Any]) -> dict[str, Any]:
    env = plan.get("execution", {}).get("environment", {})
    materialized = plan.get("materialized_config", {})
    return {
        "chain": plan.get("chain", ""),
        "rpc_mode": plan.get("rpc_mode", ""),
        "use_fake_node": plan.get("use_fake_node", False),
        "deployment": plan.get("deployment", {}),
        "runner_mode": plan.get("execution", {}).get("runner_mode", ""),
        "local_rpc_url": env.get("LOCAL_RPC_URL", ""),
        "mainnet_rpc_url": env.get("MAINNET_RPC_URL", ""),
        "ledger_device": materialized.get("LEDGER_DEVICE", ""),
        "accounts_device": materialized.get("ACCOUNTS_DEVICE", ""),
        "blockchain_process_names": materialized.get("BLOCKCHAIN_PROCESS_NAMES_STR", ""),
        "cloud_provider": materialized.get("CLOUD_PROVIDER", ""),
        "cloud_region": materialized.get("CLOUD_REGION", ""),
        "cloud_zone": materialized.get("CLOUD_ZONE", ""),
        "machine_type": materialized.get("MACHINE_TYPE", ""),
        "data_vol_type": materialized.get("DATA_VOL_TYPE", ""),
        "data_vol_size": materialized.get("DATA_VOL_SIZE", ""),
        "data_vol_max_iops": materialized.get("DATA_VOL_MAX_IOPS", ""),
        "data_vol_max_throughput": materialized.get("DATA_VOL_MAX_THROUGHPUT", ""),
        "accounts_vol_type": materialized.get("ACCOUNTS_VOL_TYPE", ""),
        "accounts_vol_size": materialized.get("ACCOUNTS_VOL_SIZE", ""),
        "accounts_vol_max_iops": materialized.get("ACCOUNTS_VOL_MAX_IOPS", ""),
        "accounts_vol_max_throughput": materialized.get("ACCOUNTS_VOL_MAX_THROUGHPUT", ""),
        "network_interface": materialized.get("NETWORK_INTERFACE", ""),
        "network_max_bandwidth_gbps": materialized.get("NETWORK_MAX_BANDWIDTH_GBPS", ""),
        "chain_template": plan.get("chain_template_requirements", {}),
    }


def _prepare_next_actions(plan: dict[str, Any], preflight: dict[str, Any]) -> list[str]:
    if plan.get("required_inputs"):
        return ["ask user to provide missing required values", "rerun prepare_benchmark_run"]
    if plan.get("required_questions"):
        return ["ask user to confirm inferred values", "rerun prepare_benchmark_run with confirmations"]
    if not preflight.get("passed"):
        return ["fix preflight blockers", "rerun prepare_benchmark_run"]
    return ["ask approval for run_smoke", "ask approval for run_fake_node_smoke_benchmark", "ask approval for submit_benchmark_job"]

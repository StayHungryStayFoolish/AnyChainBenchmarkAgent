"""Planning ADK tool wrappers for AnyChain benchmark requests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from ...onboarding.template_drafter import draft_chain_template as _draft_chain_template
    from ...planners.diff import diff_plans
    from ...planners.preflight import run_preflight as _run_preflight
    from ...planners.strategy_planner import generate_plan as _generate_plan
    from ...runners.benchmark_pipeline import _structured_request
    from ...runners.benchmark_pipeline import prepare_benchmark_run as _prepare_benchmark_run_core
    from ...runners.runbook import render_runbook as _render_runbook
    from ...runners.tool_result import tool_result as _tool_result
except ImportError:  # script execution with agent/ on sys.path
    from onboarding.template_drafter import draft_chain_template as _draft_chain_template
    from planners.diff import diff_plans
    from planners.preflight import run_preflight as _run_preflight
    from planners.strategy_planner import generate_plan as _generate_plan
    from runners.benchmark_pipeline import _structured_request
    from runners.benchmark_pipeline import prepare_benchmark_run as _prepare_benchmark_run_core
    from runners.runbook import render_runbook as _render_runbook
    from runners.tool_result import tool_result as _tool_result


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
    mainnet_rpc_url_reviewed: bool | None = None,
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
    observability_enabled: bool | None = None,
    observability_mode: str = "",
    observability_auto_stop: bool | None = None,
    workflow_type: str = "",
    sync_observe_stop_condition: str = "",
    sync_observe_duration_seconds: int | None = None,
    sync_observe_local_attribution: bool = True,
    exporter_port: str = "",
    prometheus_port: str = "",
    grafana_port: str = "",
    confirmations: list[str] | None = None,
    assumed_values: dict | None = None,
    assumed_for_smoke: bool = False,
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
    return _prepare_benchmark_run_core(
        source_prompt=source_prompt,
        chain=chain,
        goal=goal,
        rpc_mode=rpc_mode,
        use_fake_node=use_fake_node,
        deployment_type=deployment_type,
        cloud_provider=cloud_provider,
        target_rpc_url=target_rpc_url,
        mainnet_rpc_url=mainnet_rpc_url,
        mainnet_rpc_url_reviewed=mainnet_rpc_url_reviewed,
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
        observability_enabled=observability_enabled,
        observability_mode=observability_mode,
        observability_auto_stop=observability_auto_stop,
        workflow_type=workflow_type,
        sync_observe_stop_condition=sync_observe_stop_condition,
        sync_observe_duration_seconds=sync_observe_duration_seconds,
        sync_observe_local_attribution=sync_observe_local_attribution,
        exporter_port=exporter_port,
        prometheus_port=prometheus_port,
        grafana_port=grafana_port,
        confirmations=confirmations,
        assumed_values=assumed_values,
        assumed_for_smoke=assumed_for_smoke,
        output_dir=output_dir,
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
    mainnet_rpc_url_reviewed: bool | None = None,
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
    observability_enabled: bool | None = None,
    observability_mode: str = "",
    observability_auto_stop: bool | None = None,
    workflow_type: str = "",
    sync_observe_stop_condition: str = "",
    sync_observe_duration_seconds: int | None = None,
    exporter_port: str = "",
    prometheus_port: str = "",
    grafana_port: str = "",
    confirmations: list[str] | None = None,
    assumed_values: dict | None = None,
    assumed_for_smoke: bool = False,
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
        mainnet_rpc_url_reviewed=mainnet_rpc_url_reviewed,
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
        observability_enabled=observability_enabled,
        observability_mode=observability_mode,
        observability_auto_stop=observability_auto_stop,
        workflow_type=workflow_type,
        sync_observe_stop_condition=sync_observe_stop_condition,
        sync_observe_duration_seconds=sync_observe_duration_seconds,
        exporter_port=exporter_port,
        prometheus_port=prometheus_port,
        grafana_port=grafana_port,
        confirmations=confirmations,
        assumed_values=assumed_values,
        assumed_for_smoke=assumed_for_smoke,
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

    Always call this before fake-node smoke or real benchmark submission. If
    blockers are returned, explain them and ask the user for missing
    configuration instead of launching work.
    """
    preflight = _run_preflight(plan)
    return _tool_result(
        status="ok" if preflight.get("passed") else "blocked",
        data=preflight,
        warnings=preflight.get("warnings", []) + preflight.get("blockers", []),
        next_actions=["run_fake_node_smoke_benchmark", "ask_missing_required_values"] if preflight.get("passed") else ["fix blockers"],
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

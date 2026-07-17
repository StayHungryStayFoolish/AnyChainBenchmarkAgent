"""Shared benchmark preparation/execution pipeline.

Both the CLI tool-dispatch surface (`agent/tools/executor.py`) and the
LangGraph Harness (`agent/harness/domains/execution_runtime.py`) call these
functions directly, so the plan/preflight/smoke business logic lives in exactly one
place. `executor.py` adds the `approved`/confirmation gate for its callers;
the Harness calls straight in since it already gates approval itself via its
own graph state.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ..diagnostics.doctor import run_doctor as _run_doctor
from ..discovery.environment import discover_environment as _discover_environment
from ..planners.preflight import run_preflight as _run_preflight
from ..planners.strategy_planner import generate_plan as _generate_plan
from ..planners.strategy_planner import write_json
from .job_manager import submit_job as _submit_job
from .runbook import render_runbook as _render_runbook
from .tool_result import tool_result as _tool_result


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
    sync_observe_rpc_url: str = "",
    sync_observe_rpc_url_ready: bool = False,
    sync_observe_stop_condition: str = "",
    sync_observe_duration_seconds: int | None = None,
    sync_observe_source: str = "",
    node_process_identity: str = "",
    node_prometheus_metrics_url: str = "",
    exporter_port: str = "",
    prometheus_port: str = "",
    grafana_port: str = "",
    confirmations: list[str] | None = None,
    assumed_values: dict | None = None,
    assumed_for_smoke: bool = False,
    output_dir: str = ".agent/prepared",
) -> dict[str, Any]:
    """Run discovery, drafting, plan generation, preflight, and runbook rendering.

    Returns the same structured-result shape the ADK `prepare_benchmark_run`
    tool exposes to the LLM; the ADK wrapper is a thin pass-through to this
    function.
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
        mainnet_rpc_url_reviewed=mainnet_rpc_url_reviewed,
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
        observability_enabled=observability_enabled,
        observability_mode=observability_mode,
        observability_auto_stop=observability_auto_stop,
        workflow_type=workflow_type,
        sync_observe_rpc_url=sync_observe_rpc_url,
        sync_observe_rpc_url_ready=sync_observe_rpc_url_ready,
        sync_observe_stop_condition=sync_observe_stop_condition,
        sync_observe_duration_seconds=sync_observe_duration_seconds,
        sync_observe_source=sync_observe_source,
        node_process_identity=node_process_identity,
        node_prometheus_metrics_url=node_prometheus_metrics_url,
        exporter_port=exporter_port,
        prometheus_port=prometheus_port,
        grafana_port=grafana_port,
        confirmations=confirmations,
        assumed_values=assumed_values,
        assumed_for_smoke=assumed_for_smoke,
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


def run_fake_node_smoke_benchmark(
    plan_file: str,
    jobs_dir: str = ".agent/jobs",
    *,
    execution_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize the fake-node smoke plan and submit it as a detached job.

    Returns the same structured-result shape the ADK
    `run_fake_node_smoke_benchmark` tool exposes to the LLM; the ADK wrapper
    only adds the `approved` confirmation gate before calling this.
    """
    plan_path = Path(plan_file)
    if not plan_path.is_file():
        return _tool_result(
            status="blocked",
            data={"plan_file": plan_file},
            warnings=[f"plan file not found: {plan_file}"],
            next_actions=["prepare_benchmark_run", "write plan file"],
        )

    repo = Path(__file__).resolve().parents[2]
    smoke_root = Path(jobs_dir) / "fake_node_smoke" / f"{plan_path.stem}_{plan_path.stat().st_mtime_ns}"
    if not smoke_root.is_absolute():
        smoke_root = repo / smoke_root
    smoke_root.mkdir(parents=True, exist_ok=True)
    source_plan = dict(execution_plan) if execution_plan is not None else None
    smoke_plan = _fake_node_smoke_plan(plan_path, smoke_root, plan=source_plan)

    job = _submit_job(
        plan_path,
        jobs_dir=jobs_dir,
        mock=False,
        approved=True,
        execution_plan=smoke_plan,
    )
    benchmark_log = str(Path(job.get("run_dir", "")) / "benchmark.log") if job.get("run_dir") else ""
    evidence = [
        job.get("runtime_env_file", ""),
        job.get("artifact_index", ""),
        benchmark_log,
        str(smoke_root / "benchmark-data"),
    ]
    return _tool_result(
        status="ok" if job.get("status") in {"completed", "running"} else "failed",
        data={
            "job": job,
            "source_plan_file": str(plan_path),
            "smoke_plan_file": str(job.get("plan_file") or ""),
            "isolated_output_root": str(smoke_root),
            "terminal_commands": _job_terminal_commands(job),
        },
        evidence_paths=evidence,
        warnings=[job.get("error", "")] if job.get("error") else [],
        next_actions=_job_user_next_actions(job),
    )


def run_real_node_smoke_benchmark(
    plan_file: str,
    jobs_dir: str = ".agent/jobs",
    *,
    execution_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Submit a safe, isolated smoke for a prepared real-node plan."""

    plan_path = Path(plan_file)
    if not plan_path.is_file():
        return _tool_result(
            status="blocked",
            data={"plan_file": plan_file},
            warnings=[f"plan file not found: {plan_file}"],
            next_actions=["prepare_benchmark_run", "write plan file"],
        )

    repo = Path(__file__).resolve().parents[2]
    smoke_root = Path(jobs_dir) / "real_node_smoke" / f"{plan_path.stem}_{plan_path.stat().st_mtime_ns}"
    if not smoke_root.is_absolute():
        smoke_root = repo / smoke_root
    smoke_root.mkdir(parents=True, exist_ok=True)
    source_plan = dict(execution_plan) if execution_plan is not None else None
    smoke_plan = _real_node_smoke_plan(plan_path, smoke_root, plan=source_plan)

    job = _submit_job(
        plan_path,
        jobs_dir=jobs_dir,
        mock=False,
        approved=True,
        execution_plan=smoke_plan,
    )
    benchmark_log = str(Path(job.get("run_dir", "")) / "benchmark.log") if job.get("run_dir") else ""
    return _tool_result(
        status="ok" if job.get("status") in {"completed", "running"} else "failed",
        data={
            "job": job,
            "source_plan_file": str(plan_path),
            "smoke_plan_file": str(job.get("plan_file") or ""),
            "isolated_output_root": str(smoke_root),
            "terminal_commands": _job_terminal_commands(job),
        },
        evidence_paths=[
            job.get("runtime_env_file", ""),
            job.get("artifact_index", ""),
            benchmark_log,
            str(smoke_root / "benchmark-data"),
        ],
        warnings=[job.get("error", "")] if job.get("error") else [],
        next_actions=_job_user_next_actions(job),
    )


def submit_benchmark_job(
    plan_file: str,
    jobs_dir: str = ".agent/jobs",
    *,
    execution_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Submit a real benchmark job.

    Returns the same structured-result shape the ADK `submit_benchmark_job`
    tool exposes to the LLM; the ADK wrapper only adds the `approved`
    confirmation gate before calling this.
    """
    plan_path = Path(plan_file)
    if not plan_path.is_file():
        return _tool_result(
            status="blocked",
            data={"plan_file": plan_file},
            warnings=[f"plan file not found: {plan_file}"],
            next_actions=["generate_benchmark_plan", "write plan file"],
        )
    source_plan = dict(execution_plan) if execution_plan is not None else None
    executable_plan = _isolated_execution_plan(plan_path, Path(jobs_dir), plan=source_plan)
    job = _submit_job(
        plan_path,
        jobs_dir=jobs_dir,
        mock=False,
        approved=True,
        execution_plan=executable_plan,
    )
    return _tool_result(
        status="ok" if job.get("status") in {"completed", "running"} else "failed",
        data={"job": job, "terminal_commands": _job_terminal_commands(job)},
        evidence_paths=[job.get("plan_file", ""), job.get("runtime_env_file", ""), job.get("artifact_index", "")],
        warnings=[job.get("error", "")] if job.get("error") else [],
        next_actions=_job_user_next_actions(job),
    )


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
    mainnet_rpc_url_reviewed: bool | None,
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
    observability_enabled: bool | None,
    observability_mode: str,
    observability_auto_stop: bool | None,
    workflow_type: str,
    sync_observe_rpc_url: str,
    sync_observe_rpc_url_ready: bool = False,
    sync_observe_stop_condition: str,
    sync_observe_duration_seconds: int | None,
    sync_observe_source: str,
    node_process_identity: str = "",
    node_prometheus_metrics_url: str,
    exporter_port: str,
    prometheus_port: str,
    grafana_port: str,
    confirmations: list[str] | None,
    assumed_values: dict | None,
    assumed_for_smoke: bool,
) -> dict[str, Any]:
    if assumed_for_smoke:
        (
            goal,
            rpc_mode,
            use_fake_node,
            blockchain_process_names,
            ledger_device,
            accounts_device,
            data_vol_type,
            data_vol_size,
            data_vol_max_iops,
            data_vol_max_throughput,
            accounts_vol_type,
            accounts_vol_size,
            accounts_vol_max_iops,
            accounts_vol_max_throughput,
            network_interface,
            network_max_bandwidth_gbps,
            qps_initial,
            qps_max,
            qps_step,
            duration_seconds,
            observability_enabled,
            confirmations,
            assumed_values,
        ) = _apply_assumed_smoke_defaults(
            goal=goal,
            rpc_mode=rpc_mode,
            use_fake_node=use_fake_node,
            blockchain_process_names=blockchain_process_names,
            ledger_device=ledger_device,
            accounts_device=accounts_device,
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
            observability_enabled=observability_enabled,
            confirmations=confirmations,
            assumed_values=assumed_values,
        )
    observability = {
        "enabled": bool(observability_enabled) if observability_enabled is not None else False,
        "mode": observability_mode if observability_mode in {"local", "exporter"} else "local",
        "auto_stop": True if observability_auto_stop is None else bool(observability_auto_stop),
    }
    request: dict[str, Any] = {
        "chain": chain,
        "goal": goal or "baseline",
        "rpc_mode": rpc_mode or "single",
        "workflow_type": workflow_type,
        "run_mode": workflow_type,
        "deployment": {
            "type": deployment_type or "unknown",
            "provider": cloud_provider,
        },
        "observability": observability,
        "dependency_mode": "audit",
        "runner_mode": "detached",
        "bottleneck_focus": ["cpu", "memory", "disk", "network", "rpc_errors"],
        "source_prompt": source_prompt,
    }
    if confirmations:
        request["confirmations"] = list(confirmations)
    if assumed_for_smoke:
        request["assumed_for_smoke"] = True
        request["assumed_values"] = dict(assumed_values or {})
    if use_fake_node is not None:
        request["use_fake_node"] = bool(use_fake_node)
    for key, value in {
        "local_rpc_url": target_rpc_url,
        "target_rpc_url": target_rpc_url,
        "mainnet_rpc_url": mainnet_rpc_url,
        "mainnet_rpc_url_reviewed": bool(mainnet_rpc_url_reviewed) if mainnet_rpc_url_reviewed is not None else False,
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
        "exporter_port": exporter_port or "9108",
        "prometheus_port": prometheus_port or "9091",
        "grafana_port": grafana_port or "3001",
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
    if workflow_type:
        request["workflow_type"] = workflow_type
        request["run_mode"] = workflow_type
    if sync_observe_source:
        request["sync_observe_source"] = sync_observe_source
    if sync_observe_rpc_url:
        request["sync_observe_rpc_url"] = sync_observe_rpc_url
        request["local_rpc_url"] = sync_observe_rpc_url
        request["sync_observe_rpc_url_ready"] = bool(sync_observe_rpc_url_ready)
    if node_process_identity:
        request["node_process_identity"] = node_process_identity
        if not request.get("blockchain_process_names"):
            request["blockchain_process_names"] = [node_process_identity]
    if node_prometheus_metrics_url:
        request["node_prometheus_metrics_url"] = node_prometheus_metrics_url
    if sync_observe_stop_condition:
        request["sync_observe_stop_condition"] = sync_observe_stop_condition
    if sync_observe_duration_seconds is not None:
        request["sync_observe_duration_seconds"] = int(sync_observe_duration_seconds)
    if rpc_methods:
        request["rpc_methods"] = list(rpc_methods)
    if mixed_weights:
        request["mixed_weighted"] = [
            {"method": method, "weight": int(weight)}
            for method, weight in mixed_weights.items()
        ]
    return request


def _apply_assumed_smoke_defaults(
    *,
    goal: str,
    rpc_mode: str,
    use_fake_node: bool | None,
    blockchain_process_names: list[str] | None,
    ledger_device: str,
    accounts_device: str,
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
    observability_enabled: bool | None,
    confirmations: list[str] | None,
    assumed_values: dict | None,
) -> tuple[
    str,
    str,
    bool,
    list[str],
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    int,
    int,
    int,
    int,
    bool,
    list[str],
    dict,
]:
    """Materialize safe smoke-only defaults after explicit user approval.

    This is intentionally deterministic. The LLM may infer that the user wants
    a quick framework check with assumed values, but the benchmark toolchain
    owns which assumptions are safe and how they are marked. These values must
    never be promoted into a real-node benchmark because execution_gate blocks
    plans with ``assumed_for_smoke``.
    """
    merged_confirmations = set(confirmations or [])
    merged_confirmations.update(
        {
            "benchmark_mode_confirmed",
            "qps_profile_confirmed",
            "observability_choice_confirmed",
            "chain_template_reviewed",
            "rpc_workload_confirmed",
            "rpc_workload_confirmation",
            "rpc_param_samples_confirmed",
            "rpc_param_samples_confirmation",
            "custom_rpc_method_review",
            "advanced_config_review",
            "disk_inventory_confirmation",
            "ledger_device_confirmation",
            "has_accounts_device",
            "blockchain_process_names",
            "accounts_device",
            "accounts_vol_type",
            "accounts_vol_size",
            "accounts_vol_max_iops",
            "accounts_vol_max_throughput",
            "data_vol_type",
            "data_vol_size",
            "data_vol_max_iops",
            "data_vol_max_throughput",
            "network_max_bandwidth_gbps",
            "network_interface",
        }
    )
    values = dict(assumed_values or {})

    def pick(key: str, current: str, default: str) -> str:
        value = current or str(values.get(key, "")) or default
        values[key] = value
        return value

    process_names = list(blockchain_process_names or values.get("blockchain_process_names") or [])
    if not process_names:
        process_names = ["fake-node"]
    values["blockchain_process_names"] = process_names
    selected_accounts_device = pick("accounts_device", accounts_device, "") if (accounts_device or values.get("accounts_device")) else ""
    if selected_accounts_device:
        accounts_vol_type = pick("accounts_vol_type", accounts_vol_type, "assumed-smoke-accounts-disk")
        accounts_vol_size = pick("accounts_vol_size", accounts_vol_size, "100")
        accounts_vol_max_iops = pick("accounts_vol_max_iops", accounts_vol_max_iops, "3000")
        accounts_vol_max_throughput = pick("accounts_vol_max_throughput", accounts_vol_max_throughput, "125")

    return (
        "smoke",
        rpc_mode or "single",
        True if use_fake_node is None else bool(use_fake_node),
        process_names,
        pick("ledger_device", ledger_device, "assumed-smoke-ledger"),
        selected_accounts_device,
        pick("data_vol_type", data_vol_type, "assumed-smoke-disk"),
        pick("data_vol_size", data_vol_size, "100"),
        pick("data_vol_max_iops", data_vol_max_iops, "3000"),
        pick("data_vol_max_throughput", data_vol_max_throughput, "125"),
        accounts_vol_type,
        accounts_vol_size,
        accounts_vol_max_iops,
        accounts_vol_max_throughput,
        pick("network_interface", network_interface, "assumed-smoke-net0"),
        pick("network_max_bandwidth_gbps", network_max_bandwidth_gbps, "10"),
        int(qps_initial or values.get("qps_initial") or 1),
        int(qps_max or values.get("qps_max") or 1),
        int(qps_step or values.get("qps_step") or 1),
        int(duration_seconds or values.get("duration_seconds") or 10),
        False if observability_enabled is None else bool(observability_enabled),
        sorted(merged_confirmations),
        values,
    )


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
    if not preflight.get("passed"):
        return ["fix preflight blockers", "rerun prepare_benchmark_run"]
    return ["ask approval for run_fake_node_smoke_benchmark", "ask approval for submit_benchmark_job"]


def _fake_node_smoke_plan(
    plan_file: Path,
    smoke_root: Path,
    *,
    plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    plan = dict(plan) if plan is not None else json.loads(plan_file.read_text(encoding="utf-8"))
    if not smoke_root.is_absolute():
        smoke_root = Path(__file__).resolve().parents[2] / smoke_root
    smoke_output_root = (smoke_root / "benchmark-data").resolve()
    smoke_memory_dir = (smoke_root / "memory").resolve()
    plan["plan_id"] = f"{plan.get('plan_id', 'plan')}_fake_node_smoke"
    plan["strategy"] = "smoke"
    plan["goal"] = "smoke"
    plan["benchmark_mode"] = "quick"
    plan["use_fake_node"] = True
    plan["required_inputs"] = [
        item for item in plan.get("required_inputs", [])
        if item not in _FAKE_NODE_IGNORED_REQUIREMENTS
    ]
    checklist = dict(plan.get("configuration_checklist", {}))
    if checklist:
        checklist["missing_blockers"] = [
            item for item in checklist.get("missing_blockers", [])
            if item not in _FAKE_NODE_IGNORED_REQUIREMENTS
        ]
        checklist["summary"] = "fake-node smoke uses isolated job-local output; only real-node endpoint blockers are ignored."
        plan["configuration_checklist"] = checklist
    command = ["./blockchain_node_benchmark.sh", "--quick", f"--{plan.get('rpc_mode', 'single')}", "--fake-node"]
    execution = dict(plan.get("execution", {}))
    env = dict(execution.get("environment", {}))
    env.update({
        "BLOCKCHAIN_NODE": plan.get("chain", env.get("BLOCKCHAIN_NODE", "")),
        "RPC_MODE": plan.get("rpc_mode", env.get("RPC_MODE", "single")),
        "LOCAL_RPC_URL": env.get("LOCAL_RPC_URL", ""),
        "QUICK_INITIAL_QPS": "1",
        "QUICK_MAX_QPS": "1",
        "QUICK_QPS_STEP": "1",
        "QUICK_DURATION": "10",
        "QPS_WARMUP_DURATION": "0",
        "QPS_COOLDOWN": "0",
        "BLOCKCHAIN_BENCHMARK_DATA_DIR": str(smoke_output_root),
        "MEMORY_SHARE_DIR": str(smoke_memory_dir),
    })
    execution.update({
        "command": command,
        "environment": env,
        # Agent turns must not block on even a quick benchmark. Submit the
        # smoke run as a detached job, then let the terminal expose status/log
        # follow commands while the benchmark engine writes artifacts.
        "runner_mode": "detached",
        "idempotency_key": execution.get("idempotency_key") or f"fake-node-smoke:{plan.get('plan_id', plan_file.stem)}",
    })
    plan["execution"] = execution
    advanced_defaults = dict(plan.get("advanced_defaults", {}))
    advanced_defaults["qps"] = {
        "initial": 1,
        "max": 1,
        "step": 1,
        "duration_seconds": 10,
    }
    plan["advanced_defaults"] = advanced_defaults
    materialized = dict(plan.get("materialized_config", {}))
    materialized.update({
        "BLOCKCHAIN_NODE": env.get("BLOCKCHAIN_NODE", ""),
        "RPC_MODE": env.get("RPC_MODE", "single"),
        "BLOCKCHAIN_PROCESS_NAMES_STR": materialized.get("BLOCKCHAIN_PROCESS_NAMES_STR") or "fake-node",
        "BLOCKCHAIN_BENCHMARK_DATA_DIR": str(smoke_output_root),
        "MEMORY_SHARE_DIR": str(smoke_memory_dir),
    })
    plan["materialized_config"] = materialized
    artifacts = dict(plan.get("artifacts", {}))
    artifacts.update({
        "fake_node_smoke_output_root": str(smoke_output_root),
        "fake_node_smoke_memory_dir": str(smoke_memory_dir),
    })
    plan["artifacts"] = artifacts
    return plan


def _real_node_smoke_plan(
    plan_file: Path,
    smoke_root: Path,
    *,
    plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive an isolated low-traffic plan without changing the final plan."""

    import json

    plan = dict(plan) if plan is not None else dict(json.loads(plan_file.read_text(encoding="utf-8")))
    if not smoke_root.is_absolute():
        smoke_root = Path(__file__).resolve().parents[2] / smoke_root
    smoke_output_root = (smoke_root / "benchmark-data").resolve()
    smoke_memory_dir = (smoke_root / "memory").resolve()
    original_plan_id = str(plan.get("plan_id") or plan_file.stem)
    plan["plan_id"] = f"{original_plan_id}_real_node_smoke"
    plan["strategy"] = "smoke"
    plan["goal"] = "smoke"
    plan["benchmark_mode"] = "quick"
    plan["use_fake_node"] = False

    execution = dict(plan.get("execution") or {})
    env = dict(execution.get("environment") or {})
    env.update({
        "QUICK_INITIAL_QPS": "1",
        "QUICK_MAX_QPS": "1",
        "QUICK_QPS_STEP": "1",
        "QUICK_DURATION": "10",
        "QPS_WARMUP_DURATION": "0",
        "QPS_COOLDOWN": "0",
        "BLOCKCHAIN_BENCHMARK_DATA_DIR": str(smoke_output_root),
        "MEMORY_SHARE_DIR": str(smoke_memory_dir),
    })
    base_key = str(execution.get("idempotency_key") or f"real-node:{original_plan_id}")
    execution.update({
        "command": ["./blockchain_node_benchmark.sh", "--quick", f"--{plan.get('rpc_mode', 'single')}"],
        "environment": env,
        "runner_mode": "detached",
        "idempotency_key": f"{base_key}:real-node-smoke",
    })
    plan["execution"] = execution

    advanced_defaults = dict(plan.get("advanced_defaults") or {})
    advanced_defaults["qps"] = {"initial": 1, "max": 1, "step": 1, "duration_seconds": 10}
    plan["advanced_defaults"] = advanced_defaults
    materialized = dict(plan.get("materialized_config") or {})
    materialized.update({
        "BLOCKCHAIN_BENCHMARK_DATA_DIR": str(smoke_output_root),
        "MEMORY_SHARE_DIR": str(smoke_memory_dir),
    })
    plan["materialized_config"] = materialized
    artifacts = dict(plan.get("artifacts") or {})
    artifacts.update({
        "real_node_smoke_output_root": str(smoke_output_root),
        "real_node_smoke_memory_dir": str(smoke_memory_dir),
    })
    plan["artifacts"] = artifacts
    return plan


def _isolated_execution_plan(
    plan_file: Path,
    jobs_dir: Path,
    *,
    plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build final execution values; job manager owns the only written copy."""

    plan_path = plan_file.resolve()
    plan = dict(plan) if plan is not None else dict(json.loads(plan_path.read_text(encoding="utf-8")))
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()[:16]
    repo = Path(__file__).resolve().parents[2]
    root = jobs_dir / "final_benchmark" / f"{plan_path.stem}_{digest}"
    if not root.is_absolute():
        root = repo / root
    output_root = (root / "benchmark-data").resolve()
    memory_dir = (root / "memory").resolve()

    execution = dict(plan.get("execution") or {})
    env = dict(execution.get("environment") or {})
    env.update({
        "BLOCKCHAIN_BENCHMARK_DATA_DIR": str(output_root),
        "MEMORY_SHARE_DIR": str(memory_dir),
    })
    execution.update({
        "environment": env,
        "idempotency_key": str(execution.get("idempotency_key") or f"benchmark:{plan_path.stem}:{digest}"),
    })
    plan["execution"] = execution
    materialized = dict(plan.get("materialized_config") or {})
    materialized.update({
        "BLOCKCHAIN_BENCHMARK_DATA_DIR": str(output_root),
        "MEMORY_SHARE_DIR": str(memory_dir),
    })
    plan["materialized_config"] = materialized
    artifacts = dict(plan.get("artifacts") or {})
    artifacts.update({
        "execution_output_root": str(output_root),
        "execution_memory_dir": str(memory_dir),
        "source_plan_file": str(plan_path),
    })
    plan["artifacts"] = artifacts
    return plan


_FAKE_NODE_IGNORED_REQUIREMENTS = {
    "local_rpc_url",
    "mainnet_rpc_url_reviewed",
}


def _nested_job(tool_result_payload: dict[str, Any]) -> dict[str, Any]:
    data = tool_result_payload.get("data", {}) if isinstance(tool_result_payload, dict) else {}
    if isinstance(data, dict):
        job = data.get("job")
        if isinstance(job, dict):
            return job
    return {}


def _job_terminal_commands(job: dict[str, Any]) -> dict[str, str]:
    job_id = str(job.get("job_id", "") or "").strip()
    if not job_id:
        return {"status": "status", "logs": "logs", "follow": "follow", "analyze": "analyze latest job"}
    return {
        "status": f"status {job_id}",
        "logs": f"logs {job_id}",
        "follow": f"follow {job_id}",
        "analyze": "analyze latest job",
    }


def _job_user_next_actions(job: dict[str, Any]) -> list[str]:
    commands = _job_terminal_commands(job)
    return [
        f"check status with `{commands['status']}`",
        f"show recent logs with `{commands['logs']}`",
        f"stream logs with `{commands['follow']}`",
        f"after completion, ask `{commands['analyze']}`",
    ]

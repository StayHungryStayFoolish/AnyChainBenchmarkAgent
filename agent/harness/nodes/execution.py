"""Execution handoff for approved LangGraph Harness runs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

AGENT_ROOT = Path(__file__).resolve().parents[2]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

try:
    from ...runners.benchmark_pipeline import (
        prepare_benchmark_run,
        run_fake_node_smoke_benchmark,
        submit_benchmark_job,
    )
except ImportError:  # script execution with agent/ on sys.path
    from runners.benchmark_pipeline import (
        prepare_benchmark_run,
        run_fake_node_smoke_benchmark,
        submit_benchmark_job,
    )

from ..state import AgentGraphState


def run_approved_preflight_and_smoke(state: AgentGraphState) -> AgentGraphState:
    """Run deterministic preflight/smoke after explicit graph approval."""

    output: AgentGraphState = dict(state)
    if output.get("preflight", {}).get("status") in {"passed", "blocked"} or output.get("smoke"):
        return output

    # Preserve any message already queued by the caller (e.g. a demo-mode
    # disclaimer shown right before auto-triggering execution) instead of
    # overwriting it -- mirrors the prefix-then-append pattern used
    # throughout `groups.py` (e.g. `_ask_next_blocking_question`).
    prefix = list(output.get("visible_response") or [])

    if output.get("workflow_mode") == "sync_observe":
        prepared = prepare_benchmark_run(**_prepare_kwargs(output))
        data = prepared.get("data", {})
        preflight = data.get("preflight", {})
        output["plan"] = data.get("plan", {})
        output["plan_file"] = data.get("plan_file", "")
        output["preflight"] = {
            **preflight,
            "status": "passed" if preflight.get("passed") else "blocked",
            "evidence_paths": prepared.get("evidence_paths", []),
        }
        if not preflight.get("passed"):
            output["visible_response"] = prefix + [_blocked_message(prepared)]
            output["_stop_after_response"] = True
            return output
        job_result = submit_benchmark_job(str(data.get("plan_file", "")))
        output["job"] = (job_result.get("data") or {}).get("job", {})
        output["visible_response"] = prefix + [_job_message(job_result, prefix="Sync-observe job submitted")]
        output["_stop_after_response"] = True
        return output

    prepared = prepare_benchmark_run(**_prepare_kwargs(output))
    data = prepared.get("data", {})
    preflight = data.get("preflight", {})
    output["plan"] = data.get("plan", {})
    output["plan_file"] = data.get("plan_file", "")
    output["preflight"] = {
        **preflight,
        "status": "passed" if preflight.get("passed") else "blocked",
        "evidence_paths": prepared.get("evidence_paths", []),
    }
    if not preflight.get("passed"):
        output["visible_response"] = prefix + [_blocked_message(prepared)]
        output["_stop_after_response"] = True
        return output

    if output.get("target_mode") == "fake-node":
        smoke = run_fake_node_smoke_benchmark(str(data.get("plan_file", "")))
        output["smoke"] = smoke
        output["job"] = (smoke.get("data") or {}).get("job", {})
        output["visible_response"] = prefix + [_smoke_message(smoke)]
        output["_stop_after_response"] = True
        return output

    output["visible_response"] = prefix + [
        "Preflight passed. Real-node execution still requires an isolated smoke validation before final benchmark submission."
    ]
    output["_stop_after_response"] = True
    return output


def _prepare_kwargs(state: AgentGraphState) -> dict[str, Any]:
    confirmed = state.get("confirmed_config") or {}
    qps = state.get("qps_profile") or {}
    qps_overrides = qps.get("overrides") or {}
    obs = state.get("observability") or {}
    workload = state.get("workload") or {}
    target_mode = state.get("target_mode")
    chain = (state.get("chain_identity") or {}).get("canonical") or confirmed.get("BLOCKCHAIN_NODE", "")
    kwargs = {
        "source_prompt": "LangGraph Harness approved preflight/smoke",
        "chain": chain,
        "goal": _goal_from_mode(str(qps.get("mode") or "quick")),
        "rpc_mode": state.get("rpc_mode") or "single",
        "use_fake_node": target_mode == "fake-node",
        "target_rpc_url": str(confirmed.get("LOCAL_RPC_URL") or ""),
        "blockchain_process_names": ["fake-node"] if target_mode == "fake-node" else [],
        "deployment_type": str(((state.get("discovery") or {}).get("deployment") or {}).get("type") or ""),
        "cloud_provider": str(((state.get("discovery") or {}).get("cloud") or {}).get("provider") or ""),
        "ledger_device": str(confirmed.get("LEDGER_DEVICE") or ""),
        "accounts_device": str(confirmed.get("ACCOUNTS_DEVICE") or ""),
        "cloud_region": str(confirmed.get("CLOUD_REGION") or ""),
        "cloud_zone": str(confirmed.get("CLOUD_ZONE") or ""),
        "machine_type": str(confirmed.get("MACHINE_TYPE") or ""),
        "data_vol_type": str(confirmed.get("DATA_VOL_TYPE") or ""),
        "data_vol_size": str(confirmed.get("DATA_VOL_SIZE") or ""),
        "data_vol_max_iops": str(confirmed.get("DATA_VOL_MAX_IOPS") or ""),
        "data_vol_max_throughput": str(confirmed.get("DATA_VOL_MAX_THROUGHPUT") or ""),
        "accounts_vol_type": str(confirmed.get("ACCOUNTS_VOL_TYPE") or ""),
        "accounts_vol_size": str(confirmed.get("ACCOUNTS_VOL_SIZE") or ""),
        "accounts_vol_max_iops": str(confirmed.get("ACCOUNTS_VOL_MAX_IOPS") or ""),
        "accounts_vol_max_throughput": str(confirmed.get("ACCOUNTS_VOL_MAX_THROUGHPUT") or ""),
        "network_interface": str(confirmed.get("NETWORK_INTERFACE") or ""),
        "network_max_bandwidth_gbps": str(confirmed.get("NETWORK_MAX_BANDWIDTH_GBPS") or ""),
        "qps_initial": _int_or_none(qps_overrides.get("INITIAL_QPS")),
        "qps_max": _int_or_none(qps_overrides.get("MAX_QPS")),
        "qps_step": _int_or_none(qps_overrides.get("QPS_STEP")),
        "duration_seconds": _int_or_none(qps_overrides.get("DURATION")),
        "observability_enabled": obs.get("mode") not in {"", "disabled", None},
        "observability_mode": str(obs.get("mode") or "disabled"),
        "workflow_type": "sync_observe" if state.get("workflow_mode") == "sync_observe" else "rpc_benchmark",
        "sync_observe_stop_condition": str((state.get("sync_observe") or {}).get("stop_condition") or ""),
        "sync_observe_duration_seconds": _int_or_none((state.get("sync_observe") or {}).get("duration_seconds")),
        "sync_observe_source": str((state.get("sync_observe") or {}).get("source") or ""),
        "mainnet_rpc_url_reviewed": bool(confirmed.get("MAINNET_RPC_URL_REVIEWED")),
        "confirmations": [
            "benchmark_mode_confirmed",
            "qps_profile_confirmed",
            "observability_choice_confirmed",
            "chain_template_reviewed",
            "rpc_workload_confirmed",
            "rpc_workload_confirmation",
            "rpc_param_samples_confirmed",
            "rpc_param_samples_confirmation",
            "advanced_config_review",
            "disk_inventory_confirmation",
            "ledger_device_confirmation",
            "has_accounts_device",
            "sync_observe_stop_condition",
        ],
    }
    # Mixed RPC mode requires weights to be confirmed. Both the default-workload
    # path (template weights sum to 100) and the custom/adjusted path (validated
    # to sum to 100) set `workload.confirmed`; without this the checklist blocks
    # every mixed run on `mixed_weights_confirmed`.
    if state.get("rpc_mode") == "mixed" and workload.get("confirmed"):
        kwargs["confirmations"].append("mixed_weights_confirmed")
    process_names = str(confirmed.get("BLOCKCHAIN_PROCESS_NAMES") or "").strip()
    if process_names:
        kwargs["blockchain_process_names"] = [process_names]
    if confirmed.get("MAINNET_RPC_URL"):
        kwargs["mainnet_rpc_url"] = str(confirmed.get("MAINNET_RPC_URL") or "")
    methods = workload.get("methods")
    weights = workload.get("mixed_weights")
    if isinstance(methods, list) and methods:
        kwargs["rpc_methods"] = [str(method) for method in methods if str(method).strip()]
    if isinstance(weights, dict) and weights:
        kwargs["mixed_weights"] = {str(method): int(weight) for method, weight in weights.items()}
    return kwargs


def _goal_from_mode(mode: str) -> str:
    if mode == "quick":
        return "smoke"
    if mode == "intensive":
        return "stress"
    return "baseline"


def _int_or_none(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def _blocked_message(prepared: dict[str, Any]) -> str:
    data = prepared.get("data", {})
    preflight = data.get("preflight", {})
    blockers = preflight.get("blockers") or prepared.get("warnings") or ["unknown blocker"]
    evidence = ", ".join(str(item) for item in prepared.get("evidence_paths", []) if item)
    return f"Preflight is blocked: {'; '.join(str(item) for item in blockers)}. Evidence: {evidence or '<none>'}"


def _smoke_message(smoke: dict[str, Any]) -> str:
    data = smoke.get("data") or {}
    job = data.get("job") or {}
    commands = data.get("terminal_commands") or {}
    command_text = "; ".join(str(value) for value in commands.values())
    return (
        f"Fake-node smoke submitted: status={smoke.get('status')}, job_id={job.get('job_id', '<unknown>')}. "
        f"Use: {command_text or 'jobs/status/logs'}"
    )


def _job_message(result: dict[str, Any], *, prefix: str) -> str:
    data = result.get("data") or {}
    job = data.get("job") or {}
    commands = data.get("terminal_commands") or {}
    command_text = "; ".join(str(value) for value in commands.values())
    return (
        f"{prefix}: status={result.get('status')}, job_id={job.get('job_id', '<unknown>')}. "
        f"Use: {command_text or 'jobs/status/logs'}"
    )

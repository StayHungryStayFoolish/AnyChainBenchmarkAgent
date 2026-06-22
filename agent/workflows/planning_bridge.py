"""Bridge terminal workflow state into deterministic benchmark plans."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from discovery.environment import discover_environment
from planners.preflight import run_preflight
from planners.strategy_planner import generate_plan, write_json
from runners.job_manager import submit_job
from runners.runbook import render_runbook
from workflows.state import WorkflowState


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREPARED_DIR = REPO_ROOT / ".agent" / "prepared"
DEFAULT_JOBS_DIR = REPO_ROOT / ".agent" / "jobs"


def request_from_state(state: WorkflowState, goal: str = "smoke") -> dict[str, Any]:
    """Return a planner request from confirmed terminal workflow values."""
    values = state.confirmed_values
    request: dict[str, Any] = {
        "chain": str(values.get("chain", "")).strip().lower(),
        "goal": goal,
        "rpc_mode": values.get("rpc_mode", "single"),
        "use_fake_node": bool(values.get("use_fake_node", False)),
        "deployment": {
            "type": values.get("deployment_type", "unknown"),
            "provider": values.get("cloud_provider", ""),
        },
        "observability": {"enabled": False, "mode": "local"},
        "dependency_mode": "audit",
        "runner_mode": values.get("runner_mode", "detached"),
        "bottleneck_focus": ["cpu", "memory", "disk", "network", "rpc_errors"],
        "confirmations": [
            "mixed_weights_confirmation",
            "rpc_workload_confirmation",
            "rpc_param_samples_confirmation",
        ],
        "source_prompt": "terminal workflow confirmed by user",
    }
    passthrough_keys = {
        "local_rpc_url",
        "mainnet_rpc_url",
        "ledger_device",
        "accounts_device",
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
    }
    for key in passthrough_keys:
        if values.get(key):
            request[key] = values[key]
    if values.get("blockchain_process_names"):
        request["blockchain_process_names"] = values["blockchain_process_names"]
    if values.get("rpc_methods"):
        request["rpc_methods"] = values["rpc_methods"]
    if values.get("mixed_weights"):
        request["mixed_weights"] = values["mixed_weights"]
    return request


def prepare_plan_from_state(
    state: WorkflowState,
    output_dir: str | Path = DEFAULT_PREPARED_DIR,
) -> dict[str, Any]:
    """Generate plan, run preflight, and write review artifacts."""
    discovery = discover_environment()
    request = request_from_state(state)
    request["discovery"] = discovery
    plan = generate_plan(request, discovery=discovery)
    preflight = run_preflight(plan)

    prepared_dir = Path(output_dir)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    plan_file = prepared_dir / f"{plan['plan_id']}.json"
    runbook_file = prepared_dir / f"{plan['plan_id']}_runbook.md"
    write_json(plan_file, plan)
    runbook_file.write_text(render_runbook(plan), encoding="utf-8")
    return {
        "request": request,
        "plan": plan,
        "preflight": preflight,
        "plan_file": str(plan_file),
        "runbook_file": str(runbook_file),
    }


def submit_mock_smoke_from_plan(
    plan_file: str | Path,
    jobs_dir: str | Path = DEFAULT_JOBS_DIR,
) -> dict[str, Any]:
    """Submit a lifecycle-only job to verify runtime.env and artifact plumbing."""
    return submit_job(plan_file, jobs_dir=jobs_dir, mock=True, approved=True)

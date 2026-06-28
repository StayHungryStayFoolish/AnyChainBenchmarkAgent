"""Generate deterministic benchmark plans from Agent requests."""

from __future__ import annotations

import json
import hashlib
import time
import uuid
from pathlib import Path
from typing import Any

from planners.chain_template_requirements import inspect_chain_template
from planners.config_checklist import build_configuration_checklist, missing_required_from_checklist
from planners.risk import score_plan_risk
from planners.config_questions import required_questions


REPO_ROOT = Path(__file__).resolve().parents[2]


GOAL_TO_STRATEGY = {
    "smoke": "smoke",
    "baseline": "baseline",
    "max_stable_qps": "ramp",
    "stress": "stress",
    "bottleneck_confirmation": "bottleneck-confirmation",
    "regression": "regression",
}


DEFAULT_QPS = {
    "smoke": {"initial": 1000, "max": 1500, "step": 500, "duration_seconds": 60},
    "baseline": {"initial": 2000, "max": 50000, "step": 500, "duration_seconds": 600},
    "ramp": {"initial": 2000, "max": 50000, "step": 500, "duration_seconds": 600},
    "stress": {"initial": 50000, "max": 9999999, "step": 250, "duration_seconds": 600},
    "bottleneck-confirmation": {"initial": 2000, "max": 50000, "step": 500, "duration_seconds": 600},
    "regression": {"initial": 2000, "max": 50000, "step": 500, "duration_seconds": 600},
}


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def generate_plan(request: dict[str, Any], discovery: dict[str, Any] | None = None) -> dict[str, Any]:
    chain = (request.get("chain") or "").strip().lower()
    goal = request.get("goal") or "baseline"
    strategy = _strategy_from_mode(request.get("benchmark_mode")) or GOAL_TO_STRATEGY.get(goal, "baseline")
    benchmark_mode = _benchmark_mode(strategy)
    rpc_mode = request.get("rpc_mode") or "single"
    use_fake_node = request.get("use_fake_node") if isinstance(request.get("use_fake_node"), bool) else None
    qps = {**DEFAULT_QPS[strategy], **request.get("qps", {})}
    confirmations = set(request.get("confirmations", []))
    runner_mode = request.get("runner_mode", "detached")
    if runner_mode not in {"detached", "foreground"}:
        runner_mode = "detached"

    required_inputs = []
    if not chain:
        required_inputs.append("chain")
    if use_fake_node is None:
        required_inputs.append("use_fake_node")
    if use_fake_node is False and not request.get("local_rpc_url"):
        required_inputs.append("local_rpc_url")
    requires_confirmation = []
    if request.get("workload", {}).get("methods") and "mixed_weights_confirmation" not in confirmations:
        requires_confirmation.append("mixed_weights")
    if strategy == "stress" and "stress_execution_confirmation" not in confirmations:
        requires_confirmation.append("stress_execution")

    command = ["./blockchain_node_benchmark.sh", _mode_flag(strategy), f"--{rpc_mode}"]
    if use_fake_node is True:
        command.append("--fake-node")

    qps_prefix = _qps_env_prefix(strategy)
    env = {
        "BLOCKCHAIN_NODE": chain,
        "RPC_MODE": rpc_mode,
        "LOCAL_RPC_URL": request.get("local_rpc_url", ""),
        "MAINNET_RPC_URL": request.get("mainnet_rpc_url", ""),
        f"{qps_prefix}_INITIAL_QPS": str(qps["initial"]),
        f"{qps_prefix}_MAX_QPS": str(qps["max"]),
        f"{qps_prefix}_QPS_STEP": str(qps["step"]),
        f"{qps_prefix}_DURATION": str(qps["duration_seconds"]),
        "OBSERVABILITY_STACK_ENABLED": str(
            bool(request.get("observability", {}).get("enabled", False))
        ).lower(),
        "OBSERVABILITY_STACK_MODE": request.get("observability", {}).get("mode", "local"),
        "OBSERVABILITY_STACK_AUTO_STOP": str(
            request.get("observability", {}).get("auto_stop", True)
        ).lower(),
        "EXPORTER_PORT": str(request.get("exporter_port", "")),
        "PROMETHEUS_PORT": str(request.get("prometheus_port", "")),
        "GRAFANA_PORT": str(request.get("grafana_port", "")),
    }
    discovery_payload = discovery or request.get("discovery") or {
        "source": "not_collected",
        "warnings": ["Environment discovery was not collected for this plan."],
    }
    deployment = request.get("deployment", {"type": "unknown"})
    if deployment.get("type") == "unknown" and discovery_payload.get("deployment", {}).get("type"):
        deployment = {"type": discovery_payload["deployment"]["type"], "provider": discovery_payload.get("cloud", {}).get("provider", "")}

    materialized_config = {
        "CLOUD_PROVIDER": deployment.get("provider") or discovery_payload.get("cloud", {}).get("provider", ""),
        "REPORT_CLOUD_PROVIDER": deployment.get("provider") or discovery_payload.get("cloud", {}).get("provider", ""),
        "CLOUD_REGION": request.get("cloud_region", ""),
        "CLOUD_ZONE": request.get("cloud_zone", ""),
        "MACHINE_TYPE": request.get("machine_type", ""),
        "LEDGER_DEVICE": request.get("ledger_device", ""),
        "ACCOUNTS_DEVICE": request.get("accounts_device", ""),
        "BLOCKCHAIN_PROCESS_NAMES_STR": " ".join(request.get("blockchain_process_names", []))
        if isinstance(request.get("blockchain_process_names"), list)
        else request.get("blockchain_process_names", ""),
        "DATA_VOL_TYPE": request.get("data_vol_type", ""),
        "DATA_VOL_SIZE": request.get("data_vol_size", ""),
        "DATA_VOL_MAX_IOPS": request.get("data_vol_max_iops", ""),
        "DATA_VOL_MAX_THROUGHPUT": request.get("data_vol_max_throughput", request.get("data_vol_max_throughput_mibs", "")),
        "ACCOUNTS_VOL_TYPE": request.get("accounts_vol_type", ""),
        "ACCOUNTS_VOL_SIZE": request.get("accounts_vol_size", ""),
        "ACCOUNTS_VOL_MAX_IOPS": request.get("accounts_vol_max_iops", ""),
        "ACCOUNTS_VOL_MAX_THROUGHPUT": request.get("accounts_vol_max_throughput", ""),
        "NETWORK_INTERFACE": request.get("network_interface", ""),
        "NETWORK_MAX_BANDWIDTH_GBPS": request.get("network_max_bandwidth_gbps", ""),
        "OBSERVABILITY_STACK_ENABLED": str(bool(request.get("observability", {}).get("enabled", False))).lower(),
        "OBSERVABILITY_STACK_MODE": request.get("observability", {}).get("mode", "local"),
        "OBSERVABILITY_STACK_AUTO_STOP": str(request.get("observability", {}).get("auto_stop", True)).lower(),
        "EXPORTER_PORT": str(request.get("exporter_port", "")),
        "PROMETHEUS_PORT": str(request.get("prometheus_port", "")),
        "GRAFANA_PORT": str(request.get("grafana_port", "")),
        "CHAIN_REST_URL": request.get("chain_rest_url", ""),
        "CHAIN_INDEXER_URL": request.get("chain_indexer_url", ""),
        "CHAIN_SIDECAR_URL": request.get("chain_sidecar_url", ""),
        "CHAIN_EVM_RPC_URL": request.get("chain_evm_rpc_url", ""),
        "CHAIN_JSON_RPC_URL": request.get("chain_json_rpc_url", ""),
        "CHAIN_MIRROR_URL": request.get("chain_mirror_url", ""),
        "RPC_API_KEY": request.get("rpc_api_key", ""),
        "TARGET_ADDRESS": request.get("target_address", ""),
        "TARGET_TX_HASH": request.get("target_tx_hash", ""),
        "TARGET_TXID": request.get("target_txid", ""),
        "TARGET_BLOCK_HASH": request.get("target_block_hash", ""),
        "TARGET_BLOCK": request.get("target_block", ""),
        "TARGET_HEIGHT": request.get("target_height", ""),
        "TARGET_ROUND": request.get("target_round", ""),
        "TARGET_ASSET_ID": request.get("target_asset_id", ""),
        "TARGET_ASSET": request.get("target_asset", ""),
        "TARGET_EPOCH": request.get("target_epoch", ""),
        "TARGET_VP": request.get("target_vp", ""),
        "TARGET_POOL_ID": request.get("target_pool_id", ""),
        "TARGET_TOKEN_ACCOUNT": request.get("target_token_account", ""),
        "TARGET_TOKEN_MINT": request.get("target_token_mint", ""),
        "TARGET_CONTRACT_ADDRESS": request.get("target_contract_address", ""),
        "TARGET_EVM_ADDRESS": request.get("target_evm_address", ""),
        "TARGET_SIGNER_ID": request.get("target_signer_id", ""),
        "TARGET_STORAGE_SLOT": request.get("target_storage_slot", ""),
    }

    plan_id = f"plan_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    plan = {
        "plan_id": plan_id,
        "chain": chain,
        "strategy": strategy,
        "benchmark_mode": benchmark_mode,
        "goal": goal,
        "rpc_mode": rpc_mode,
        "use_fake_node": use_fake_node,
        "deployment": deployment,
        "dependency_mode": request.get("dependency_mode", "audit"),
        "discovery": discovery_payload,
        "confidence": {
            "chain": 0.95 if chain else 0.0,
            "deployment": _deployment_confidence(deployment, discovery_payload),
            "ledger_device": 0.95 if request.get("ledger_device") else discovery_payload.get("disks", {}).get("confidence", 0.0),
            "workload": 0.6 if request.get("workload", {}).get("methods") else 0.4,
        },
        "requires_confirmation": requires_confirmation,
        "approval_checkpoints": _approval_checkpoints(strategy, request.get("dependency_mode", "audit")),
        "plan_diff": {"baseline_plan_id": "", "changed": [], "added": [], "removed": []},
        "config_snapshot": _config_snapshot(chain),
        "redaction_policy": {
            "enabled": True,
            "patterns": ["RPC_API_KEY", "Authorization", "Bearer", "password", "token"],
        },
        "bottleneck_focus": request.get("bottleneck_focus", []),
        "confirmed_inputs": sorted(confirmations),
        "materialized_config": materialized_config,
        "chain_template_requirements": inspect_chain_template(chain),
        "required_inputs": required_inputs,
        "advanced_defaults": {
            "qps": qps,
            "observability": request.get("observability", {"enabled": False, "mode": "local"}),
        },
        "execution": {
            "working_dir": str(REPO_ROOT),
            "command": command,
            "environment": env,
            "runner_mode": runner_mode,
        },
        "preflight_checks": [
            "chain_template_exists",
            "required_inputs_present",
            "benchmark_entry_exists",
            "fake_node_available_when_requested",
            "output_directories_writable",
        ],
        "artifacts": {
            "runtime_env_file": "<job_run_dir>/runtime.env",
            "current_reports_glob": "current/reports/performance_report_*.html",
            "archive_summary_glob": "archives/*/test_summary.json",
            "proxy_method_csv": "current/logs/proxy_method.csv",
            "performance_latest_csv": "current/logs/performance_latest.csv",
        },
    }
    checklist = build_configuration_checklist(request, plan)
    plan["configuration_checklist"] = checklist
    combined_required = _ordered_required_inputs(set(plan["required_inputs"]) | set(missing_required_from_checklist(checklist)))
    plan["required_inputs"] = combined_required
    plan["required_questions"] = required_questions(plan)
    plan["risk"] = score_plan_risk(plan)
    return plan


def validate_plan_shape(plan: dict[str, Any]) -> list[str]:
    errors = []
    for field in ("plan_id", "chain", "strategy", "execution", "required_inputs", "artifacts"):
        if field not in plan:
            errors.append(f"missing field: {field}")
    execution = plan.get("execution", {})
    if not isinstance(execution.get("command"), list) or not execution.get("command"):
        errors.append("execution.command must be a non-empty list")
    if not isinstance(execution.get("environment"), dict):
        errors.append("execution.environment must be an object")
    return errors


def _mode_flag(strategy: str) -> str:
    if strategy == "smoke":
        return "--quick"
    if strategy == "stress":
        return "--intensive"
    return "--standard"


def _strategy_from_mode(mode: Any) -> str:
    normalized = str(mode or "").strip().lower()
    if normalized == "quick":
        return "smoke"
    if normalized == "standard":
        return "baseline"
    if normalized == "intensive":
        return "stress"
    return ""


def _benchmark_mode(strategy: str) -> str:
    if strategy == "smoke":
        return "quick"
    if strategy == "stress":
        return "intensive"
    return "standard"


def _qps_env_prefix(strategy: str) -> str:
    if strategy == "smoke":
        return "QUICK"
    if strategy == "stress":
        return "INTENSIVE"
    return "STANDARD"


def _approval_checkpoints(strategy: str, dependency_mode: str) -> list[str]:
    checkpoints = ["plan_execution"]
    if dependency_mode == "managed":
        checkpoints.append("dependency_install")
    if strategy == "stress":
        checkpoints.append("stress_execution")
    return checkpoints


def _deployment_confidence(deployment: dict[str, Any], discovery: dict[str, Any]) -> float:
    if deployment.get("type") == "unknown":
        return 0.5
    cloud_confidence = float(discovery.get("cloud", {}).get("confidence", 0.6) or 0.6)
    return min(0.95, max(0.6, cloud_confidence))


def _config_snapshot(chain: str) -> dict[str, list[dict[str, str | float]]]:
    files = [REPO_ROOT / "config" / "user_config.sh"]
    if chain:
        files.append(REPO_ROOT / "config" / "chains" / f"{chain}.json")
    snapshots = []
    for path in files:
        if not path.is_file():
            continue
        stat = path.stat()
        snapshots.append({
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "mtime": stat.st_mtime,
        })
    return {"files": snapshots}


def _ordered_required_inputs(items: set[str]) -> list[str]:
    priority = [
        "chain",
        "use_fake_node",
        "rpc_mode",
        "benchmark_mode_confirmed",
        "qps_profile_confirmed",
        "observability_choice_confirmed",
        "chain_template_reviewed",
        "local_rpc_url",
        "mainnet_rpc_url_reviewed",
        "blockchain_process_names",
        "ledger_device",
        "has_accounts_device",
        "data_vol_type",
        "data_vol_size",
        "data_vol_max_iops",
        "data_vol_max_throughput",
        "accounts_device",
        "accounts_vol_type",
        "accounts_vol_size",
        "accounts_vol_max_iops",
        "accounts_vol_max_throughput",
        "network_interface",
        "network_max_bandwidth_gbps",
        "rpc_workload_confirmed",
        "mixed_weights_confirmed",
        "rpc_param_samples_confirmed",
    ]
    known = [item for item in priority if item in items]
    unknown = sorted(item for item in items if item not in priority)
    return known + unknown

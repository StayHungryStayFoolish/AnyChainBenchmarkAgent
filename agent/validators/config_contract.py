"""Benchmark configuration contract validators."""

from __future__ import annotations

from typing import Any

from knowledge.entry_contract import OPTIONAL_ACCOUNTS_FIELDS
from workflows.requirements import ENVIRONMENT_BLOCKERS, REAL_NODE_BLOCKERS, missing_smoke_blockers


OPTIONAL_ACCOUNTS_KEYS = tuple(field.key for field in OPTIONAL_ACCOUNTS_FIELDS if field.key != "accounts_device")


def validate_required_config(target_mode: str | None, confirmed_config: dict[str, Any]) -> dict[str, Any]:
    """Return required-config status for fake-node or real-node benchmarks."""
    values = dict(confirmed_config)
    if target_mode == "fake-node":
        values["use_fake_node"] = True
    elif target_mode == "real-node":
        values["use_fake_node"] = False
    missing = missing_smoke_blockers(values)
    return {
        "target_mode": target_mode or _target_from_values(values),
        "ready": not missing,
        "missing": missing,
        "required_groups": {
            "environment": list(ENVIRONMENT_BLOCKERS),
            "real_node": list(REAL_NODE_BLOCKERS),
            "optional_accounts": list(OPTIONAL_ACCOUNTS_KEYS),
        },
    }


def build_missing_config_questions(
    target_mode: str | None,
    confirmed_config: dict[str, Any],
    discovery: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build user questions for missing or ambiguous config values."""
    validation = validate_required_config(target_mode, confirmed_config)
    discovery = discovery or {}
    disks = discovery.get("disks", {})
    candidates = _disk_candidates(disks)
    questions = []
    for key in validation["missing"]:
        question = {
            "id": key,
            "severity": "blocker",
            "prompt": _prompt_for_key(key),
            "manual_input_allowed": True,
            "manual_input_hint": "Reply with a listed number/id when candidates are shown, or type a custom value.",
        }
        if key == "ledger_device" and candidates:
            question["candidates"] = candidates
            question["prompt"] = "Choose LEDGER_DEVICE from the detected disk inventory or enter a device name."
        if key == "benchmark_mode_confirmed":
            question["candidates"] = [
                {"id": "quick", "description": "Short smoke/sanity run."},
                {"id": "standard", "description": "Normal benchmark run."},
                {"id": "intensive", "description": "Long bottleneck discovery run."},
            ]
        if key == "observability_choice_confirmed":
            question["candidates"] = [
                {"id": "disabled", "description": "Do not start observability stack."},
                {"id": "local", "description": "Start exporter, local Prometheus, and local Grafana."},
                {"id": "exporter", "description": "Start only exporter for an existing Prometheus/Grafana environment."},
            ]
        if key == "qps_profile_confirmed":
            question["interaction_mode"] = "accept_defaults_or_adjust_item"
            question["prompt"] = (
                "Show the selected mode's default QPS profile with parameter meanings, then ask whether "
                "to keep the defaults. Only if the user wants changes, ask which item to adjust."
            )
            question["parameter_descriptions"] = {
                "initial_qps": "Starting request rate for the first QPS level.",
                "max_qps": "Highest request rate the mode will attempt before stopping or hitting a bottleneck.",
                "qps_step": "Increment added between QPS levels.",
                "duration_seconds": "How long each QPS level runs before moving to the next level.",
            }
            question["adjustable_items"] = [
                {"id": "initial_qps", "env_suffix": "INITIAL_QPS"},
                {"id": "max_qps", "env_suffix": "MAX_QPS"},
                {"id": "qps_step", "env_suffix": "QPS_STEP"},
                {"id": "duration_seconds", "env_suffix": "DURATION"},
            ]
        questions.append(question)
    if confirmed_config.get("has_accounts_device") is None:
        questions.append({
            "id": "has_accounts_device",
            "severity": "confirm",
            "prompt": "Does this node have a separate accounts/state disk?",
            "candidates": candidates,
            "manual_input_allowed": True,
            "manual_input_hint": "Reply yes/no, choose a listed disk, or type the accounts device name if it exists.",
        })
    return {
        "ready": validation["ready"],
        "missing": validation["missing"],
        "questions": questions,
        "disk_candidates": candidates,
    }


def _target_from_values(values: dict[str, Any]) -> str:
    if values.get("use_fake_node") is True:
        return "fake-node"
    if values.get("use_fake_node") is False:
        return "real-node"
    return "unknown"


def _prompt_for_key(key: str) -> str:
    prompts = {
        "chain": "Which blockchain node should be benchmarked?",
        "rpc_mode": "Choose RPC mode: single or mixed.",
        "rpc_workload_confirmed": "Confirm the RPC methods and workload weights.",
        "rpc_param_samples_confirmed": "Confirm TARGET_* sample values for the selected RPC methods.",
        "benchmark_mode_confirmed": "Choose quick, standard, or intensive benchmark mode.",
        "qps_profile_confirmed": "Confirm INITIAL_QPS, MAX_QPS, QPS_STEP, and DURATION for the selected mode.",
        "observability_choice_confirmed": "Choose disabled, local Prometheus/Grafana, or exporter-only observability mode.",
        "chain_template_reviewed": "Review selected chain template endpoints, TARGET_* sample variables, and default workload.",
        "use_fake_node": "Choose fake-node closed-loop testing or real-node testing.",
        "local_rpc_url": "Provide LOCAL_RPC_URL for the node under test.",
        "mainnet_rpc_url_reviewed": "Provide MAINNET_RPC_URL or confirm template/default sync-health handling.",
        "cloud_region": "Confirm CLOUD_REGION; use the detected value or enter a custom region.",
        "cloud_zone": "Confirm CLOUD_ZONE; use the detected value or enter a custom zone.",
        "machine_type": "Confirm MACHINE_TYPE or instance type for report metadata.",
        "blockchain_process_names": "Provide node process names or command keywords.",
        "ledger_device": "Confirm the ledger/data disk device.",
        "data_vol_type": "Confirm DATA_VOL_TYPE for the ledger/data disk.",
        "data_vol_size": "Confirm DATA_VOL_SIZE in GiB for the ledger/data disk.",
        "data_vol_max_iops": "Confirm DATA_VOL_MAX_IOPS for the ledger/data disk.",
        "data_vol_max_throughput": "Confirm DATA_VOL_MAX_THROUGHPUT in MiB/s for the ledger/data disk.",
        "network_interface": "Confirm the network interface used by the node.",
        "network_max_bandwidth_gbps": "Confirm NETWORK_MAX_BANDWIDTH_GBPS for saturation analysis.",
    }
    return prompts.get(key, f"Provide required value: {key}")


def _disk_candidates(disks: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in disks.get("candidates", []):
        name = item.get("name")
        if not name or item.get("mountpoint") in {"/boot", "/boot/efi"}:
            continue
        rows.append({
            "name": name,
            "type": item.get("type", ""),
            "size": item.get("size", ""),
            "mountpoint": item.get("mountpoint", ""),
            "fstype": item.get("fstype", ""),
            "label": item.get("label", ""),
        })
    return rows

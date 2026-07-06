"""Benchmark configuration contract validators."""

from __future__ import annotations

from typing import Any

from knowledge.entry_contract import OPTIONAL_ACCOUNTS_FIELDS
from workflows.group_registry import GROUP_QUESTION_ORDER
from workflows.group_registry import question_keys_for_group
from workflows.requirements import ENVIRONMENT_BLOCKERS, REAL_NODE_BLOCKERS, missing_smoke_blockers


OPTIONAL_ACCOUNTS_KEYS = tuple(field.key for field in OPTIONAL_ACCOUNTS_FIELDS if field.key != "accounts_device")
WORKLOAD_CONFIRMATION_KEYS = frozenset({
    "chain_template_reviewed",
    "rpc_workload_confirmed",
    "rpc_param_samples_confirmed",
    "mixed_weights_confirmed",
})

SYNC_OBSERVE_BLOCKERS = (
    "chain",
    "sync_observe_stop_condition",
    "ledger_device",
    "data_vol_type",
    "data_vol_size",
    "data_vol_max_iops",
    "data_vol_max_throughput",
    "network_interface",
    "network_max_bandwidth_gbps",
    "node_process_identity",
    "mainnet_rpc_url_reviewed",
)

ENV_TO_KEY = {
    "BLOCKCHAIN_NODE": "chain",
    "RPC_MODE": "rpc_mode",
    "LOCAL_RPC_URL": "local_rpc_url",
    "MAINNET_RPC_URL": "mainnet_rpc_url_reviewed",
    "BLOCKCHAIN_PROCESS_NAMES": "blockchain_process_names",
    "BLOCKCHAIN_PROCESS_NAMES_STR": "blockchain_process_names",
    "LEDGER_DEVICE": "ledger_device",
    "DATA_VOL_TYPE": "data_vol_type",
    "DATA_VOL_SIZE": "data_vol_size",
    "DATA_VOL_MAX_IOPS": "data_vol_max_iops",
    "DATA_VOL_MAX_THROUGHPUT": "data_vol_max_throughput",
    "ACCOUNTS_DEVICE": "accounts_device",
    "ACCOUNTS_VOL_TYPE": "accounts_vol_type",
    "ACCOUNTS_VOL_SIZE": "accounts_vol_size",
    "ACCOUNTS_VOL_MAX_IOPS": "accounts_vol_max_iops",
    "ACCOUNTS_VOL_MAX_THROUGHPUT": "accounts_vol_max_throughput",
    "NETWORK_INTERFACE": "network_interface",
    "NETWORK_MAX_BANDWIDTH_GBPS": "network_max_bandwidth_gbps",
    "CLOUD_REGION": "cloud_region",
    "CLOUD_ZONE": "cloud_zone",
    "MACHINE_TYPE": "machine_type",
    "OBSERVABILITY_STACK_MODE": "observability_choice_confirmed",
}

def validate_required_config(target_mode: str | None, confirmed_config: dict[str, Any]) -> dict[str, Any]:
    """Return required-config status for fake-node or real-node benchmarks."""
    values = _canonical_values(confirmed_config)
    if _is_sync_observe(values):
        missing = [key for key in SYNC_OBSERVE_BLOCKERS if key not in values or _is_missing_value(values.get(key))]
        return {
            "target_mode": "sync-observe",
            "ready": not missing,
            "missing": missing,
            "required_groups": {
                "sync_observe": list(SYNC_OBSERVE_BLOCKERS),
                "environment": list(ENVIRONMENT_BLOCKERS),
            },
        }
    if target_mode == "fake-node":
        values["use_fake_node"] = True
    elif target_mode == "real-node":
        values["use_fake_node"] = False
    _apply_fake_node_defaults(values)
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
    preferred_group: str | None = None,
) -> dict[str, Any]:
    """Build user questions for missing or ambiguous config values."""
    values = _canonical_values(confirmed_config)
    if _is_sync_observe(values):
        return _build_sync_observe_missing_questions(values, discovery or {}, preferred_group=preferred_group)
    if target_mode == "fake-node":
        values["use_fake_node"] = True
    elif target_mode == "real-node":
        values["use_fake_node"] = False
    _apply_fake_node_defaults(values)
    validation = validate_required_config(target_mode, values)
    discovery = discovery or {}
    disks = discovery.get("disks", {})
    candidates = _disk_candidates(disks)
    questions = []
    for key in validation["missing"]:
        question = _typed_question({
            "id": key,
            "severity": "blocker",
            "prompt": _prompt_for_key(key),
            "manual_input_allowed": True,
            "allow_manual_input": True,
            "manual_input_hint": "Reply with a listed number/id when candidates are shown, or type a custom value.",
        })
        if key == "ledger_device" and candidates:
            question["candidates"] = candidates
            question["options"] = _options_from_candidates(candidates)
            question["prompt"] = "Choose LEDGER_DEVICE from the detected disk inventory or enter a device name."
        if key == "benchmark_mode_confirmed":
            question["candidates"] = [
                {"id": "quick", "description": "Short smoke/sanity run."},
                {"id": "standard", "description": "Normal benchmark run."},
                {"id": "intensive", "description": "Long bottleneck discovery run."},
            ]
            question["options"] = _options_from_candidates(question["candidates"])
        if key == "observability_choice_confirmed":
            question["candidates"] = [
                {"id": "disabled", "description": "Do not start observability stack."},
                {"id": "local", "description": "Start exporter, local Prometheus, and local Grafana."},
                {"id": "exporter", "description": "Start only exporter for an existing Prometheus/Grafana environment."},
            ]
            question["options"] = _options_from_candidates(question["candidates"])
        if key == "qps_profile_confirmed":
            question["interaction_mode"] = "accept_defaults_or_adjust_item"
            question["prompt"] = _qps_profile_prompt(values)
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
    next_question = _build_next_question(validation["missing"], values, discovery, candidates, preferred_group=preferred_group)
    if "has_accounts_device" not in values:
        question = _typed_question({
            "id": "has_accounts_device",
            "severity": "confirm",
            "prompt": "Does this node have a separate accounts/state disk?",
            "candidates": candidates,
            "options": _options_from_candidates(candidates),
            "manual_input_allowed": True,
            "allow_manual_input": True,
            "manual_input_hint": "Reply yes/no, choose a listed disk, or type the accounts device name if it exists.",
        })
        questions.append(question)
    return {
        "ready": validation["ready"],
        "missing": validation["missing"],
        "questions": questions,
        "next_question": next_question,
        "disk_candidates": candidates,
    }


def _canonical_values(confirmed_config: dict[str, Any]) -> dict[str, Any]:
    values = dict(confirmed_config or {})
    for env_key, logical_key in ENV_TO_KEY.items():
        if logical_key not in values and env_key in values:
            values[logical_key] = values[env_key]
    if "target_mode" in values and "use_fake_node" not in values:
        target = str(values.get("target_mode") or "").strip().lower().replace("_", "-")
        if target == "fake-node":
            values["use_fake_node"] = True
        elif target == "real-node":
            values["use_fake_node"] = False
    _apply_fake_node_defaults(values)
    if "NODE_PROMETHEUS_METRICS_URL" in values and "node_prometheus_metrics_url" not in values:
        values["node_prometheus_metrics_url"] = values["NODE_PROMETHEUS_METRICS_URL"]
    if "NODE_PROCESS_PID" in values and "node_process_identity" not in values:
        values["node_process_identity"] = values["NODE_PROCESS_PID"]
    if "BLOCKCHAIN_PROCESS_NAMES" in values and "node_process_identity" not in values:
        values["node_process_identity"] = values["BLOCKCHAIN_PROCESS_NAMES"]
    if "SYNC_OBSERVE_STOP_CONDITION" in values and "sync_observe_stop_condition" not in values:
        values["sync_observe_stop_condition"] = values["SYNC_OBSERVE_STOP_CONDITION"]
    return values


def _is_sync_observe(values: dict[str, Any]) -> bool:
    value = values.get("workflow_type") or values.get("run_mode") or values.get("mode")
    return str(value or "").strip().lower().replace("-", "_") in {"sync_observe", "sync", "observe_sync"}


def _build_sync_observe_missing_questions(
    values: dict[str, Any],
    discovery: dict[str, Any],
    preferred_group: str | None = None,
) -> dict[str, Any]:
    missing = [key for key in SYNC_OBSERVE_BLOCKERS if key not in values or _is_missing_value(values.get(key))]
    disks = discovery.get("disks", {}) if isinstance(discovery.get("disks"), dict) else {}
    candidates = _disk_candidates(disks)
    questions = [
        _typed_question({
            "id": key,
            "severity": "blocker",
            "prompt": _prompt_for_key(key),
            "manual_input_allowed": True,
            "allow_manual_input": True,
        })
        for key in missing
    ]
    next_question = {}
    preferred_keys = _question_keys_for_group(preferred_group)
    for key in preferred_keys:
        if key in missing:
            next_question = _question_for_next_key(key, values, discovery, candidates)
            break
    if not next_question and missing:
        next_question = _question_for_next_key(missing[0], values, discovery, candidates)
    return {
        "ready": not missing,
        "missing": missing,
        "questions": questions,
        "next_question": next_question,
        "disk_candidates": candidates,
    }


def _apply_fake_node_defaults(values: dict[str, Any]) -> None:
    if values.get("use_fake_node") is True and _is_missing_value(values.get("blockchain_process_names")):
        values["blockchain_process_names"] = ["fake-node"]
        values["BLOCKCHAIN_PROCESS_NAMES"] = ["fake-node"]
        values["BLOCKCHAIN_PROCESS_NAMES_STR"] = "fake-node"


def _target_from_values(values: dict[str, Any]) -> str:
    if values.get("use_fake_node") is True:
        return "fake-node"
    if values.get("use_fake_node") is False:
        return "real-node"
    return "unknown"


def _build_next_question(
    missing: list[str],
    values: dict[str, Any],
    discovery: dict[str, Any],
    disk_candidates: list[dict[str, Any]],
    preferred_group: str | None = None,
) -> dict[str, Any]:
    missing_set = set(missing)
    if "has_accounts_device" not in values:
        missing_set.add("has_accounts_device")
    if values.get("has_accounts_device") is True:
        if _is_missing_value(values.get("accounts_device")):
            missing_set.add("accounts_device")
        for key in OPTIONAL_ACCOUNTS_KEYS:
            if _is_missing_value(values.get(key)):
                missing_set.add(key)
    elif values.get("has_accounts_device") is False:
        missing_set.difference_update(OPTIONAL_ACCOUNTS_KEYS)
        missing_set.discard("accounts_device")
    if missing_set & WORKLOAD_CONFIRMATION_KEYS:
        if str(values.get("rpc_mode") or "").strip().lower() in {"single", "mixed"}:
            missing_set.difference_update(WORKLOAD_CONFIRMATION_KEYS)
            missing_set.add("workload_customization_choice")

    preferred_keys = _question_keys_for_group(preferred_group)
    for key in preferred_keys:
        if key in missing_set:
            return _question_for_next_key(key, values, discovery, disk_candidates)
    for _group, keys in GROUP_QUESTION_ORDER:
        for key in keys:
            if key in missing_set:
                return _question_for_next_key(key, values, discovery, disk_candidates)
    return {}


def _question_keys_for_group(group: str | None) -> tuple[str, ...]:
    return question_keys_for_group(group)


def _question_for_next_key(
    key: str,
    values: dict[str, Any],
    discovery: dict[str, Any],
    disk_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    if key == "use_fake_node":
        return _typed_question({
            "id": "target_mode",
            "kind": "numbered_choice",
            "field": "target_mode",
            "prompt": "Choose the target mode: fake-node closed-loop test or real-node benchmark.",
            "options": [
                {
                    "id": "1",
                    "value": "fake-node",
                    "label": "fake-node",
                    "state_patch": {"target_mode": "fake-node", "confirmed_config": {"use_fake_node": True}},
                },
                {
                    "id": "2",
                    "value": "real-node",
                    "label": "real-node",
                    "state_patch": {"target_mode": "real-node", "confirmed_config": {"use_fake_node": False}},
                },
            ],
        })
    if key == "ledger_device":
        return _typed_question({
            "id": "disk_ledger_choice",
            "kind": "device",
            "field": "LEDGER_DEVICE",
            "prompt": "Choose the ledger/data disk from the detected inventory, or type the device name.",
            "options": _options_from_candidates(disk_candidates),
            "manual_input_allowed": True,
        })
    if key == "has_accounts_device":
        return _typed_question({
            "id": "disk_accounts_exists",
            "kind": "yes_no",
            "field": "has_accounts_device",
            "prompt": "Does this node have a separate accounts/state disk? Reply Y or N.",
            "next_on_yes": {"workflow_step": "accounts_device_required", "tool": "build_missing_config_questions"},
            "next_on_no": {
                "workflow_step": "accounts_device_skipped",
                "tool": "build_missing_config_questions",
                "state_patch": {"confirmed_config": {"ACCOUNTS_DEVICE": "", "has_accounts_device": False}},
            },
        })
    if key == "accounts_device":
        return _typed_question({
            "id": "disk_accounts_choice",
            "kind": "device",
            "field": "ACCOUNTS_DEVICE",
            "prompt": "Choose the accounts/state disk from the detected inventory, or type the device name.",
            "options": _options_from_candidates(disk_candidates),
            "manual_input_allowed": True,
        })
    if key == "benchmark_mode_confirmed":
        return _typed_question({
            "id": "benchmark_profile_choice",
            "kind": "numbered_choice",
            "field": "benchmark_mode_confirmed",
            "prompt": "Choose benchmark mode: 1 quick, 2 standard, or 3 intensive.",
            "options": [
                {
                    "id": "1",
                    "value": "quick",
                    "label": "quick",
                    "description": "Short validation benchmark.",
                    "state_patch": {"benchmark_profile": {"mode": "quick"}},
                },
                {
                    "id": "2",
                    "value": "standard",
                    "label": "standard",
                    "description": "Normal benchmark run.",
                    "state_patch": {"benchmark_profile": {"mode": "standard"}},
                },
                {
                    "id": "3",
                    "value": "intensive",
                    "label": "intensive",
                    "description": "Long bottleneck discovery run.",
                    "state_patch": {"benchmark_profile": {"mode": "intensive"}},
                },
            ],
        })
    if key == "qps_profile_confirmed":
        return _typed_question({
            "id": "benchmark_profile_confirm",
            "kind": "yes_no",
            "field": "qps_profile_confirmed",
            "prompt": _qps_profile_prompt(values),
            "next_on_no": {
                "workflow_step": "benchmark_profile_adjust_item",
                "next_question_id": "benchmark_profile_adjust_item",
            },
        })
    if key == "workload_customization_choice":
        chain = str(values.get("chain") or values.get("BLOCKCHAIN_NODE") or "selected chain").strip() or "selected chain"
        rpc_mode = str(values.get("rpc_mode") or values.get("RPC_MODE") or "selected").strip().lower() or "selected"
        return _typed_question({
            "id": "workload_customization_choice",
            "kind": "numbered_choice",
            "prompt": _workload_customization_prompt(chain, rpc_mode),
            "branch": "rpc_workload",
            "workflow_step": "workload_customization_choice",
            "manual_input_allowed": True,
            "allow_manual_input": True,
            "options": _workload_customization_options(),
        })
    if key == "observability_choice_confirmed":
        return _typed_question({
            "id": "observability_mode_choice",
            "kind": "numbered_choice",
            "field": "observability_choice_confirmed",
            "prompt": "Choose observability mode: 1 disabled, 2 local Prometheus/Grafana, or 3 exporter-only.",
            "options": [
                {
                    "id": "1",
                    "value": "disabled",
                    "label": "disabled",
                    "state_patch": {
                        "observability": {"mode": "disabled", "enabled": False},
                        "confirmed_config": {"OBSERVABILITY_STACK_MODE": "disabled"},
                    },
                },
                {
                    "id": "2",
                    "value": "local",
                    "label": "local Prometheus/Grafana",
                    "state_patch": {
                        "observability": {"mode": "local", "enabled": True},
                        "confirmed_config": {"OBSERVABILITY_STACK_MODE": "local"},
                    },
                },
                {
                    "id": "3",
                    "value": "exporter",
                    "label": "exporter-only",
                    "state_patch": {
                        "observability": {"mode": "exporter", "enabled": True},
                        "confirmed_config": {"OBSERVABILITY_STACK_MODE": "exporter"},
                    },
                },
            ],
        })
    if key == "sync_observe_stop_condition":
        return _typed_question({
            "id": "sync_observe_stop_condition",
            "kind": "numbered_choice",
            "field": "sync_observe_stop_condition",
            "prompt": "Choose sync-observe stop condition: 1 until stopped, 2 fixed duration, or 3 until synced.",
            "options": [
                {
                    "id": "1",
                    "value": "until_stopped",
                    "label": "until stopped",
                    "state_patch": {
                        "confirmed_config": {"sync_observe_stop_condition": "until_stopped"},
                    },
                },
                {
                    "id": "2",
                    "value": "duration",
                    "label": "fixed duration",
                    "state_patch": {
                        "confirmed_config": {"sync_observe_stop_condition": "duration"},
                    },
                },
                {
                    "id": "3",
                    "value": "until_synced",
                    "label": "until synced",
                    "state_patch": {
                        "confirmed_config": {"sync_observe_stop_condition": "until_synced"},
                    },
                },
            ],
        })
    return _typed_question({
        "id": key,
        "kind": _expected_answer_for(key),
        "field": _field_for(key),
        "prompt": _prompt_for_key(key),
        "current_value": _current_value_for_key(key, values, discovery),
        "manual_input_allowed": True,
    })


def _workload_customization_prompt(chain: str, rpc_mode: str) -> str:
    return (
        f"Review the {chain} {rpc_mode} RPC workload. Choose the next step:\n"
        "1. Continue with the chain template defaults\n"
        "2. Add a custom RPC method\n"
        "3. Adjust mixed weights\n"
        "4. Change chain or target mode\n"
        "Reply with `1`, `2`, `3`, or `4`, or describe the change manually."
    )


def _workload_customization_options() -> list[dict[str, Any]]:
    return [
        {
            "id": "1",
            "value": "use_defaults",
            "label": "continue with defaults",
            "state_patch": {
                "workflow_step": "workload_default_confirmed",
                "confirmed_config": {
                    "chain_template_reviewed": True,
                    "rpc_workload_confirmed": True,
                    "rpc_param_samples_confirmed": True,
                    "mixed_weights_confirmed": True,
                },
            },
            "transition": {
                "workflow_step": "workload_default_confirmed",
                "tool": "validate_rpc_workload",
            },
        },
        {
            "id": "2",
            "value": "add_custom_rpc",
            "label": "add a custom RPC method",
            "state_patch": {"workflow_step": "custom_rpc_requested", "fixture_status": {"status": "needs_endpoint"}},
            "transition": {
                "workflow_step": "custom_rpc_endpoint_gate",
                "next_question_id": "custom_rpc_endpoint_gate",
            },
        },
        {
            "id": "3",
            "value": "adjust_weights",
            "label": "adjust mixed weights",
            "state_patch": {"workflow_step": "mixed_weight_adjustment_requested"},
            "transition": {
                "workflow_step": "mixed_weights_adjust",
                "next_question_id": "mixed_weights_confirm",
            },
        },
        {
            "id": "4",
            "value": "change_chain_or_mode",
            "label": "change chain or target mode",
            "state_patch": {"workflow_step": "chain_selection"},
            "transition": {
                "workflow_step": "chain_selection",
                "next_question_id": "chain_selection",
            },
        },
    ]


def _current_value_for_key(key: str, values: dict[str, Any], discovery: dict[str, Any]) -> Any:
    if key in values:
        return values.get(key)
    cloud = discovery.get("cloud", {}) if isinstance(discovery.get("cloud"), dict) else {}
    network = discovery.get("network", {}) if isinstance(discovery.get("network"), dict) else {}
    disks = discovery.get("disks", {}) if isinstance(discovery.get("disks"), dict) else {}
    inferred = {
        "cloud_region": cloud.get("region"),
        "cloud_zone": cloud.get("zone"),
        "machine_type": cloud.get("machine_type"),
        "network_interface": network.get("default_interface"),
        "data_vol_size": _selected_disk_size_gib(values.get("ledger_device") or values.get("LEDGER_DEVICE"), disks),
        "accounts_vol_size": _selected_disk_size_gib(values.get("accounts_device") or values.get("ACCOUNTS_DEVICE"), disks),
    }
    return inferred.get(key, "")


def _selected_disk_size_gib(device: Any, disks: dict[str, Any]) -> str:
    name = _normalize_device_name(device)
    if not name:
        return ""
    for item in disks.get("candidates", []) or []:
        if _normalize_device_name(item.get("name")) == name:
            return _size_to_gib(item.get("size"))
    return ""


def _normalize_device_name(value: Any) -> str:
    text = str(value or "").strip()
    if text.startswith("/dev/"):
        text = text.removeprefix("/dev/")
    return text


def _size_to_gib(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    compact = text.replace(" ", "").upper()
    multiplier = 1.0
    number = compact
    if compact.endswith("IB"):
        number = compact[:-2]
    if number.endswith("B") and len(number) > 1:
        number = number[:-1]
    if number.endswith("K"):
        multiplier = 1.0 / 1024 / 1024
        number = number[:-1]
    elif number.endswith("M"):
        multiplier = 1.0 / 1024
        number = number[:-1]
    elif number.endswith("G"):
        multiplier = 1.0
        number = number[:-1]
    elif number.endswith("T"):
        multiplier = 1024.0
        number = number[:-1]
    elif number.endswith("P"):
        multiplier = 1024.0 * 1024
        number = number[:-1]
    try:
        gib = float(number) * multiplier
    except ValueError:
        return ""
    if gib <= 0:
        return ""
    if gib >= 10:
        rounded = round(gib)
        return str(int(rounded))
    return f"{gib:.2f}".rstrip("0").rstrip(".")


def _prompt_for_key(key: str) -> str:
    prompts = {
        "chain": "Which blockchain node should be benchmarked?",
        "rpc_mode": "Choose RPC mode: single or mixed.",
        "rpc_workload_confirmed": "Confirm the RPC methods and workload weights.",
        "mixed_weights_confirmed": "Confirm that mixed RPC method weights are explicit and sum to 100%.",
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
        "blockchain_process_names": "Provide node process names or command-line fragments used for attribution.",
        "ledger_device": "Confirm the ledger/data disk device.",
        "data_vol_type": "Confirm DATA_VOL_TYPE for the ledger/data disk, for example hyperdisk-balanced, hyperdisk-extreme, pd-ssd, pd-balanced, local-ssd, ssd, or nvme.",
        "data_vol_size": "Confirm DATA_VOL_SIZE in GiB for the ledger/data disk.",
        "data_vol_max_iops": "Confirm DATA_VOL_MAX_IOPS for the ledger/data disk.",
        "data_vol_max_throughput": "Confirm DATA_VOL_MAX_THROUGHPUT in MiB/s for the ledger/data disk.",
        "accounts_vol_type": "Confirm ACCOUNTS_VOL_TYPE for the accounts/state disk, for example hyperdisk-balanced, hyperdisk-extreme, pd-ssd, pd-balanced, local-ssd, ssd, or nvme.",
        "accounts_vol_size": "Confirm ACCOUNTS_VOL_SIZE in GiB for the accounts/state disk.",
        "accounts_vol_max_iops": "Confirm ACCOUNTS_VOL_MAX_IOPS for the accounts/state disk.",
        "accounts_vol_max_throughput": "Confirm ACCOUNTS_VOL_MAX_THROUGHPUT in MiB/s for the accounts/state disk.",
        "network_interface": "Confirm the network interface used by the node.",
        "network_max_bandwidth_gbps": "Confirm NETWORK_MAX_BANDWIDTH_GBPS for saturation analysis.",
        "sync_observe_stop_condition": "Choose sync-observe stop condition: run until stopped, fixed duration, or until synced.",
        "node_prometheus_metrics_url": "Confirm NODE_PROMETHEUS_METRICS_URL if the node exposes Prometheus metrics, or leave it unavailable.",
        "node_process_identity": "Confirm the node PID or process-name/command-line fragment for CPU/thread attribution.",
    }
    return prompts.get(key, f"Provide required value: {key}")


def _qps_profile_prompt(values: dict[str, Any]) -> str:
    mode = str(values.get("benchmark_mode_confirmed") or values.get("benchmark_mode") or "").strip().lower()
    if mode not in {"quick", "standard", "intensive"}:
        mode = "selected"
    fake_node = values.get("use_fake_node") is True
    if fake_node:
        return (
            "Use the selected mode's default QPS profile? Reply `Y` to keep it, or `N` to adjust "
            "INITIAL_QPS, MAX_QPS, QPS_STEP, or DURATION.\n"
            "Note: fake-node smoke uses a safe low-traffic override during execution (1 QPS, 10s) "
            "to validate the full closed loop. This is not a real-node performance test. "
            "Real-node benchmarks use the selected mode's full QPS profile."
        )
    return (
        f"Review the {mode} QPS profile defaults from config/user_config.sh. Reply `Y` to keep them, "
        "or `N` to adjust INITIAL_QPS, MAX_QPS, QPS_STEP, or DURATION."
    )


def _typed_question(question: dict[str, Any]) -> dict[str, Any]:
    qid = str(question.get("id", ""))
    enriched = dict(question)
    enriched.setdefault("expected_answer", _expected_answer_for(qid))
    enriched.setdefault("field", _field_for(qid))
    enriched.setdefault("branch", "benchmark_setup")
    enriched.setdefault("workflow_step", qid)
    enriched.setdefault("validation_tool", "validate_required_config")
    enriched.setdefault("next_on_yes", {"workflow_step": f"{qid}:confirmed", "tool": "validate_required_config"})
    enriched.setdefault("next_on_no", {"workflow_step": f"{qid}:manual_input", "tool": "build_missing_config_questions"})
    enriched.setdefault("next_on_manual", {"workflow_step": f"{qid}:manual_value", "tool": "validate_required_config"})
    return enriched


def _expected_answer_for(question_id: str) -> str:
    if question_id in {"benchmark_mode_confirmed", "observability_choice_confirmed", "workload_customization_choice"}:
        return "numbered_choice"
    if question_id == "has_accounts_device":
        return "yes_no"
    if question_id in {"ledger_device", "accounts_device"}:
        return "device"
    if question_id in {"local_rpc_url", "mainnet_rpc_url"}:
        return "url"
    if question_id in {"rpc_workload_confirmed", "mixed_weights_confirmed", "rpc_param_samples_confirmed", "chain_template_reviewed", "qps_profile_confirmed"}:
        return "yes_no"
    return "manual_value"


def _field_for(question_id: str) -> str:
    mapping = {
        "chain": "BLOCKCHAIN_NODE",
        "rpc_mode": "RPC_MODE",
        "use_fake_node": "TARGET_MODE",
        "local_rpc_url": "LOCAL_RPC_URL",
        "mainnet_rpc_url": "MAINNET_RPC_URL",
        "mainnet_rpc_url_reviewed": "MAINNET_RPC_URL",
        "ledger_device": "LEDGER_DEVICE",
        "accounts_device": "ACCOUNTS_DEVICE",
        "has_accounts_device": "has_accounts_device",
        "benchmark_mode_confirmed": "benchmark_mode_confirmed",
        "qps_profile_confirmed": "qps_profile_confirmed",
        "observability_choice_confirmed": "observability_choice_confirmed",
        "chain_template_reviewed": "chain_template_reviewed",
        "rpc_workload_confirmed": "rpc_workload_confirmed",
        "rpc_param_samples_confirmed": "rpc_param_samples_confirmed",
        "mixed_weights_confirmed": "mixed_weights_confirmed",
    }
    return mapping.get(question_id, question_id.upper())


def _is_missing_value(value: Any) -> bool:
    if value is None or value is False:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, set, dict)):
        return not value
    return False


def _options_from_candidates(candidates: Any) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    if not isinstance(candidates, list):
        return options
    for index, item in enumerate(candidates, start=1):
        if isinstance(item, dict):
            value = item.get("id") or item.get("name") or item.get("value") or str(index)
            label = _candidate_option_label(item, value)
            option = dict(item)
            option["id"] = str(index)
            option["value"] = value
            option["label"] = str(label)
            options.append(option)
        else:
            options.append({"id": str(index), "value": str(item), "label": str(item)})
    return options


def _candidate_option_label(item: dict[str, Any], value: Any) -> str:
    if item.get("name"):
        parts = []
        for key in ("size", "type", "mountpoint", "fstype", "label"):
            raw = str(item.get(key) or "").strip()
            if raw:
                if key == "mountpoint":
                    raw = f"mount={raw}"
                elif key == "label":
                    raw = f"label={raw}"
                parts.append(raw)
        suffix = f" ({', '.join(parts)})" if parts else ""
        return f"{item.get('name')}{suffix}"
    return item.get("description") or item.get("label") or value


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

"""Generate required configuration questions from benchmark plans."""

from __future__ import annotations

from typing import Any


def required_questions(plan: dict[str, Any]) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []

    for item in plan.get("required_inputs", []):
        questions.append({
            "id": item,
            "category": "required_input",
            "severity": "blocker",
            "prompt": _required_prompt(item),
        })

    confidence = plan.get("confidence", {})
    confirmed = set(plan.get("confirmed_inputs", []))
    if (
        not plan.get("use_fake_node")
        and confidence.get("ledger_device", 1.0) < 0.6
        and "ledger_device_confirmation" not in confirmed
    ):
        disks = plan.get("discovery", {}).get("disks", {})
        questions.append({
            "id": "ledger_device_confirmation",
            "category": "environment",
            "severity": "blocker",
            "prompt": "Confirm the ledger/data disk device before running the benchmark.",
            "candidates": disks.get("ambiguous_candidates") or _candidate_disk_names(disks),
        })

    discovery = plan.get("discovery", {})
    disks = discovery.get("disks", {})
    disk_candidates = _disk_candidates(disks)
    if (
        not plan.get("use_fake_node")
        and len(disk_candidates) > 1
        and "disk_inventory_confirmation" not in confirmed
    ):
        questions.append({
            "id": "disk_inventory_confirmation",
            "category": "storage",
            "severity": "confirm",
            "prompt": (
                "Multiple disk candidates were detected from lsblk. Show the disk inventory "
                "and ask the user to confirm which device is LEDGER_DEVICE, whether a separate "
                "ACCOUNTS_DEVICE exists, and the provisioned disk baselines."
            ),
            "candidates": disk_candidates,
            "proposed_ledger_device": disks.get("proposed_ledger_device", ""),
            "proposed_accounts_device": disks.get("proposed_accounts_device", ""),
        })
    missing_required = discovery.get("dependencies", {}).get("missing_required", [])
    if missing_required and "dependency_mode_confirmation" not in confirmed:
        questions.append({
            "id": "dependency_mode_confirmation",
            "category": "dependency",
            "severity": "warning",
            "prompt": (
                "Required dependencies are missing. Explain what would be installed, then ask for "
                "explicit approval before calling install_dependencies. Keep audit-only mode unless approved."
            ),
            "missing": missing_required,
        })

    if "mixed_weights" in plan.get("requires_confirmation", []):
        questions.append({
            "id": "mixed_weights_confirmation",
            "category": "workload",
            "severity": "blocker",
            "prompt": "Confirm mixed RPC method weights and parameter samples.",
        })

    checklist = plan.get("configuration_checklist", {})
    for item in checklist.get("environment", []):
        if item.get("id") not in confirmed:
            questions.append({
                "id": item["id"],
                "category": "environment",
                "severity": "confirm",
                "prompt": f"Confirm {item['description']}",
                "current_value": _current_value(plan, item["id"]),
            })

    accounts_items = checklist.get("accounts_optional", [])
    if accounts_items and "has_accounts_device" not in confirmed:
        questions.append({
            "id": "has_accounts_device",
            "category": "storage",
            "severity": "confirm",
            "prompt": (
                "Does this node use a second accounts/state disk? If yes, confirm ACCOUNTS_DEVICE "
                "from the lsblk inventory and provide ACCOUNTS_VOL_* baselines."
            ),
            "current_value": _current_value(plan, "accounts_device"),
            "candidates": disk_candidates,
        })
    for item in accounts_items:
        if item.get("severity") == "blocker" and not item.get("present"):
            questions.append({
                "id": item["id"],
                "category": "storage",
                "severity": "blocker",
                "prompt": item["description"],
            })

    chain_requirements = plan.get("chain_template_requirements", {})
    if chain_requirements.get("exists"):
        if "rpc_workload_confirmation" not in confirmed:
            questions.append({
                "id": "rpc_workload_confirmation",
                "category": "workload",
                "severity": "confirm",
                "prompt": "Confirm the RPC methods and weights to test from the selected chain template.",
                "single": chain_requirements.get("single_method"),
                "mixed_weighted": chain_requirements.get("mixed_weighted", []),
            })
        if "custom_rpc_method_review" not in confirmed:
            questions.append({
                "id": "custom_rpc_method_review",
                "category": "workload",
                "severity": "confirm",
                "prompt": (
                    "Ask whether the user wants to add custom RPC methods before execution. "
                    "If yes, collect method name, parameter shape, sample TARGET_* values, "
                    "single/mixed inclusion, mixed weight, and fake-node fixture expectations."
                ),
                "extension_fields": chain_requirements.get("custom_rpc_extension_fields", []),
                "param_formats": chain_requirements.get("param_formats", {}),
                "param_spec_methods": chain_requirements.get("param_spec_methods", []),
            })
        sample_vars = chain_requirements.get("runtime_sample_variables", [])
        if sample_vars and "rpc_param_samples_confirmation" not in confirmed:
            questions.append({
                "id": "rpc_param_samples_confirmation",
                "category": "workload",
                "severity": "confirm",
                "prompt": "Confirm whether these chain template sample variables should use defaults or user-provided values.",
                "variables": sample_vars,
            })
        endpoint_vars = chain_requirements.get("runtime_endpoint_variables", [])
        if endpoint_vars and "chain_endpoint_overrides_confirmation" not in confirmed:
            questions.append({
                "id": "chain_endpoint_overrides_confirmation",
                "category": "endpoint",
                "severity": "confirm",
                "prompt": "Confirm whether this chain needs endpoint overrides beyond LOCAL_RPC_URL.",
                "variables": endpoint_vars,
            })

    if "advanced_config_review" not in confirmed:
        questions.append({
            "id": "advanced_config_review",
            "category": "advanced",
            "severity": "info",
            "prompt": (
                "Advanced thresholds use config/internal_config.sh defaults. "
                "Ask whether the user wants a short explanation or wants to adjust bottleneck, latency, success-rate, or sync-health thresholds."
            ),
            "variables": [
                "BOTTLENECK_CPU_THRESHOLD",
                "BOTTLENECK_MEMORY_THRESHOLD",
                "BOTTLENECK_DISK_UTIL_THRESHOLD",
                "BOTTLENECK_DISK_LATENCY_THRESHOLD",
                "BOTTLENECK_NETWORK_THRESHOLD",
                "BOTTLENECK_ERROR_RATE_THRESHOLD",
                "SUCCESS_RATE_THRESHOLD",
                "MAX_LATENCY_THRESHOLD",
                "BLOCK_HEIGHT_DIFF_THRESHOLD",
                "BLOCK_HEIGHT_TIME_THRESHOLD",
            ],
        })

    if "stress_execution" in plan.get("requires_confirmation", []):
        questions.append({
            "id": "stress_execution_confirmation",
            "category": "safety",
            "severity": "blocker",
            "prompt": "Confirm stress/intensive benchmark execution; it may affect the target node.",
        })

    return _dedupe_questions(questions)


def _required_prompt(item: str) -> str:
    prompts = {
        "chain": "Which blockchain node should be tested?",
        "local_rpc_url": "Provide the local RPC endpoint, or choose fake-node for closed-loop testing.",
        "blockchain_process_names": "Provide blockchain node process names or command keywords for resource attribution.",
        "ledger_device": "Confirm the ledger/data disk device used by the node.",
        "data_vol_type": "Provide the ledger/data disk type.",
        "data_vol_size": "Provide the ledger/data disk size in GiB.",
        "data_vol_max_iops": "Provide the provisioned ledger/data disk IOPS baseline.",
        "data_vol_max_throughput": "Provide the provisioned ledger/data disk throughput baseline in MiB/s.",
        "network_interface": "Confirm the network interface used by the node.",
        "network_max_bandwidth_gbps": "Provide the instance or pod network bandwidth baseline in Gbps.",
        "rpc_mode": "Choose single or mixed RPC workload mode.",
    }
    return prompts.get(item, f"Provide required value: {item}")


def _current_value(plan: dict[str, Any], item_id: str) -> Any:
    inferred = {
        "cloud_provider": plan.get("materialized_config", {}).get("CLOUD_PROVIDER"),
        "deployment_platform": plan.get("deployment", {}).get("type"),
        "cloud_region": plan.get("materialized_config", {}).get("CLOUD_REGION"),
        "cloud_zone": plan.get("materialized_config", {}).get("CLOUD_ZONE"),
        "machine_type": plan.get("materialized_config", {}).get("MACHINE_TYPE"),
        "accounts_device": plan.get("materialized_config", {}).get("ACCOUNTS_DEVICE"),
    }
    return inferred.get(item_id, "")


def _candidate_disk_names(disks: dict[str, Any]) -> list[str]:
    names = []
    for item in disks.get("candidates", []):
        name = item.get("name")
        if name and item.get("mountpoint") not in {"", "/", "/boot", "/boot/efi"}:
            names.append(name)
    return names


def _disk_candidates(disks: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    for item in disks.get("candidates", []):
        name = item.get("name")
        if not name or item.get("mountpoint") in {"/boot", "/boot/efi"}:
            continue
        candidates.append({
            "name": name,
            "type": item.get("type", ""),
            "size": item.get("size", ""),
            "mountpoint": item.get("mountpoint", ""),
            "fstype": item.get("fstype", ""),
            "label": item.get("label", ""),
        })
    return candidates


def _dedupe_questions(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    result = []
    for question in questions:
        qid = question["id"]
        if qid in seen:
            continue
        seen.add(qid)
        result.append(question)
    return result

"""Generate required configuration questions from benchmark plans."""

from __future__ import annotations

from typing import Any


def required_questions(plan: dict[str, Any]) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    if _is_sync_observe_plan(plan):
        return _dedupe_questions(_sync_observe_questions(plan))

    for item in plan.get("required_inputs", []):
        if item in _SPECIALIZED_REQUIRED_QUESTIONS:
            continue
        questions.append(_with_manual_input({
            "id": item,
            "category": "required_input",
            "severity": "blocker",
            "prompt": _required_prompt(item),
        }))

    confidence = plan.get("confidence", {})
    confirmed = set(plan.get("confirmed_inputs", []))
    if (
        confidence.get("ledger_device", 1.0) < 0.6
        and "ledger_device_confirmation" not in confirmed
    ):
        disks = plan.get("discovery", {}).get("disks", {})
        questions.append(_with_manual_input({
            "id": "ledger_device_confirmation",
            "category": "environment",
            "severity": "blocker",
            "prompt": "Confirm the ledger/data disk device before running the benchmark.",
            "candidates": disks.get("ambiguous_candidates") or _candidate_disk_names(disks),
        }))

    discovery = plan.get("discovery", {})
    disks = discovery.get("disks", {})
    disk_candidates = _disk_candidates(disks)
    if (
        len(disk_candidates) > 1
        and "disk_inventory_confirmation" not in confirmed
    ):
        questions.append(_with_manual_input({
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
        }))
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
        questions.append(_with_manual_input({
            "id": "mixed_weights_confirmation",
            "category": "workload",
            "severity": "blocker",
            "prompt": "Confirm mixed RPC method weights and parameter samples.",
        }))

    if "benchmark_mode_confirmed" not in confirmed:
        questions.append(_with_manual_input({
            "id": "benchmark_mode_confirmed",
            "category": "execution",
            "severity": "blocker",
            "prompt": (
                "Choose benchmark mode. quick is a short smoke/sanity run, standard is the normal "
                "performance benchmark, and intensive searches for bottlenecks and can run much longer."
            ),
            "candidates": [
                {"id": "quick", "description": "Short validation run; safest first step."},
                {"id": "standard", "description": "Normal benchmark run using standard QPS settings."},
                {"id": "intensive", "description": "Long bottleneck discovery run with auto-stop when configured."},
            ],
            "current_value": plan.get("benchmark_mode") or plan.get("strategy"),
        }))

    if "qps_profile_confirmed" not in confirmed:
        qps = plan.get("advanced_defaults", {}).get("qps", {})
        questions.append(_with_manual_input({
            "id": "qps_profile_confirmed",
            "category": "execution",
            "severity": "blocker",
            "prompt": (
                "Show the selected mode's default QPS profile with parameter meanings, then ask whether "
                "to keep the defaults. Only if the user wants changes, ask which item to adjust."
            ),
            "interaction_mode": "accept_defaults_or_adjust_item",
            "accepted_reply_examples": ["keep defaults", "use defaults", "yes"],
            "adjust_reply_examples": ["adjust max_qps", "change duration", "set qps_step to 100"],
            "current_value": {
                "initial_qps": qps.get("initial"),
                "max_qps": qps.get("max"),
                "qps_step": qps.get("step"),
                "duration_seconds": qps.get("duration_seconds"),
            },
            "parameter_descriptions": {
                "initial_qps": "Starting request rate for the first QPS level.",
                "max_qps": "Highest request rate the mode will attempt before stopping or hitting a bottleneck.",
                "qps_step": "Increment added between QPS levels.",
                "duration_seconds": "How long each QPS level runs before moving to the next level.",
            },
            "adjustable_items": [
                {"id": "initial_qps", "env_suffix": "INITIAL_QPS"},
                {"id": "max_qps", "env_suffix": "MAX_QPS"},
                {"id": "qps_step", "env_suffix": "QPS_STEP"},
                {"id": "duration_seconds", "env_suffix": "DURATION"},
            ],
        }))

    if "observability_choice_confirmed" not in confirmed:
        observability = plan.get("advanced_defaults", {}).get("observability", {})
        questions.append(_with_manual_input({
            "id": "observability_choice_confirmed",
            "category": "observability",
            "severity": "confirm",
            "prompt": (
                "Choose observability mode: disabled, local Prometheus/Grafana, or exporter-only "
                "for an existing Prometheus/Grafana environment."
            ),
            "candidates": [
                {"id": "disabled", "description": "Do not start the optional observability stack."},
                {
                    "id": "local",
                    "description": "Start exporter, local Prometheus, and local Grafana; confirm EXPORTER_PORT, PROMETHEUS_PORT, and GRAFANA_PORT.",
                },
                {
                    "id": "exporter",
                    "description": "Start only the read-only exporter; configure the user's existing Prometheus to scrape http://<benchmark-host>:EXPORTER_PORT/metrics.",
                },
            ],
            "current_value": observability,
        }))

    checklist = plan.get("configuration_checklist", {})
    for item in checklist.get("environment", []):
        if item.get("id") not in confirmed:
            question = {
                "id": item["id"],
                "category": "environment",
                "severity": "confirm",
                "prompt": f"Confirm {item['description']}",
                "current_value": _current_value(plan, item["id"]),
            }
            if item["id"] == "deployment_platform":
                question["candidates"] = [
                    {"id": "gce", "description": "Google Compute Engine VM"},
                    {"id": "ec2", "description": "Amazon EC2 VM"},
                    {"id": "gke", "description": "Google Kubernetes Engine"},
                    {"id": "eks", "description": "Amazon Elastic Kubernetes Service"},
                    {"id": "self-hosted-k8s", "description": "Self-hosted Kubernetes cluster"},
                    {"id": "container", "description": "Container or Docker runtime outside a Kubernetes cluster"},
                    {"id": "vm", "description": "Generic virtual machine or bare-metal host"},
                ]
            questions.append(_with_manual_input(question))

    accounts_items = checklist.get("accounts_optional", [])
    if accounts_items and "has_accounts_device" not in confirmed:
        questions.append(_with_manual_input({
            "id": "has_accounts_device",
            "category": "storage",
            "severity": "confirm",
            "prompt": (
                "Does this node use a second accounts/state disk? If yes, confirm ACCOUNTS_DEVICE "
                "from the lsblk inventory and provide ACCOUNTS_VOL_* baselines."
            ),
            "current_value": _current_value(plan, "accounts_device"),
            "candidates": disk_candidates,
        }))
    for item in accounts_items:
        if item.get("severity") == "blocker" and not item.get("present"):
            questions.append(_with_manual_input({
                "id": item["id"],
                "category": "storage",
                "severity": "blocker",
                "prompt": item["description"],
            }))

    chain_requirements = plan.get("chain_template_requirements", {})
    if chain_requirements.get("exists"):
        workload_menu_needed = _workload_menu_required(confirmed)
        if workload_menu_needed:
            questions.append(_workload_customization_question(chain_requirements))
        sample_vars = chain_requirements.get("runtime_sample_variables", [])
        if sample_vars and not workload_menu_needed and "rpc_param_samples_confirmation" not in confirmed:
            questions.append(_with_manual_input({
                "id": "rpc_param_samples_confirmation",
                "category": "workload",
                "severity": "confirm",
                "prompt": "Confirm whether these chain template sample variables should use defaults or user-provided values.",
                "variables": sample_vars,
            }))
        endpoint_vars = chain_requirements.get("runtime_endpoint_variables", [])
        if endpoint_vars and "chain_endpoint_overrides_confirmation" not in confirmed:
            questions.append(_with_manual_input({
                "id": "chain_endpoint_overrides_confirmation",
                "category": "endpoint",
                "severity": "confirm",
                "prompt": "Confirm whether this chain needs endpoint overrides beyond LOCAL_RPC_URL.",
                "variables": endpoint_vars,
            }))

    if "advanced_config_review" not in confirmed:
        questions.append(_with_manual_input({
            "id": "advanced_config_review",
            "category": "advanced",
            "severity": "info",
            "prompt": (
                "Advanced thresholds use config/internal_config.sh defaults. "
                "Ask whether the user wants a short explanation or wants to adjust bottleneck, latency, success-rate, or sync-health thresholds."
            ),
            "variables": [
                "ACCOUNT_COUNT",
                "ACCOUNT_MAX_SIGNATURES",
                "ACCOUNT_TX_BATCH_SIZE",
                "ACCOUNT_SEMAPHORE_LIMIT",
                "MONITOR_INTERVAL",
                "DISK_MONITOR_RATE",
                "QPS_COOLDOWN",
                "QPS_WARMUP_DURATION",
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
        }))

    if "stress_execution" in plan.get("requires_confirmation", []):
        questions.append(_with_manual_input({
            "id": "stress_execution_confirmation",
            "category": "safety",
            "severity": "blocker",
            "prompt": "Confirm stress/intensive benchmark execution; it may affect the target node.",
        }))

    return _dedupe_questions(questions)


def _required_prompt(item: str) -> str:
    prompts = {
        "chain": "Which blockchain node should be tested?",
        "local_rpc_url": "Provide the local RPC endpoint, or choose fake-node for closed-loop testing.",
        "use_fake_node": "Choose fake-node closed-loop testing or real-node testing.",
        "blockchain_process_names": "Provide blockchain node process names or command-line fragments for resource attribution.",
        "ledger_device": "Confirm the ledger/data disk device used by the node.",
        "data_vol_type": "Provide the ledger/data disk type.",
        "data_vol_size": "Provide the ledger/data disk size in GiB.",
        "data_vol_max_iops": "Provide the provisioned ledger/data disk IOPS baseline.",
        "data_vol_max_throughput": "Provide the provisioned ledger/data disk throughput baseline in MiB/s.",
        "network_interface": "Confirm the network interface used by the node.",
        "network_max_bandwidth_gbps": "Provide the instance or pod network bandwidth baseline in Gbps.",
        "rpc_mode": "Choose single or mixed RPC workload mode.",
        "benchmark_mode_confirmed": "Choose quick, standard, or intensive benchmark mode.",
        "qps_profile_confirmed": "Confirm INITIAL_QPS, MAX_QPS, QPS_STEP, and DURATION for the selected mode.",
        "observability_choice_confirmed": "Choose disabled, local Prometheus/Grafana, or exporter-only observability mode.",
        "chain_template_reviewed": "Review selected chain template endpoints, TARGET_* sample variables, and default workload.",
        "sync_observe_stop_condition": "Choose how sync-observe should stop: run until stopped, fixed duration, or until synced.",
        "node_prometheus_metrics_url": "Provide the node Prometheus metrics endpoint if available, or leave it unset.",
        "node_process_identity": "Provide the node process PID or command-line fragments for CPU/thread attribution.",
    }
    return prompts.get(item, f"Provide required value: {item}")


def _is_sync_observe_plan(plan: dict[str, Any]) -> bool:
    value = plan.get("workflow_type") or plan.get("run_mode") or plan.get("mode")
    return str(value or "").strip().lower().replace("-", "_") in {"sync_observe", "sync", "observe_sync"}


def _sync_observe_questions(plan: dict[str, Any]) -> list[dict[str, Any]]:
    confirmed = set(plan.get("confirmed_inputs", []))
    questions: list[dict[str, Any]] = []
    if "sync_observe_stop_condition" not in confirmed:
        questions.append(_with_manual_input({
            "id": "sync_observe_stop_condition",
            "category": "sync_observe",
            "severity": "blocker",
            "prompt": (
                "Choose sync-observe stop condition. This mode observes node sync/resource behavior "
                "without RPC workload, proxy, Vegeta, or QPS profile."
            ),
            "candidates": [
                {"id": "until_stopped", "description": "Run until the user stops it; default for long sync observation."},
                {"id": "duration", "description": "Run for a fixed duration in seconds."},
                {"id": "until_synced", "description": "Run until the chain sync-health model reports synced."},
            ],
        }))
    if "node_prometheus_metrics_url" not in confirmed:
        questions.append(_with_manual_input({
            "id": "node_prometheus_metrics_url",
            "category": "sync_observe",
            "severity": "confirm",
            "prompt": (
                "Confirm whether the node exposes a Prometheus metrics endpoint for MGas/s or "
                "client-native execution metrics. If unavailable, reports still show block height, CPU, disk, and network."
            ),
            "manual_input_hint": "Enter a metrics URL such as http://127.0.0.1:6060/debug/metrics/prometheus, or reply N if unavailable.",
        }))
    if "node_process_identity" not in confirmed:
        questions.append(_with_manual_input({
            "id": "node_process_identity",
            "category": "sync_observe",
            "severity": "blocker",
            "prompt": "Confirm the node process PID or process-name/command-line fragment for CPU and thread attribution.",
        }))
    if "mainnet_rpc_url_reviewed" not in confirmed:
        questions.append(_with_manual_input({
            "id": "mainnet_rpc_url_reviewed",
            "category": "sync_observe",
            "severity": "blocker",
            "prompt": "Confirm MAINNET_RPC_URL or the chain-template sync-health behavior used for target-height comparison.",
        }))
    return questions


_SPECIALIZED_REQUIRED_QUESTIONS = {
    "benchmark_mode_confirmed",
    "qps_profile_confirmed",
    "observability_choice_confirmed",
    "chain_template_reviewed",
}


_WORKLOAD_MENU_CONFIRMATIONS = {
    "chain_template_reviewed",
    "rpc_workload_confirmation",
    "rpc_workload_confirmed",
    "custom_rpc_method_review",
}


def _workload_menu_required(confirmed: set[str]) -> bool:
    return not _WORKLOAD_MENU_CONFIRMATIONS.issubset(confirmed)


def _workload_customization_question(chain_requirements: dict[str, Any]) -> dict[str, Any]:
    return _with_manual_input({
        "id": "workload_customization_choice",
        "category": "workload",
        "severity": "blocker",
        "prompt": (
            "Review the selected chain template defaults and choose one workload path: "
            "continue with defaults, add a custom RPC method, adjust mixed weights, or change chain/mode."
        ),
        "runtime_endpoint_variables": chain_requirements.get("runtime_endpoint_variables", []),
        "runtime_sample_variables": chain_requirements.get("runtime_sample_variables", []),
        "single": chain_requirements.get("single_method"),
        "mixed_weighted": chain_requirements.get("mixed_weighted", []),
        "extension_fields": chain_requirements.get("custom_rpc_extension_fields", []),
        "param_formats": chain_requirements.get("param_formats", {}),
        "param_spec_methods": chain_requirements.get("param_spec_methods", []),
        "options": [
            {
                "id": "1",
                "value": "use_defaults",
                "label": "Continue with chain template defaults",
                "state_patch": {
                    "confirmed_config": {
                        "chain_template_reviewed": True,
                        "rpc_workload_confirmed": True,
                        "rpc_workload_confirmation": True,
                        "rpc_param_samples_confirmed": True,
                        "rpc_param_samples_confirmation": True,
                        "custom_rpc_method_review": True,
                    }
                },
            },
            {"id": "2", "value": "add_custom_rpc", "label": "Add a custom RPC method"},
            {"id": "3", "value": "adjust_weights", "label": "Adjust mixed weights"},
            {"id": "4", "value": "change_chain_or_mode", "label": "Change chain or target mode"},
        ],
        "workflow_step": "workload_customization_choice",
        "branch": "rpc_workload",
    })


def _with_manual_input(question: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(question)
    enriched["manual_input_allowed"] = True
    enriched["allow_manual_input"] = True
    enriched.setdefault("expected_answer", _expected_answer_for(enriched))
    enriched.setdefault("field", _field_for(enriched["id"]))
    enriched.setdefault("branch", "benchmark_setup")
    enriched.setdefault("workflow_step", enriched["id"])
    enriched.setdefault("validation_tool", "validate_required_config")
    if "candidates" in enriched and "options" not in enriched:
        enriched["options"] = _options_from_candidates(enriched.get("candidates", []))
    enriched.setdefault(
        "next_on_yes",
        {"workflow_step": f"{enriched['id']}:confirmed", "tool": "validate_required_config"},
    )
    enriched.setdefault(
        "next_on_no",
        {"workflow_step": f"{enriched['id']}:manual_input", "tool": "build_missing_config_questions"},
    )
    enriched.setdefault(
        "next_on_manual",
        {"workflow_step": f"{enriched['id']}:manual_value", "tool": "validate_required_config"},
    )
    enriched.setdefault(
        "manual_input_hint",
        "The user may reply with a listed number/id or provide a custom value.",
    )
    return enriched


def _expected_answer_for(question: dict[str, Any]) -> str:
    qid = str(question.get("id", ""))
    if qid in {"benchmark_mode_confirmed", "observability_choice_confirmed", "disk_inventory_confirmation", "has_accounts_device", "workload_customization_choice"}:
        return "numbered_choice"
    if qid in {"ledger_device_confirmation", "ledger_device", "accounts_device"}:
        return "device"
    if qid in {"local_rpc_url", "mainnet_rpc_url", "target_rpc_url"}:
        return "url"
    if qid in {"mixed_weights_confirmation", "rpc_workload_confirmation"}:
        return "multi_select"
    if qid.endswith("_confirmed") or qid.endswith("_confirmation") or qid.endswith("_review"):
        return "yes_no"
    return "manual_value"


def _field_for(question_id: str) -> str:
    mapping = {
        "ledger_device_confirmation": "LEDGER_DEVICE",
        "disk_inventory_confirmation": "LEDGER_DEVICE",
        "has_accounts_device": "ACCOUNTS_DEVICE",
        "benchmark_mode_confirmed": "BENCHMARK_MODE",
        "qps_profile_confirmed": "QPS_PROFILE",
        "observability_choice_confirmed": "OBSERVABILITY_STACK_MODE",
        "chain_template_reviewed": "CHAIN_TEMPLATE",
        "workload_customization_choice": "RPC_WORKLOAD",
        "rpc_workload_confirmation": "RPC_WORKLOAD",
        "mixed_weights_confirmation": "MIXED_WEIGHTS",
        "rpc_param_samples_confirmation": "TARGET_SAMPLES",
        "chain_endpoint_overrides_confirmation": "CHAIN_ENDPOINT_OVERRIDES",
        "advanced_config_review": "ADVANCED_CONFIG",
    }
    return mapping.get(question_id, question_id.upper())


def _options_from_candidates(candidates: Any) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    if not isinstance(candidates, list):
        return options
    for index, item in enumerate(candidates, start=1):
        if isinstance(item, dict):
            value = item.get("id") or item.get("name") or item.get("value") or str(index)
            label = item.get("description") or item.get("label") or value
            option = dict(item)
            option["id"] = str(index)
            option["value"] = value
            option["label"] = str(label)
            options.append(option)
        else:
            options.append({"id": str(index), "value": str(item), "label": str(item)})
    return options


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

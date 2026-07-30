"""Generate deterministic benchmark plans from Agent requests."""

from __future__ import annotations

import json
import hashlib
import re
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

from agent.runners.private_files import atomic_write_private_text

from .chain_template_requirements import inspect_chain_template
from .config_checklist import build_configuration_checklist, missing_required_from_checklist
from .risk import score_plan_risk
from ..knowledge.entry_contract import field_specs_for
from ..knowledge.qps_profiles import (
    STRATEGY_BENCHMARK_MODE,
    strategy_qps_defaults,
)


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
    strategy: strategy_qps_defaults(strategy)
    for strategy in STRATEGY_BENCHMARK_MODE
}


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    atomic_write_private_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def generate_plan(request: dict[str, Any], discovery: dict[str, Any] | None = None) -> dict[str, Any]:
    chain = (request.get("chain") or "").strip().lower()
    workflow_type = _workflow_type(request)
    is_sync_observe = workflow_type == "sync_observe"
    goal = request.get("goal") or "baseline"
    strategy = "sync_observe" if is_sync_observe else (_strategy_from_mode(request.get("benchmark_mode")) or GOAL_TO_STRATEGY.get(goal, "baseline"))
    benchmark_mode = _benchmark_mode(strategy)
    rpc_mode = "sync_observe" if is_sync_observe else (request.get("rpc_mode") or "single")
    use_fake_node = False if is_sync_observe else (request.get("use_fake_node") if isinstance(request.get("use_fake_node"), bool) else None)
    qps = {} if is_sync_observe else {**DEFAULT_QPS[strategy], **request.get("qps", {})}
    confirmations = set(request.get("confirmations", []))
    assumed_values = dict(request.get("assumed_values") or {})
    assumed_for_smoke = bool(request.get("assumed_for_smoke"))
    runner_mode = request.get("runner_mode", "detached")
    if runner_mode not in {"detached", "foreground"}:
        runner_mode = "detached"

    required_inputs = []
    if not chain:
        required_inputs.append("chain")
    if use_fake_node is None and not is_sync_observe:
        required_inputs.append("use_fake_node")
    if use_fake_node is False and not is_sync_observe and not request.get("local_rpc_url"):
        required_inputs.append("local_rpc_url")
    requires_confirmation = []
    if request.get("workload", {}).get("methods") and "mixed_weights_confirmation" not in confirmations:
        requires_confirmation.append("mixed_weights")
    if strategy == "stress" and "stress_execution_confirmation" not in confirmations:
        requires_confirmation.append("stress_execution")

    if is_sync_observe:
        command = ["./blockchain_node_benchmark.sh", "--sync-observe"]
        stop_condition = str(request.get("sync_observe_stop_condition") or "").strip()
        if stop_condition == "until_synced":
            command.append("--until-synced")
        if stop_condition == "duration" and request.get("sync_observe_duration_seconds") is not None:
            command.extend(["--duration", str(int(request.get("sync_observe_duration_seconds") or 0))])
    else:
        command = ["./blockchain_node_benchmark.sh", _mode_flag(strategy), f"--{rpc_mode}"]
        if use_fake_node is True:
            command.append("--fake-node")

    qps_prefix = _qps_env_prefix(strategy) if not is_sync_observe else "SYNC_OBSERVE"
    env = {
        "BLOCKCHAIN_NODE": chain,
        "RPC_MODE": rpc_mode,
        "LOCAL_RPC_URL": request.get("local_rpc_url", ""),
        "SYNC_OBSERVE_RPC_URL": request.get("sync_observe_rpc_url", ""),
        "NODE_PROMETHEUS_METRICS_URL": request.get("node_prometheus_metrics_url", ""),
        "MAINNET_RPC_URL": request.get("mainnet_rpc_url", ""),
        "MAINNET_RPC_URL_DISABLED": str(
            bool(request.get("mainnet_rpc_url_disabled"))
        ).lower(),
        f"{qps_prefix}_INITIAL_QPS": str(qps.get("initial", "")),
        f"{qps_prefix}_MAX_QPS": str(qps.get("max", "")),
        f"{qps_prefix}_QPS_STEP": str(qps.get("step", "")),
        f"{qps_prefix}_DURATION": str(qps.get("duration_seconds", "")),
        "SYNC_OBSERVE_MODE": str(is_sync_observe).lower(),
        "SYNC_OBSERVE_STOP_CONDITION": str(request.get("sync_observe_stop_condition", "")),
        "SYNC_OBSERVE_DURATION": str(request.get("sync_observe_duration_seconds", "")),
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
        "SYNC_OBSERVE_RPC_URL": request.get("sync_observe_rpc_url", ""),
        "NODE_PROMETHEUS_METRICS_URL": request.get("node_prometheus_metrics_url", ""),
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
        "workflow_type": workflow_type,
        "run_mode": workflow_type,
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
        "assumed_for_smoke": assumed_for_smoke,
        "assumed_values": assumed_values,
        "materialized_config": materialized_config,
        "chain_template_requirements": inspect_chain_template(chain),
        "required_inputs": required_inputs,
        "advanced_defaults": {
            "qps": qps,
            "observability": request.get("observability", {"enabled": False, "mode": "local"}),
            "sync_observe": {
                "stop_condition": request.get("sync_observe_stop_condition", ""),
                "duration_seconds": request.get("sync_observe_duration_seconds", ""),
            },
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
    chain_override = _chain_config_override(chain, request)
    if chain_override:
        plan["chain_config_override"] = chain_override
        plan["artifacts"]["chain_config_override_file"] = "<job_run_dir>/chain_template.override.json"
        plan["chain_template_requirements"] = (
            template_requirements_from_override(chain, chain_override)
        )
    checklist = build_configuration_checklist(request, plan)
    plan["configuration_checklist"] = checklist
    combined_required = _ordered_required_inputs(set(plan["required_inputs"]) | set(missing_required_from_checklist(checklist)))
    plan["required_inputs"] = combined_required
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


def _workflow_type(request: dict[str, Any]) -> str:
    text = str(request.get("workflow_type") or request.get("run_mode") or "").strip().lower().replace("-", "_")
    return "sync_observe" if text in {"sync_observe", "sync", "observe_sync"} else "rpc_benchmark"


def _benchmark_mode(strategy: str) -> str:
    if strategy == "sync_observe":
        return "sync_observe"
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


def _chain_config_override(chain: str, request: dict[str, Any]) -> dict[str, Any]:
    """Return a job-local chain template override when workload changed.

    The benchmark engine reads workloads from CHAIN_CONFIG. Agent-only plan
    fields are therefore not enough: if the user confirms mixed weights or a
    single-method override, this function materializes that workload into the
    same chain-template shape consumed by config_loader.sh and target_generator.sh.
    """
    if not chain:
        return {}
    explicit = request.get("chain_config_override")
    if isinstance(explicit, dict) and explicit:
        return json.loads(json.dumps(explicit))
    chain_file = REPO_ROOT / "config" / "chains" / f"{chain}.json"
    if not chain_file.is_file():
        return {}
    mixed_weighted = request.get("mixed_weighted")
    rpc_methods = request.get("rpc_methods")
    if not mixed_weighted and not rpc_methods:
        return {}

    data = load_json(chain_file)
    rpc = dict(data.get("rpc_methods") or {})
    if mixed_weighted:
        rows = [
            {"method": str(item.get("method", "")).strip(), "weight": int(item.get("weight", 0) or 0)}
            for item in mixed_weighted
            if isinstance(item, dict) and str(item.get("method", "")).strip()
        ]
        if rows:
            rpc["mixed_weighted"] = rows
            rpc["mixed"] = ",".join(item["method"] for item in rows)
    if rpc_methods:
        methods = [str(method).strip() for method in rpc_methods if str(method).strip()]
        if methods and (request.get("rpc_mode") or "single") == "single":
            rpc["single"] = methods[0]
    data["rpc_methods"] = rpc
    return data


def materialize_custom_rpc_template(
    *,
    chain: str,
    adapter_family: str,
    rpc_mode: str,
    workload: dict[str, Any],
    validated_methods: list[dict[str, Any]],
    contract_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the complete job-local template for a validated custom workload.

    This works for both Case1 (overlay a canonical template) and Case2 (create
    a complete runtime-only template for a new chain in an existing family).
    Concrete probed params become literal ``param_spec`` values so target
    generation sends the exact reviewed wire payload.
    """

    chain = str(chain or "").strip().lower()
    family = str(adapter_family or "").strip().lower()
    if not chain:
        return {}
    chain_file = REPO_ROOT / "config" / "chains" / f"{chain}.json"
    has_canonical_template = chain_file.is_file()
    template = load_json(chain_file) if has_canonical_template else {
        "chain_type": chain,
        "rpc_url": "LOCAL_RPC_URL",
        "params": {},
        "param_formats": {},
        "param_spec": {},
        "rpc_methods": {},
        "_meta": {"source": "agent-job-local-case2", "adapter_family": family},
    }
    family = family or str((template.get("_meta") or {}).get("adapter_family") or "").strip().lower()
    if not family:
        return {}
    template["chain_type"] = str(template.get("chain_type") or chain)
    template["rpc_url"] = str(template.get("rpc_url") or "LOCAL_RPC_URL")
    meta = dict(template.get("_meta") or {})
    meta["adapter_family"] = family
    meta["job_local_override"] = True
    template["_meta"] = meta

    selected = [str(method).strip() for method in workload.get("methods") or [] if str(method).strip()]
    weights = {
        str(method).strip(): int(weight)
        for method, weight in (workload.get("mixed_weights") or {}).items()
        if str(method).strip()
    }
    if rpc_mode == "single":
        if len(selected) != 1:
            return {}
        rpc_methods = {"single": selected[0], "mixed": selected[0], "mixed_weighted": [{"method": selected[0], "weight": 100}]}
    else:
        if not selected or set(selected) != set(weights) or sum(weights.values()) != 100 or any(weight <= 0 for weight in weights.values()):
            return {}
        rpc_methods = {
            "single": selected[0],
            "mixed": ",".join(selected),
            "mixed_weighted": [{"method": method, "weight": weights[method]} for method in selected],
        }
    contracts = {
        str(item.get("method") or "").strip(): item
        for item in validated_methods
        if isinstance(item, dict) and str(item.get("method") or "").strip()
    }
    canonical_rpc = template.get("rpc_methods") if isinstance(template.get("rpc_methods"), dict) else {}
    canonical_methods = {
        str(canonical_rpc.get("single") or "").strip(),
        *(str(method).strip() for method in str(canonical_rpc.get("mixed") or "").split(",")),
        *(
            str(item.get("method") or "").strip()
            for item in canonical_rpc.get("mixed_weighted") or []
            if isinstance(item, dict)
        ),
    }
    canonical_methods.discard("")
    contract_required = set(selected) if not has_canonical_template else set(selected) - canonical_methods
    if any(method not in contracts for method in contract_required):
        return {}
    selected_contracts = [
        contracts[method]
        for method in selected
        if method in contracts
    ]
    if selected_contracts:
        if contract_state is None:
            return {}
        from agent.harness.domains.rpc_catalog import (
            validated_method_contract_is_current,
        )

        if any(
            not validated_method_contract_is_current(
                contract_state,
                contract,
            )
            for contract in selected_contracts
        ):
            return {}

    template["rpc_methods"] = rpc_methods
    param_formats = dict(template.get("param_formats") or {})
    param_spec = dict(template.get("param_spec") or {})
    for method in selected:
        contract = contracts.get(method)
        if not contract:
            continue
        params = contract.get("params", (contract.get("schema") or {}).get("params_json"))
        if not isinstance(params, (list, dict)):
            return {}
        if isinstance(params, list):
            param_formats[method] = "no_params" if not params else "param_spec"
            param_spec[method] = {
                "transport": "jsonrpc_list",
                "params": [{"literal": value} for value in params],
            }
        elif isinstance(params, dict):
            param_formats[method] = "param_spec"
            param_spec[method] = {
                "transport": "jsonrpc_dict",
                "fields": {name: {"literal": value} for name, value in params.items()},
            }
    template["param_formats"] = param_formats
    template["param_spec"] = param_spec
    proxy_extraction = _materialize_proxy_extraction(template, family, selected)
    if not proxy_extraction:
        return {}
    template["proxy_extraction"] = proxy_extraction
    materialization_evidence = {
        "source_kind": (
            "canonical_template_overlay"
            if has_canonical_template
            else "new_chain_runtime_template"
        ),
        "chain": chain,
        "adapter_family": family,
        "rpc_mode": rpc_mode,
        "method_hashes": [_stable_digest(method) for method in selected],
        "mixed_weight_entries": [
            {
                "method_hash": _stable_digest(method),
                "weight": weights[method],
            }
            for method in selected
            if method in weights
        ],
        "replace_defaults": bool(workload.get("replace_defaults")),
        "required_contract_method_hashes": [
            _stable_digest(method) for method in sorted(contract_required)
        ],
        "parameter_contracts": [
            {
                "method_hash": _stable_digest(method),
                "contract_hash": _stable_digest(param_spec.get(method)),
            }
            for method in selected
            if method in param_spec
        ],
        "effective_rpc_methods_hash": _stable_digest(rpc_methods),
        "template_hash": _stable_digest(template),
        "template_hash_scope": "template_without_materialization_evidence",
    }
    meta["materialization_evidence"] = materialization_evidence
    template["_meta"] = meta
    return template


def _materialize_proxy_extraction(
    template: dict[str, Any],
    family: str,
    selected_methods: list[str],
) -> dict[str, Any]:
    """Complete the proxy DSL from adapter-family and validated method facts."""

    existing = template.get("proxy_extraction")
    extractors = [
        dict(item)
        for item in ((existing or {}).get("extractors") if isinstance(existing, dict) else []) or []
        if isinstance(item, dict)
    ]
    jsonrpc_families = {"jsonrpc", "substrate", "tendermint", "bitcoin_jsonrpc"}
    needs_jsonrpc = family in jsonrpc_families or (
        family == "hedera_dual" and any(not method.startswith("/") for method in selected_methods)
    )
    if needs_jsonrpc and not any(item.get("protocol") == "json_rpc" for item in extractors):
        extractors.append({
            "protocol": "json_rpc",
            "method_source": "body.method",
            "id_source": "body.id",
            "params_source": "body.params",
            "url_pattern": "^/$",
            "batch_handling": "split",
        })

    rest_methods = [method for method in selected_methods if method.startswith("/")]
    if family in {"rest", "hedera_dual"} and rest_methods:
        rest_extractor = next(
            (item for item in extractors if item.get("protocol") == "rest"),
            None,
        )
        if rest_extractor is None:
            rest_extractor = {"protocol": "rest", "url_patterns": []}
            extractors.insert(0, rest_extractor)
        patterns = [
            dict(item)
            for item in rest_extractor.get("url_patterns") or []
            if isinstance(item, dict)
        ]
        known_names = {str(item.get("method_name") or "") for item in patterns}
        for method in rest_methods:
            if method in known_names:
                continue
            path = method.split("?", 1)[0]
            patterns.append({"pattern": f"^{re.escape(path)}$", "method_name": method})
        rest_extractor["url_patterns"] = patterns

    if family == "rest" and not rest_methods:
        return {}
    return {"extractors": extractors} if extractors else {}


def template_requirements_from_override(chain: str, template: dict[str, Any]) -> dict[str, Any]:
    """Return the preflight workload facts for a validated runtime template."""

    rpc = template.get("rpc_methods") if isinstance(template.get("rpc_methods"), dict) else {}
    weighted = [
        {"method": str(item.get("method") or ""), "weight": int(item.get("weight") or 0)}
        for item in rpc.get("mixed_weighted") or []
        if isinstance(item, dict) and str(item.get("method") or "").strip()
    ]
    return {
        "chain": chain,
        "exists": bool(template),
        "path": "<job-local-chain-template>",
        "adapter_family": str((template.get("_meta") or {}).get("adapter_family") or ""),
        "single_method": str(rpc.get("single") or ""),
        "mixed_weighted": weighted,
        "param_formats": dict(template.get("param_formats") or {}),
        "param_spec_methods": sorted((template.get("param_spec") or {}).keys()),
        "runtime_sample_variables": [],
        "runtime_endpoint_variables": ["LOCAL_RPC_URL"],
        "sync_health_mode": str(((template.get("_meta") or {}).get("sync_health") or {}).get("mode") or ""),
    }


def _stable_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _ordered_required_inputs(items: set[str]) -> list[str]:
    known = [item for item in _REQUIRED_INPUT_PRIORITY if item in items]
    unknown = sorted(item for item in items if item not in _REQUIRED_INPUT_PRIORITY)
    return known + unknown


# The order questions are asked in for a real-node plan. Kept as an explicit
# ordered list (rather than derived from `entry_contract.field_specs_for`,
# whose grouping order differs) because reordering this list is a live
# behavior change, not just a catalog-membership one. `has_accounts_device`
# and `mixed_weights_confirmed` are not modeled as `RuntimeField`s (they are
# derived confirmation flags, not entrypoint values), hence the explicit
# exception set below. The check below only catches a key being REMOVED or
# renamed out of `entry_contract.py`'s canonical catalog (a stale reference);
# it cannot catch a key being ADDED to the catalog and never added here, since
# that still satisfies the subset check — such a field would silently fall
# into `_ordered_required_inputs`'s alphabetical "unknown" tail instead of a
# deliberate position. `cloud_region`/`cloud_zone`/`machine_type` are a known,
# pre-existing case of this: they are in the canonical catalog for real_node
# but were never part of this priority list.
_REQUIRED_INPUT_PRIORITY = [
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
_known_real_node_keys = {field.key for field in field_specs_for("real_node")} | {"has_accounts_device", "mixed_weights_confirmed"}
if not set(_REQUIRED_INPUT_PRIORITY) <= _known_real_node_keys:
    # Not an `assert` on purpose: `python -O` strips asserts, and this
    # invariant must hold even in optimized runs.
    raise RuntimeError("_REQUIRED_INPUT_PRIORITY references a key removed from entry_contract.field_specs_for('real_node')")

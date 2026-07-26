"""Preflight checks for Agent benchmark plans."""

from __future__ import annotations

import os
import json
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agent.knowledge.entry_contract import (
    ENTRYPOINT_SCRIPTS,
    dependency_names,
    validate_mixed_weighted,
)
from agent.planners.chain_template_requirements import inspect_chain_template
from agent.planners.strategy_planner import template_requirements_from_override
from agent.validators.fixture_checks import validate_plan_fake_node_fixtures

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_preflight(plan: dict[str, Any]) -> dict[str, Any]:
    checks = []
    warnings = []

    chain = plan.get("chain", "")
    is_sync_observe = str(plan.get("workflow_type") or plan.get("run_mode") or "").strip().lower().replace("-", "_") == "sync_observe"
    checks.append(_check(
        "required_inputs_present",
        not plan.get("required_inputs"),
        f"missing: {', '.join(plan.get('required_inputs', []))}" if plan.get("required_inputs") else "",
    ))
    checklist_missing = plan.get("configuration_checklist", {}).get("missing_blockers", [])
    checks.append(_check(
        "configuration_checklist_complete",
        not checklist_missing,
        f"missing: {', '.join(checklist_missing)}" if checklist_missing else "",
    ))

    chain_template = REPO_ROOT / "config" / "chains" / f"{chain}.json"
    chain_template_override = plan.get("chain_config_override")
    chain_template_requirements = _resolved_chain_template_requirements(plan, chain)
    if isinstance(chain_template_override, dict) and chain_template_override:
        checks.append(_check(
            "chain_template_exists",
            bool(chain_template_override),
            "chain template provided by runtime override",
        ))
        checks.append(_check(
            "chain_template_json_valid",
            _dict_has_required_runtime_template_fields(chain_template_override, chain_template_requirements),
            "missing required runtime override fields",
        ))
    else:
        checks.append(_check("chain_template_exists", chain_template.is_file(), str(chain_template)))
        checks.append(_check("chain_template_json_valid", _json_valid(chain_template), str(chain_template)))
    checks.append(_check(
        "chain_template_requirements_available",
        bool(chain_template_requirements),
        "chain template workload requirements could not be resolved",
    ))

    entry = REPO_ROOT / "blockchain_node_benchmark.sh"
    checks.append(_check("benchmark_entry_exists", entry.is_file() and os.access(entry, os.X_OK), str(entry)))
    for script in ENTRYPOINT_SCRIPTS:
        path = REPO_ROOT / script
        checks.append(_check(f"entry_script_exists:{script}", path.is_file(), str(path)))

    fake_node_dir = REPO_ROOT / "tools" / "fake-node"
    fake_node_ok = not plan.get("use_fake_node") or fake_node_dir.is_dir()
    checks.append(_check("fake_node_available_when_requested", fake_node_ok, str(fake_node_dir)))
    if plan.get("use_fake_node"):
        checks.append(_check("fake_node_fixtures_available", (fake_node_dir / "fixtures").is_dir(), str(fake_node_dir / "fixtures")))
        fixture_result = validate_plan_fake_node_fixtures(plan)
        missing = ", ".join(
            f"{item['method']} -> {item['fixture']}" for item in fixture_result.get("missing", [])
        )
        checks.append(_check(
            "effective_workload_fixtures_available",
            bool(fixture_result.get("passed")),
            missing or f"methods={', '.join(fixture_result.get('methods', []))}",
        ))

    rpc_mode = plan.get("rpc_mode", "")
    checks.append(_check("rpc_mode_valid", rpc_mode in {"single", "mixed"} or is_sync_observe, rpc_mode))
    use_fake_node = bool(plan.get("use_fake_node"))
    workload_checks = _workload_checks(
        rpc_mode,
        chain_template_requirements,
        chain_template_override,
        is_sync_observe,
        use_fake_node=use_fake_node,
    )
    checks.extend(workload_checks)
    if rpc_mode == "mixed" and not is_sync_observe:
        ok, detail = validate_mixed_weighted(plan.get("chain_template_requirements", {}))
        checks.append(_check("mixed_weighted_total_valid", ok, detail))

    if not use_fake_node and not is_sync_observe:
        local_rpc_url = plan.get("execution", {}).get("environment", {}).get("LOCAL_RPC_URL", "")
        checks.append(_check("local_rpc_url_valid", _valid_endpoint(local_rpc_url), local_rpc_url))

    dependency_check = _dependency_check(plan)
    if dependency_check["blockers"]:
        checks.append(_check("entrypoint_dependencies_available", False, ", ".join(dependency_check["blockers"])))
    else:
        checks.append(_check("entrypoint_dependencies_available", True, "required tools available or not reported missing"))
    warnings.extend(dependency_check["warnings"])

    fidelity = _fidelity_checks(plan)
    checks.extend(fidelity["checks"])
    warnings.extend(fidelity["warnings"])

    current_dir = REPO_ROOT / "current"
    current_dir.mkdir(exist_ok=True)
    checks.append(_check("output_directories_writable", os.access(current_dir, os.W_OK), str(current_dir)))

    passed = all(item["passed"] for item in checks)
    blockers = [f"{item['name']}: {item['detail']}" for item in checks if not item["passed"]]
    return {"passed": passed, "checks": checks, "warnings": warnings, "blockers": blockers}


def _check(name: str, passed: bool, detail: str = "") -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _resolved_chain_template_requirements(plan: dict[str, Any], chain: str) -> dict[str, Any]:
    requirements = plan.get("chain_template_requirements")
    if isinstance(requirements, dict) and requirements:
        return requirements
    override = plan.get("chain_config_override")
    if isinstance(override, dict) and override:
        return _requirements_from_override(chain, override)
    return inspect_chain_template(chain)


def _requirements_from_override(chain: str, template: dict[str, Any]) -> dict[str, Any]:
    return template_requirements_from_override(chain, template)


def _json_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except Exception:
        return False


def _dict_has_required_runtime_template_fields(payload: dict[str, Any], requirements: dict[str, Any]) -> bool:
    required_keys = {"rpc_methods"}
    if not isinstance(payload, dict):
        return False
    if not required_keys.issubset(set(payload.keys())):
        return False
    rpc_methods = payload.get("rpc_methods")
    if not isinstance(rpc_methods, dict):
        return False
    adapter_family = str((payload.get("_meta") or {}).get("adapter_family") or "").strip()
    if not adapter_family and not str(requirements.get("adapter_family") or "").strip():
        return False
    return True


def _workload_checks(
    rpc_mode: str,
    requirements: dict[str, Any],
    chain_template_override: Any,
    is_sync_observe: bool,
    *,
    use_fake_node: bool = False,
) -> list[dict[str, Any]]:
    if is_sync_observe:
        return []

    checks: list[dict[str, Any]] = []
    mode = str(rpc_mode or "").strip().lower()
    if mode not in {"single", "mixed"}:
        return [
            _check("chain_workload_requires_rpc_mode", False, f"unsupported rpc_mode={rpc_mode}"),
        ]

    methods = []
    if mode == "single":
        method = str(requirements.get("single_method") or "").strip()
        if not method and isinstance(chain_template_override, dict):
            method = str((chain_template_override.get("rpc_methods") or {}).get("single") or "").strip()
        checks.append(_check("single_rpc_method_present", bool(method), f"single method={method or '<empty>'}"))
        if method:
            methods = [method]
    else:
        weighted = requirements.get("mixed_weighted") if isinstance(requirements.get("mixed_weighted"), list) else []
        entries: list[tuple[str, int]] = []
        for item in weighted:
            if not isinstance(item, dict):
                continue
            method = str(item.get("method") or "").strip()
            weight = _safe_int(item.get("weight"))
            if not method:
                checks.append(_check("mixed_weighted_method", False, "method is required"))
                continue
            if weight < 0:
                checks.append(_check("mixed_weighted_weight", False, f"invalid weight for {method}: {item.get('weight')}"))
                continue
            entries.append((method, weight))
        duplicate = [method for method in set(method for method, _ in entries) if [m for m, _ in entries].count(method) > 1]
        if duplicate:
            checks.append(_check("mixed_weighted_no_duplicate_methods", False, f"duplicate methods: {', '.join(sorted(set(duplicate)))}"))
        non_positive = [method for method, weight in entries if weight <= 0]
        if non_positive:
            checks.append(_check("mixed_weighted_positive_weights", False, f"weights must be > 0; invalid methods: {', '.join(sorted(set(non_positive)))}"))
        if not entries:
            checks.append(_check("mixed_weighted_entries", False, "rpc_mode=mixed requires mixed_weighted entries"))
        total = sum(weight for _, weight in entries)
        checks.append(_check("mixed_weighted_total_valid", total == 100, f"mixed weights must total 100, got {total}"))
        checks.append(_check("mixed_weighted_methods_present", bool(entries), f"methods={', '.join(method for method, _ in entries)}"))
        methods = [method for method, _ in entries]

    param_spec_methods = [str(m) for m in (requirements.get("param_spec_methods") or [])]
    if use_fake_node:
        return checks

    for method in methods:
        if method and requirements.get("runtime_sample_variables") and method in ("",):
            # compatibility branch for historical data where runtime sample variables are not yet generated.
            continue
        if method and method not in param_spec_methods and not _method_has_default_contract(requirements, method):
            checks.append(_check(
                "rpc_method_contract_available",
                False,
                f"method {method} missing from param spec/contract metadata",
            ))

    return checks


def _method_has_default_contract(requirements: dict[str, Any], method: str) -> bool:
    if not method:
        return False
    param_formats = requirements.get("param_formats") or {}
    if isinstance(param_formats, dict) and method in param_formats:
        return True
    return False


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return -1


def _valid_endpoint(value: str) -> bool:
    parsed = urlparse(value or "")
    return parsed.scheme in {"http", "https", "ws", "wss"} and bool(parsed.netloc)


def _dependency_check(plan: dict[str, Any]) -> dict[str, list[str]]:
    discovery = plan.get("discovery", {})
    reported = discovery.get("dependencies", {}).get("tools", {})
    required = dependency_names(bool(plan.get("use_fake_node")))
    blockers: list[str] = []
    warnings: list[str] = []
    dependency_mode = plan.get("dependency_mode", "audit")
    for name in required:
        available = _tool_available(name, reported)
        if available:
            continue
        message = f"{name} is required by benchmark entrypoint"
        if dependency_mode == "audit":
            warnings.append(message)
        else:
            blockers.append(message)
    return {"blockers": blockers, "warnings": warnings}


def _tool_available(name: str, reported: dict[str, Any]) -> bool:
    if name in reported:
        return bool(reported.get(name, {}).get("available"))
    return bool(shutil.which(name))


def _fidelity_checks(plan: dict[str, Any]) -> dict[str, list[Any]]:
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []
    assumed_for_smoke = bool(plan.get("assumed_for_smoke"))
    discovery = plan.get("discovery", {})
    candidates = _candidate_names(discovery.get("disks", {}))
    materialized = plan.get("materialized_config", {})
    ledger = materialized.get("LEDGER_DEVICE", "")
    accounts = materialized.get("ACCOUNTS_DEVICE", "")
    network = materialized.get("NETWORK_INTERFACE", "")
    iface = discovery.get("network", {}).get("default_interface", "")

    if candidates:
        if assumed_for_smoke:
            warnings.append(
                "assumed_for_smoke plan uses smoke-only disk assumptions; "
                f"ledger={ledger}, accounts={accounts or '<none>'}, candidates={', '.join(candidates)}"
            )
        else:
            checks.append(_check("ledger_device_candidate_known", ledger in candidates, f"ledger={ledger}, candidates={', '.join(candidates)}"))
            if accounts:
                checks.append(_check("accounts_device_candidate_known", accounts in candidates, f"accounts={accounts}, candidates={', '.join(candidates)}"))
    else:
        warnings.append("disk inventory was unavailable; disk charts may be degraded unless the selected devices are visible to iostat")

    if iface:
        if assumed_for_smoke and network and network != iface:
            warnings.append(
                "assumed_for_smoke plan uses a smoke-only network interface assumption; "
                f"selected={network}, detected={iface}"
            )
        else:
            checks.append(_check("network_interface_detected_or_confirmed", network == iface or bool(network), f"selected={network}, detected={iface}"))
    else:
        warnings.append("network interface could not be detected; network charts depend on the confirmed NETWORK_INTERFACE")

    return {"checks": checks, "warnings": warnings}


def _candidate_names(disks: dict[str, Any]) -> list[str]:
    names = []
    for item in disks.get("candidates", []) or []:
        name = item.get("name")
        if name:
            names.append(str(name))
    return names

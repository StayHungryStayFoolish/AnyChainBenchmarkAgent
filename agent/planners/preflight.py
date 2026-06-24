"""Preflight checks for Agent benchmark plans."""

from __future__ import annotations

import os
import json
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from knowledge.entry_contract import (
    ENTRYPOINT_SCRIPTS,
    dependency_names,
    validate_mixed_weighted,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_preflight(plan: dict[str, Any]) -> dict[str, Any]:
    checks = []
    warnings = []

    chain = plan.get("chain", "")
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
    checks.append(_check("chain_template_exists", chain_template.is_file(), str(chain_template)))
    checks.append(_check("chain_template_json_valid", _json_valid(chain_template), str(chain_template)))

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

    rpc_mode = plan.get("rpc_mode", "")
    checks.append(_check("rpc_mode_valid", rpc_mode in {"single", "mixed"}, rpc_mode))
    if rpc_mode == "mixed":
        ok, detail = validate_mixed_weighted(plan.get("chain_template_requirements", {}))
        checks.append(_check("mixed_weighted_total_valid", ok, detail))

    if not plan.get("use_fake_node"):
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


def _json_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except Exception:
        return False


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
    discovery = plan.get("discovery", {})
    candidates = _candidate_names(discovery.get("disks", {}))
    materialized = plan.get("materialized_config", {})
    ledger = materialized.get("LEDGER_DEVICE", "")
    accounts = materialized.get("ACCOUNTS_DEVICE", "")
    network = materialized.get("NETWORK_INTERFACE", "")
    iface = discovery.get("network", {}).get("default_interface", "")

    if candidates:
        checks.append(_check("ledger_device_candidate_known", ledger in candidates, f"ledger={ledger}, candidates={', '.join(candidates)}"))
        if accounts:
            checks.append(_check("accounts_device_candidate_known", accounts in candidates, f"accounts={accounts}, candidates={', '.join(candidates)}"))
    else:
        warnings.append("disk inventory was unavailable; disk charts may be degraded unless the selected devices are visible to iostat")

    if iface:
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

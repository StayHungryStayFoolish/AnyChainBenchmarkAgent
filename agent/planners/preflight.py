"""Preflight checks for Agent benchmark plans."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def run_preflight(plan: dict[str, Any]) -> dict[str, Any]:
    checks = []

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

    entry = REPO_ROOT / "blockchain_node_benchmark.sh"
    checks.append(_check("benchmark_entry_exists", entry.is_file() and os.access(entry, os.X_OK), str(entry)))

    fake_node_dir = REPO_ROOT / "tools" / "fake-node"
    fake_node_ok = not plan.get("use_fake_node") or fake_node_dir.is_dir()
    checks.append(_check("fake_node_available_when_requested", fake_node_ok, str(fake_node_dir)))

    current_dir = REPO_ROOT / "current"
    current_dir.mkdir(exist_ok=True)
    checks.append(_check("output_directories_writable", os.access(current_dir, os.W_OK), str(current_dir)))

    passed = all(item["passed"] for item in checks)
    return {"passed": passed, "checks": checks}


def _check(name: str, passed: bool, detail: str = "") -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}

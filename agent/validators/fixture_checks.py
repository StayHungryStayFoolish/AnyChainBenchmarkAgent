"""Fake-node fixture coverage/authenticity checks, shared by the CLI tool-call surface."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from agent.runners.tool_result import tool_result as _tool_result

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_COVERAGE = REPO_ROOT / "tools" / "fake-node" / "check_fixture_coverage.py"
FIXTURE_AUTHENTICITY = REPO_ROOT / "tools" / "fake-node" / "validate_fixture_authenticity.py"
FIXTURES_DIR = REPO_ROOT / "tools" / "fake-node" / "fixtures"
FAKE_NODE_CONFIGS_DIR = REPO_ROOT / "tools" / "fake-node" / "configs"


def validate_plan_fake_node_fixtures(plan: dict[str, Any]) -> dict[str, Any]:
    """Validate fixtures for the effective job-local workload in ``plan``.

    The repository-wide coverage tool reads committed chain templates. Agent
    runs may use a job-local chain override, so preflight must inspect that
    override instead of assuming a successful endpoint probe created a fixture.
    """

    if not plan.get("use_fake_node"):
        return {"passed": True, "chain": "", "family": "", "methods": [], "missing": []}
    chain = str(plan.get("chain") or "").strip()
    template = plan.get("chain_config_override") or _load_chain_template(chain)
    if not isinstance(template, dict):
        template = {}
    family = str((template.get("_meta") or {}).get("adapter_family") or "").strip()
    methods = _effective_methods(template, str(plan.get("rpc_mode") or "single"))
    fixture_map = _load_fixture_map(family)
    rows: list[dict[str, Any]] = []
    for method in methods:
        fixture = fixture_map.get(method) or f"{_safe_name(method)}.json"
        path = FIXTURES_DIR / chain / fixture
        rows.append({
            "method": method,
            "fixture": str(path),
            "exists": path.is_file() and path.stat().st_size > 0,
        })
    missing = [row for row in rows if not row["exists"]]
    return {
        "passed": bool(chain and family and methods and not missing),
        "chain": chain,
        "family": family,
        "methods": methods,
        "rows": rows,
        "missing": missing,
    }


def validate_effective_fake_node_workload(
    chain: str,
    rpc_mode: str,
    methods: list[str],
    mixed_weights: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Validate a Harness-selected workload before a benchmark plan exists."""

    template = _load_chain_template(chain)
    if not template:
        return {"passed": False, "chain": chain, "family": "", "methods": methods, "rows": [], "missing": []}
    override = dict(template)
    rpc = dict(template.get("rpc_methods") or {})
    selected = [str(method).strip() for method in methods if str(method).strip()]
    if rpc_mode == "single":
        rpc["single"] = selected[0] if selected else ""
    else:
        weights = mixed_weights or {}
        rpc["mixed_weighted"] = [
            {"method": method, "weight": int(weights.get(method, 0))}
            for method in selected
        ]
        rpc["mixed"] = ",".join(selected)
    override["rpc_methods"] = rpc
    return validate_plan_fake_node_fixtures({
        "chain": chain,
        "use_fake_node": True,
        "rpc_mode": rpc_mode,
        "chain_config_override": override,
    })


def validate_fake_node_fixture_coverage(
    chains: str = "all",
    modes: str = "single,mixed",
    strict: bool = True,
) -> dict[str, Any]:
    """Validate fake-node fixture coverage through the canonical fake-node checker."""
    command = ["python3", str(FIXTURE_COVERAGE), "--chains", chains or "all", "--modes", modes or "single,mixed", "--json"]
    if strict:
        command.append("--strict")
    payload = _run_json_tool(command, allow_failure=True)
    incomplete = _fixture_payload_incomplete(payload)
    return _tool_result(
        status="blocked" if incomplete else "ok",
        data=payload,
        warnings=_fixture_warnings(payload),
        next_actions=["record missing or placeholder fixtures"] if incomplete else ["continue fake-node smoke gate"],
    )


def validate_fake_node_fixture_authenticity(
    modes: str = "single,mixed",
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Validate that fixture files have matching recorded request/response evidence."""
    command = ["python3", str(FIXTURE_AUTHENTICITY), "--modes", modes or "single,mixed", "--json"]
    if allow_incomplete:
        command.append("--allow-incomplete")
    payload = _run_json_tool(command, allow_failure=True)
    incomplete = bool(payload.get("exit_code")) or bool(payload.get("incomplete"))
    warnings = [f"{key}: {value}" for key, value in sorted((payload.get("statuses") or {}).items()) if key != "real-recorded"]
    if payload.get("exit_code"):
        warnings.append(f"fixture checker exit_code={payload['exit_code']}")
    return _tool_result(
        status="blocked" if incomplete else "ok",
        data=payload,
        warnings=warnings,
        next_actions=["record real request/response evidence"] if incomplete else ["continue fake-node smoke gate"],
    )


def _run_json_tool(command: list[str], allow_failure: bool = False) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        payload = {}
    payload.setdefault("command", " ".join(command))
    payload["exit_code"] = completed.returncode
    if completed.stderr:
        payload["stderr"] = completed.stderr.strip()
    if completed.returncode and not allow_failure:
        raise RuntimeError(f"{command[1]} failed with exit_code={completed.returncode}: {completed.stderr}")
    return payload


def _fixture_payload_incomplete(payload: dict[str, Any]) -> bool:
    if payload.get("exit_code"):
        return True
    statuses = payload.get("statuses") or {}
    return any(key != "ok" and count for key, count in statuses.items())


def _fixture_warnings(payload: dict[str, Any]) -> list[str]:
    statuses = payload.get("statuses") or {}
    warnings = [f"{key}: {value}" for key, value in sorted(statuses.items()) if key != "ok"]
    if payload.get("exit_code"):
        warnings.append(f"fixture checker exit_code={payload['exit_code']}")
    return warnings


def _load_chain_template(chain: str) -> dict[str, Any]:
    path = REPO_ROOT / "config" / "chains" / f"{chain}.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _effective_methods(template: dict[str, Any], rpc_mode: str) -> list[str]:
    configured = template.get("rpc_methods") or {}
    if rpc_mode == "single":
        method = str(configured.get("single") or "").strip()
        return [method] if method else []
    weighted = configured.get("mixed_weighted") or []
    methods = []
    for item in weighted:
        if not isinstance(item, dict):
            continue
        try:
            weight = int(item.get("weight") or 0)
        except (TypeError, ValueError):
            continue
        method = str(item.get("method") or "").strip()
        if method and weight > 0:
            methods.append(method)
    if methods:
        return list(dict.fromkeys(method for method in methods if method))
    raw = configured.get("mixed") or ""
    if isinstance(raw, list):
        return list(dict.fromkeys(str(item).strip() for item in raw if str(item).strip()))
    return list(dict.fromkeys(item.strip() for item in str(raw).split(",") if item.strip()))


def _load_fixture_map(family: str) -> dict[str, str]:
    path = FAKE_NODE_CONFIGS_DIR / f"{family}.yaml"
    if not path.is_file():
        return {}
    mapping: dict[str, str] = {}
    in_methods = False
    current = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.split("#", 1)[0].rstrip()
        if not raw.strip():
            continue
        if raw.strip() == "methods:":
            in_methods = True
            continue
        if in_methods and not raw.startswith(" "):
            break
        if not in_methods:
            continue
        method_match = re.match(r"^  ([^:\n]+):\s*$", raw)
        if method_match:
            current = method_match.group(1).strip().strip("'\"")
            continue
        fixture_match = re.match(r"^\s+fixture:\s*(.+?)\s*$", raw)
        if fixture_match and current:
            mapping[current] = fixture_match.group(1).strip().strip("'\"")
    return mapping


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.replace("/", "_")).strip("_") or "method"

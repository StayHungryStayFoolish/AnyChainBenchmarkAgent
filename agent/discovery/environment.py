"""Safe read-only environment discovery.

Discovery is intentionally conservative. It collects hints and confidence
signals but does not mutate host state or install dependencies.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable


CommandRunner = Callable[[list[str], float], tuple[int, str, str]]
REPO_ROOT = Path(__file__).resolve().parents[2]


def discover_environment(command_runner: CommandRunner | None = None) -> dict[str, Any]:
    runner = command_runner or _run_command
    host = _discover_host()
    container = _discover_container()
    kubernetes = _discover_kubernetes()
    cloud = _discover_cloud(runner, kubernetes)
    network = _discover_network(runner)
    disks = _discover_disks(runner)
    dependencies = _discover_dependencies(runner)

    deployment_type = "kubernetes" if kubernetes["detected"] else "container" if container["detected"] else "vm"
    if cloud["provider"] == "other" and deployment_type == "container":
        cloud["confidence"] = min(cloud["confidence"], 0.4)

    return {
        "source": "agent.discovery.environment",
        "mode": "read_only",
        "host": host,
        "deployment": {
            "type": deployment_type,
            "container": container,
            "kubernetes": kubernetes,
        },
        "cloud": cloud,
        "network": network,
        "disks": disks,
        "dependencies": dependencies,
        "warnings": _warnings(cloud, disks, dependencies),
    }


def _discover_host() -> dict[str, Any]:
    memory_gib = None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemTotal:"):
                parts = line.split()
                if len(parts) >= 2:
                    memory_gib = round(int(parts[1]) / 1024 / 1024, 2)
                break
    return {
        "os": platform.system().lower(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "memory_gib": memory_gib,
    }


def _discover_container() -> dict[str, Any]:
    cgroup = Path("/proc/1/cgroup")
    text = cgroup.read_text(encoding="utf-8", errors="replace") if cgroup.is_file() else ""
    detected = Path("/.dockerenv").exists() or any(token in text for token in ("docker", "kubepods", "containerd"))
    return {"detected": detected, "cgroup_hint": _shorten(text)}


def _discover_kubernetes() -> dict[str, Any]:
    token_file = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
    namespace_file = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")
    detected = bool(os.environ.get("KUBERNETES_SERVICE_HOST")) or token_file.is_file()
    namespace = ""
    if namespace_file.is_file():
        namespace = namespace_file.read_text(encoding="utf-8", errors="replace").strip()
    return {
        "detected": detected,
        "namespace": namespace,
        "service_host": os.environ.get("KUBERNETES_SERVICE_HOST", ""),
        "service_account_token": token_file.is_file(),
    }


def _discover_cloud(runner: CommandRunner, kubernetes: dict[str, Any]) -> dict[str, Any]:
    provider = "other"
    confidence = 0.2
    variant = ""

    code, stdout, _ = runner(["bash", "-lc", "source config/cloud_provider.sh >/dev/null 2>&1; printf '%s,%s,%s' \"${CLOUD_PROVIDER:-other}\" \"${CLOUD_PROVIDER_VARIANT:-}\" \"${NETWORK_INTERFACE:-}\""], 5)
    if code == 0 and stdout.strip():
        parts = stdout.strip().split(",")
        provider = parts[0] or "other"
        variant = parts[1] if len(parts) > 1 else ""
        confidence = 0.8 if provider in {"gcp", "aws"} else 0.3

    if kubernetes["detected"] and provider == "gcp":
        platform_name = "gke"
    elif kubernetes["detected"] and provider == "aws":
        platform_name = "eks"
    elif provider == "gcp":
        platform_name = "gce"
    elif provider == "aws":
        platform_name = "ec2"
    else:
        platform_name = "unknown"

    return {
        "provider": provider,
        "platform": platform_name,
        "variant": variant,
        "confidence": confidence,
    }


def _discover_network(runner: CommandRunner) -> dict[str, Any]:
    iface = ""
    driver = ""
    code, stdout, _ = runner(["bash", "-lc", "ip route 2>/dev/null | awk '/^default/ {print $5; exit}'"], 2)
    if code == 0:
        iface = stdout.strip()
    if iface and shutil.which("ethtool"):
        code, stdout, _ = runner(["ethtool", "-i", iface], 2)
        if code == 0:
            for line in stdout.splitlines():
                if line.startswith("driver:"):
                    driver = line.split(":", 1)[1].strip()
                    break
    return {"default_interface": iface, "driver": driver}


def _discover_disks(runner: CommandRunner) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    code, stdout, _ = runner(["lsblk", "-J", "-o", "NAME,TYPE,SIZE,MOUNTPOINT,FSTYPE,LABEL"], 3)
    if code == 0 and stdout.strip():
        try:
            payload = json.loads(stdout)
            for item in payload.get("blockdevices", []):
                _collect_disk_candidates(item, candidates)
        except json.JSONDecodeError:
            pass

    non_root = [item for item in candidates if item.get("mountpoint") not in {"", "/", "/boot", "/boot/efi"}]
    proposed_ledger = ""
    proposed_accounts = ""
    confidence = 0.0
    ambiguous = []

    scored = sorted((_score_disk(item), item) for item in non_root)
    if len(scored) == 1:
        proposed_ledger = scored[0][1]["name"]
        confidence = 0.65
    elif scored:
        best_score, best = scored[-1]
        second_score = scored[-2][0] if len(scored) > 1 else -1
        if best_score >= 2 and best_score > second_score:
            proposed_ledger = best["name"]
            confidence = min(0.9, 0.55 + best_score / 10)
        else:
            ambiguous = [item["name"] for _, item in scored]
            confidence = 0.3
        accounts = [item for _, item in scored if "account" in _disk_text(item)]
        if accounts:
            proposed_accounts = accounts[-1]["name"]

    return {
        "candidates": candidates,
        "proposed_ledger_device": proposed_ledger,
        "proposed_accounts_device": proposed_accounts,
        "confidence": confidence,
        "ambiguous_candidates": ambiguous,
    }


def _collect_disk_candidates(item: dict[str, Any], candidates: list[dict[str, Any]]) -> None:
    if item.get("type") in {"disk", "part", "lvm"}:
        candidates.append({
            "name": item.get("name", ""),
            "type": item.get("type", ""),
            "size": item.get("size", ""),
            "mountpoint": item.get("mountpoint") or "",
            "fstype": item.get("fstype") or "",
            "label": item.get("label") or "",
        })
    for child in item.get("children", []) or []:
        _collect_disk_candidates(child, candidates)


def _score_disk(item: dict[str, Any]) -> int:
    text = _disk_text(item)
    score = 0
    for token in ("ledger", "data", "datadir", "chaindata", "ancient", "blocks", "state", "validator", "database", "rocksdb", "leveldb"):
        if token in text:
            score += 2
    for token in ("snapshot", "snapshots", "db"):
        if token in text:
            score += 1
    if "account" in text:
        score += 2
    return score


def _disk_text(item: dict[str, Any]) -> str:
    return " ".join(str(item.get(key, "")) for key in ("name", "mountpoint", "fstype", "label")).lower()


def _discover_dependencies(runner: CommandRunner) -> dict[str, Any]:
    tools = {}
    for name in ("bash", "python3", "jq", "curl", "vegeta", "docker", "kubectl", "go", "iostat", "ethtool", "ip", "lsblk"):
        code, stdout, _ = runner(["bash", "-lc", f"command -v {name}"], 1)
        tools[name] = {"available": code == 0, "path": stdout.strip() if code == 0 else ""}
    missing_required = [name for name in ("bash", "python3", "jq", "curl", "vegeta") if not tools[name]["available"]]
    return {
        "mode": "audit",
        "tools": tools,
        "missing_required": missing_required,
        "missing_optional": [name for name, meta in tools.items() if not meta["available"] and name not in missing_required],
    }


def _warnings(cloud: dict[str, Any], disks: dict[str, Any], dependencies: dict[str, Any]) -> list[str]:
    warnings = []
    if cloud["provider"] == "other":
        warnings.append("Cloud provider could not be detected with high confidence.")
    if disks.get("ambiguous_candidates"):
        warnings.append("Multiple plausible data disks were found; confirm ledger/accounts devices.")
    if dependencies.get("missing_required"):
        warnings.append("Required dependencies are missing; dependency_mode remains audit-only.")
    return warnings


def _run_command(command: list[str], timeout: float) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return completed.returncode, completed.stdout, completed.stderr
    except Exception as exc:  # pragma: no cover - defensive discovery guard
        return 1, "", str(exc)


def _shorten(text: str, limit: int = 500) -> str:
    text = text.strip()
    return text[:limit] + ("..." if len(text) > limit else "")

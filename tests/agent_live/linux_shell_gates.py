"""Versioned Linux shell-gate inventory and executor."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping


SHELL_GATE_SCHEMA_VERSION = 1
SHELL_GATE_CLASSIFICATIONS = frozenset({
    "required",
    "equivalently_covered",
    "diagnostic",
    "retired",
})
DEFAULT_MANIFEST = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "linux_shell_gate_manifest.json"
)


def load_linux_shell_gate_manifest(
    repo_root: Path,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    """Load the manifest and prove complete tracked recursive classification."""

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != SHELL_GATE_SCHEMA_VERSION
        or payload.get("scope") != "tracked_tests_recursive_shell"
        or not isinstance(payload.get("entries"), list)
    ):
        raise ValueError("Linux shell-gate manifest schema is invalid")
    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in payload["entries"]:
        if not isinstance(raw, Mapping):
            raise ValueError("Linux shell-gate entry must be an object")
        allowed = {"path", "classification", "covered_by", "reason"}
        if set(raw) - allowed:
            raise ValueError("Linux shell-gate entry has unknown fields")
        path = str(raw.get("path") or "").strip()
        classification = str(raw.get("classification") or "").strip()
        if (
            not path.startswith("tests/")
            or not path.endswith(".sh")
            or classification not in SHELL_GATE_CLASSIFICATIONS
            or path in seen
        ):
            raise ValueError(f"Linux shell-gate entry is invalid: {path}")
        if classification != "required" and not str(
            raw.get("reason") or ""
        ).strip():
            raise ValueError(f"non-required shell gate lacks reason: {path}")
        if classification == "equivalently_covered" and not str(
            raw.get("covered_by") or ""
        ).strip():
            raise ValueError(f"equivalent shell gate lacks coverage owner: {path}")
        seen.add(path)
        entries.append({
            key: str(raw[key])
            for key in raw
            if key in allowed
        })
    by_path = {entry["path"]: entry for entry in entries}
    for entry in entries:
        if entry["classification"] != "equivalently_covered":
            continue
        covered_by = str(entry["covered_by"])
        owner = by_path.get(covered_by)
        if owner is None or owner["classification"] != "required":
            raise ValueError(
                f"equivalent shell gate has invalid coverage owner: "
                f"{entry['path']} -> {covered_by}"
            )
    completed = subprocess.run(
        (
            "git",
            "-C",
            str(repo_root),
            "ls-files",
            "-z",
            "tests/*.sh",
            "tests/**/*.sh",
        ),
        capture_output=True,
        check=True,
    )
    observed = {
        item.decode("utf-8")
        for item in completed.stdout.split(b"\0")
        if item
    }
    if observed != seen:
        missing = sorted(observed - seen)
        stale = sorted(seen - observed)
        raise ValueError(
            "Linux shell-gate classification drift: "
            f"unclassified={missing}, missing={stale}"
        )
    return {
        "schema_version": SHELL_GATE_SCHEMA_VERSION,
        "manifest": str(manifest_path.relative_to(repo_root)),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "denominator": len(entries),
        "entries": entries,
    }


def execute_required_linux_shell_gates(
    repo_root: Path,
    manifest: Mapping[str, Any],
    *,
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    """Execute every required entry and retain bounded auditable results."""

    results = []
    for entry in manifest.get("entries") or ():
        if entry.get("classification") != "required":
            continue
        path = str(entry["path"])
        try:
            completed = subprocess.run(
                ("bash", path),
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
            result = {
                "path": path,
                "exit_code": completed.returncode,
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
                "passed": completed.returncode == 0,
            }
        except subprocess.TimeoutExpired as exc:
            result = {
                "path": path,
                "exit_code": None,
                "stdout": str(exc.stdout or "")[-4000:],
                "stderr": str(exc.stderr or "")[-4000:],
                "passed": False,
                "timed_out": True,
            }
        results.append(result)
    required = len(results)
    passed = sum(result["passed"] for result in results)
    return {
        "manifest": str(manifest.get("manifest") or ""),
        "manifest_sha256": str(manifest.get("manifest_sha256") or ""),
        "classified_denominator": int(manifest.get("denominator") or 0),
        "required_denominator": required,
        "passed": passed,
        "failed": required - passed,
        "results": results,
        "status": "passed" if required > 0 and passed == required else "failed",
    }

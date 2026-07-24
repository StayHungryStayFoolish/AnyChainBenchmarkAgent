"""Strict projection proof for a job's materialized runtime environment."""

from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path
from typing import Any, Mapping

from agent.runners.materialize import build_runtime_env


RUNTIME_ENV_PROJECTION_SCHEMA_VERSION = 1


def validate_runtime_env_projection(job: Mapping[str, Any]) -> dict[str, Any]:
    job_id = str(job.get("job_id") or "")
    run_dir = Path(str(job.get("run_dir") or "")).resolve(strict=True)
    plan_file = Path(str(job.get("plan_file") or ""))
    runtime_env_file = Path(str(job.get("runtime_env_file") or ""))
    if (
        not job_id
        or run_dir.name != job_id
        or plan_file != run_dir / "plan.json"
        or runtime_env_file != run_dir / "runtime.env"
        or plan_file.is_symlink()
        or runtime_env_file.is_symlink()
    ):
        raise ValueError("job runtime projection paths violate the run_dir contract")
    try:
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("job plan is not valid JSON") from exc
    if not isinstance(plan, dict):
        raise ValueError("job plan must be a JSON object")

    observed = _parse_generated_runtime_env(runtime_env_file)
    expected = build_runtime_env(plan)
    override = plan.get("chain_config_override")
    if isinstance(override, Mapping) and override:
        override_file = run_dir / "chain_template.override.json"
        if override_file.is_symlink() or not override_file.is_file():
            raise ValueError("job chain override file is missing or symlinked")
        expected["CHAIN_CONFIG_OVERRIDE_FILE"] = str(override_file)
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        changed = sorted(
            key
            for key in set(expected) & set(observed)
            if expected[key] != observed[key]
        )
        raise ValueError(
            "runtime.env disagrees with the materialized job plan: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )

    receipt = {
        "schema_version": RUNTIME_ENV_PROJECTION_SCHEMA_VERSION,
        "job_id": job_id,
        "plan_file": str(plan_file),
        "plan_sha256": _sha256(plan_file),
        "runtime_env_file": str(runtime_env_file),
        "runtime_env_sha256": _sha256(runtime_env_file),
        "environment_keys": sorted(observed),
        "environment_sha256": _content_hash(observed),
        "execution_profile": _execution_profile(plan, observed),
    }
    return {
        **receipt,
        "projection_receipt_id": _content_hash(receipt),
    }


def _parse_generated_runtime_env(path: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith("export ") or "=" not in line:
            raise ValueError(
                f"runtime.env line {line_number} is not a generated export"
            )
        key, raw_value = line[len("export ") :].split("=", 1)
        key = key.strip()
        if not key or key in observed:
            raise ValueError(
                f"runtime.env contains a duplicate or empty key at line {line_number}"
            )
        try:
            values = shlex.split(raw_value, posix=True)
        except ValueError as exc:
            raise ValueError(
                f"runtime.env contains invalid shell quoting at line {line_number}"
            ) from exc
        if len(values) != 1:
            raise ValueError(
                f"runtime.env line {line_number} must contain one scalar value"
            )
        observed[key] = values[0]
    return observed


def _execution_profile(
    plan: Mapping[str, Any],
    environment: Mapping[str, str],
) -> dict[str, Any]:
    command = [
        str(item)
        for item in ((plan.get("execution") or {}).get("command") or ())
    ]
    mode = str(plan.get("benchmark_mode") or "").strip().lower()
    if not mode:
        mode = next(
            (
                candidate
                for candidate in ("quick", "standard", "intensive")
                if f"--{candidate}" in command
            ),
            "",
        )
    qps: dict[str, int] = {}
    if mode in {"quick", "standard", "intensive"}:
        prefix = mode.upper()
        fields = {
            "initial": f"{prefix}_INITIAL_QPS",
            "max": f"{prefix}_MAX_QPS",
            "step": f"{prefix}_QPS_STEP",
            "duration_seconds": f"{prefix}_DURATION",
        }
        for name, key in fields.items():
            raw = str(environment.get(key) or "").strip()
            if raw:
                try:
                    qps[name] = int(raw)
                except ValueError as exc:
                    raise ValueError(f"runtime QPS value is not an integer: {key}") from exc
        if set(qps) != set(fields):
            raise ValueError(f"runtime QPS profile is incomplete for mode {mode}")
        if (
            qps["initial"] <= 0
            or qps["max"] < qps["initial"]
            or qps["step"] <= 0
            or qps["duration_seconds"] <= 0
        ):
            raise ValueError(f"runtime QPS profile is invalid for mode {mode}")
    output_root = str(
        environment.get("BLOCKCHAIN_BENCHMARK_DATA_DIR") or ""
    )
    return {
        "workflow_type": str(plan.get("workflow_type") or plan.get("run_mode") or ""),
        "benchmark_mode": mode,
        "command_sha256": _content_hash(command),
        "qps": qps,
        "minimum_request_seconds": (
            qps.get("max", 0) * qps.get("duration_seconds", 0)
        ),
        "output_root_sha256": _content_hash(output_root) if output_root else "",
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

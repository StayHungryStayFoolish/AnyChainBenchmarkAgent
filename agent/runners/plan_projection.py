"""Typed projection from approved benchmark intent to one job-local plan."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


PROJECTION_SCHEMA_VERSION = 1
JOB_LOCAL_ENV_FIELDS = frozenset({
    "BLOCKCHAIN_BENCHMARK_DATA_DIR",
    "MEMORY_SHARE_DIR",
})
JOB_LOCAL_ARTIFACT_FIELDS = frozenset({
    "execution_output_root",
    "execution_memory_dir",
    "fake_node_smoke_output_root",
    "fake_node_smoke_memory_dir",
    "real_node_smoke_output_root",
    "real_node_smoke_memory_dir",
    "source_plan_file",
})


def validate_execution_plan_projection(
    approved_plan: Mapping[str, Any],
    materialized_plan: Mapping[str, Any],
    *,
    approved_plan_file: str | Path,
    operation: str,
    scenario_id: str,
    job_id: str = "",
    job_plan_file: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and describe the only admitted execution-plan projection."""

    if job_id and job_plan_file is not None:
        return _validate_persisted_job_projection(
            approved_plan,
            materialized_plan,
            approved_plan_file=approved_plan_file,
            operation=operation,
            scenario_id=scenario_id,
            job_id=job_id,
            job_plan_file=job_plan_file,
        )

    approved = deepcopy(dict(approved_plan))
    materialized = deepcopy(dict(materialized_plan))
    approved_execution = dict(approved.get("execution") or {})
    materialized_execution = dict(materialized.get("execution") or {})
    idempotency_key = str(
        materialized_execution.pop("idempotency_key", "") or ""
    ).strip()
    approved_execution.pop("idempotency_key", None)
    for execution in (approved_execution, materialized_execution):
        environment = dict(execution.get("environment") or {})
        for field in JOB_LOCAL_ENV_FIELDS:
            environment.pop(field, None)
        execution["environment"] = environment
    approved["execution"] = approved_execution
    materialized["execution"] = materialized_execution
    for plan in (approved, materialized):
        materialized_config = dict(plan.get("materialized_config") or {})
        for field in JOB_LOCAL_ENV_FIELDS:
            materialized_config.pop(field, None)
        if materialized_config:
            plan["materialized_config"] = materialized_config
        else:
            plan.pop("materialized_config", None)
        artifacts = dict(plan.get("artifacts") or {})
        for field in JOB_LOCAL_ARTIFACT_FIELDS:
            artifacts.pop(field, None)
        if artifacts:
            plan["artifacts"] = artifacts
        else:
            plan.pop("artifacts", None)
    provenance = materialized.pop("execution_provenance", None)
    approved.pop("execution_provenance", None)
    if materialized != approved:
        raise ValueError(
            "materialized execution plan changed approved business semantics"
        )
    if not idempotency_key or not isinstance(provenance, Mapping):
        raise ValueError(
            "materialized execution plan lacks typed execution provenance"
        )
    source = Path(approved_plan_file).resolve()
    expected = {
        "approved_plan_file": str(source),
        "approved_plan_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "operation": str(operation),
        "scenario_id": str(scenario_id),
        "runtime_overrides": list(provenance.get("runtime_overrides") or ()),
    }
    for field, value in expected.items():
        if provenance.get(field) != value:
            raise ValueError(
                f"materialized execution provenance mismatch: {field}"
            )
    allowed_fields = set(expected)
    if job_id or job_plan_file is not None:
        expected.update({
            "job_id": str(job_id),
            "job_plan_file": str(Path(job_plan_file).resolve()),
        })
        allowed_fields.update({"job_id", "job_plan_file"})
        for field in ("job_id", "job_plan_file"):
            if provenance.get(field) != expected[field]:
                raise ValueError(
                    f"materialized execution provenance mismatch: {field}"
                )
    if set(provenance) != allowed_fields:
        raise ValueError(
            "materialized execution provenance contains unsupported fields"
        )
    receipt = {
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "approved_plan_sha256": expected["approved_plan_sha256"],
        "materialized_plan_sha256": _content_hash(materialized_plan),
        "operation": str(operation),
        "scenario_id": str(scenario_id),
        "idempotency_key_sha256": _content_hash(idempotency_key),
        "runtime_overrides": list(expected["runtime_overrides"]),
        "job_id": str(job_id),
        "job_plan_file": (
            str(Path(job_plan_file).resolve())
            if job_plan_file is not None
            else ""
        ),
    }
    return {**receipt, "projection_receipt_id": _content_hash(receipt)}


def _validate_persisted_job_projection(
    approved_plan: Mapping[str, Any],
    materialized_plan: Mapping[str, Any],
    *,
    approved_plan_file: str | Path,
    operation: str,
    scenario_id: str,
    job_id: str,
    job_plan_file: str | Path,
) -> dict[str, Any]:
    from agent.runners.benchmark_pipeline import (
        _fake_node_smoke_plan,
        _isolated_execution_plan,
        _real_node_smoke_plan,
        _smoke_execution_root,
    )

    source = Path(approved_plan_file).resolve()
    job_plan_path = Path(job_plan_file).resolve()
    jobs_dir = job_plan_path.parent.parent
    materialized = deepcopy(dict(materialized_plan))
    provenance = dict(materialized.get("execution_provenance") or {})
    expected_provenance = {
        "approved_plan_file": str(source),
        "approved_plan_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "operation": str(operation),
        "scenario_id": str(scenario_id),
        "runtime_overrides": list(provenance.get("runtime_overrides") or ()),
        "job_id": str(job_id),
        "job_plan_file": str(job_plan_path),
    }
    if provenance != expected_provenance:
        raise ValueError("persisted job execution provenance is not canonical")
    idempotency_key = str(
        ((materialized.get("execution") or {}).get("idempotency_key") or "")
    ).strip()
    if not idempotency_key:
        raise ValueError("persisted job plan has no idempotency key")

    base = deepcopy(dict(approved_plan))
    execution = dict(base.get("execution") or {})
    source_idempotency_key = idempotency_key
    if operation == "real_node_smoke":
        suffix = ":real-node-smoke"
        if not idempotency_key.endswith(suffix):
            raise ValueError("real-node smoke idempotency key has no runner suffix")
        source_idempotency_key = idempotency_key[: -len(suffix)]
    execution["idempotency_key"] = source_idempotency_key
    base["execution"] = execution
    base["execution_provenance"] = {
        key: value
        for key, value in expected_provenance.items()
        if key not in {"job_id", "job_plan_file"}
    }
    if operation == "fake_node_smoke":
        smoke_root = _smoke_execution_root(
            source,
            jobs_dir,
            kind="fake_node_smoke",
        )
        expected = _fake_node_smoke_plan(source, smoke_root, plan=base)
    elif operation == "real_node_smoke":
        smoke_root = _smoke_execution_root(
            source,
            jobs_dir,
            kind="real_node_smoke",
        )
        expected = _real_node_smoke_plan(source, smoke_root, plan=base)
    elif operation in {"final_benchmark", "sync_observe"}:
        expected = _isolated_execution_plan(source, jobs_dir, plan=base)
    else:
        raise ValueError(f"unsupported persisted job projection operation: {operation}")
    expected["execution_provenance"] = expected_provenance
    if materialized != expected:
        raise ValueError(
            "persisted job plan is not the canonical operation projection"
        )

    receipt = {
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "approved_plan_sha256": expected_provenance["approved_plan_sha256"],
        "materialized_plan_sha256": _content_hash(materialized_plan),
        "operation": str(operation),
        "scenario_id": str(scenario_id),
        "idempotency_key_sha256": _content_hash(idempotency_key),
        "runtime_overrides": list(expected_provenance["runtime_overrides"]),
        "job_id": str(job_id),
        "job_plan_file": str(job_plan_path),
    }
    return {**receipt, "projection_receipt_id": _content_hash(receipt)}


def _content_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()

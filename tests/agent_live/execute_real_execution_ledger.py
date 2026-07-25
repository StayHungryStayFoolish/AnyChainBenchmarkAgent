#!/usr/bin/env python3
"""Produce revision-bound evidence for required real execution edges."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from agent.harness.runtime_identity import repository_revision
from agent.runners.artifact_ownership import (
    ARTIFACT_MANIFEST_SCHEMA_VERSION,
    derive_artifact_owner_roots,
    required_artifact_owner,
    validate_owned_artifact_path,
)
from agent.runners.application_service import (
    BenchmarkExecutionService,
    ExecutionOperation,
    ExecutionRequest,
)
from agent.runners.job_manager import get_job
from agent.runners.runtime_env_projection import validate_runtime_env_projection
from agent.runners.execution_scenarios import (
    EXECUTION_SCENARIOS,
    RPC_BENCHMARK_WORKFLOW,
    SYNC_OBSERVE_WORKFLOW,
    ExecutionScenarioSpec,
    scenario_by_id,
    workflow_type_from_plan,
)
from agent.utils.redaction import redact
from tests.agent_live.coverage_evidence import (
    G5_SCENARIO_ADMISSION,
    G5_RUNTIME_CONTRACT,
    build_real_execution_evidence_artifact,
    g5_scenario_admission,
    validate_real_execution_ledger_artifacts,
    validate_real_execution_predecessor,
    validate_real_execution_evidence_artifact,
    write_evidence_artifact,
)
from tests.agent_live.generate_harness_coverage_ledger import build_ledger
from tests.agent_live.g5_collection_manifest import (
    create_g5_attempt,
    publish_g5_collection,
)
from tests.agent_live.real_execution_host_attestation import (
    validate_host_attestation_file,
)
from tests.agent_live.export_approved_plan import validate_approved_plan_artifact
from tests.agent_live.real_execution_host_supervisor import (
    ATTESTATION_ENV,
    TARGET_SERVICE,
)

TERMINAL_JOB_STATUSES = frozenset({"completed", "failed", "partial", "cancelled"})
OPERATION_BY_VALUE = {operation.value: operation for operation in ExecutionOperation}
G5_FAILURE_ARTIFACT_SCHEMA_VERSION = 4
G5_BOOTSTRAP_FAILURE_SCHEMA_VERSION = 1
G5_FAILURE_STAGES = frozenset({
    "worker_admission",
    "plan_admission",
    "revision_guard",
    "endpoint_probe",
    "runtime_attestation",
    "predecessor_admission",
    "submission",
    "job_wait",
    "terminal_job",
    "artifact_collection",
    "evidence_validation",
    "ledger_validation",
})


class G5ExecutionStageError(RuntimeError):
    def __init__(
        self,
        scenario_id: str,
        stage: str,
        cause: Exception | str,
        *,
        evidence_path: Path | None = None,
    ) -> None:
        self.scenario_id = str(scenario_id)
        self.stage = str(stage)
        self.cause = cause
        self.evidence_path = evidence_path
        super().__init__(f"{self.scenario_id or '<global>'}:{self.stage}: {cause}")


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _parse_timestamp(value: Any, *, label: str) -> datetime:
    raw = str(value or "").strip()
    if not raw:
        raise RuntimeError(f"{label} timestamp is missing")
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError(f"{label} timestamp is invalid: {raw}") from exc
    if parsed.tzinfo is None:
        raise RuntimeError(f"{label} timestamp has no timezone: {raw}")
    return parsed.astimezone(timezone.utc)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_host_attestation(
    path: str | Path,
    *,
    revision: Mapping[str, str],
    worker_argv: Sequence[str],
) -> dict[str, Any]:
    try:
        return validate_host_attestation_file(
            path,
            repo_root=REPO_ROOT,
            revision=revision,
            worker_argv=worker_argv,
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(str(exc)) from exc


def _exact_descendant(path: str | Path, *, run_dir: Path) -> Path:
    raw = Path(path)
    root = run_dir.resolve(strict=True)
    artifact = raw.resolve(strict=True)
    try:
        artifact.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"execution artifact escapes fresh run_dir: {artifact}") from exc
    current = raw if raw.is_absolute() else Path.cwd() / raw
    while current != root:
        if current.is_symlink():
            raise RuntimeError(f"execution artifact path contains a symlink: {current}")
        parent = current.parent
        if parent == current:
            raise RuntimeError(f"execution artifact is not rooted in fresh run_dir: {artifact}")
        current = parent
    if artifact == root:
        raise RuntimeError("execution artifact cannot be the run_dir itself")
    return artifact


def _hashed_file(path: str | Path, *, job_id: str, run_dir: Path) -> dict[str, Any]:
    artifact = _exact_descendant(path, run_dir=run_dir)
    if not artifact.is_file():
        raise RuntimeError(f"required execution artifact does not exist: {artifact}")
    return {
        "job_id": job_id,
        "path": str(artifact),
        "sha256": _sha256(artifact),
        "size_bytes": artifact.stat().st_size,
    }


def _artifact_owner_roots(job: Mapping[str, Any]) -> dict[str, Path]:
    try:
        return derive_artifact_owner_roots(job)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"invalid execution artifact owner roots: {exc}") from exc


def _owned_hashed_file(
    path: str | Path,
    *,
    job_id: str,
    expected_owner: str,
    owner_roots: Mapping[str, Path],
) -> dict[str, Any]:
    try:
        artifact = validate_owned_artifact_path(
            path,
            expected_owner=expected_owner,
            owner_roots=owner_roots,
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"invalid owned execution artifact: {exc}") from exc
    root = owner_roots[expected_owner].resolve()
    lexical = Path(path)
    if not lexical.is_absolute():
        lexical = Path.cwd() / lexical
    return {
        "job_id": job_id,
        "owner": expected_owner,
        "owner_root": str(root),
        "relative_path": str(lexical.relative_to(root)),
        "path": str(lexical),
        "resolved_path": str(artifact),
        "link_target": os.readlink(lexical) if lexical.is_symlink() else "",
        "sha256": _sha256(artifact),
        "size_bytes": artifact.stat().st_size,
    }


def _read_plan(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"approved plan is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"approved plan must be a JSON object: {path}")
    return payload


def _validate_source_plan(
    plan_file: Path,
    *,
    workflow_type: str,
    use_fake_node: bool,
) -> None:
    plan = _read_plan(plan_file)
    observed_workflow = workflow_type_from_plan(plan)
    if observed_workflow != workflow_type:
        raise RuntimeError(
            f"approved plan has workflow {observed_workflow}; expected {workflow_type}: {plan_file}"
        )
    observed_fake_node = plan.get("use_fake_node")
    if observed_fake_node is not use_fake_node:
        expected = "true" if use_fake_node else "false"
        raise RuntimeError(
            f"approved plan must declare use_fake_node={expected}: {plan_file}"
        )


def _validate_approved_plan_binding(
    plan_file: Path,
    *,
    scenario: ExecutionScenarioSpec,
) -> dict[str, Any]:
    plan = _read_plan(plan_file)
    admission = g5_scenario_admission(scenario.scenario_id)
    if not admission.endpoint_env_var:
        return plan
    chain = str(plan.get("chain") or "").strip()
    execution_env = dict((plan.get("execution") or {}).get("environment") or {})
    if not chain:
        raise RuntimeError(f"approved plan has no chain identity: {plan_file}")
    endpoint = str(execution_env.get(admission.endpoint_env_var) or "").strip()
    if not endpoint:
        raise RuntimeError(
            f"approved plan has no {admission.endpoint_env_var}: {plan_file}"
        )
    if endpoint != G5_RUNTIME_CONTRACT.rpc_url:
        raise RuntimeError(
            f"approved plan endpoint is outside the frozen G5 runtime contract: {plan_file}"
        )
    if admission.endpoint_env_var == "SYNC_OBSERVE_RPC_URL":
        metrics_url = str(
            execution_env.get("NODE_PROMETHEUS_METRICS_URL") or ""
        ).strip()
        if not metrics_url:
            raise RuntimeError(f"approved sync-observe plan has no metrics endpoint: {plan_file}")
        if metrics_url != G5_RUNTIME_CONTRACT.metrics_url:
            raise RuntimeError(
                "approved sync-observe plan metrics endpoint is outside "
                f"the frozen G5 runtime contract: {plan_file}"
            )
    return plan


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fchmod(handle.fileno(), 0o400)
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(path)


def _stage_call(
    scenario_id: str,
    stage: str,
    operation: Any,
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    try:
        return operation(*args, **kwargs)
    except G5ExecutionStageError:
        raise
    except Exception as exc:
        raise G5ExecutionStageError(scenario_id, stage, exc) from exc


def _write_g5_failure_artifact(
    *,
    evidence_dir: Path,
    revision: Mapping[str, str],
    scenario_id: str,
    stage: str,
    error: Exception | str,
    plan_files: Sequence[Path],
    approval_files: Sequence[Path],
    host_attestation: Mapping[str, Any],
    attempt_id: str = "00000000000000000000000000000000",
) -> Path:
    plans = [_failure_input_binding(path) for path in plan_files]
    approvals = [_failure_input_binding(path) for path in approval_files]
    retained_by_scenario: dict[str, dict[str, Any]] = {}
    ledger: Mapping[str, Any] | None = None
    if evidence_dir.is_dir():
        for path in sorted(evidence_dir.glob("*.json")):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                candidate = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            candidate_scenario_id = str(
                candidate.get("scenario_id") or ""
            ) if isinstance(candidate, Mapping) else ""
            try:
                if ledger is None:
                    ledger = build_ledger(revision=revision)
                scenario = scenario_by_id(candidate_scenario_id)
                edge = _edge_by_action(ledger, scenario.action_type)
                valid, _reason = validate_real_execution_evidence_artifact(
                    candidate,
                    edge=edge,
                    revision=revision,
                    allow_observed_failure=True,
                )
            except (KeyError, OSError, RuntimeError, TypeError, ValueError):
                valid = False
            if valid:
                retained_by_scenario[candidate_scenario_id] = {
                    "path": str(path.resolve()),
                    "sha256": _sha256(path),
                    "evidence_id": str(candidate.get("evidence_id") or ""),
                    "scenario_id": candidate_scenario_id,
                }
    expected_predecessors: tuple[str, ...] = ()
    if scenario_id:
        admission = g5_scenario_admission(scenario_id)
        expected_predecessors = tuple(
            candidate_id
            for candidate_id, candidate_admission in sorted(
                (
                    (item.scenario_id, g5_scenario_admission(item.scenario_id))
                    for item in EXECUTION_SCENARIOS
                    if item.real_evidence_required
                ),
                key=lambda item: item[1].sequence_index,
            )
            if candidate_admission.sequence_index < admission.sequence_index
        )
    if set(retained_by_scenario) != set(expected_predecessors):
        raise RuntimeError(
            "G5 retained evidence is not the exact legal predecessor prefix"
        )
    retained_evidence = [
        retained_by_scenario[predecessor]
        for predecessor in expected_predecessors
    ]
    host_path = str(host_attestation.get("attestation_file") or "").strip()
    payload: dict[str, Any] = {
        "artifact_type": "g5_real_execution_failure",
        "schema_version": G5_FAILURE_ARTIFACT_SCHEMA_VERSION,
        "repository_revision": dict(revision),
        "attempt_id": str(attempt_id),
        "scenario_id": str(scenario_id),
        "stage": str(stage),
        "error_type": type(error).__name__,
        "error": str(redact(str(error))),
        "approved_plans": plans,
        "approved_plan_provenance": approvals,
        "retained_scenario_evidence": retained_evidence,
        "host_attestation": (
            _failure_input_binding(Path(host_path))
            if host_path
            else {
                "path": "",
                "status": "missing",
                "sha256": "",
                "size_bytes": 0,
                "error": f"{ATTESTATION_ENV} is not set",
            }
        ),
        "observed_at": _utc_timestamp(),
    }
    payload["failure_id"] = _canonical_hash(payload)
    serialized = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(serialized).hexdigest()
    path = evidence_dir / f"g5-real-execution-failure-{digest}.json"
    _write_immutable_json(path, payload)
    path.chmod(0o400)
    valid, reason = validate_g5_failure_artifact(path, revision=revision)
    if not valid:
        raise RuntimeError(f"persisted G5 failure evidence is invalid: {reason}")
    return path


def _write_g5_bootstrap_failure_artifact(
    *,
    evidence_dir: Path,
    stage: str,
    error: Exception | str,
    plan_files: Sequence[Path],
    approval_files: Sequence[Path],
    host_attestation_file: str,
) -> Path:
    """Persist a non-qualifying failure when revision identity is unavailable."""

    host_path = str(host_attestation_file or "").strip()
    payload: dict[str, Any] = {
        "artifact_type": "g5_worker_bootstrap_failure",
        "schema_version": G5_BOOTSTRAP_FAILURE_SCHEMA_VERSION,
        "qualifying_evidence": False,
        "repository_revision_status": "unavailable",
        "stage": str(stage),
        "error_type": type(error).__name__,
        "error": str(redact(str(error))),
        "approved_plans": [
            _failure_input_binding(path) for path in plan_files
        ],
        "approved_plan_provenance": [
            _failure_input_binding(path) for path in approval_files
        ],
        "host_attestation": (
            _failure_input_binding(Path(host_path))
            if host_path
            else {
                "path": "",
                "status": "missing",
                "sha256": "",
                "size_bytes": 0,
                "error": f"{ATTESTATION_ENV} is not set",
            }
        ),
        "observed_at": _utc_timestamp(),
    }
    payload["failure_id"] = _canonical_hash(payload)
    serialized = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(serialized).hexdigest()
    path = evidence_dir / f"g5-worker-bootstrap-failure-{digest}.json"
    _write_immutable_json(path, payload)
    path.chmod(0o400)
    if _sha256(path) != digest:
        raise RuntimeError("persisted G5 bootstrap failure identity mismatch")
    return path


def _failure_input_binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=False)
    record: dict[str, Any] = {
        "path": str(resolved),
        "status": "missing",
        "sha256": "",
        "size_bytes": 0,
        "error": "",
    }
    try:
        if path.is_symlink():
            record.update(
                status="unreadable",
                error="input path is a symlink",
            )
        elif not resolved.exists():
            record["error"] = "input path does not exist"
        elif not resolved.is_file():
            record.update(
                status="unreadable",
                error="input path is not a regular file",
            )
        else:
            record.update(
                status="present",
                sha256=_sha256(resolved),
                size_bytes=resolved.stat().st_size,
            )
    except OSError as exc:
        record.update(
            status="unreadable",
            error=str(redact(str(exc))),
        )
    return record


def _validate_failure_input_binding(
    item: Mapping[str, Any],
    *,
    label: str,
) -> tuple[bool, str]:
    if set(item) != {
        "path",
        "status",
        "sha256",
        "size_bytes",
        "error",
    }:
        return False, f"G5 failure evidence {label} binding shape is invalid"
    path = Path(str(item.get("path") or ""))
    status = str(item.get("status") or "")
    if status not in {"present", "missing", "unreadable"}:
        return False, f"G5 failure evidence {label} status is invalid"
    if status == "present":
        if (
            path.is_symlink()
            or not path.is_file()
            or _sha256(path) != str(item.get("sha256") or "")
            or path.stat().st_size != item.get("size_bytes")
            or item.get("error")
        ):
            return False, f"G5 failure evidence {label} binding is invalid"
    elif (
        item.get("sha256")
        or item.get("size_bytes") != 0
        or not str(item.get("error") or "")
    ):
        return False, f"G5 failure evidence {label} absence is invalid"
    return True, ""


def validate_g5_failure_artifact(
    path: str | Path,
    *,
    revision: Mapping[str, str],
) -> tuple[bool, str]:
    raw_path = Path(path)
    if raw_path.is_symlink() or not raw_path.is_file():
        return False, "G5 failure evidence is missing or symlinked"
    try:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"G5 failure evidence is invalid JSON: {exc}"
    if (
        not isinstance(payload, Mapping)
        or payload.get("artifact_type") != "g5_real_execution_failure"
        or payload.get("schema_version") != G5_FAILURE_ARTIFACT_SCHEMA_VERSION
    ):
        return False, "G5 failure evidence schema is invalid"
    if dict(payload.get("repository_revision") or {}) != dict(revision):
        return False, "G5 failure evidence revision mismatch"
    attempt_id = str(payload.get("attempt_id") or "")
    if (
        len(attempt_id) != 32
        or any(character not in "0123456789abcdef" for character in attempt_id)
    ):
        return False, "G5 failure evidence attempt identity is invalid"
    scenario_id = str(payload.get("scenario_id") or "")
    if scenario_id:
        try:
            failed_admission = g5_scenario_admission(scenario_id)
        except ValueError:
            return False, "G5 failure evidence scenario is invalid"
    elif str(payload.get("stage") or "") != "worker_admission":
        return False, "G5 failure evidence global scenario is invalid"
    if str(payload.get("stage") or "") not in G5_FAILURE_STAGES:
        return False, "G5 failure evidence stage is invalid"
    if not str(payload.get("error") or ""):
        return False, "G5 failure evidence has no error"
    plans = list(payload.get("approved_plans") or ())
    if len(plans) != 3 or len({str(item.get("path") or "") for item in plans}) != 3:
        return False, "G5 failure evidence plan set is invalid"
    for item in plans:
        valid, reason = _validate_failure_input_binding(item, label="plan")
        if not valid:
            return False, reason
    approvals = list(payload.get("approved_plan_provenance") or ())
    if (
        len(approvals) != 4
        or len({str(item.get("path") or "") for item in approvals}) != 4
    ):
        return False, "G5 failure evidence approval provenance set is invalid"
    for item in approvals:
        valid, reason = _validate_failure_input_binding(
            item,
            label="approval provenance",
        )
        if not valid:
            return False, reason
    retained = list(payload.get("retained_scenario_evidence") or ())
    ledger: Mapping[str, Any] | None = None
    if retained:
        try:
            ledger = build_ledger(revision=revision)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return False, f"G5 retained evidence ledger is unavailable: {exc}"
    retained_scenarios: set[str] = set()
    for item in retained:
        if set(item) != {"path", "sha256", "evidence_id", "scenario_id"}:
            return False, "G5 retained scenario evidence shape is invalid"
        retained_path = Path(str(item.get("path") or ""))
        scenario = str(item.get("scenario_id") or "")
        retained_payload: Mapping[str, Any] = {}
        try:
            retained_payload = json.loads(retained_path.read_text(encoding="utf-8"))
            retained_spec = scenario_by_id(scenario)
            retained_edge = _edge_by_action(
                ledger or {},
                retained_spec.action_type,
            )
            retained_valid, _retained_reason = (
                validate_real_execution_evidence_artifact(
                    retained_payload,
                    edge=retained_edge,
                    revision=revision,
                    allow_observed_failure=True,
                )
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            retained_valid = False
        if (
            retained_path.is_symlink()
            or not retained_path.is_file()
            or retained_path.stat().st_mode & 0o222
            or _sha256(retained_path) != str(item.get("sha256") or "")
            or not _is_sha256(str(item.get("evidence_id") or ""))
            or not scenario
            or scenario in retained_scenarios
            or not retained_valid
            or str(retained_payload.get("evidence_id") or "")
            != str(item.get("evidence_id") or "")
        ):
            return False, "G5 retained scenario evidence binding is invalid"
        retained_scenarios.add(scenario)
    expected_predecessors: tuple[str, ...] = ()
    if scenario_id:
        expected_predecessors = tuple(
            candidate_id
            for candidate_id, candidate_admission in sorted(
                G5_SCENARIO_ADMISSION.items(),
                key=lambda item: item[1].sequence_index,
            )
            if candidate_admission.sequence_index
            < failed_admission.sequence_index
        )
    if tuple(
        str(item.get("scenario_id") or "") for item in retained
    ) != expected_predecessors:
        return False, "G5 retained evidence is not the legal predecessor prefix"
    host_binding = payload.get("host_attestation")
    if not isinstance(host_binding, Mapping):
        return False, "G5 failure evidence host binding is invalid"
    valid, reason = _validate_failure_input_binding(
        host_binding,
        label="host attestation",
    )
    if not valid:
        return False, reason
    unsigned = dict(payload)
    failure_id = str(unsigned.pop("failure_id", ""))
    if _canonical_hash(unsigned) != failure_id:
        return False, "G5 failure evidence identity mismatch"
    observed_hash = _sha256(raw_path)
    if raw_path.name != f"g5-real-execution-failure-{observed_hash}.json":
        return False, "G5 failure evidence filename is not content-addressed"
    try:
        observed_at = _parse_timestamp(payload.get("observed_at"), label="failure")
    except RuntimeError as exc:
        return False, str(exc)
    if observed_at.tzinfo is None:
        return False, "G5 failure evidence timestamp has no timezone"
    return True, ""


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value.lower()
    )


def _build_admission_envelope(
    plan_file: Path,
    plan: Mapping[str, Any],
    scenario: ExecutionScenarioSpec,
    *,
    revision: Mapping[str, str],
    jobs_dir: Path,
) -> tuple[Path, dict[str, Any], str]:
    admission = g5_scenario_admission(scenario.scenario_id)
    execution_env = dict((plan.get("execution") or {}).get("environment") or {})
    endpoint = str(execution_env.get(admission.endpoint_env_var) or "").strip()
    endpoint_contract: dict[str, Any] = {}
    container_requirements: dict[str, Any] = {}
    if admission.endpoint_env_var:
        endpoint_contract = {
            "chain": str(plan.get("chain") or ""),
            "env_var": admission.endpoint_env_var,
            "endpoint_sha256": hashlib.sha256(endpoint.encode("utf-8")).hexdigest(),
            "probe_method": "eth_chainId",
            "params_sha256": _canonical_hash([]),
            "expected_identity": G5_RUNTIME_CONTRACT.chain_id,
        }
        runtime_contract = G5_RUNTIME_CONTRACT.to_dict()
        container_requirements = {
            "compose_service": G5_RUNTIME_CONTRACT.compose_service,
            "image_digest": G5_RUNTIME_CONTRACT.image_digest,
            "image_reference": G5_RUNTIME_CONTRACT.image_reference,
            "rpc_url_sha256": hashlib.sha256(
                G5_RUNTIME_CONTRACT.rpc_url.encode("utf-8")
            ).hexdigest(),
            "metrics_env_var": "NODE_PROMETHEUS_METRICS_URL",
            "metrics_url_sha256": hashlib.sha256(
                G5_RUNTIME_CONTRACT.metrics_url.encode("utf-8")
            ).hexdigest(),
            "metrics_required": True,
        }
    else:
        runtime_contract = {}
    envelope = {
        "schema_version": 1,
        "repository_revision": dict(revision),
        "scenario_id": scenario.scenario_id,
        "sequence_index": admission.sequence_index,
        "approved_plan_file": str(plan_file.resolve()),
        "approved_plan_sha256": _sha256(plan_file),
        "endpoint_identity_contract": endpoint_contract,
        "container_metrics_requirements": container_requirements,
        "g5_runtime_contract": runtime_contract,
        "g5_runtime_contract_sha256": (
            _canonical_hash(runtime_contract) if runtime_contract else ""
        ),
        "created_at": _utc_timestamp(),
    }
    path = jobs_dir / ".g5-admission" / f"{admission.sequence_index:02d}-{scenario.scenario_id}.json"
    return path, envelope, _write_immutable_json(path, envelope)


def _probe_endpoint_identity(
    plan: Mapping[str, Any],
    scenario: ExecutionScenarioSpec,
    envelope: Mapping[str, Any],
    *,
    timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    admission = g5_scenario_admission(scenario.scenario_id)
    if not admission.endpoint_env_var:
        return {}
    identity = dict(envelope.get("endpoint_identity_contract") or {})
    execution_env = dict((plan.get("execution") or {}).get("environment") or {})
    endpoint = str(execution_env.get(admission.endpoint_env_var) or "").strip()
    method = str(identity.get("probe_method") or "").strip()
    expected = str(identity.get("expected_identity") or "").strip()
    if not endpoint:
        raise RuntimeError(
            f"approved plan has no {admission.endpoint_env_var} endpoint to probe"
        )
    request_id = 1
    params: list[Any] = []
    request_body = json.dumps({
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params,
    }).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = int(response.status)
            response_body = response.read()
            payload = json.loads(response_body.decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"{scenario.scenario_id} endpoint identity probe failed"
        ) from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError(
            f"{scenario.scenario_id} endpoint identity response is not an object"
        )
    if set(payload) != {"jsonrpc", "id", "result"}:
        raise RuntimeError(
            f"{scenario.scenario_id} endpoint identity response has an unexpected shape"
        )
    if payload.get("jsonrpc") != "2.0" or payload.get("id") != request_id:
        raise RuntimeError(
            f"{scenario.scenario_id} endpoint identity response contract mismatch"
        )
    raw_result = payload.get("result")
    if not isinstance(raw_result, str) or re.fullmatch(r"0x[0-9a-fA-F]+", raw_result) is None:
        raise RuntimeError(
            f"{scenario.scenario_id} endpoint identity result is not a chain ID"
        )
    response_contract = {
        "jsonrpc": "2.0",
        "id": request_id,
        "result_path": "$.result",
        "result_type": "hex_quantity",
        "result": raw_result,
    }
    request_contract = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params_sha256": _canonical_hash(params),
    }
    observed = str(response_contract["result"] or "").strip()
    if status < 200 or status >= 300 or not observed:
        raise RuntimeError(
            f"{scenario.scenario_id} endpoint identity probe returned no usable identity"
        )
    if observed.lower() != expected.lower():
        raise RuntimeError(
            f"{scenario.scenario_id} endpoint identity mismatch: "
            f"expected={expected} observed={observed}"
        )
    return {
        "chain": str(plan.get("chain") or ""),
        "endpoint_env_var": admission.endpoint_env_var,
        "endpoint_sha256": hashlib.sha256(endpoint.encode("utf-8")).hexdigest(),
        "probe_method": method,
        "expected_identity": expected,
        "observed_identity": observed,
        "request_contract": request_contract,
        "request_contract_sha256": _canonical_hash(request_contract),
        "response_contract": response_contract,
        "response_contract_sha256": _canonical_hash(response_contract),
        "response_sha256": _canonical_hash(response_contract),
        "http_status": status,
        "verified": True,
        "probed_at": _utc_timestamp(),
    }


def _probe_metrics_endpoint(
    endpoint: str,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    request = urllib.request.Request(endpoint, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = int(response.status)
            body = response.read()
            content_type = str(response.headers.get("Content-Type") or "")
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError("geth-dev metrics probe failed") from exc
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("geth-dev metrics response is not UTF-8 exposition") from exc
    metric_names: set[str] = set()
    sample_count = 0
    sample_pattern = re.compile(
        r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
        r"(?:\s*\{[^\r\n]*\})?\s+"
        r"(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|NaN|[+-]Inf)"
        r"(?:\s+\d+)?$"
    )
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = sample_pattern.fullmatch(stripped)
        if match is None:
            raise RuntimeError("geth-dev metrics response is invalid Prometheus exposition")
        metric_names.add(match.group("name"))
        sample_count += 1
    if status < 200 or status >= 300 or not body or sample_count <= 0:
        raise RuntimeError("geth-dev metrics probe returned no Prometheus samples")
    return {
        "metrics_url_sha256": hashlib.sha256(endpoint.encode("utf-8")).hexdigest(),
        "http_status": status,
        "content_type": content_type,
        "body_sha256": hashlib.sha256(body).hexdigest(),
        "body_size_bytes": len(body),
        "non_comment_sample_count": sample_count,
        "metric_family_count": len(metric_names),
        "parser": "prometheus_text_v0.0.4",
        "probed_at": _utc_timestamp(),
    }


def _resolve_endpoint_route(endpoint: str) -> dict[str, Any]:
    parsed = urllib.parse.urlsplit(endpoint)
    if parsed.scheme != "http" or not parsed.hostname or parsed.port is None:
        raise RuntimeError("G5 endpoint must use an explicit HTTP host and port")
    try:
        addresses = sorted({
            item[4][0]
            for item in socket.getaddrinfo(
                parsed.hostname,
                parsed.port,
                type=socket.SOCK_STREAM,
            )
        })
    except OSError as exc:
        raise RuntimeError(f"G5 endpoint host cannot be resolved: {parsed.hostname}") from exc
    if not addresses:
        raise RuntimeError(f"G5 endpoint host resolved to no addresses: {parsed.hostname}")
    return {
        "scheme": parsed.scheme,
        "hostname": parsed.hostname,
        "port": parsed.port,
        "path": parsed.path or "/",
        "resolved_addresses": addresses,
    }


def _attest_geth_dev_runtime(
    plan: Mapping[str, Any],
    endpoint_probe: Mapping[str, Any],
    envelope: Mapping[str, Any],
    *,
    revision: Mapping[str, str],
    host_attestation: Mapping[str, Any],
    metrics_probe: Any = None,
    route_resolver: Any = None,
    timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    metrics_probe = metrics_probe or _probe_metrics_endpoint
    route_resolver = route_resolver or _resolve_endpoint_route
    requirements = dict(envelope.get("container_metrics_requirements") or {})
    compose_service = str(requirements.get("compose_service") or "")
    container = dict(host_attestation.get("target_container") or {})
    networks = dict(container.get("networks") or {})
    container_id = str(container.get("container_id") or "")
    image_digest = str(container.get("image_digest") or "")
    image_reference = str(container.get("image_reference") or "")
    container_addresses = sorted(
        str(details.get("ip_address") or "")
        for details in networks.values()
        if str(details.get("ip_address") or "")
    )
    network_aliases = sorted(
        {
            str(alias)
            for details in networks.values()
            for alias in (details.get("aliases") or ())
            if str(alias)
        }
    )
    if (
        not container_id
        or image_digest != str(requirements.get("image_digest") or "")
        or image_reference != str(requirements.get("image_reference") or "")
        or container.get("compose_service") != compose_service
        or compose_service not in network_aliases
        or not container_addresses
    ):
        raise RuntimeError("host attestation identity does not describe geth-dev")
    execution_env = dict((plan.get("execution") or {}).get("environment") or {})
    metrics_url = G5_RUNTIME_CONTRACT.metrics_url
    endpoint_env_var = str(
        (envelope.get("endpoint_identity_contract") or {}).get("env_var") or ""
    )
    rpc_url = str(execution_env.get(endpoint_env_var) or "").strip()
    rpc_route = route_resolver(rpc_url)
    metrics_route = route_resolver(metrics_url)
    for label, route in (("RPC", rpc_route), ("metrics", metrics_route)):
        if route.get("hostname") != compose_service:
            raise RuntimeError(f"geth-dev {label} endpoint does not use compose service DNS")
        if not set(route.get("resolved_addresses") or ()) & set(container_addresses):
            raise RuntimeError(f"geth-dev {label} endpoint does not route to inspected container")
    metrics = metrics_probe(metrics_url, timeout_seconds=timeout_seconds)
    container_contract = {
        "container_id": container_id,
        "image_digest": image_digest,
        "image_reference": image_reference,
        "compose_service": str(container.get("compose_service") or ""),
        "compose_project": str(container.get("compose_project") or ""),
        "network_addresses": container_addresses,
        "network_aliases": network_aliases,
    }
    attestation: dict[str, Any] = {
        "schema_version": 1,
        "repository_revision": dict(revision),
        "container": container_contract,
        "container_contract_sha256": _canonical_hash(container_contract),
        "host_attestation_file": str(
            host_attestation.get("attestation_file") or ""
        ),
        "host_attestation_sha256": str(
            host_attestation.get("attestation_file_sha256") or ""
        ),
        "docker_inspect_sha256": str(
            (host_attestation.get("docker_inspect_sha256") or {}).get(
                TARGET_SERVICE
            )
            or ""
        ),
        "rpc_endpoint_sha256": str(endpoint_probe.get("endpoint_sha256") or ""),
        "rpc_chain_id": str(endpoint_probe.get("observed_identity") or ""),
        "rpc_response_sha256": str(endpoint_probe.get("response_sha256") or ""),
        "rpc_route": rpc_route,
        "metrics_route": metrics_route,
        "metrics_probe": dict(metrics),
        "attested_at": _utc_timestamp(),
    }
    attestation["attestation_sha256"] = _canonical_hash(attestation)
    return attestation


def _plan_file_for_scenario(
    scenario: ExecutionScenarioSpec,
    *,
    fake_plan_file: Path,
    rpc_plan_file: Path,
    sync_plan_file: Path,
) -> Path:
    if scenario.scenario_id == "rpc_fake_node_smoke":
        return fake_plan_file
    if scenario.workflow_type == SYNC_OBSERVE_WORKFLOW:
        return sync_plan_file
    return rpc_plan_file


def _wait_for_job(job_id: str, *, jobs_dir: Path, timeout_seconds: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        job = get_job(job_id, jobs_dir=jobs_dir)
        status = str(job.get("status") or "")
        if status in TERMINAL_JOB_STATUSES:
            return job
        if time.monotonic() >= deadline:
            raise TimeoutError(f"job {job_id} did not finish within {timeout_seconds:g}s")
        time.sleep(0.5)


def _job_evidence_files(job: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    job_id = str(job.get("job_id") or "")
    run_dir = Path(str(job.get("run_dir") or "")).resolve()
    if not job_id or run_dir.name != job_id:
        raise RuntimeError("execution result has an invalid job identity")
    runtime_env_file = str(job.get("runtime_env_file") or "").strip()
    artifact_index = str(job.get("artifact_index") or "").strip()
    if not runtime_env_file:
        raise RuntimeError(f"job {job_id} has no runtime.env evidence")
    if not artifact_index:
        raise RuntimeError(f"job {job_id} has no artifact index evidence")
    job_files = [
        run_dir / "job.json",
        run_dir / "plan.json",
        Path(runtime_env_file),
        Path(artifact_index),
    ]
    log_file = run_dir / "benchmark.log"
    return (
        [_hashed_file(path, job_id=job_id, run_dir=run_dir) for path in job_files],
        [_hashed_file(log_file, job_id=job_id, run_dir=run_dir)],
    )


def _required_artifact_manifest(
    job: Mapping[str, Any],
    scenario: ExecutionScenarioSpec,
) -> dict[str, Any]:
    job_id = str(job.get("job_id") or "")
    run_dir = Path(str(job.get("run_dir") or "")).resolve()
    if not job_id or run_dir.name != job_id:
        raise RuntimeError("execution result has an invalid job identity")
    observed = dict(job.get("artifacts") or {})
    owner_roots = _artifact_owner_roots(job)
    records: list[dict[str, Any]] = []
    for name in scenario.required_artifacts:
        raw_path = str(observed.get(name) or "").strip()
        if not raw_path:
            raise RuntimeError(f"required execution artifact is missing: {name}")
        record = _owned_hashed_file(
            raw_path,
            job_id=job_id,
            expected_owner=required_artifact_owner(scenario, name),
            owner_roots=owner_roots,
        )
        records.append({"name": name, **record})
    manifest = {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "job_id": job_id,
        "scenario_id": scenario.scenario_id,
        "job_plan_sha256": _sha256(Path(str(job.get("plan_file") or ""))),
        "owner_roots_sha256": _canonical_hash(
            {
                name: str(path)
                for name, path in sorted(owner_roots.items())
            }
        ),
        "owner_roots": {
            name: str(path)
            for name, path in sorted(owner_roots.items())
        },
        "artifacts": records,
    }
    manifest_path = run_dir / "required-artifacts.sha256.json"
    _write_immutable_json(manifest_path, manifest)
    return {
        "manifest": _hashed_file(manifest_path, job_id=job_id, run_dir=run_dir),
        "records": records,
    }


def _validate_fresh_job(
    job: Mapping[str, Any],
    *,
    jobs_dir: Path,
    admitted_at: str,
    submitted_at: str = "",
) -> None:
    job_id = str(job.get("job_id") or "")
    raw_run_dir = Path(str(job.get("run_dir") or ""))
    run_dir = raw_run_dir.resolve()
    jobs_root = jobs_dir.resolve()
    if (
        not job_id
        or raw_run_dir.is_symlink()
        or run_dir.name != job_id
        or run_dir.parent != jobs_root
    ):
        raise RuntimeError(f"job {job_id or '<missing>'} is outside the admitted fresh jobs root")
    created_at = _parse_timestamp(job.get("created_at"), label=f"job {job_id} created_at")
    if created_at < _parse_timestamp(admitted_at, label="jobs root admission"):
        raise RuntimeError(f"job {job_id} predates the admitted fresh jobs root")
    if submitted_at and created_at < _parse_timestamp(
        submitted_at,
        label=f"job {job_id} submission",
    ):
        raise RuntimeError(f"job {job_id} predates its execution submission")


def _admit_fresh_jobs_root(jobs_dir: Path) -> str:
    if jobs_dir.exists():
        raise RuntimeError(f"real execution jobs root must not already exist: {jobs_dir}")
    admitted_at = _utc_timestamp()
    jobs_dir.mkdir(parents=True, exist_ok=False)
    return admitted_at


def _require_current_revision(expected: Mapping[str, str]) -> None:
    if dict(repository_revision(REPO_ROOT)) != dict(expected):
        raise RuntimeError("repository revision changed during real execution acceptance")


def _edge_by_action(ledger: Mapping[str, Any], action_type: str) -> Mapping[str, Any]:
    matches = [
        edge
        for edge in ledger.get("edges") or []
        if edge.get("action_type") == action_type
        and bool(((edge.get("evidence") or {}).get("real_execution") or {}).get("required"))
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one real execution edge for {action_type}, found {len(matches)}")
    return matches[0]


def execute_required_edges(
    *,
    fake_plan_file: Path,
    rpc_plan_file: Path,
    sync_plan_file: Path,
    fake_approval_file: Path,
    rpc_smoke_approval_file: Path,
    rpc_final_approval_file: Path,
    sync_approval_file: Path,
    jobs_dir: Path,
    evidence_dir: Path,
    timeout_seconds: float,
    host_attestation: Mapping[str, Any],
) -> list[Path]:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("real execution evidence is supported only on Linux")
    for plan_file in (fake_plan_file, rpc_plan_file, sync_plan_file):
        if not plan_file.is_file():
            raise FileNotFoundError(f"approved plan not found: {plan_file}")
    revision = repository_revision(REPO_ROOT)
    approval_contracts = (
        (
            fake_approval_file,
            fake_plan_file,
            RPC_BENCHMARK_WORKFLOW,
            "fake-node",
            "approve_preflight_smoke",
        ),
        (
            rpc_smoke_approval_file,
            rpc_plan_file,
            RPC_BENCHMARK_WORKFLOW,
            "real-node",
            "approve_preflight_smoke",
        ),
        (
            rpc_final_approval_file,
            rpc_plan_file,
            RPC_BENCHMARK_WORKFLOW,
            "real-node",
            "approve_final_benchmark",
        ),
        (
            sync_approval_file,
            sync_plan_file,
            SYNC_OBSERVE_WORKFLOW,
            "sync-observe",
            "approve_preflight_smoke",
        ),
    )
    for (
        approval_file,
        plan_file,
        workflow,
        target_mode,
        approval_action,
    ) in approval_contracts:
        valid, reason = validate_approved_plan_artifact(
            approval_file,
            expected_revision=revision,
            expected_plan_file=plan_file,
            expected_workflow=workflow,
            expected_target_mode=target_mode,
            expected_approval_action=approval_action,
        )
        if not valid:
            raise RuntimeError(f"Agent approved-plan provenance is invalid: {reason}")
    _validate_source_plan(
        fake_plan_file,
        workflow_type=RPC_BENCHMARK_WORKFLOW,
        use_fake_node=True,
    )
    _validate_source_plan(
        rpc_plan_file,
        workflow_type=RPC_BENCHMARK_WORKFLOW,
        use_fake_node=False,
    )
    _validate_source_plan(
        sync_plan_file,
        workflow_type=SYNC_OBSERVE_WORKFLOW,
        use_fake_node=False,
    )
    if len({fake_plan_file.resolve(), rpc_plan_file.resolve(), sync_plan_file.resolve()}) != 3:
        raise RuntimeError("fake-node, real-node, and sync-observe require distinct approved plans")
    jobs_root_admitted_at = _admit_fresh_jobs_root(jobs_dir)
    ledger = build_ledger(revision=revision)
    service = BenchmarkExecutionService()
    written: list[Path] = []
    artifacts: list[dict[str, Any]] = []
    observed_job_ids: set[str] = set()
    scenarios = tuple(
        sorted(
            (item for item in EXECUTION_SCENARIOS if item.real_evidence_required),
            key=lambda item: g5_scenario_admission(item.scenario_id).sequence_index,
        )
    )
    for scenario in scenarios:
        admission = g5_scenario_admission(scenario.scenario_id)
        plan_file = _plan_file_for_scenario(
            scenario,
            fake_plan_file=fake_plan_file,
            rpc_plan_file=rpc_plan_file,
            sync_plan_file=sync_plan_file,
        )
        approval_file = (
            rpc_final_approval_file.resolve()
            if scenario.action_type == "approve_final_benchmark"
            else {
                fake_plan_file.resolve(): fake_approval_file.resolve(),
                rpc_plan_file.resolve(): rpc_smoke_approval_file.resolve(),
                sync_plan_file.resolve(): sync_approval_file.resolve(),
            }[plan_file.resolve()]
        )
        approved_plan = _stage_call(
            scenario.scenario_id,
            "plan_admission",
            _validate_approved_plan_binding,
            plan_file,
            scenario=scenario,
        )
        plan_hash = _sha256(plan_file)
        envelope_file, envelope, envelope_hash = _stage_call(
            scenario.scenario_id,
            "plan_admission",
            _build_admission_envelope,
            plan_file,
            approved_plan,
            scenario,
            revision=revision,
            jobs_dir=jobs_dir,
        )
        _stage_call(
            scenario.scenario_id,
            "revision_guard",
            _require_current_revision,
            revision,
        )
        endpoint_probe = _stage_call(
            scenario.scenario_id,
            "endpoint_probe",
            _probe_endpoint_identity,
            approved_plan,
            scenario,
            envelope,
        )
        runtime_attestation = (
            _stage_call(
                scenario.scenario_id,
                "runtime_attestation",
                _attest_geth_dev_runtime,
                approved_plan,
                endpoint_probe,
                envelope,
                revision=revision,
                host_attestation=host_attestation,
            )
            if admission.endpoint_env_var
            else {}
        )
        if _sha256(plan_file) != plan_hash:
            raise G5ExecutionStageError(
                scenario.scenario_id,
                "revision_guard",
                f"approved plan changed during endpoint admission: {plan_file}",
            )
        _stage_call(
            scenario.scenario_id,
            "revision_guard",
            _require_current_revision,
            revision,
        )
        service_operation = OPERATION_BY_VALUE[scenario.operation]
        edge = _edge_by_action(ledger, scenario.action_type)
        started_at = _utc_timestamp()
        predecessor_valid, predecessor_reason = _stage_call(
            scenario.scenario_id,
            "predecessor_admission",
            validate_real_execution_predecessor,
            artifacts,
            scenario=scenario,
            started_at=started_at,
            endpoint_probe=endpoint_probe,
            runtime_attestation=runtime_attestation,
        )
        if not predecessor_valid:
            raise G5ExecutionStageError(
                scenario.scenario_id,
                "predecessor_admission",
                f"real execution predecessor admission failed: {predecessor_reason}",
            )
        request = ExecutionRequest(
            operation=service_operation,
            approved=True,
            plan_file=plan_file,
            jobs_dir=jobs_dir,
        )
        service_result = _stage_call(
            scenario.scenario_id,
            "submission",
            service.execute,
            request,
        )
        result_payload = service_result.to_dict()
        submitted_job = dict(result_payload.get("data", {}).get("job") or {})
        job_id = str(submitted_job.get("job_id") or "")
        if not service_result.succeeded or not job_id:
            raise G5ExecutionStageError(
                scenario.scenario_id,
                "submission",
                f"{service_operation.value} submission failed: "
                f"{json.dumps(redact(result_payload), ensure_ascii=False)}",
            )
        if service_result.reused or bool(submitted_job.get("submission_reused")):
            raise G5ExecutionStageError(
                scenario.scenario_id,
                "submission",
                f"reused job {job_id}; fresh real execution is required",
            )
        if job_id in observed_job_ids:
            raise G5ExecutionStageError(
                scenario.scenario_id,
                "submission",
                f"real execution scenarios reused job identity: {job_id}",
            )
        observed_job_ids.add(job_id)
        final_job = _stage_call(
            scenario.scenario_id,
            "job_wait",
            _wait_for_job,
            job_id,
            jobs_dir=jobs_dir,
            timeout_seconds=timeout_seconds,
        )
        _stage_call(
            scenario.scenario_id,
            "job_wait",
            _validate_fresh_job,
            final_job,
            jobs_dir=jobs_dir,
            admitted_at=jobs_root_admitted_at,
            submitted_at=started_at,
        )
        observed_status = str(final_job.get("status") or "")
        observed_exit_status = final_job.get("exit_code")
        observed_error = str(final_job.get("error") or "")
        if not isinstance(observed_exit_status, int):
            raise G5ExecutionStageError(
                scenario.scenario_id,
                "terminal_job",
                "persisted job has no integer exit_code",
            )
        succeeded = observed_status == "completed" and observed_exit_status == 0
        if not succeeded and (observed_exit_status == 0 or not observed_error):
            raise G5ExecutionStageError(
                scenario.scenario_id,
                "terminal_job",
                "persisted failure has incomplete outcome evidence",
            )
        required_artifacts: dict[str, Any] = {"records": []}
        runtime_env_projection = _stage_call(
            scenario.scenario_id,
            "artifact_collection",
            validate_runtime_env_projection,
            final_job,
        )
        job_artifacts, log_artifacts = _stage_call(
            scenario.scenario_id,
            "artifact_collection",
            _job_evidence_files,
            final_job,
        )
        if succeeded:
            required_artifacts = _stage_call(
                scenario.scenario_id,
                "artifact_collection",
                _required_artifact_manifest,
                final_job,
                scenario,
            )
            job_artifacts.append(required_artifacts["manifest"])
        request_payload = {
            "action_type": scenario.action_type,
            "scenario_id": scenario.scenario_id,
            "service_operation": service_operation.value,
            "approved": True,
            "approved_plan_file": str(plan_file.resolve()),
            "approved_plan_sha256": plan_hash,
            "approved_plan_revision": dict(revision),
            "approved_plan_provenance_file": str(approval_file),
            "approved_plan_provenance_sha256": _sha256(approval_file),
            "admission_envelope_file": str(envelope_file.resolve()),
            "admission_envelope_sha256": envelope_hash,
            "jobs_dir": str(jobs_dir.resolve()),
            "jobs_root_admitted_at": jobs_root_admitted_at,
            "required_artifact_hashes": required_artifacts["records"],
            "endpoint_probe": endpoint_probe,
            "runtime_attestation": runtime_attestation,
            "runtime_env_projection": runtime_env_projection,
            "ledger_sequence": admission.sequence_index,
            "predecessor_scenario_id": admission.predecessor_scenario_id,
        }
        result_payload["observed_job"] = final_job
        outcome = "passed" if succeeded else "observed-fail"
        exit_status = observed_exit_status
        error = "" if succeeded else observed_error
        artifact = build_real_execution_evidence_artifact(
            edge=edge,
            revision=revision,
            scenario_id=scenario.scenario_id,
            operation_kind=scenario.operation_kind,
            request=request_payload,
            result=result_payload,
            job_id=job_id,
            job_artifacts=job_artifacts,
            log_artifacts=log_artifacts,
            started_at=started_at,
            outcome=outcome,
            exit_status=exit_status,
            error=error,
            finished_at=_utc_timestamp(),
        )
        valid, reason = validate_real_execution_evidence_artifact(
            artifact,
            edge=edge,
            revision=revision,
            allow_observed_failure=True,
        )
        if not valid:
            raise G5ExecutionStageError(
                scenario.scenario_id,
                "evidence_validation",
                f"real execution evidence failed validation: {reason}",
            )
        artifacts.append(artifact)
        written_path = write_evidence_artifact(artifact, evidence_dir)
        if written_path.is_file():
            written_path.chmod(0o400)
        written.append(written_path)
        if not succeeded:
            raise G5ExecutionStageError(
                scenario.scenario_id,
                "terminal_job",
                f"revision-bound observed-fail evidence: {written_path}",
                evidence_path=written_path,
            )
    _stage_call(
        "<ledger>",
        "revision_guard",
        _require_current_revision,
        revision,
    )
    valid, reason = _stage_call(
        "<ledger>",
        "ledger_validation",
        validate_real_execution_ledger_artifacts,
        artifacts,
        revision=revision,
    )
    if not valid:
        raise G5ExecutionStageError(
            "<ledger>",
            "ledger_validation",
            f"real execution ledger failed validation: {reason}",
        )
    return written


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fake-plan", required=True, type=Path)
    parser.add_argument("--rpc-plan", required=True, type=Path)
    parser.add_argument("--sync-plan", required=True, type=Path)
    parser.add_argument("--fake-approval", required=True, type=Path)
    parser.add_argument("--rpc-smoke-approval", required=True, type=Path)
    parser.add_argument("--rpc-final-approval", required=True, type=Path)
    parser.add_argument("--sync-approval", required=True, type=Path)
    parser.add_argument("--jobs-dir", required=True, type=Path)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args(argv)


def _attempt_evidence_paths(attempt_dir: Path) -> list[Path]:
    paths: list[tuple[int, Path]] = []
    for path in attempt_dir.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("evidence_class") != "real_execution":
            continue
        admission = g5_scenario_admission(
            str(payload.get("scenario_id") or "")
        )
        paths.append((admission.sequence_index, path))
    return [path for _sequence, path in sorted(paths)]


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(argv if argv is not None else sys.argv[1:])
    args = _parse_args(raw_argv)
    attestation_file = os.environ.get(ATTESTATION_ENV, "").strip()
    plan_files = (
        args.fake_plan.resolve(),
        args.rpc_plan.resolve(),
        args.sync_plan.resolve(),
    )
    approval_files = (
        args.fake_approval.resolve(),
        args.rpc_smoke_approval.resolve(),
        args.rpc_final_approval.resolve(),
        args.sync_approval.resolve(),
    )
    revision: dict[str, str] | None = None
    host_attestation: dict[str, Any] = {
        "attestation_file": attestation_file,
    }
    g5_root = args.evidence_dir.resolve().parent
    attempt_id, attempt_evidence_dir = create_g5_attempt(
        args.evidence_dir.resolve()
    )
    try:
        revision = repository_revision(REPO_ROOT)
        if not attestation_file:
            raise RuntimeError(
                f"{ATTESTATION_ENV} is required; launch this worker through "
                "real_execution_host_supervisor"
            )
        host_attestation = _load_host_attestation(
            attestation_file,
            revision=revision,
            worker_argv=raw_argv,
        )
        paths = execute_required_edges(
            fake_plan_file=plan_files[0],
            rpc_plan_file=plan_files[1],
            sync_plan_file=plan_files[2],
            fake_approval_file=args.fake_approval.resolve(),
            rpc_smoke_approval_file=args.rpc_smoke_approval.resolve(),
            rpc_final_approval_file=args.rpc_final_approval.resolve(),
            sync_approval_file=args.sync_approval.resolve(),
            jobs_dir=args.jobs_dir.resolve(),
            evidence_dir=attempt_evidence_dir,
            timeout_seconds=args.timeout,
            host_attestation=host_attestation,
        )
    except G5ExecutionStageError as exc:
        if exc.scenario_id == "<ledger>":
            publish_g5_collection(
                g5_root=g5_root,
                attempt_id=attempt_id,
                revision=revision,
                evidence_paths=_attempt_evidence_paths(
                    attempt_evidence_dir
                ),
                status="failed",
            )
            raise RuntimeError(
                f"G5 ledger validation failed: {exc.cause}"
            ) from exc
        if exc.evidence_path is not None:
            publish_g5_collection(
                g5_root=g5_root,
                attempt_id=attempt_id,
                revision=revision,
                evidence_paths=_attempt_evidence_paths(
                    attempt_evidence_dir
                ),
                status="failed",
                failure_path=exc.evidence_path,
            )
            raise RuntimeError(
                f"G5 execution failed with retained evidence: {exc.evidence_path}"
            ) from exc
        failure_path = _write_g5_failure_artifact(
            evidence_dir=attempt_evidence_dir,
            revision=revision,
            scenario_id=exc.scenario_id,
            stage=exc.stage,
            error=exc.cause,
            plan_files=plan_files,
            approval_files=approval_files,
            host_attestation=host_attestation,
            attempt_id=attempt_id,
        )
        retained_paths = [
            Path(str(item["path"]))
            for item in json.loads(
                failure_path.read_text(encoding="utf-8")
            ).get("retained_scenario_evidence", [])
        ]
        publish_g5_collection(
            g5_root=g5_root,
            attempt_id=attempt_id,
            revision=revision,
            evidence_paths=retained_paths,
            status="failed",
            failure_path=failure_path,
        )
        raise RuntimeError(
            f"G5 execution failed with retained stage evidence: {failure_path}"
        ) from exc
    except Exception as exc:
        if revision is None:
            failure_path = _write_g5_bootstrap_failure_artifact(
                evidence_dir=attempt_evidence_dir,
                stage="worker_admission",
                error=exc,
                plan_files=plan_files,
                approval_files=approval_files,
                host_attestation_file=attestation_file,
            )
            raise RuntimeError(
                "G5 worker bootstrap failed without a repository revision; "
                f"non-qualifying evidence: {failure_path}"
            ) from exc
        failure_path = _write_g5_failure_artifact(
            evidence_dir=attempt_evidence_dir,
            revision=revision,
            scenario_id="",
            stage="worker_admission",
            error=exc,
            plan_files=plan_files,
            approval_files=approval_files,
            host_attestation=host_attestation,
            attempt_id=attempt_id,
        )
        publish_g5_collection(
            g5_root=g5_root,
            attempt_id=attempt_id,
            revision=revision,
            evidence_paths=(),
            status="failed",
            failure_path=failure_path,
        )
        raise RuntimeError(
            f"G5 worker failed with retained evidence: {failure_path}"
        ) from exc
    manifest = publish_g5_collection(
        g5_root=g5_root,
        attempt_id=attempt_id,
        revision=revision,
        evidence_paths=paths,
        status="complete",
    )
    print(json.dumps({
        "executed": len(paths),
        "evidence": [str(path) for path in paths],
        "collection_manifest": str(manifest),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

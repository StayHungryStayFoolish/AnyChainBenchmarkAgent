"""Typed application boundary for benchmark preparation and execution.

Transport adapters (CLI, tools, and the LangGraph Harness) submit requests to
this service.  Pipeline functions and the file-backed job manager are
infrastructure details and must not be called by those adapters directly.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from agent.planners.preflight import run_preflight
from agent.runners.benchmark_pipeline import (
    prepare_benchmark_run,
    run_fake_node_smoke_benchmark,
    run_real_node_smoke_benchmark,
    submit_benchmark_job,
)
from agent.runners.job_manager import DEFAULT_JOBS_DIR, submit_job


class ExecutionOperation(str, Enum):
    PREPARE = "prepare"
    PREFLIGHT = "preflight"
    FAKE_NODE_SMOKE = "fake_node_smoke"
    REAL_NODE_SMOKE = "real_node_smoke"
    FINAL_BENCHMARK = "final_benchmark"
    SYNC_OBSERVE = "sync_observe"


@dataclass(frozen=True)
class ExecutionOperationSpec:
    operation: ExecutionOperation
    runner_kind: str
    requires_approval: bool = False
    requires_plan_file: bool = False


EXECUTION_OPERATION_SPECS: dict[ExecutionOperation, ExecutionOperationSpec] = {
    spec.operation: spec
    for spec in (
        ExecutionOperationSpec(ExecutionOperation.PREPARE, "prepare"),
        ExecutionOperationSpec(ExecutionOperation.PREFLIGHT, "preflight"),
        ExecutionOperationSpec(ExecutionOperation.FAKE_NODE_SMOKE, "fake_node_smoke", True, True),
        ExecutionOperationSpec(ExecutionOperation.REAL_NODE_SMOKE, "real_node_smoke", True, True),
        ExecutionOperationSpec(ExecutionOperation.FINAL_BENCHMARK, "benchmark", True, True),
        ExecutionOperationSpec(ExecutionOperation.SYNC_OBSERVE, "benchmark", True, True),
    )
}


class ExecutionStatus(str, Enum):
    OK = "ok"
    BLOCKED = "blocked"
    FAILED = "failed"


class ExecutionFailureCode(str, Enum):
    APPROVAL_REQUIRED = "approval_required"
    INVALID_REQUEST = "invalid_request"
    PLAN_NOT_FOUND = "plan_not_found"
    PREFLIGHT_BLOCKED = "preflight_blocked"
    SUBMISSION_FAILED = "submission_failed"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True)
class ExecutionFailure:
    code: ExecutionFailureCode
    message: str
    retryable: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "retryable": self.retryable,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class ExecutionRequest:
    operation: ExecutionOperation
    approved: bool = False
    idempotency_key: str = ""
    plan_file: str | Path | None = None
    plan: Mapping[str, Any] | None = None
    prepare_kwargs: Mapping[str, Any] = field(default_factory=dict)
    jobs_dir: str | Path = DEFAULT_JOBS_DIR
    mock: bool = False
    runtime_override_sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutionResult:
    operation: ExecutionOperation
    status: ExecutionStatus
    data: Mapping[str, Any] = field(default_factory=dict)
    evidence_paths: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    next_actions: tuple[str, ...] = ()
    requires_user_confirmation: bool = False
    idempotency_key: str = ""
    reused: bool = False
    failure: ExecutionFailure | None = None

    @property
    def succeeded(self) -> bool:
        return self.status is ExecutionStatus.OK

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation.value,
            "status": self.status.value,
            "data": dict(self.data),
            "evidence_paths": list(self.evidence_paths),
            "warnings": list(self.warnings),
            "next_actions": list(self.next_actions),
            "requires_user_confirmation": self.requires_user_confirmation,
            "idempotency_key": self.idempotency_key,
            "reused": self.reused,
            "failure": self.failure.to_dict() if self.failure else None,
        }


class BenchmarkExecutionService:
    """Single application service for all benchmark execution requests."""

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        operation = request.operation
        spec = EXECUTION_OPERATION_SPECS[operation]
        if spec.requires_approval and not request.approved:
            return self._failure(
                operation,
                ExecutionStatus.BLOCKED,
                ExecutionFailureCode.APPROVAL_REQUIRED,
                f"explicit approval is required for {operation.value}",
                requires_user_confirmation=True,
                next_actions=("obtain explicit user approval",),
            )

        try:
            if spec.runner_kind == "prepare":
                return self._from_envelope(operation, prepare_benchmark_run(**dict(request.prepare_kwargs)))
            if spec.runner_kind == "preflight":
                plan = self._load_plan(request)
                preflight = run_preflight(plan)
                status = ExecutionStatus.OK if preflight.get("passed") else ExecutionStatus.BLOCKED
                failure = None
                if status is ExecutionStatus.BLOCKED:
                    failure = ExecutionFailure(
                        code=ExecutionFailureCode.PREFLIGHT_BLOCKED,
                        message="; ".join(str(item) for item in preflight.get("blockers") or [])
                        or "preflight checks did not pass",
                        retryable=True,
                        details={"blockers": list(preflight.get("blockers") or [])},
                    )
                return ExecutionResult(
                    operation=operation,
                    status=status,
                    data={"preflight": preflight},
                    warnings=tuple(str(item) for item in preflight.get("warnings") or []),
                    next_actions=("correct blockers and rerun preflight",) if failure else (),
                    failure=failure,
                )

            approved_plan_file = self._require_plan_file(request)
            plan = self._load_approved_submission_plan(request, approved_plan_file)
            operation = self._effective_submission_operation(operation, plan)
            spec = EXECUTION_OPERATION_SPECS[operation]
            idempotency_key = self._idempotency_key(
                approved_plan_file,
                plan,
                operation=operation,
                requested=request.idempotency_key,
            )
            execution_plan = self._build_execution_plan(
                approved_plan_file,
                plan,
                operation=operation,
                idempotency_key=idempotency_key,
                runtime_override_sources=request.runtime_override_sources,
            )
            jobs_dir = str(request.jobs_dir)
            if spec.runner_kind == "fake_node_smoke":
                payload = run_fake_node_smoke_benchmark(
                    str(approved_plan_file), jobs_dir=jobs_dir, execution_plan=execution_plan
                )
            elif spec.runner_kind == "real_node_smoke":
                payload = run_real_node_smoke_benchmark(
                    str(approved_plan_file), jobs_dir=jobs_dir, execution_plan=execution_plan
                )
            elif request.mock:
                job = submit_job(
                    approved_plan_file,
                    jobs_dir=jobs_dir,
                    mock=True,
                    approved=True,
                    execution_plan=execution_plan,
                )
                payload = self._job_envelope(job)
            elif spec.runner_kind == "benchmark":
                payload = submit_benchmark_job(
                    str(approved_plan_file), jobs_dir=jobs_dir, execution_plan=execution_plan
                )
            else:  # pragma: no cover - registry completeness guard
                raise RuntimeError(f"unsupported execution runner kind: {spec.runner_kind}")
            return self._from_envelope(operation, payload, idempotency_key=idempotency_key)
        except FileNotFoundError as exc:
            return self._failure(
                operation,
                ExecutionStatus.BLOCKED,
                ExecutionFailureCode.PLAN_NOT_FOUND,
                str(exc),
                next_actions=("prepare a benchmark plan",),
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            return self._failure(
                operation,
                ExecutionStatus.BLOCKED,
                ExecutionFailureCode.INVALID_REQUEST,
                str(exc),
            )
        except Exception as exc:  # pragma: no cover - application boundary guard
            return self._failure(
                operation,
                ExecutionStatus.FAILED,
                ExecutionFailureCode.INTERNAL_ERROR,
                str(exc),
                retryable=True,
            )

    @staticmethod
    def _load_plan(request: ExecutionRequest) -> dict[str, Any]:
        if request.plan is not None:
            return dict(request.plan)
        return json.loads(BenchmarkExecutionService._require_plan_file(request).read_text(encoding="utf-8"))

    @staticmethod
    def _load_approved_submission_plan(
        request: ExecutionRequest,
        approved_plan_file: Path,
    ) -> dict[str, Any]:
        """Load the approved baseline and admit only typed job-local overrides."""

        baseline = json.loads(approved_plan_file.read_text(encoding="utf-8"))
        if request.plan is None:
            return baseline
        proposed = dict(request.plan)
        changed = _changed_paths(baseline, proposed)
        if not changed:
            return baseline
        if not request.runtime_override_sources:
            raise ValueError("runtime plan overrides require an explicit typed source")
        forbidden = sorted(
            ".".join(path)
            for path in changed
            if not _runtime_override_path_allowed(path)
        )
        if forbidden:
            raise ValueError(
                "runtime plan attempted to replace approved content: " + ", ".join(forbidden)
            )
        admitted = deepcopy(baseline)
        for path in changed:
            _copy_override_path(admitted, proposed, path)
        return admitted

    @staticmethod
    def _require_plan_file(request: ExecutionRequest) -> Path:
        if request.plan_file in {None, ""}:
            raise ValueError(f"plan_file is required for {request.operation.value}")
        path = Path(str(request.plan_file)).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"plan file not found: {path}")
        return path

    @staticmethod
    def _effective_submission_operation(
        operation: ExecutionOperation,
        plan: Mapping[str, Any],
    ) -> ExecutionOperation:
        if operation is not ExecutionOperation.FINAL_BENCHMARK:
            return operation
        workflow = str(plan.get("workflow_type") or (plan.get("request") or {}).get("workflow_type") or "")
        return ExecutionOperation.SYNC_OBSERVE if workflow == "sync_observe" else operation

    @staticmethod
    def _idempotency_key(
        plan_file: Path,
        plan: Mapping[str, Any],
        *,
        operation: ExecutionOperation,
        requested: str,
    ) -> str:
        execution = dict(plan.get("execution") or {})
        current = str(execution.get("idempotency_key") or "").strip()
        key = requested.strip() or current
        if not key:
            digest = hashlib.sha256(plan_file.read_bytes()).hexdigest()[:24]
            key = f"{operation.value}:{digest}"
        return key

    @staticmethod
    def _build_execution_plan(
        approved_plan_file: Path,
        plan: Mapping[str, Any],
        *,
        operation: ExecutionOperation,
        idempotency_key: str,
        runtime_override_sources: tuple[str, ...],
    ) -> dict[str, Any]:
        """Build the admitted execution view without creating a second plan file."""

        approved_plan_file = approved_plan_file.resolve()
        approved_bytes = approved_plan_file.read_bytes()
        approved_sha256 = hashlib.sha256(approved_bytes).hexdigest()
        execution_plan = dict(plan)
        execution = dict(execution_plan.get("execution") or {})
        previous_key = str(execution.get("idempotency_key") or "").strip()
        execution["idempotency_key"] = idempotency_key
        execution_plan["execution"] = execution

        execution_plan.pop("execution_provenance", None)
        source_file = str(approved_plan_file)
        source_sha256 = approved_sha256
        overrides = [str(item) for item in runtime_override_sources if str(item)]
        if previous_key != idempotency_key:
            overrides.append("execution.idempotency_key")
        if execution_plan.get("chain_config_override"):
            overrides.append("custom_rpc.chain_config_override")
        overrides = list(dict.fromkeys(overrides))

        execution_plan["execution_provenance"] = {
            "approved_plan_file": source_file,
            "approved_plan_sha256": source_sha256,
            "operation": operation.value,
            "runtime_overrides": overrides,
        }
        return execution_plan

    def _from_envelope(
        self,
        operation: ExecutionOperation,
        payload: Mapping[str, Any],
        *,
        idempotency_key: str = "",
    ) -> ExecutionResult:
        raw_status = str(payload.get("status") or "failed")
        status = {
            "ok": ExecutionStatus.OK,
            "completed": ExecutionStatus.OK,
            "running": ExecutionStatus.OK,
            "blocked": ExecutionStatus.BLOCKED,
        }.get(raw_status, ExecutionStatus.FAILED)
        data = dict(payload.get("data") or {})
        job = data.get("job") if isinstance(data.get("job"), dict) else {}
        reused = bool(job.get("submission_reused"))
        warnings = tuple(str(item) for item in payload.get("warnings") or [] if str(item))
        failure = None
        if status is not ExecutionStatus.OK:
            failure = ExecutionFailure(
                code=(
                    ExecutionFailureCode.PREFLIGHT_BLOCKED
                    if operation in {ExecutionOperation.PREPARE, ExecutionOperation.PREFLIGHT}
                    else ExecutionFailureCode.SUBMISSION_FAILED
                ),
                message="; ".join(warnings) or str(job.get("error") or f"{operation.value} failed"),
                retryable=status is ExecutionStatus.FAILED,
                details={"job_id": str(job.get("job_id") or "")},
            )
        return ExecutionResult(
            operation=operation,
            status=status,
            data=data,
            evidence_paths=tuple(str(item) for item in payload.get("evidence_paths") or [] if str(item)),
            warnings=warnings,
            next_actions=tuple(str(item) for item in payload.get("next_actions") or []),
            requires_user_confirmation=bool(payload.get("requires_user_confirmation")),
            idempotency_key=idempotency_key,
            reused=reused,
            failure=failure,
        )

    @staticmethod
    def _job_envelope(job: Mapping[str, Any]) -> dict[str, Any]:
        status = "ok" if job.get("status") in {"completed", "running"} else "failed"
        return {
            "status": status,
            "data": {"job": dict(job)},
            "evidence_paths": [str(job.get("runtime_env_file") or ""), str(job.get("artifact_index") or "")],
            "warnings": [str(job.get("error") or "")] if job.get("error") else [],
            "next_actions": [],
        }

    @staticmethod
    def _failure(
        operation: ExecutionOperation,
        status: ExecutionStatus,
        code: ExecutionFailureCode,
        message: str,
        *,
        retryable: bool = False,
        requires_user_confirmation: bool = False,
        next_actions: tuple[str, ...] = (),
    ) -> ExecutionResult:
        failure = ExecutionFailure(code=code, message=message, retryable=retryable)
        return ExecutionResult(
            operation=operation,
            status=status,
            warnings=(message,),
            next_actions=next_actions,
            requires_user_confirmation=requires_user_confirmation,
            failure=failure,
        )


execution_service = BenchmarkExecutionService()


_RUNTIME_OVERRIDE_PREFIXES = (
    ("chain_config_override",),
    ("chain_template_requirements",),
    ("artifacts", "chain_config_override_file"),
)


def _runtime_override_path_allowed(path: tuple[str, ...]) -> bool:
    return any(path[: len(prefix)] == prefix for prefix in _RUNTIME_OVERRIDE_PREFIXES)


def _changed_paths(
    baseline: Mapping[str, Any],
    proposed: Mapping[str, Any],
    prefix: tuple[str, ...] = (),
) -> set[tuple[str, ...]]:
    changed: set[tuple[str, ...]] = set()
    for key in set(baseline) | set(proposed):
        path = (*prefix, str(key))
        if key not in baseline:
            value = proposed[key]
            if isinstance(value, Mapping):
                changed.update(_changed_paths({}, value, path))
            else:
                changed.add(path)
            continue
        if key not in proposed:
            value = baseline[key]
            if isinstance(value, Mapping):
                changed.update(_changed_paths(value, {}, path))
            else:
                changed.add(path)
            continue
        left = baseline[key]
        right = proposed[key]
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            changed.update(_changed_paths(left, right, path))
        elif left != right:
            changed.add(path)
    return changed


def _copy_override_path(
    target: dict[str, Any],
    source: Mapping[str, Any],
    path: tuple[str, ...],
) -> None:
    target_parent: dict[str, Any] = target
    source_parent: Mapping[str, Any] = source
    for part in path[:-1]:
        source_child = source_parent.get(part)
        if not isinstance(source_child, Mapping):
            target_parent.pop(part, None)
            return
        target_child = target_parent.get(part)
        if not isinstance(target_child, dict):
            target_child = {}
            target_parent[part] = target_child
        target_parent = target_child
        source_parent = source_child
    leaf = path[-1]
    if leaf in source_parent:
        target_parent[leaf] = deepcopy(source_parent[leaf])
    else:
        target_parent.pop(leaf, None)

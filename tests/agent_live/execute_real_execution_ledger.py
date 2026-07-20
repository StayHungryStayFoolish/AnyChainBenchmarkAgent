#!/usr/bin/env python3
"""Produce revision-bound evidence for required real execution edges."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from agent.harness.runtime_identity import repository_revision
from agent.runners.application_service import (
    BenchmarkExecutionService,
    ExecutionOperation,
    ExecutionRequest,
)
from agent.runners.job_manager import get_job
from tests.agent_live.coverage_evidence import (
    build_real_execution_evidence_artifact,
    validate_real_execution_evidence_artifact,
    write_evidence_artifact,
)
from tests.agent_live.generate_harness_coverage_ledger import build_ledger

TERMINAL_JOB_STATUSES = frozenset({"completed", "failed", "partial", "cancelled"})
EXECUTION_CASES = (
    ("approve_preflight_smoke", "preflight_smoke", ExecutionOperation.REAL_NODE_SMOKE),
    ("approve_final_benchmark", "final_benchmark", ExecutionOperation.FINAL_BENCHMARK),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hashed_file(path: str | Path, *, job_id: str) -> dict[str, Any]:
    artifact = Path(path).resolve()
    if not artifact.is_file():
        raise RuntimeError(f"required execution artifact does not exist: {artifact}")
    if job_id not in artifact.parts:
        raise RuntimeError(f"execution artifact is not owned by {job_id}: {artifact}")
    return {
        "job_id": job_id,
        "path": str(artifact),
        "sha256": _sha256(artifact),
        "size_bytes": artifact.stat().st_size,
    }


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
    job_files = [run_dir / "job.json", run_dir / "plan.json"]
    for candidate in (job.get("runtime_env_file"), job.get("artifact_index")):
        if candidate and Path(str(candidate)).is_file():
            job_files.append(Path(str(candidate)))
    log_file = run_dir / "benchmark.log"
    return (
        [_hashed_file(path, job_id=job_id) for path in job_files],
        [_hashed_file(log_file, job_id=job_id)],
    )


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
    plan_file: Path,
    jobs_dir: Path,
    evidence_dir: Path,
    timeout_seconds: float,
) -> list[Path]:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("real execution evidence is supported only on Linux")
    if not plan_file.is_file():
        raise FileNotFoundError(f"approved plan not found: {plan_file}")
    if jobs_dir.exists() and any(jobs_dir.iterdir()):
        raise RuntimeError(f"real execution jobs directory must be fresh: {jobs_dir}")
    jobs_dir.mkdir(parents=True, exist_ok=True)
    revision = repository_revision(REPO_ROOT)
    ledger = build_ledger(revision=revision)
    service = BenchmarkExecutionService()
    written: list[Path] = []
    plan_hash = _sha256(plan_file)

    for action_type, operation_kind, service_operation in EXECUTION_CASES:
        edge = _edge_by_action(ledger, action_type)
        started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        request = ExecutionRequest(
            operation=service_operation,
            approved=True,
            plan_file=plan_file,
            jobs_dir=jobs_dir,
        )
        service_result = service.execute(request)
        result_payload = service_result.to_dict()
        submitted_job = dict(result_payload.get("data", {}).get("job") or {})
        job_id = str(submitted_job.get("job_id") or "")
        if not service_result.succeeded or not job_id:
            raise RuntimeError(
                f"{service_operation.value} submission failed: {json.dumps(result_payload, ensure_ascii=False)}"
            )
        final_job = _wait_for_job(job_id, jobs_dir=jobs_dir, timeout_seconds=timeout_seconds)
        if final_job.get("status") != "completed" or int(final_job.get("exit_code", 0)) != 0:
            raise RuntimeError(
                f"job {job_id} did not complete successfully: "
                f"status={final_job.get('status')} error={final_job.get('error') or '<none>'}"
            )
        job_artifacts, log_artifacts = _job_evidence_files(final_job)
        request_payload = {
            "action_type": action_type,
            "service_operation": service_operation.value,
            "approved": True,
            "approved_plan_file": str(plan_file.resolve()),
            "approved_plan_sha256": plan_hash,
            "jobs_dir": str(jobs_dir.resolve()),
        }
        result_payload["observed_job"] = final_job
        artifact = build_real_execution_evidence_artifact(
            edge=edge,
            revision=revision,
            operation_kind=operation_kind,
            request=request_payload,
            result=result_payload,
            job_id=job_id,
            job_artifacts=job_artifacts,
            log_artifacts=log_artifacts,
            started_at=started_at,
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        valid, reason = validate_real_execution_evidence_artifact(
            artifact,
            edge=edge,
            revision=revision,
        )
        if not valid:
            raise RuntimeError(f"real execution evidence failed validation: {reason}")
        written.append(write_evidence_artifact(artifact, evidence_dir))
    return written


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--jobs-dir", required=True, type=Path)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = execute_required_edges(
        plan_file=args.plan.resolve(),
        jobs_dir=args.jobs_dir.resolve(),
        evidence_dir=args.evidence_dir.resolve(),
        timeout_seconds=args.timeout,
    )
    print(json.dumps({"executed": len(paths), "evidence": [str(path) for path in paths]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

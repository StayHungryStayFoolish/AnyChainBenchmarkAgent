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
from agent.runners.execution_scenarios import (
    EXECUTION_SCENARIOS,
    RPC_BENCHMARK_WORKFLOW,
    SYNC_OBSERVE_WORKFLOW,
    ExecutionScenarioSpec,
    workflow_type_from_plan,
)
from tests.agent_live.coverage_evidence import (
    build_real_execution_evidence_artifact,
    validate_real_execution_evidence_artifact,
    write_evidence_artifact,
)
from tests.agent_live.generate_harness_coverage_ledger import build_ledger

TERMINAL_JOB_STATUSES = frozenset({"completed", "failed", "partial", "cancelled"})
OPERATION_BY_VALUE = {operation.value: operation for operation in ExecutionOperation}


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
        [_hashed_file(path, job_id=job_id) for path in job_files],
        [_hashed_file(log_file, job_id=job_id)],
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
    records: list[dict[str, Any]] = []
    for name in scenario.required_artifacts:
        raw_path = str(observed.get(name) or "").strip()
        if not raw_path:
            raise RuntimeError(f"required execution artifact is missing: {name}")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise RuntimeError(f"required execution artifact does not exist: {name}: {path}")
        records.append({
            "name": name,
            "path": str(path),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        })
    manifest = {
        "job_id": job_id,
        "scenario_id": scenario.scenario_id,
        "artifacts": records,
    }
    manifest_path = run_dir / "required-artifacts.sha256.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "manifest": _hashed_file(manifest_path, job_id=job_id),
        "records": records,
    }


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
    jobs_dir: Path,
    evidence_dir: Path,
    timeout_seconds: float,
) -> list[Path]:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("real execution evidence is supported only on Linux")
    for plan_file in (fake_plan_file, rpc_plan_file, sync_plan_file):
        if not plan_file.is_file():
            raise FileNotFoundError(f"approved plan not found: {plan_file}")
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
    if jobs_dir.exists() and any(jobs_dir.iterdir()):
        raise RuntimeError(f"real execution jobs directory must be fresh: {jobs_dir}")
    jobs_dir.mkdir(parents=True, exist_ok=True)
    revision = repository_revision(REPO_ROOT)
    ledger = build_ledger(revision=revision)
    service = BenchmarkExecutionService()
    written: list[Path] = []
    observed_job_ids: set[str] = set()
    for scenario in EXECUTION_SCENARIOS:
        if not scenario.real_evidence_required:
            continue
        plan_file = _plan_file_for_scenario(
            scenario,
            fake_plan_file=fake_plan_file,
            rpc_plan_file=rpc_plan_file,
            sync_plan_file=sync_plan_file,
        )
        plan_hash = _sha256(plan_file)
        service_operation = OPERATION_BY_VALUE[scenario.operation]
        edge = _edge_by_action(ledger, scenario.action_type)
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
        if service_result.reused or bool(submitted_job.get("submission_reused")):
            raise RuntimeError(
                f"{scenario.scenario_id} reused job {job_id}; fresh real execution is required"
            )
        if job_id in observed_job_ids:
            raise RuntimeError(f"real execution scenarios reused job identity: {job_id}")
        observed_job_ids.add(job_id)
        final_job = _wait_for_job(job_id, jobs_dir=jobs_dir, timeout_seconds=timeout_seconds)
        if final_job.get("status") != "completed" or int(final_job.get("exit_code", 0)) != 0:
            raise RuntimeError(
                f"job {job_id} did not complete successfully: "
                f"status={final_job.get('status')} error={final_job.get('error') or '<none>'}"
            )
        required_artifacts = _required_artifact_manifest(final_job, scenario)
        job_artifacts, log_artifacts = _job_evidence_files(final_job)
        job_artifacts.append(required_artifacts["manifest"])
        request_payload = {
            "action_type": scenario.action_type,
            "scenario_id": scenario.scenario_id,
            "service_operation": service_operation.value,
            "approved": True,
            "approved_plan_file": str(plan_file.resolve()),
            "approved_plan_sha256": plan_hash,
            "jobs_dir": str(jobs_dir.resolve()),
            "required_artifact_hashes": required_artifacts["records"],
        }
        result_payload["observed_job"] = final_job
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
    parser.add_argument("--fake-plan", required=True, type=Path)
    parser.add_argument("--rpc-plan", required=True, type=Path)
    parser.add_argument("--sync-plan", required=True, type=Path)
    parser.add_argument("--jobs-dir", required=True, type=Path)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = execute_required_edges(
        fake_plan_file=args.fake_plan.resolve(),
        rpc_plan_file=args.rpc_plan.resolve(),
        sync_plan_file=args.sync_plan.resolve(),
        jobs_dir=args.jobs_dir.resolve(),
        evidence_dir=args.evidence_dir.resolve(),
        timeout_seconds=args.timeout,
    )
    print(json.dumps({"executed": len(paths), "evidence": [str(path) for path in paths]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

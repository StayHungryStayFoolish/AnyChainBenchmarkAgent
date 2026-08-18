"""Authoritative ownership rules for execution control and benchmark artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from agent.runners.execution_scenarios import ExecutionScenarioSpec
from agent.runners.materialize import load_runtime_env_file


JOB_CONTROL_OWNER = "job_control"
BENCHMARK_DATA_OWNER = "benchmark_data"
ARTIFACT_MANIFEST_SCHEMA_VERSION = 1


def required_artifact_owner(
    scenario: ExecutionScenarioSpec,
    artifact_name: str,
) -> str:
    if artifact_name not in scenario.required_artifacts:
        raise ValueError(
            f"artifact is not required by scenario {scenario.scenario_id}: "
            f"{artifact_name}"
        )
    owner = str(scenario.required_artifact_owner or "")
    if owner not in {JOB_CONTROL_OWNER, BENCHMARK_DATA_OWNER}:
        raise ValueError(
            f"scenario {scenario.scenario_id} has invalid artifact owner: {owner}"
        )
    return owner


def derive_artifact_owner_roots(job: Mapping[str, Any]) -> dict[str, Path]:
    job_id = str(job.get("job_id") or "")
    raw_run_dir = Path(str(job.get("run_dir") or ""))
    if raw_run_dir.is_symlink():
        raise ValueError("job run_dir cannot be a symlink")
    run_dir = raw_run_dir.resolve(strict=True)
    if not job_id or run_dir.name != job_id or not run_dir.is_dir():
        raise ValueError("execution result has an invalid job run_dir")

    runtime_env_file = Path(str(job.get("runtime_env_file") or ""))
    plan_file = Path(str(job.get("plan_file") or ""))
    if (
        runtime_env_file != run_dir / "runtime.env"
        or plan_file != run_dir / "plan.json"
        or runtime_env_file.is_symlink()
        or plan_file.is_symlink()
        or not runtime_env_file.is_file()
        or not plan_file.is_file()
    ):
        raise ValueError("job control files do not match the exact run_dir contract")

    runtime_env = load_runtime_env_file(runtime_env_file)
    try:
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("materialized job plan is not valid JSON") from exc
    if not isinstance(plan, Mapping):
        raise ValueError("materialized job plan must be a JSON object")
    execution = dict(plan.get("execution") or {})
    plan_env = dict(execution.get("environment") or {})
    materialized = dict(plan.get("materialized_config") or {})
    raw_data_root = str(
        runtime_env.get("BLOCKCHAIN_BENCHMARK_DATA_DIR") or ""
    ).strip()
    if not raw_data_root:
        raise ValueError("runtime.env has no BLOCKCHAIN_BENCHMARK_DATA_DIR")
    if str(plan_env.get("BLOCKCHAIN_BENCHMARK_DATA_DIR") or "") != raw_data_root:
        raise ValueError("runtime.env benchmark data root disagrees with the job plan")
    materialized_root = str(
        materialized.get("BLOCKCHAIN_BENCHMARK_DATA_DIR") or ""
    )
    if materialized_root and materialized_root != raw_data_root:
        raise ValueError(
            "materialized_config benchmark data root disagrees with runtime.env"
        )

    data_root = Path(raw_data_root)
    if not data_root.is_absolute():
        working_dir = Path(str(execution.get("working_dir") or Path.cwd()))
        data_root = working_dir / data_root
    if data_root.is_symlink():
        raise ValueError("benchmark data root cannot be a symlink")
    data_root = data_root.resolve(strict=True)
    if not data_root.is_dir():
        raise ValueError("benchmark data root is not a directory")

    jobs_root = run_dir.parent
    try:
        data_root.relative_to(jobs_root)
    except ValueError as exc:
        raise ValueError("benchmark data root is outside the admitted jobs root") from exc
    if data_root == run_dir or data_root in run_dir.parents or run_dir in data_root.parents:
        raise ValueError("execution artifact owner roots overlap")
    _reject_symlink_components(data_root, stop=jobs_root)
    return {
        JOB_CONTROL_OWNER: run_dir,
        BENCHMARK_DATA_OWNER: data_root,
    }


def validate_owned_artifact_path(
    path: str | Path,
    *,
    expected_owner: str,
    owner_roots: Mapping[str, Path],
) -> Path:
    if expected_owner not in owner_roots:
        raise ValueError(f"unknown artifact owner: {expected_owner}")
    raw = Path(path)
    lexical = raw if raw.is_absolute() else Path.cwd() / raw
    artifact = raw.resolve(strict=True)
    root = Path(owner_roots[expected_owner]).resolve(strict=True)
    if artifact == root:
        raise ValueError("artifact cannot be an owner root")
    try:
        artifact.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"artifact is outside expected owner {expected_owner}: {artifact}"
        ) from exc
    _reject_symlink_components(lexical.parent, stop=root)
    if not artifact.is_file():
        raise ValueError(f"artifact is not a regular file: {artifact}")
    matches = [
        name
        for name, candidate_root in owner_roots.items()
        if artifact != candidate_root.resolve()
        and candidate_root.resolve() in artifact.parents
    ]
    if matches != [expected_owner]:
        raise ValueError(
            f"artifact ownership is ambiguous or incorrect: {artifact}: {matches}"
        )
    return artifact


def _reject_symlink_components(path: Path, *, stop: Path) -> None:
    current = path
    root = stop.resolve(strict=True)
    while current != root:
        if current.is_symlink():
            raise ValueError(f"path contains a symlink: {current}")
        parent = current.parent
        if parent == current:
            raise ValueError(f"path is not rooted at {root}: {path}")
        current = parent

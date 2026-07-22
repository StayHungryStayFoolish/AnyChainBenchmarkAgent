"""Immutable batch orchestration for response-driven Agent Chaos shards.

The orchestrator owns process isolation, terminal classification, and artifact
indexing.  It deliberately does not generate simulator messages or change the
coverage, discovery, or bridge protocols.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Protocol, Sequence

from agent.harness.runtime_identity import repository_revision
from agent.utils.redaction import redact
from tests.agent_live.chaos_scheduler import (
    CHAOS_SCHEDULE_SCHEMA_VERSION,
    build_chaos_schedule,
    build_journey_schedule,
    journey_schedule_payload,
    schedule_payload,
    validate_chaos_schedule,
    validate_journey_schedule,
)
from tests.agent_live.codex_simulator_bridge import (
    CONTEXT_FRAME,
    DECISION_FRAME,
    RESULT_FRAME,
)
from tests.agent_live.journey_simulator_bridge import (
    CONTEXT_FRAME as JOURNEY_CONTEXT_FRAME,
    DECISION_FRAME as JOURNEY_DECISION_FRAME,
    RESULT_FRAME as JOURNEY_RESULT_FRAME,
    load_verifier_registry,
)
from tests.agent_live.dynamic_dual_ai_chaos import (
    JourneyTerminalClassification,
    validate_journey_evidence_artifact,
)
from tests.agent_live.container_process_guard import (
    EXECUTION_ID_ENV,
    RECEIPT_DIR_ENV,
    CleanupReceiptArtifact,
    ContainerProcessGuard,
)
from tests.agent_live.discovery_ledger import (
    DISCOVERY_LEDGER_SCHEMA_VERSION,
    RESULT_CLASSIFICATIONS,
    append_discovery_attempt,
    build_discovery_attempt,
)
from tests.agent_live.coverage_evidence import (
    load_valid_evidence_reference,
    validate_pty_diagnostic_artifact,
)
from tests.agent_live.generate_harness_coverage_ledger import build_ledger


BATCH_MANIFEST_SCHEMA_VERSION = 3
BATCH_RESULT_SCHEMA_VERSION = 2
CLEANUP_RECEIPT_SCHEMA_VERSION = 2
BATCH_SURVIVOR_PROOF_SCHEMA_VERSION = 1
DEFAULT_SHARD_COUNT = 32
FORMAL_EDGE_SHARD_COUNT = 24
FORMAL_JOURNEY_SHARD_COUNT = 8
SHARD_LANES = frozenset({"edge", "journey"})


class ExternalDecisionBlocked(RuntimeError):
    """The external simulator could not provide a decision for this response."""


class DecisionBroker(Protocol):
    def __call__(
        self, shard_id: str, context: Mapping[str, Any]
    ) -> Mapping[str, Any] | Awaitable[Mapping[str, Any]]:
        """Return exactly one decision bound to the supplied response context."""


CommandFactory = Callable[[int, Path, Path, str, int, str], Sequence[str]]


@dataclass(frozen=True)
class TimeoutPolicy:
    shard_seconds: float = 900.0
    decision_seconds: float = 240.0
    cleanup_seconds: float = 5.0


@dataclass(frozen=True)
class FrozenShardSpec:
    shard_id: str
    index: int
    seed: int
    session_id: str
    target_path: str
    target_hash: str
    schedule_id: str
    runtime_root: str
    execution_id: str
    container_cleanup_receipt_dir: str
    command: tuple[str, ...]
    target_ids: tuple[str, ...]
    edge_keys: tuple[str, ...]
    personas: tuple[str, ...]
    goals: tuple[str, ...]
    input_classes: tuple[str, ...]
    sequence_ids: tuple[str, ...]
    factor_ids: tuple[str, ...]
    lane: str = "edge"
    verifier_registry_import: str = ""
    verifier_registry_id: str = ""


@dataclass(frozen=True)
class FrozenBatchManifest:
    manifest_id: str
    batch_id: str
    created_at_ns: int
    manifest_path: str
    discovery_ledger_path: str
    repo_root: str
    revision: Mapping[str, str]
    shard_count: int
    shards: tuple[FrozenShardSpec, ...]
    required_env_names: tuple[str, ...]
    timeout_policy: TimeoutPolicy
    stderr_cap_bytes: int
    worker_runtime: str = "docker"
    formal_profile: bool = False
    scheduler_schema_version: int = CHAOS_SCHEDULE_SCHEMA_VERSION
    discovery_schema_version: int = DISCOVERY_LEDGER_SCHEMA_VERSION
    schema_version: int = BATCH_MANIFEST_SCHEMA_VERSION


@dataclass(frozen=True)
class CleanupReceipt:
    receipt_id: str
    schema_version: int
    shard_id: str
    execution_id: str
    host_proof: Mapping[str, Any]
    container_proof: Mapping[str, Any]
    worker_identity: Mapping[str, Any]
    exit_code: int | None
    actions: tuple[str, ...]
    errors: tuple[str, ...]
    cleaned: bool
    started_at_ns: int
    finished_at_ns: int


@dataclass(frozen=True)
class ShardResult:
    shard_id: str
    classification: str
    attempt_count: int
    started_at_ns: int
    finished_at_ns: int
    exit_code: int | None
    target_hash: str
    response_hashes: tuple[str, ...]
    decision_hashes: tuple[str, ...]
    transcript_hash: str
    schedule_result_hash: str
    evidence_hashes: tuple[str, ...]
    diagnostic_hashes: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    diagnostic_ids: tuple[str, ...]
    stderr_hash: str
    stderr_path: str
    stderr_truncated: bool
    cleanup_receipt_path: str
    cleanup_receipt_hash: str
    reason: str


@dataclass(frozen=True)
class BatchResultIndex:
    index_id: str
    batch_id: str
    manifest_id: str
    revision: Mapping[str, str]
    execution_status: str
    release_status: str
    scheduled: int
    started: int
    completed: int
    discovery_attempt_ids: tuple[str, ...]
    classification_counts: Mapping[str, int]
    batch_survivor_proof: Mapping[str, Any]
    shards: tuple[ShardResult, ...]
    schema_version: int = BATCH_RESULT_SCHEMA_VERSION


@dataclass
class _RunState:
    response_hashes: list[str] = field(default_factory=list)
    decision_hashes: list[str] = field(default_factory=list)
    result_payload: Mapping[str, Any] | None = None
    forced_classification: str = ""
    reason: str = ""


@dataclass(frozen=True)
class _HostCleanupOutcome:
    artifact: CleanupReceiptArtifact | None
    cleaned: bool
    exit_code: int | None
    actions: tuple[str, ...]
    errors: tuple[str, ...]


def freeze_batch_manifest(
    *,
    repo_root: str | Path,
    targets_dir: str | Path,
    manifest_path: str | Path,
    runtime_base: str | Path,
    shard_count: int = DEFAULT_SHARD_COUNT,
    seed_base: int = 10_000,
    command_factory: CommandFactory | None = None,
    expected_revision: Mapping[str, str] | None = None,
    required_env_names: Sequence[str] = ("DEEPSEEK_API_KEY",),
    timeout_policy: TimeoutPolicy = TimeoutPolicy(),
    stderr_cap_bytes: int = 65_536,
    discovery_ledger_path: str | Path | None = None,
    formal_profile: bool = False,
    worker_runtime: str = "docker",
) -> FrozenBatchManifest:
    """Validate every target and atomically freeze the complete spawn manifest."""

    root = Path(repo_root).resolve()
    target_root = Path(targets_dir).resolve()
    output = Path(manifest_path).resolve()
    runtime = Path(runtime_base).resolve()
    discovery_path = Path(
        discovery_ledger_path
        or (root / ".agent" / "discovery" / "attempts.jsonl")
    ).resolve()
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if stderr_cap_bytes < 1:
        raise ValueError("stderr_cap_bytes must be positive")
    if worker_runtime not in {"docker", "linux"}:
        raise ValueError("worker_runtime must be docker or linux")
    if formal_profile and worker_runtime != "linux":
        raise ValueError("formal batches must run inside the Linux control plane")
    if output.exists():
        raise FileExistsError(f"batch manifest is immutable: {output}")
    revision = repository_revision(root)
    if expected_revision is not None and dict(expected_revision) != revision:
        raise ValueError("repository revision does not match the requested batch revision")

    ledger = build_ledger(revision=revision)
    edge_index = {
        str(edge.get("edge_key") or ""): edge
        for edge in ledger.get("edges") or ()
    }
    factory = command_factory or _default_command_factory(root, worker_runtime=worker_runtime)
    batch_nonce = hashlib.sha256(
        f"{revision}|{seed_base}|{shard_count}|{time.time_ns()}".encode()
    ).hexdigest()[:16]
    shards: list[FrozenShardSpec] = []
    for index in range(1, shard_count + 1):
        target_path = target_root / f"{index:02d}.json"
        try:
            target_path = target_path.resolve(strict=True)
            target_path.relative_to(target_root)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(
                f"target file must be contained by the frozen target directory: {target_path}"
            ) from exc
        payload = _load_json(target_path)
        lane = _target_lane(payload)
        shard_id = f"{batch_nonce}-{index:02d}"
        session_id = f"batch-{shard_id}"
        shard_runtime = runtime / shard_id
        target_hash = _sha256_file(target_path)
        execution_id = "anychain-chaos-" + hashlib.sha256(
            f"{batch_nonce}|{index}|{session_id}|{target_hash}".encode("utf-8")
        ).hexdigest()
        container_receipt_dir = shard_runtime / "container-cleanup-receipts"
        verifier_registry_import = ""
        verifier_registry_id = ""
        if lane == "edge":
            targets = _target_rows(payload)
            if not targets:
                raise ValueError(f"target file has no scheduled targets: {target_path}")
            schedule = build_chaos_schedule(
                ledger,
                revision=revision,
                seed=seed_base + index,
                targets=targets,
            )
            validate_chaos_schedule(schedule, ledger=ledger, revision=revision)
            target_ids = tuple(target.target_id for target in schedule.targets)
            edge_keys = tuple(target.edge_key for target in schedule.targets)
            personas = tuple(target.persona for target in schedule.targets)
            goals = tuple(target.goal for target in schedule.targets)
            edges = [edge_index[target.edge_key] for target in schedule.targets]
            input_classes = tuple(str(edge.get("input_class") or "") for edge in edges)
            sequence_ids = tuple(
                target.sequence_id or f"target:{target.target_id}"
                for target in schedule.targets
            )
            factor_ids = tuple(
                "|".join((*target.tuple_ids, target.scenario_id))
                or f"edge:{target.edge_key}"
                for target in schedule.targets
            )
        else:
            journey, verifier_registry_import = _journey_target(payload)
            schedule = build_journey_schedule(
                revision=revision,
                seed=seed_base + index,
                journey=journey,
            )
            validate_journey_schedule(schedule, revision=revision)
            registry = load_verifier_registry(verifier_registry_import)
            verifier_registry_id = registry.registry_id
            target_ids = (schedule.journey_id,)
            edge_keys = ()
            personas = (schedule.persona,)
            goals = (schedule.mission,)
            input_classes = ("response_driven_journey",)
            sequence_ids = (f"journey:{schedule.journey_id}",)
            factor_ids = tuple(schedule.allowed_risk_factors) or (
                f"journey:{schedule.journey_id}",
            )
        command = tuple(str(item) for item in factory(
            index, target_path, shard_runtime, session_id, seed_base + index,
            schedule.schedule_id,
        ))
        command = _bind_docker_execution_environment(
            command,
            execution_id=execution_id,
            container_receipt_dir=container_receipt_dir,
            repo_root=root,
        )
        if not command or any(not item for item in command):
            raise ValueError(f"shard {index:02d} has an invalid command")
        _reject_secret_bearing_command(command)
        shards.append(FrozenShardSpec(
            shard_id=shard_id,
            index=index,
            seed=seed_base + index,
            session_id=session_id,
            target_path=str(target_path),
            target_hash=target_hash,
            schedule_id=schedule.schedule_id,
            runtime_root=str(shard_runtime),
            execution_id=execution_id,
            container_cleanup_receipt_dir=str(container_receipt_dir),
            command=command,
            target_ids=target_ids,
            edge_keys=edge_keys,
            personas=personas,
            goals=goals,
            input_classes=input_classes,
            sequence_ids=sequence_ids,
            factor_ids=factor_ids,
            lane=lane,
            verifier_registry_import=verifier_registry_import,
            verifier_registry_id=verifier_registry_id,
        ))

    lane_counts = {lane: sum(item.lane == lane for item in shards) for lane in SHARD_LANES}
    if formal_profile and lane_counts != {
        "edge": FORMAL_EDGE_SHARD_COUNT,
        "journey": FORMAL_JOURNEY_SHARD_COUNT,
    }:
        raise ValueError(
            "formal batch requires exactly 24 edge shards and 8 journey shards"
        )

    unsigned = {
        "created_at_ns": time.time_ns(),
        "manifest_path": str(output),
        "discovery_ledger_path": str(discovery_path),
        "repo_root": str(root),
        "revision": revision,
        "shard_count": shard_count,
        "shards": [_shard_payload(item) for item in shards],
        "required_env_names": sorted({str(item) for item in required_env_names}),
        "timeout_policy": asdict(timeout_policy),
        "stderr_cap_bytes": stderr_cap_bytes,
        "worker_runtime": worker_runtime,
        "formal_profile": formal_profile,
        "scheduler_schema_version": CHAOS_SCHEDULE_SCHEMA_VERSION,
        "discovery_schema_version": DISCOVERY_LEDGER_SCHEMA_VERSION,
        "schema_version": BATCH_MANIFEST_SCHEMA_VERSION,
    }
    batch_id = _content_hash(unsigned)
    identity = {**unsigned, "batch_id": batch_id}
    manifest = FrozenBatchManifest(
        manifest_id=_content_hash(identity),
        batch_id=batch_id,
        created_at_ns=int(unsigned["created_at_ns"]),
        manifest_path=str(output),
        discovery_ledger_path=str(discovery_path),
        repo_root=str(root),
        revision=revision,
        shard_count=shard_count,
        shards=tuple(shards),
        required_env_names=tuple(unsigned["required_env_names"]),
        timeout_policy=timeout_policy,
        stderr_cap_bytes=stderr_cap_bytes,
        worker_runtime=worker_runtime,
        formal_profile=formal_profile,
    )
    _write_immutable_json(output, manifest_payload(manifest))
    validate_frozen_manifest(manifest, manifest_path=output)
    return manifest


def load_frozen_manifest(path: str | Path) -> FrozenBatchManifest:
    payload = _load_json(Path(path))
    manifest = FrozenBatchManifest(
        manifest_id=str(payload["manifest_id"]),
        batch_id=str(payload["batch_id"]),
        created_at_ns=int(payload["created_at_ns"]),
        manifest_path=str(payload["manifest_path"]),
        discovery_ledger_path=str(payload["discovery_ledger_path"]),
        repo_root=str(payload["repo_root"]),
        revision=dict(payload["revision"]),
        shard_count=int(payload["shard_count"]),
        shards=tuple(FrozenShardSpec(
            **{**row, "command": tuple(row["command"]),
               "target_ids": tuple(row["target_ids"]),
               "edge_keys": tuple(row["edge_keys"]),
               "personas": tuple(row["personas"]),
               "goals": tuple(row["goals"]),
               "input_classes": tuple(row["input_classes"]),
               "sequence_ids": tuple(row["sequence_ids"]),
               "factor_ids": tuple(row["factor_ids"])}
        ) for row in payload["shards"]),
        required_env_names=tuple(payload["required_env_names"]),
        timeout_policy=TimeoutPolicy(**payload["timeout_policy"]),
        stderr_cap_bytes=int(payload["stderr_cap_bytes"]),
        worker_runtime=str(payload.get("worker_runtime") or "docker"),
        formal_profile=bool(payload.get("formal_profile", False)),
        scheduler_schema_version=int(payload["scheduler_schema_version"]),
        discovery_schema_version=int(payload["discovery_schema_version"]),
        schema_version=int(payload["schema_version"]),
    )
    validate_frozen_manifest(manifest, manifest_path=Path(path))
    return manifest


def validate_frozen_manifest(
    manifest: FrozenBatchManifest,
    *,
    manifest_path: str | Path | None = None,
    verify_revision: bool = True,
) -> None:
    if manifest.schema_version != BATCH_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported batch manifest schema")
    if manifest.scheduler_schema_version != CHAOS_SCHEDULE_SCHEMA_VERSION:
        raise ValueError("scheduler schema changed after manifest creation")
    if manifest.discovery_schema_version != DISCOVERY_LEDGER_SCHEMA_VERSION:
        raise ValueError("discovery schema changed after manifest creation")
    if manifest.shard_count != len(manifest.shards):
        raise ValueError("manifest shard_count does not match its shard records")
    if manifest.worker_runtime not in {"docker", "linux"}:
        raise ValueError("manifest worker runtime is invalid")
    if manifest.formal_profile and manifest.worker_runtime != "linux":
        raise ValueError("formal manifest escaped the Linux control plane")
    if any(shard.lane not in SHARD_LANES for shard in manifest.shards):
        raise ValueError("manifest contains an unsupported shard lane")
    lane_counts = {
        lane: sum(shard.lane == lane for shard in manifest.shards)
        for lane in SHARD_LANES
    }
    if manifest.formal_profile and lane_counts != {
        "edge": FORMAL_EDGE_SHARD_COUNT,
        "journey": FORMAL_JOURNEY_SHARD_COUNT,
    }:
        raise ValueError("formal batch lane cardinality changed after manifest creation")
    indexes = [item.index for item in manifest.shards]
    if indexes != list(range(1, manifest.shard_count + 1)):
        raise ValueError("manifest shard indexes are not contiguous")
    for attribute in (
        "shard_id", "session_id", "runtime_root", "target_path", "execution_id",
        "container_cleanup_receipt_dir",
    ):
        values = [getattr(item, attribute) for item in manifest.shards]
        if len(values) != len(set(values)):
            raise ValueError(f"manifest has duplicate {attribute}")
    for shard in manifest.shards:
        target = Path(shard.target_path)
        if _sha256_file(target) != shard.target_hash:
            raise ValueError(f"frozen target changed: {target}")
        if not shard.execution_id.startswith("anychain-chaos-"):
            raise ValueError(f"shard execution id is invalid: {shard.shard_id}")
        expected_receipt_dir = Path(shard.runtime_root) / "container-cleanup-receipts"
        if Path(shard.container_cleanup_receipt_dir) != expected_receipt_dir:
            raise ValueError(f"container cleanup receipt directory changed: {shard.shard_id}")
        _validate_bound_docker_environment(shard, Path(manifest.repo_root))
        if manifest.formal_profile:
            _validate_formal_linux_command(shard, Path(manifest.repo_root))
    ledger = build_ledger(revision=manifest.revision)
    for shard in manifest.shards:
        target_payload = _load_json(Path(shard.target_path))
        if _target_lane(target_payload) != shard.lane:
            raise ValueError(f"frozen shard lane changed: {shard.shard_id}")
        if shard.lane == "edge":
            schedule = build_chaos_schedule(
                ledger,
                revision=manifest.revision,
                seed=shard.seed,
                targets=_target_rows(target_payload),
            )
            validate_chaos_schedule(schedule, ledger=ledger, revision=manifest.revision)
            if shard.verifier_registry_import or shard.verifier_registry_id:
                raise ValueError("edge shard cannot bind a Journey verifier registry")
        else:
            journey, registry_import = _journey_target(target_payload)
            schedule = build_journey_schedule(
                revision=manifest.revision,
                seed=shard.seed,
                journey=journey,
            )
            validate_journey_schedule(schedule, revision=manifest.revision)
            if registry_import != shard.verifier_registry_import:
                raise ValueError("frozen Journey verifier import changed")
            if load_verifier_registry(registry_import).registry_id != shard.verifier_registry_id:
                raise ValueError("frozen Journey verifier registry identity changed")
        if schedule.schedule_id != shard.schedule_id:
            raise ValueError(f"frozen scheduler identity changed: {shard.shard_id}")
    unsigned = _manifest_unsigned_payload(manifest)
    if _content_hash(unsigned) != manifest.batch_id:
        raise ValueError("batch manifest content hash is stale")
    if _content_hash({**unsigned, "batch_id": manifest.batch_id}) != manifest.manifest_id:
        raise ValueError("batch manifest identity is stale")
    if verify_revision and repository_revision(manifest.repo_root) != dict(manifest.revision):
        raise ValueError("repository revision changed after manifest freeze")
    path = Path(manifest_path or manifest.manifest_path).resolve()
    if path != Path(manifest.manifest_path).resolve():
        raise ValueError("manifest path does not match its frozen identity")
    if not path.is_file() or path.stat().st_mode & 0o222:
        raise ValueError("frozen manifest must exist and be read-only")
    if _load_json(path) != manifest_payload(manifest):
        raise ValueError("on-disk manifest does not match the in-memory manifest")


async def run_batch(
    manifest: FrozenBatchManifest | str | Path,
    *,
    broker: DecisionBroker,
    result_index_path: str | Path,
    verify_revision: bool = True,
    interruption_event: asyncio.Event | None = None,
) -> BatchResultIndex:
    """Run every frozen shard once and wait for every shard to terminate."""

    frozen = load_frozen_manifest(manifest) if isinstance(manifest, (str, Path)) else manifest
    _require_linux_control_plane()
    validate_frozen_manifest(frozen, verify_revision=verify_revision)
    index_path = Path(result_index_path).resolve()
    if index_path.exists():
        raise FileExistsError(f"batch result index is immutable: {index_path}")

    tasks = [asyncio.create_task(_run_shard(frozen, shard, broker)) for shard in frozen.shards]
    batch_interrupted = False
    try:
        gather = asyncio.gather(*tasks, return_exceptions=True)
        if interruption_event is None:
            raw_results = await gather
        else:
            interruption_wait = asyncio.create_task(interruption_event.wait())
            try:
                done, _pending = await asyncio.wait(
                    (gather, interruption_wait),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if interruption_wait in done and interruption_event.is_set():
                    batch_interrupted = True
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                raw_results = await gather
            finally:
                if not interruption_wait.done():
                    interruption_wait.cancel()
                await asyncio.gather(interruption_wait, return_exceptions=True)
    except asyncio.CancelledError:
        batch_interrupted = True
        for task in tasks:
            if not task.done():
                task.cancel()
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    results: list[ShardResult] = []
    for shard, raw in zip(frozen.shards, raw_results, strict=True):
        if isinstance(raw, BaseException):
            results.append(_synthetic_interruption_result(frozen, shard, raw))
        else:
            results.append(raw)
    validated_results: list[ShardResult] = []
    for shard, result in zip(frozen.shards, results, strict=True):
        try:
            _validate_composite_cleanup_receipt(shard, result)
        except Exception as exc:
            result = replace(
                result,
                classification="infrastructure_interrupted",
                reason=f"cleanup receipt validation failed: {type(exc).__name__}: {exc}",
            )
        validated_results.append(result)
    results = validated_results

    batch_survivor_proof = await _final_batch_survivor_proof(frozen)
    if not batch_survivor_proof["cleaned"]:
        failed_execution_ids = {
            str(row.get("execution_id") or "")
            for scan in batch_survivor_proof["scans"]
            for row in scan["survivors"]
        }
        fail_all = bool(batch_survivor_proof["errors"])
        results = [
            replace(
                result,
                classification="infrastructure_interrupted",
                reason="final batch-wide execution-id survivor proof failed",
            )
            if fail_all or shard.execution_id in failed_execution_ids
            else result
            for shard, result in zip(frozen.shards, results, strict=True)
        ]
    validate_frozen_manifest(frozen, verify_revision=verify_revision)
    attempt_ids = _append_discovery_results(frozen, results)
    counts = {name: 0 for name in sorted(RESULT_CLASSIFICATIONS)}
    for result in results:
        counts[result.classification] += 1
    cleanup_complete = bool(batch_survivor_proof["cleaned"]) and all(
        _load_json(Path(result.cleanup_receipt_path)).get("cleaned") is True
        for result in results
    )
    execution_status = (
        "discovery_complete"
        if cleanup_complete and not batch_interrupted
        else "infrastructure_interrupted"
    )
    unsigned = {
        "batch_id": frozen.batch_id,
        "manifest_id": frozen.manifest_id,
        "revision": dict(frozen.revision),
        "execution_status": execution_status,
        "release_status": "not_evaluated",
        "scheduled": frozen.shard_count,
        "started": frozen.shard_count,
        "completed": len(results),
        "discovery_attempt_ids": list(attempt_ids),
        "classification_counts": counts,
        "batch_survivor_proof": batch_survivor_proof,
        "shards": [_result_payload(item) for item in results],
        "schema_version": BATCH_RESULT_SCHEMA_VERSION,
    }
    index = BatchResultIndex(
        index_id=_content_hash(unsigned),
        batch_id=frozen.batch_id,
        manifest_id=frozen.manifest_id,
        revision=dict(frozen.revision),
        execution_status=execution_status,
        release_status="not_evaluated",
        scheduled=frozen.shard_count,
        started=frozen.shard_count,
        completed=len(results),
        discovery_attempt_ids=tuple(attempt_ids),
        classification_counts=counts,
        batch_survivor_proof=batch_survivor_proof,
        shards=tuple(results),
    )
    _write_immutable_json(index_path, result_index_payload(index))
    return index


async def _run_shard(
    manifest: FrozenBatchManifest,
    shard: FrozenShardSpec,
    broker: DecisionBroker,
) -> ShardResult:
    started = time.time_ns()
    runtime_root = Path(shard.runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=False)
    stderr_path = runtime_root / "worker.stderr.redacted.txt"
    receipt_path = runtime_root / "cleanup-receipt.json"
    host_receipt_dir = runtime_root / "host-cleanup-receipts"
    host_guard = ContainerProcessGuard(
        shard.execution_id,
        receipt_dir=host_receipt_dir,
        term_grace_seconds=max(manifest.timeout_policy.cleanup_seconds / 2, 0.01),
        kill_grace_seconds=max(manifest.timeout_policy.cleanup_seconds / 2, 0.01),
        scan_interval_seconds=min(
            0.05, max(manifest.timeout_policy.cleanup_seconds / 20, 0.005)
        ),
    )
    state = _RunState()
    process: asyncio.subprocess.Process | None = None
    worker_identity: Mapping[str, Any] = {}
    stderr_task: asyncio.Task[tuple[bytes, bool]] | None = None
    exit_code: int | None = None
    cleanup_outcome = _HostCleanupOutcome(None, False, None, (), ())
    stderr_bytes = b""
    stderr_truncated = False
    try:
        worker_env = os.environ.copy()
        worker_env[EXECUTION_ID_ENV] = shard.execution_id
        worker_env[RECEIPT_DIR_ENV] = shard.container_cleanup_receipt_dir
        process = await asyncio.create_subprocess_exec(
            *shard.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=manifest.repo_root,
            start_new_session=True,
            env=worker_env,
        )
        identity = host_guard.register_pid(process.pid, role="host_worker")
        if _is_docker_exec_command(shard.command):
            host_guard.register_pid(process.pid, role="docker_exec")
        worker_identity = identity.payload()
        assert process.stderr is not None
        stderr_task = asyncio.create_task(
            _read_capped(process.stderr, manifest.stderr_cap_bytes)
        )
        await asyncio.wait_for(
            _exchange_frames(
                process, manifest, shard, broker, state, manifest.timeout_policy
            ),
            timeout=manifest.timeout_policy.shard_seconds,
        )
        exit_code = await asyncio.wait_for(
            process.wait(), timeout=manifest.timeout_policy.shard_seconds
        )
    except ExternalDecisionBlocked as exc:
        state.forced_classification = "externally_blocked"
        state.reason = str(exc)
    except _SimulatorInvalid as exc:
        state.forced_classification = "simulator_invalid"
        state.reason = str(exc)
    except asyncio.TimeoutError:
        state.forced_classification = "infrastructure_interrupted"
        state.reason = "shard wall-clock timeout"
    except asyncio.CancelledError:
        state.forced_classification = "infrastructure_interrupted"
        state.reason = "batch cancellation requested"
    except Exception as exc:
        state.forced_classification = "infrastructure_interrupted"
        state.reason = f"{type(exc).__name__}: {exc}"
    finally:
        cleanup_outcome = await _cleanup_process(
            process,
            host_guard,
            manifest.timeout_policy.cleanup_seconds,
        )
        exit_code = cleanup_outcome.exit_code
        if stderr_task is not None:
            try:
                stderr_bytes, stderr_truncated = await stderr_task
            except Exception as exc:
                state.forced_classification = "infrastructure_interrupted"
                state.reason = f"stderr collection failed: {type(exc).__name__}: {exc}"
        if not cleanup_outcome.cleaned:
            state.forced_classification = "infrastructure_interrupted"
            state.reason = "host worker cleanup could not prove process termination"

    redacted_stderr = str(redact(stderr_bytes.decode("utf-8", errors="replace")))
    persisted_stderr = redacted_stderr.encode("utf-8")
    if len(persisted_stderr) > manifest.stderr_cap_bytes:
        persisted_stderr = persisted_stderr[:manifest.stderr_cap_bytes]
        stderr_truncated = True
    _atomic_write(stderr_path, persisted_stderr, mode=0o600)
    cleanup_errors = list(cleanup_outcome.errors)
    host_proof: Mapping[str, Any] = {}
    container_proof: Mapping[str, Any] = {}
    if cleanup_outcome.artifact is not None:
        try:
            host_proof = _validated_guard_proof(
                cleanup_outcome.artifact.path,
                execution_id=shard.execution_id,
                required_roles=(
                    ("host_worker", "docker_exec")
                    if _is_docker_exec_command(shard.command)
                    else ("host_worker",)
                ),
                allowed_roots=(host_receipt_dir,),
            )
        except Exception as exc:
            cleanup_errors.append(f"host proof invalid: {type(exc).__name__}: {exc}")
    else:
        cleanup_errors.append("host process guard did not produce a receipt")
    try:
        inner_path = _single_guard_receipt_path(
            Path(shard.container_cleanup_receipt_dir)
        )
        container_proof = _validated_guard_proof(
            inner_path,
            execution_id=shard.execution_id,
            required_roles=("container_bridge", "agent_process_group_leader"),
            allowed_roots=(Path(shard.container_cleanup_receipt_dir),),
        )
    except Exception as exc:
        cleanup_errors.append(f"container proof invalid: {type(exc).__name__}: {exc}")

    composite_cleaned = (
        cleanup_outcome.cleaned
        and bool(host_proof.get("cleaned"))
        and bool(container_proof.get("cleaned"))
        and not cleanup_errors
    )
    receipt_unsigned = {
        "schema_version": CLEANUP_RECEIPT_SCHEMA_VERSION,
        "shard_id": shard.shard_id,
        "execution_id": shard.execution_id,
        "host_proof": host_proof,
        "container_proof": container_proof,
        "worker_identity": dict(worker_identity),
        "exit_code": exit_code,
        "actions": list(cleanup_outcome.actions),
        "errors": cleanup_errors,
        "cleaned": composite_cleaned,
        "started_at_ns": started,
        "finished_at_ns": time.time_ns(),
    }
    receipt = CleanupReceipt(
        receipt_id=_content_hash(receipt_unsigned),
        schema_version=CLEANUP_RECEIPT_SCHEMA_VERSION,
        shard_id=shard.shard_id,
        execution_id=shard.execution_id,
        host_proof=host_proof,
        container_proof=container_proof,
        worker_identity=worker_identity,
        exit_code=exit_code,
        actions=cleanup_outcome.actions,
        errors=tuple(cleanup_errors),
        cleaned=receipt_unsigned["cleaned"],
        started_at_ns=started,
        finished_at_ns=receipt_unsigned["finished_at_ns"],
    )
    _write_immutable_json(receipt_path, asdict(receipt))
    artifacts = _artifact_hashes(manifest, state.result_payload, shard)
    classification, reason = _classify(
        manifest, shard, state, exit_code, receipt.cleaned, artifacts
    )
    diagnostic_ids = tuple(artifacts["diagnostic_ids"]) or (receipt.receipt_id,)
    return ShardResult(
        shard_id=shard.shard_id,
        classification=classification,
        attempt_count=1,
        started_at_ns=started,
        finished_at_ns=time.time_ns(),
        exit_code=exit_code,
        target_hash=shard.target_hash,
        response_hashes=tuple(state.response_hashes),
        decision_hashes=tuple(state.decision_hashes),
        transcript_hash=artifacts["transcript_hash"],
        schedule_result_hash=artifacts["schedule_result_hash"],
        evidence_hashes=tuple(artifacts["evidence_hashes"]),
        diagnostic_hashes=tuple(artifacts["diagnostic_hashes"]),
        evidence_ids=tuple(artifacts["evidence_ids"]),
        diagnostic_ids=diagnostic_ids,
        stderr_hash=_sha256_file(stderr_path),
        stderr_path=str(stderr_path),
        stderr_truncated=stderr_truncated,
        cleanup_receipt_path=str(receipt_path),
        cleanup_receipt_hash=_sha256_file(receipt_path),
        reason=str(redact(reason))[:1000],
    )


async def _exchange_frames(
    process: asyncio.subprocess.Process,
    manifest: FrozenBatchManifest,
    shard: FrozenShardSpec,
    broker: DecisionBroker,
    state: _RunState,
    policy: TimeoutPolicy,
) -> None:
    assert process.stdout is not None
    assert process.stdin is not None
    while True:
        line = await process.stdout.readline()
        if not line:
            return
        text = line.decode("utf-8", errors="replace").rstrip("\r\n")
        context_frame = JOURNEY_CONTEXT_FRAME if shard.lane == "journey" else CONTEXT_FRAME
        decision_frame = JOURNEY_DECISION_FRAME if shard.lane == "journey" else DECISION_FRAME
        result_frame = JOURNEY_RESULT_FRAME if shard.lane == "journey" else RESULT_FRAME
        if text.startswith(context_frame):
            context = _parse_frame(text, context_frame)
            _validate_context_frame(shard, context, len(state.response_hashes))
            response_hash = str(context.get("previous_response_hash") or "")
            if not response_hash:
                raise RuntimeError("simulator context has no response hash")
            state.response_hashes.append(response_hash)
            try:
                decision = await _call_broker_with_timeout(
                    broker,
                    shard.shard_id,
                    context,
                    timeout_seconds=policy.decision_seconds,
                )
            except ExternalDecisionBlocked:
                raise
            except asyncio.TimeoutError as exc:
                raise ExternalDecisionBlocked("simulator decision timed out") from exc
            if decision is None:
                raise ExternalDecisionBlocked("simulator returned no decision")
            normalized = _validate_decision(context, decision, lane=shard.lane)
            state.decision_hashes.append(_content_hash(normalized))
            process.stdin.write(
                (decision_frame + _canonical_json(normalized) + "\n").encode("utf-8")
            )
            await process.stdin.drain()
        elif text.startswith(result_frame):
            if state.result_payload is not None:
                raise RuntimeError("worker emitted more than one terminal result frame")
            result = _parse_frame(text, result_frame)
            _validate_result_frame(manifest, shard, result)
            state.result_payload = result


async def _call_broker_with_timeout(
    broker: DecisionBroker,
    shard_id: str,
    context: Mapping[str, Any],
    *,
    timeout_seconds: float,
) -> Mapping[str, Any]:
    async_call = inspect.iscoroutinefunction(broker) or inspect.iscoroutinefunction(
        getattr(broker, "__call__", None)
    )
    if async_call:
        return await asyncio.wait_for(
            broker(shard_id, context), timeout=timeout_seconds
        )
    # A synchronous Codex broker can block on external input. Running it on the
    # event-loop thread would stall every shard and defeat independent timeouts.
    result = await asyncio.wait_for(
        asyncio.to_thread(broker, shard_id, context),
        timeout=timeout_seconds,
    )
    if inspect.isawaitable(result):
        return await asyncio.wait_for(result, timeout=timeout_seconds)
    return result


class _SimulatorInvalid(RuntimeError):
    pass


def _validate_decision(
    context: Mapping[str, Any], decision: Mapping[str, Any], *, lane: str = "edge"
) -> dict[str, Any]:
    if not isinstance(decision, Mapping):
        raise _SimulatorInvalid("simulator decision is not an object")
    if lane == "journey":
        required = {
            "previous_response_hash", "user_message", "persona", "mission",
            "rationale", "risk_factor_ids",
        }
    else:
        required = {
            "previous_response_hash", "user_message", "persona", "goal",
            "rationale", "target_coverage_ids",
        }
    missing = required.difference(decision)
    if missing:
        raise _SimulatorInvalid(f"simulator decision is missing: {sorted(missing)}")
    expected_hash = str(context.get("previous_response_hash") or "")
    if str(decision.get("previous_response_hash") or "") != expected_hash:
        raise _SimulatorInvalid("simulator decision is bound to a stale response")
    if lane == "journey":
        schedule = context.get("schedule") or {}
        if str(decision.get("persona") or "") != str(schedule.get("persona") or ""):
            raise _SimulatorInvalid("simulator changed the Journey persona")
        if str(decision.get("mission") or "") != str(schedule.get("mission") or ""):
            raise _SimulatorInvalid("simulator changed the Journey mission")
        risks = [str(item) for item in decision.get("risk_factor_ids") or ()]
        if len(risks) != len(set(risks)) or set(risks) - set(
            schedule.get("allowed_risk_factors") or ()
        ):
            raise _SimulatorInvalid("simulator selected invalid Journey risk factors")
    else:
        target = context.get("scheduled_target") or {}
        if str(decision.get("persona") or "") != str(target.get("persona") or ""):
            raise _SimulatorInvalid("simulator changed the scheduled persona")
        if str(decision.get("goal") or "") != str(target.get("goal") or ""):
            raise _SimulatorInvalid("simulator changed the scheduled goal")
        coverage_ids = [str(item) for item in decision.get("target_coverage_ids") or ()]
        if str(target.get("edge_key") or "") not in coverage_ids:
            raise _SimulatorInvalid("simulator dropped the scheduled coverage edge")
    if not str(decision.get("user_message") or ""):
        raise _SimulatorInvalid("simulator decision has an empty user message")
    return dict(decision)


def _validate_context_frame(
    shard: FrozenShardSpec,
    context: Mapping[str, Any],
    target_index: int,
) -> None:
    if str(context.get("session_id") or "") != shard.session_id:
        raise RuntimeError("simulator context belongs to another session")
    if shard.lane == "journey":
        schedule = context.get("schedule")
        if not isinstance(schedule, Mapping):
            raise RuntimeError("Journey simulator context has no frozen schedule")
        if str(schedule.get("schedule_id") or "") != shard.schedule_id:
            raise RuntimeError("Journey simulator context schedule identity changed")
        if str(schedule.get("persona") or "") != shard.personas[0]:
            raise RuntimeError("Journey simulator context persona changed")
        if str(schedule.get("mission") or "") != shard.goals[0]:
            raise RuntimeError("Journey simulator context mission changed")
        if int(context.get("turn_index") or 0) != target_index + 1:
            raise RuntimeError("Journey simulator context turn order is invalid")
        return
    if target_index >= len(shard.target_ids):
        raise RuntimeError("worker requested more decisions than the frozen schedule")
    target = context.get("scheduled_target")
    if not isinstance(target, Mapping):
        raise RuntimeError("simulator context has no scheduled target")
    expected = {
        "target_id": shard.target_ids[target_index],
        "edge_key": shard.edge_keys[target_index],
        "persona": shard.personas[target_index],
        "goal": shard.goals[target_index],
    }
    for key, value in expected.items():
        if str(target.get(key) or "") != value:
            raise RuntimeError(f"simulator context {key} differs from the frozen target")


async def _cleanup_process(
    process: asyncio.subprocess.Process | None,
    guard: ContainerProcessGuard,
    cleanup_seconds: float,
) -> _HostCleanupOutcome:
    actions: list[str] = []
    errors: list[str] = []
    if process is not None and process.stdin is not None and not process.stdin.is_closing():
        process.stdin.close()
        actions.append("stdin_closed")
    if process is None:
        actions.append("spawn_failed_no_process")
    reapers = (
        {process.pid: lambda: process.returncode}
        if process is not None
        else {}
    )
    artifact: CleanupReceiptArtifact | None = None
    try:
        artifact = await asyncio.to_thread(guard.cleanup, reapers=reapers)
        actions.append("host_process_guard_completed")
    except Exception as exc:
        errors.append(f"host process guard failed: {type(exc).__name__}: {exc}")
    if process is not None and process.returncode is None:
        try:
            await asyncio.wait_for(process.wait(), timeout=cleanup_seconds)
        except asyncio.TimeoutError:
            actions.append("worker_wait_timeout_after_guard")
            errors.append("worker process was not reaped after host guard cleanup")
    worker_reaped = process is not None and process.returncode is not None
    if worker_reaped:
        actions.append("worker_reaped")
    cleaned = bool(artifact and artifact.cleaned and worker_reaped and not errors)
    return _HostCleanupOutcome(
        artifact=artifact,
        cleaned=cleaned,
        exit_code=process.returncode if process is not None else None,
        actions=tuple(actions),
        errors=tuple(errors),
    )


async def _read_capped(
    stream: asyncio.StreamReader, cap: int
) -> tuple[bytes, bool]:
    retained = bytearray()
    truncated = False
    while True:
        chunk = await stream.read(8192)
        if not chunk:
            break
        remaining = cap - len(retained)
        if remaining > 0:
            retained.extend(chunk[:remaining])
        if len(chunk) > max(remaining, 0):
            truncated = True
    return bytes(retained), truncated


def _classify(
    manifest: FrozenBatchManifest,
    shard: FrozenShardSpec,
    state: _RunState,
    exit_code: int | None,
    cleaned: bool,
    artifacts: Mapping[str, Any],
) -> tuple[str, str]:
    if not cleaned:
        return "infrastructure_interrupted", "worker process cleanup was incomplete"
    if state.forced_classification:
        return state.forced_classification, state.reason
    if artifacts.get("validation_error"):
        return "infrastructure_interrupted", str(artifacts["validation_error"])
    if artifacts.get("product_failure"):
        return "product_failed", str(artifacts["product_failure"])
    if shard.lane == "journey":
        status = str(artifacts.get("execution_status") or "")
        if status == JourneyTerminalClassification.PASSED.value and exit_code == 0:
            if not state.response_hashes or len(state.response_hashes) != len(
                state.decision_hashes
            ):
                return "infrastructure_interrupted", "Journey decision ledger is incomplete"
            return "passed", "Journey terminal postconditions passed"
        if status == JourneyTerminalClassification.PRODUCT_FAILED.value:
            return "product_failed", str(artifacts.get("product_failure") or status)
        if status == JourneyTerminalClassification.SIMULATOR_INVALID.value:
            return "simulator_invalid", status
        if status == JourneyTerminalClassification.EXTERNALLY_BLOCKED.value:
            return "externally_blocked", status
        return "infrastructure_interrupted", state.reason or status or "Journey did not finish"
    if (
        exit_code == 0
        and artifacts.get("execution_status") == "complete"
        and len(state.response_hashes) == len(shard.target_ids)
        and len(state.decision_hashes) == len(shard.target_ids)
    ):
        return "passed", "all scheduled targets passed"
    return "infrastructure_interrupted", state.reason or "worker did not complete cleanly"


def _artifact_hashes(
    manifest: FrozenBatchManifest,
    payload: Mapping[str, Any] | None,
    shard: FrozenShardSpec | None = None,
) -> dict[str, Any]:
    data = payload if isinstance(payload, Mapping) else {}
    empty = {
        "transcript_hash": "", "schedule_result_hash": "",
        "evidence_hashes": (), "diagnostic_hashes": (),
        "evidence_ids": (), "diagnostic_ids": (),
        "execution_status": "", "product_failure": "", "validation_error": "",
    }
    if shard is None:
        return empty
    if shard.lane == "journey":
        return _journey_artifact_hashes(manifest, data, shard, empty)
    try:
        runtime_roots = _worker_runtime_roots(manifest, shard)
        schedule_path = _known_artifact_path(
            manifest, shard, data, "schedule_path", "schedule.json"
        )
        result_path = _known_artifact_path(
            manifest, shard, data, "schedule_result_path", "schedule-result.json"
        )
        transcript = _known_artifact_path(
            manifest, shard, data, "transcript_path", "transcript.txt"
        )
        for name, path in (
            ("schedule", schedule_path),
            ("schedule result", result_path),
            ("transcript", transcript),
        ):
            if path is None or not path.is_file():
                raise ValueError(f"worker {name} is missing")
            _require_contained(path, runtime_roots, label=name)

        expected_schedule = build_chaos_schedule(
            build_ledger(revision=manifest.revision),
            revision=manifest.revision,
            seed=shard.seed,
            targets=_target_rows(_load_json(Path(shard.target_path))),
        )
        if _load_json(schedule_path) != schedule_payload(expected_schedule):
            raise ValueError("worker schedule does not match the frozen authoritative schedule")
        schedule_result = _load_json(result_path)
        _validate_schedule_result(manifest, shard, schedule_result, data, runtime_roots)

        ledger = build_ledger(revision=manifest.revision)
        edge_index = {
            str(edge.get("edge_key") or ""): edge
            for edge in ledger.get("edges") or ()
        }
        evidence_paths = [
            _resolve_required_path(manifest, item, runtime_roots, label="evidence")
            for item in data.get("evidence_paths") or ()
        ]
        evidence_hashes: list[str] = []
        evidence_ids: list[str] = []
        for path in evidence_paths:
            artifact = _load_json(path)
            edge_key = str(artifact.get("edge_key") or "")
            if edge_key not in shard.edge_keys:
                raise ValueError("evidence references a target outside this shard")
            edge = edge_index.get(edge_key)
            if edge is None:
                raise ValueError("evidence references an unknown authoritative edge")
            validated, reason = load_valid_evidence_reference(
                str(path), edge=edge, revision=manifest.revision
            )
            if validated is None:
                raise ValueError(f"invalid evidence artifact: {reason}")
            evidence_id = str(validated.get("evidence_id") or "")
            if not evidence_id:
                raise ValueError("validated evidence has no evidence_id")
            evidence_ids.append(evidence_id)
            evidence_hashes.append(_sha256_file(path))

        diagnostic_paths = _diagnostic_paths_from_result(
            manifest, schedule_result, runtime_roots
        )
        diagnostic_hashes: list[str] = []
        diagnostic_ids: list[str] = []
        product_failure = ""
        for row, path in diagnostic_paths:
            artifact = _load_json(path)
            valid, reason = validate_pty_diagnostic_artifact(artifact)
            if not valid:
                raise ValueError(f"invalid diagnostic artifact: {reason}")
            if dict(artifact.get("revision") or {}) != dict(manifest.revision):
                raise ValueError("diagnostic repository revision mismatch")
            if str(artifact.get("target_id") or "") != str(row.get("target_id") or ""):
                raise ValueError("diagnostic target identity mismatch")
            if str(artifact.get("target_edge_key") or "") != str(row.get("edge_key") or ""):
                raise ValueError("diagnostic target edge mismatch")
            diagnostic_ids.append(str(artifact["diagnostic_id"]))
            diagnostic_hashes.append(_sha256_file(path))
            if artifact.get("verification_status") == "postcondition_failed":
                product_failure = str(row.get("reason") or "postcondition failed")

        status = str(schedule_result.get("execution_status") or "")
        if status == "complete" and not evidence_ids:
            raise ValueError("completed schedule has no validated evidence")
        return {
            "transcript_hash": _sha256_file(transcript),
            "schedule_result_hash": _sha256_file(result_path),
            "evidence_hashes": tuple(evidence_hashes),
            "diagnostic_hashes": tuple(diagnostic_hashes),
            "evidence_ids": tuple(evidence_ids),
            "diagnostic_ids": tuple(diagnostic_ids),
            "execution_status": status,
            "product_failure": product_failure,
            "validation_error": "",
        }
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return {**empty, "validation_error": f"artifact validation failed: {exc}"}


def _journey_artifact_hashes(
    manifest: FrozenBatchManifest,
    data: Mapping[str, Any],
    shard: FrozenShardSpec,
    empty: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        roots = _worker_runtime_roots(manifest, shard)
        schedule_path = _resolve_required_path(
            manifest, data.get("schedule_path"), roots, label="Journey schedule"
        )
        result_path = _resolve_required_path(
            manifest, data.get("journey_result_path"), roots, label="Journey result"
        )
        transcript_path = _resolve_required_path(
            manifest, data.get("transcript_path"), roots, label="Journey transcript"
        )
        evidence_path = _resolve_required_path(
            manifest, data.get("evidence_path"), roots, label="Journey evidence"
        )
        for label, path in (
            ("schedule", schedule_path), ("result", result_path),
            ("transcript", transcript_path), ("evidence", evidence_path),
        ):
            if not path.is_file():
                raise ValueError(f"Journey {label} is missing")
        journey, registry_import = _journey_target(
            _load_json(Path(shard.target_path))
        )
        schedule = build_journey_schedule(
            revision=manifest.revision,
            seed=shard.seed,
            journey=journey,
        )
        if _load_json(schedule_path) != journey_schedule_payload(schedule):
            raise ValueError("worker Journey schedule differs from the frozen schedule")
        registry = load_verifier_registry(registry_import)
        if registry.registry_id != shard.verifier_registry_id:
            raise ValueError("worker Journey verifier registry identity changed")
        evidence = _load_json(evidence_path)
        result = _load_json(result_path)
        status = str(result.get("terminal_classification") or "")
        if str(data.get("terminal_classification") or "") != status:
            raise ValueError("Journey frame and result classification disagree")
        if str(data.get("schedule_id") or "") != shard.schedule_id:
            raise ValueError("Journey result frame schedule identity changed")
        if str(data.get("evidence_id") or "") != str(result.get("evidence_id") or ""):
            raise ValueError("Journey result frame evidence identity changed")
        product_failure = ""
        evidence_ids: tuple[str, ...] = ()
        evidence_hashes: tuple[str, ...] = ()
        if status == JourneyTerminalClassification.PASSED.value:
            validate_journey_evidence_artifact(
                evidence,
                schedule=schedule,
                verifier_registry=registry,
                revision=manifest.revision,
            )
            evidence_ids = (str(evidence["evidence_id"]),)
            evidence_hashes = (_sha256_file(evidence_path),)
        elif status == JourneyTerminalClassification.PRODUCT_FAILED.value:
            product_failure = str(result.get("failure_reason") or "Journey product failure")
        return {
            "transcript_hash": _sha256_file(transcript_path),
            "schedule_result_hash": _sha256_file(result_path),
            "evidence_hashes": evidence_hashes,
            "diagnostic_hashes": (),
            "evidence_ids": evidence_ids,
            "diagnostic_ids": (),
            "execution_status": status,
            "product_failure": product_failure,
            "validation_error": "",
        }
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return {**empty, "validation_error": f"artifact validation failed: {exc}"}


def _validate_result_frame(
    manifest: FrozenBatchManifest,
    shard: FrozenShardSpec,
    payload: Mapping[str, Any],
) -> None:
    if shard.lane == "journey":
        required = {
            "session_id", "terminal_classification", "schedule_id", "schedule_path",
            "journey_result_path", "transcript_path", "evidence_id", "evidence_path",
        }
        if set(payload) != required:
            raise RuntimeError("Journey worker result frame fields mismatch")
        if str(payload.get("session_id") or "") != shard.session_id:
            raise RuntimeError("Journey worker result belongs to another session")
        try:
            JourneyTerminalClassification(str(payload.get("terminal_classification") or ""))
        except ValueError as exc:
            raise RuntimeError("Journey worker terminal classification is invalid") from exc
        roots = _worker_runtime_roots(manifest, shard)
        for key in (
            "schedule_path", "journey_result_path", "transcript_path", "evidence_path"
        ):
            _resolve_required_path(manifest, payload[key], roots, label=key)
        return
    required = {
        "session_id", "execution_status", "schedule_path", "schedule_result_path",
        "transcript_path", "evidence_paths",
    }
    if set(payload) != required:
        raise RuntimeError(
            "worker result frame fields mismatch: "
            f"missing={sorted(required - set(payload))}, extra={sorted(set(payload) - required)}"
        )
    if str(payload.get("session_id") or "") != shard.session_id:
        raise RuntimeError("worker result belongs to another session")
    if payload.get("execution_status") not in {"complete", "incomplete"}:
        raise RuntimeError("worker result has an invalid execution status")
    if not isinstance(payload.get("evidence_paths"), list):
        raise RuntimeError("worker result evidence_paths must be a list")
    roots = _worker_runtime_roots(manifest, shard)
    for key in ("schedule_path", "schedule_result_path", "transcript_path"):
        _resolve_required_path(manifest, payload[key], roots, label=key)
    for value in payload["evidence_paths"]:
        _resolve_required_path(manifest, value, roots, label="evidence")


def _validate_schedule_result(
    manifest: FrozenBatchManifest,
    shard: FrozenShardSpec,
    result: Mapping[str, Any],
    frame: Mapping[str, Any],
    runtime_roots: Sequence[Path],
) -> None:
    required = {
        "schema_version", "schedule_id", "revision", "execution_status",
        "required_target_count", "passed_target_count", "targets",
    }
    if set(result) != required:
        raise ValueError("schedule result fields do not match the authoritative schema")
    if result.get("schema_version") != 1:
        raise ValueError("unsupported schedule result schema")
    if str(result.get("schedule_id") or "") != shard.schedule_id:
        raise ValueError("schedule result identity mismatch")
    if dict(result.get("revision") or {}) != dict(manifest.revision):
        raise ValueError("schedule result repository revision mismatch")
    if frame and result.get("execution_status") != frame.get("execution_status"):
        raise ValueError("result frame and schedule result status disagree")
    rows = result.get("targets")
    if not isinstance(rows, list) or len(rows) != len(shard.target_ids):
        raise ValueError("schedule result target cardinality mismatch")
    observed = []
    passed = 0
    lane_paths: list[Path] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("schedule result target row is not an object")
        identity = (str(row.get("target_id") or ""), str(row.get("edge_key") or ""))
        observed.append(identity)
        status = str(row.get("status") or "")
        if status not in {"passed", "failed"}:
            raise ValueError("schedule result target status is invalid")
        passed += status == "passed"
        if status == "passed":
            lane = row.get("lane_evidence")
            if not isinstance(lane, Mapping) or "dynamic_dual_ai" not in lane:
                raise ValueError("passed target has no dynamic_dual_ai evidence binding")
            for lane_name, lane_item in lane.items():
                if lane_name not in {"dynamic_dual_ai", "real_cli"} or not isinstance(
                    lane_item, Mapping
                ):
                    raise ValueError("passed target has an invalid evidence lane binding")
                lane_paths.append(_resolve_required_path(
                    manifest,
                    lane_item.get("evidence_path"),
                    runtime_roots,
                    label="lane evidence",
                ))
        elif not str(row.get("diagnostic_path") or ""):
            raise ValueError("failed target has no diagnostic path")
    if observed != list(zip(shard.target_ids, shard.edge_keys, strict=True)):
        raise ValueError("schedule result targets differ from the frozen target order")
    if result.get("required_target_count") != len(rows):
        raise ValueError("schedule result required target count is false")
    if result.get("passed_target_count") != passed:
        raise ValueError("schedule result passed target count is false")
    status = result.get("execution_status")
    if (status == "complete") != (passed == len(rows)):
        raise ValueError("schedule completion status contradicts target outcomes")
    declared_paths = [
        _resolve_required_path(manifest, item, runtime_roots, label="evidence")
        for item in frame.get("evidence_paths") or ()
    ]
    if sorted(map(str, declared_paths)) != sorted(map(str, lane_paths)):
        raise ValueError("result evidence paths differ from target lane bindings")


def _diagnostic_paths_from_result(
    manifest: FrozenBatchManifest,
    schedule_result: Mapping[str, Any],
    runtime_roots: Sequence[Path],
) -> list[tuple[Mapping[str, Any], Path]]:
    diagnostics: list[tuple[Mapping[str, Any], Path]] = []
    for row in schedule_result.get("targets") or ():
        if isinstance(row, Mapping) and row.get("status") == "failed":
            diagnostics.append((row, _resolve_required_path(
                manifest, row.get("diagnostic_path"), runtime_roots, label="diagnostic"
            )))
    return diagnostics


def _known_artifact_path(
    manifest: FrozenBatchManifest,
    shard: FrozenShardSpec | None,
    payload: Mapping[str, Any],
    payload_key: str,
    filename: str,
) -> Path | None:
    declared = _resolve_artifact_path(manifest, payload.get(payload_key))
    if declared is not None:
        return declared
    if shard is None:
        return None
    candidates = (
        Path(shard.runtime_root) / filename,
        Path(manifest.repo_root) / ".agent" / "dynamic-chaos" / shard.session_id / filename,
    )
    return next((path.resolve() for path in candidates if path.is_file()), None)


def _worker_runtime_roots(
    manifest: FrozenBatchManifest, shard: FrozenShardSpec
) -> tuple[Path, ...]:
    return (
        Path(shard.runtime_root).resolve(),
        (
            Path(manifest.repo_root)
            / ".agent" / "dynamic-chaos" / shard.session_id
        ).resolve(),
    )


def _resolve_required_path(
    manifest: FrozenBatchManifest,
    value: Any,
    roots: Sequence[Path],
    *,
    label: str,
) -> Path:
    path = _resolve_artifact_path(manifest, value)
    if path is None or not path.is_file():
        raise ValueError(f"{label} path is missing")
    _require_contained(path, roots, label=label)
    return path


def _require_contained(path: Path, roots: Sequence[Path], *, label: str) -> None:
    resolved = path.resolve(strict=True)
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
            return
        except ValueError:
            continue
    raise ValueError(f"{label} path escapes the shard runtime roots")


def _resolve_artifact_path(
    manifest: FrozenBatchManifest, value: Any
) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if raw == "/workspace" or raw.startswith("/workspace/"):
        path = Path(manifest.repo_root) / Path(raw).relative_to("/workspace")
    elif not path.is_absolute():
        path = Path(manifest.repo_root) / path
    return path.resolve()


def _synthetic_interruption_result(
    manifest: FrozenBatchManifest, shard: FrozenShardSpec, exc: BaseException
) -> ShardResult:
    now = time.time_ns()
    runtime = Path(shard.runtime_root)
    runtime.mkdir(parents=True, exist_ok=True)
    stderr_path = runtime / "worker.stderr.redacted.txt"
    receipt_path = runtime / "synthetic-cleanup-receipt.json"
    if not stderr_path.exists():
        _atomic_write(stderr_path, b"", mode=0o600)
    receipt = {
        "receipt_id": "",
        "schema_version": CLEANUP_RECEIPT_SCHEMA_VERSION,
        "shard_id": shard.shard_id,
        "execution_id": shard.execution_id,
        "host_proof": {},
        "container_proof": {},
        "worker_identity": {},
        "exit_code": None,
        "actions": ["orchestrator_task_exception"],
        "errors": ["synthetic interruption has no process cleanup proof"],
        "cleaned": False,
        "started_at_ns": now,
        "finished_at_ns": now,
    }
    receipt["receipt_id"] = _content_hash({k: v for k, v in receipt.items() if k != "receipt_id"})
    _write_immutable_json(receipt_path, receipt)
    return ShardResult(
        shard_id=shard.shard_id, classification="infrastructure_interrupted",
        attempt_count=1, started_at_ns=now, finished_at_ns=now, exit_code=None,
        target_hash=shard.target_hash, response_hashes=(), decision_hashes=(),
        transcript_hash="", schedule_result_hash="", evidence_hashes=(),
        diagnostic_hashes=(), evidence_ids=(),
        diagnostic_ids=(str(receipt["receipt_id"]),),
        stderr_hash=_sha256_file(stderr_path),
        stderr_path=str(stderr_path), stderr_truncated=False,
        cleanup_receipt_path=str(receipt_path),
        cleanup_receipt_hash=_sha256_file(receipt_path),
        reason=str(redact(f"{type(exc).__name__}: {exc}"))[:1000],
    )


def _append_discovery_results(
    manifest: FrozenBatchManifest,
    results: Sequence[ShardResult],
) -> tuple[str, ...]:
    by_shard = {shard.shard_id: shard for shard in manifest.shards}
    attempt_ids: list[str] = []
    empty_hash = hashlib.sha256(b"").hexdigest()
    for result in results:
        shard = by_shard[result.shard_id]
        diagnostic_ids = result.diagnostic_ids or (result.cleanup_receipt_hash,)
        stable_coverage_ids = shard.edge_keys if shard.lane == "edge" else ()
        journey_ids = shard.target_ids if shard.lane == "journey" else ()
        attempt = build_discovery_attempt(
            batch_id=manifest.batch_id,
            shard_id=result.shard_id,
            stable_coverage_ids=stable_coverage_ids,
            journey_ids=journey_ids,
            revision=manifest.revision,
            schedule_hash=shard.schedule_id,
            response_hashes=result.response_hashes,
            decision_hashes=result.decision_hashes,
            transcript_hash=result.transcript_hash or empty_hash,
            persona=" | ".join(shard.personas),
            input_classes=shard.input_classes,
            sequence_ids=shard.sequence_ids,
            factor_ids=shard.factor_ids,
            started_at=_timestamp_from_ns(result.started_at_ns),
            finished_at=_timestamp_from_ns(result.finished_at_ns),
            classification=result.classification,
            diagnostic_ids=diagnostic_ids,
            evidence_ids=result.evidence_ids,
            defect_id=(
                diagnostic_ids[0] if result.classification == "product_failed" else ""
            ),
        )
        append_discovery_attempt(manifest.discovery_ledger_path, attempt)
        attempt_ids.append(attempt.attempt_id)
    return tuple(attempt_ids)


def _timestamp_from_ns(value: int) -> str:
    return datetime.fromtimestamp(value / 1_000_000_000, tz=timezone.utc).isoformat()


def manifest_payload(manifest: FrozenBatchManifest) -> dict[str, Any]:
    return {
        "manifest_id": manifest.manifest_id,
        "batch_id": manifest.batch_id,
        **_manifest_unsigned_payload(manifest),
    }


def result_index_payload(index: BatchResultIndex) -> dict[str, Any]:
    return {
        "index_id": index.index_id,
        "batch_id": index.batch_id,
        "manifest_id": index.manifest_id,
        "revision": dict(index.revision),
        "execution_status": index.execution_status,
        "release_status": index.release_status,
        "scheduled": index.scheduled,
        "started": index.started,
        "completed": index.completed,
        "discovery_attempt_ids": list(index.discovery_attempt_ids),
        "classification_counts": dict(index.classification_counts),
        "batch_survivor_proof": dict(index.batch_survivor_proof),
        "shards": [_result_payload(item) for item in index.shards],
        "schema_version": index.schema_version,
    }


def _manifest_unsigned_payload(manifest: FrozenBatchManifest) -> dict[str, Any]:
    return {
        "created_at_ns": manifest.created_at_ns,
        "manifest_path": manifest.manifest_path,
        "discovery_ledger_path": manifest.discovery_ledger_path,
        "repo_root": manifest.repo_root,
        "revision": dict(manifest.revision),
        "shard_count": manifest.shard_count,
        "shards": [_shard_payload(item) for item in manifest.shards],
        "required_env_names": list(manifest.required_env_names),
        "timeout_policy": asdict(manifest.timeout_policy),
        "stderr_cap_bytes": manifest.stderr_cap_bytes,
        "worker_runtime": manifest.worker_runtime,
        "formal_profile": manifest.formal_profile,
        "scheduler_schema_version": manifest.scheduler_schema_version,
        "discovery_schema_version": manifest.discovery_schema_version,
        "schema_version": manifest.schema_version,
    }


def _shard_payload(shard: FrozenShardSpec) -> dict[str, Any]:
    payload = asdict(shard)
    payload["command"] = list(shard.command)
    payload["target_ids"] = list(shard.target_ids)
    payload["edge_keys"] = list(shard.edge_keys)
    payload["personas"] = list(shard.personas)
    payload["goals"] = list(shard.goals)
    payload["input_classes"] = list(shard.input_classes)
    payload["sequence_ids"] = list(shard.sequence_ids)
    payload["factor_ids"] = list(shard.factor_ids)
    return payload


def _result_payload(result: ShardResult) -> dict[str, Any]:
    payload = asdict(result)
    for key in (
        "response_hashes", "decision_hashes", "evidence_hashes", "diagnostic_hashes",
        "evidence_ids", "diagnostic_ids",
    ):
        payload[key] = list(payload[key])
    return payload


def _bind_docker_execution_environment(
    command: Sequence[str],
    *,
    execution_id: str,
    container_receipt_dir: Path,
    repo_root: Path,
) -> tuple[str, ...]:
    frozen = tuple(command)
    insert_at = _docker_exec_insert_index(frozen)
    if insert_at is None:
        return frozen
    try:
        receipt_in_container = Path("/workspace") / container_receipt_dir.relative_to(
            repo_root
        )
    except ValueError as exc:
        raise ValueError("Docker cleanup receipts must be contained by the repository") from exc
    bindings = (
        "-e", f"{EXECUTION_ID_ENV}={execution_id}",
        "-e", f"{RECEIPT_DIR_ENV}={receipt_in_container}",
    )
    return (*frozen[:insert_at], *bindings, *frozen[insert_at:])


def _validate_bound_docker_environment(
    shard: FrozenShardSpec,
    repo_root: Path,
) -> None:
    insert_at = _docker_exec_insert_index(shard.command)
    if insert_at is None:
        return
    expected_execution = f"{EXECUTION_ID_ENV}={shard.execution_id}"
    try:
        expected_receipt = Path("/workspace") / Path(
            shard.container_cleanup_receipt_dir
        ).relative_to(repo_root)
    except ValueError as exc:
        raise ValueError("Docker cleanup receipt directory escapes the repository") from exc
    values = tuple(shard.command)
    if expected_execution not in values:
        raise ValueError(f"Docker command lost its execution id: {shard.shard_id}")
    receipt_values = [
        item for item in values if item.startswith(f"{RECEIPT_DIR_ENV}=")
    ]
    if receipt_values != [f"{RECEIPT_DIR_ENV}={expected_receipt}"]:
        raise ValueError(f"Docker command lost its cleanup receipt directory: {shard.shard_id}")


def _validate_formal_linux_command(shard: FrozenShardSpec, repo_root: Path) -> None:
    """Keep formal workers on the frozen in-container Linux implementation."""

    expected_module = (
        "tests.agent_live.journey_simulator_bridge"
        if shard.lane == "journey"
        else "tests.agent_live.codex_simulator_bridge"
    )
    command = tuple(shard.command)
    expected_prefix = (
        str(repo_root / ".venv-adk" / "bin" / "python"),
        "-m",
        expected_module,
    )
    if command[:3] != expected_prefix:
        raise ValueError(
            f"formal shard escaped its Linux worker implementation: {shard.shard_id}"
        )
    required_pairs = {
        "--repo-root": str(repo_root),
        "--runtime": "linux",
        "--session-id": shard.session_id,
    }
    for flag, expected in required_pairs.items():
        positions = [index for index, value in enumerate(command) if value == flag]
        if len(positions) != 1 or positions[0] + 1 >= len(command):
            raise ValueError(f"formal shard command lost {flag}: {shard.shard_id}")
        if command[positions[0] + 1] != expected:
            raise ValueError(f"formal shard command changed {flag}: {shard.shard_id}")


def _docker_exec_insert_index(command: Sequence[str]) -> int | None:
    if tuple(command[:2]) == ("docker", "exec"):
        return 2
    if tuple(command[:3]) == ("docker", "compose", "exec"):
        return 3
    return None


def _is_docker_exec_command(command: Sequence[str]) -> bool:
    return _docker_exec_insert_index(command) is not None


def _single_guard_receipt_path(receipt_dir: Path) -> Path:
    try:
        paths = tuple(receipt_dir.glob("container-cleanup-receipt-*.json"))
    except OSError as exc:
        raise ValueError(f"cannot enumerate process guard receipts: {exc}") from exc
    if len(paths) != 1:
        raise ValueError(f"expected exactly one process guard receipt, found {len(paths)}")
    return paths[0]


def _validated_guard_proof(
    path: Path,
    *,
    execution_id: str,
    required_roles: Sequence[str],
    allowed_roots: Sequence[Path],
) -> dict[str, Any]:
    _require_contained(path, allowed_roots, label="process guard receipt")
    payload = _load_json(path)
    required = {
        "receipt_id", "schema_version", "execution_id", "container_id",
        "pid_namespace_inode", "bridge_identity", "registered_processes",
        "term_decisions", "kill_decisions", "reap_results", "survivor_scans",
        "zero_survivor_scans", "errors", "cleaned", "started_at_ns",
        "finished_at_ns",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("process guard receipt fields do not match the authoritative schema")
    if payload["schema_version"] != 1:
        raise ValueError("unsupported process guard receipt schema")
    if payload["execution_id"] != execution_id:
        raise ValueError("process guard receipt execution id mismatch")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_id"}
    receipt_id = hashlib.sha256(_guard_canonical_json(unsigned)).hexdigest()
    if payload["receipt_id"] != receipt_id:
        raise ValueError("process guard receipt content hash is stale")
    if path.name != f"container-cleanup-receipt-{receipt_id}.json":
        raise ValueError("process guard receipt filename is not content-addressed")
    if not str(payload["container_id"]):
        raise ValueError("process guard receipt has no container/host identity")
    if int(payload["pid_namespace_inode"]) <= 0:
        raise ValueError("process guard receipt has no PID namespace identity")

    registered_rows = payload["registered_processes"]
    if not isinstance(registered_rows, list) or not registered_rows:
        raise ValueError("process guard receipt has no registered processes")
    registered: set[tuple[int, int, int, int]] = set()
    roles: set[str] = set()
    for row in registered_rows:
        identity = _identity_tuple(row)
        if identity in registered:
            raise ValueError("process guard receipt repeats a registered identity")
        registered.add(identity)
        row_roles = row.get("roles") if isinstance(row, Mapping) else None
        if not isinstance(row_roles, list) or not all(
            isinstance(role, str) and role for role in row_roles
        ):
            raise ValueError("registered process roles are invalid")
        roles.update(row_roles)
    if not set(required_roles).issubset(roles):
        raise ValueError("process guard receipt is missing required registered roles")
    if _identity_tuple(payload["bridge_identity"]) not in registered:
        raise ValueError("process guard bridge/observer identity was not registered")

    term_identities = _validate_signal_decisions(
        payload["term_decisions"], registered, expected_signal="SIGTERM"
    )
    kill_identities = _validate_signal_decisions(
        payload["kill_decisions"], registered, expected_signal="SIGKILL"
    )
    if not kill_identities.issubset(term_identities):
        raise ValueError("process guard issued KILL without a TERM decision")

    zero_scans = payload["zero_survivor_scans"]
    if not isinstance(zero_scans, list) or len(zero_scans) != 2:
        raise ValueError("process guard receipt lacks two stable zero-survivor scans")
    for scan in zero_scans:
        if not isinstance(scan, Mapping) or set(scan) != {
            "scan_index", "purpose", "scanned_at_ns", "survivors", "scan_errors"
        }:
            raise ValueError("process guard zero-survivor scan is malformed")
        if (
            scan["purpose"] != "zero_survivor_verification"
            or scan["survivors"]
            or scan["scan_errors"]
        ):
            raise ValueError("process guard zero-survivor scans are not clean")
    if (
        int(zero_scans[1]["scan_index"]) <= int(zero_scans[0]["scan_index"])
        or int(zero_scans[1]["scanned_at_ns"]) <= int(zero_scans[0]["scanned_at_ns"])
    ):
        raise ValueError("process guard zero-survivor scans are not stable and ordered")
    survivor_scans = payload["survivor_scans"]
    if not isinstance(survivor_scans, list) or survivor_scans[-2:] != zero_scans:
        raise ValueError("process guard final scans differ from its zero-survivor proof")

    reap_results = payload["reap_results"]
    if not isinstance(reap_results, list) or not all(
        isinstance(row, Mapping)
        and set(row) == {"pid", "reaped", "return_code", "error"}
        and int(row.get("pid") or 0) > 0
        and isinstance(row.get("reaped"), bool)
        for row in reap_results
    ):
        raise ValueError("process guard reap results are malformed")
    if payload["cleaned"]:
        if payload["errors"]:
            raise ValueError("clean process guard receipt contains errors")
        if any(not bool(row.get("reaped")) for row in reap_results):
            raise ValueError("clean process guard receipt contains an unreaped process")

    return {
        "receipt_id": receipt_id,
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "execution_id": execution_id,
        "container_id": str(payload["container_id"]),
        "pid_namespace_inode": int(payload["pid_namespace_inode"]),
        "registered_processes": registered_rows,
        "term_decisions": payload["term_decisions"],
        "kill_decisions": payload["kill_decisions"],
        "reap_results": payload["reap_results"],
        "zero_survivor_scans": zero_scans,
        "errors": payload["errors"],
        "cleaned": bool(payload["cleaned"]),
    }


def _validate_signal_decisions(
    rows: Any,
    registered: set[tuple[int, int, int, int]],
    *,
    expected_signal: str,
) -> set[tuple[int, int, int, int]]:
    if not isinstance(rows, list):
        raise ValueError("process guard signal decisions are not a list")
    identities: set[tuple[int, int, int, int]] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "identity", "signal", "sent", "outcome", "decided_at_ns"
        } or row.get("signal") != expected_signal:
            raise ValueError("process guard signal decision is malformed")
        if not isinstance(row.get("sent"), bool) or int(row.get("decided_at_ns") or 0) <= 0:
            raise ValueError("process guard signal decision metadata is malformed")
        if row["sent"] != (row.get("outcome") == "sent"):
            raise ValueError("process guard signal outcome contradicts its sent flag")
        identity = _identity_tuple(row.get("identity"))
        if identity not in registered:
            raise ValueError("process guard signaled an unregistered identity")
        identities.add(identity)
    return identities


def _identity_tuple(value: Any) -> tuple[int, int, int, int]:
    if not isinstance(value, Mapping):
        raise ValueError("process identity is not an object")
    try:
        identity = (
            int(value["pid"]), int(value["pgid"]), int(value["start_ticks"]),
            int(value["pid_namespace_inode"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("process identity fields are invalid") from exc
    if any(item <= 0 for item in identity):
        raise ValueError("process identity fields must be positive")
    return identity


def _validate_composite_cleanup_receipt(
    shard: FrozenShardSpec,
    result: ShardResult,
) -> None:
    path = Path(result.cleanup_receipt_path)
    _require_contained(path, (Path(shard.runtime_root),), label="composite cleanup receipt")
    if _sha256_file(path) != result.cleanup_receipt_hash:
        raise ValueError("composite cleanup receipt file hash mismatch")
    payload = _load_json(path)
    required = {
        "receipt_id", "schema_version", "shard_id", "execution_id", "host_proof",
        "container_proof", "worker_identity", "exit_code", "actions", "errors",
        "cleaned", "started_at_ns", "finished_at_ns",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("composite cleanup receipt fields do not match the schema")
    if payload["schema_version"] != CLEANUP_RECEIPT_SCHEMA_VERSION:
        raise ValueError("unsupported composite cleanup receipt schema")
    if payload["shard_id"] != shard.shard_id or payload["execution_id"] != shard.execution_id:
        raise ValueError("composite cleanup receipt identity mismatch")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_id"}
    if payload["receipt_id"] != _content_hash(unsigned):
        raise ValueError("composite cleanup receipt content hash is stale")
    if not payload["cleaned"]:
        details = "; ".join(str(item) for item in payload["errors"])
        raise ValueError(
            "composite cleanup receipt does not prove cleanup"
            + (f": {details}" if details else "")
        )
    if payload["errors"]:
        raise ValueError("clean composite cleanup receipt contains errors")
    proof_contracts = {
        "host_proof": (
            ("host_worker", "docker_exec")
            if _is_docker_exec_command(shard.command)
            else ("host_worker",),
            (Path(shard.runtime_root) / "host-cleanup-receipts",),
        ),
        "container_proof": (
            ("container_bridge", "agent_process_group_leader"),
            (Path(shard.container_cleanup_receipt_dir),),
        ),
    }
    for key, (roles, roots) in proof_contracts.items():
        proof = payload[key]
        if not isinstance(proof, Mapping) or proof.get("execution_id") != shard.execution_id:
            raise ValueError(f"composite {key} execution id mismatch")
        if proof.get("cleaned") is not True:
            raise ValueError(f"composite {key} is incomplete")
        proof_path = Path(str(proof.get("path") or ""))
        if not proof_path.is_file() or _sha256_file(proof_path) != proof.get("sha256"):
            raise ValueError(f"composite {key} file binding is invalid")
        revalidated = _validated_guard_proof(
            proof_path,
            execution_id=shard.execution_id,
            required_roles=roles,
            allowed_roots=roots,
        )
        if revalidated != proof:
            raise ValueError(f"composite {key} summary differs from its guard receipt")
    worker = _identity_tuple(payload["worker_identity"])
    host_registered = {
        _identity_tuple(row) for row in payload["host_proof"]["registered_processes"]
    }
    if worker not in host_registered:
        raise ValueError("composite worker identity is not in the host proof")


async def _final_batch_survivor_proof(
    manifest: FrozenBatchManifest,
) -> dict[str, Any]:
    started_at_ns = time.time_ns()
    expected = tuple(shard.execution_id for shard in manifest.shards)
    first = await asyncio.to_thread(_scan_batch_execution_ids, expected)
    await asyncio.sleep(0.05)
    second = await asyncio.to_thread(_scan_batch_execution_ids, expected)
    errors = sorted(set((*first["scan_errors"], *second["scan_errors"])))
    cleaned = not errors and not first["survivors"] and not second["survivors"]
    unsigned = {
        "schema_version": BATCH_SURVIVOR_PROOF_SCHEMA_VERSION,
        "execution_ids": list(expected),
        "scans": [first, second],
        "errors": errors,
        "cleaned": cleaned,
        "started_at_ns": started_at_ns,
        "finished_at_ns": time.time_ns(),
    }
    return {"proof_id": _content_hash(unsigned), **unsigned}


def _scan_batch_execution_ids(execution_ids: Sequence[str]) -> dict[str, Any]:
    expected = {value.encode("utf-8"): value for value in execution_ids}
    token_name = EXECUTION_ID_ENV.encode("ascii")
    survivors: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        entries = tuple(Path("/proc").iterdir())
    except OSError as exc:
        return {"scanned_at_ns": time.time_ns(), "survivors": [], "scan_errors": [str(exc)]}
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        pid = int(entry.name)
        try:
            before = _read_host_identity(pid)
            environ = (entry / "environ").read_bytes()
            after = _read_host_identity(pid)
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError as exc:
            errors.append(f"cannot inspect host process {pid}: {exc}")
            continue
        if before != after:
            errors.append(f"host pid {pid} changed identity during batch scan")
            continue
        for token in environ.split(b"\0"):
            name, separator, value = token.partition(b"=")
            if separator and name == token_name and value in expected:
                survivors.append({
                    "execution_id": expected[value],
                    "identity": dict(after),
                })
                break
    survivors.sort(key=lambda row: (row["execution_id"], row["identity"]["pid"]))
    return {
        "scanned_at_ns": time.time_ns(),
        "survivors": survivors,
        "scan_errors": errors,
    }


def _read_host_identity(pid: int) -> dict[str, int]:
    root = Path("/proc") / str(pid)
    stat_text = (root / "stat").read_text(encoding="utf-8")
    namespace_inode = (root / "ns" / "pid").stat().st_ino
    closing_paren = stat_text.rfind(")")
    fields = stat_text[closing_paren + 2:].split() if closing_paren >= 0 else []
    if len(fields) <= 19:
        raise OSError(f"malformed /proc/{pid}/stat")
    try:
        return {
            "pid": pid,
            "pgid": int(fields[2]),
            "start_ticks": int(fields[19]),
            "pid_namespace_inode": namespace_inode,
        }
    except ValueError as exc:
        raise OSError(f"malformed /proc/{pid}/stat identity fields") from exc


def _guard_canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _default_command_factory(repo_root: Path, *, worker_runtime: str) -> CommandFactory:
    def factory(
        index: int,
        target: Path,
        runtime: Path,
        session: str,
        seed: int,
        schedule_id: str,
    ) -> Sequence[str]:
        del index, runtime
        payload = _load_json(target)
        lane = _target_lane(payload)
        if worker_runtime == "docker":
            try:
                target_for_worker = Path("/workspace") / target.relative_to(repo_root)
            except ValueError as exc:
                raise ValueError("Docker target must be contained by the repository") from exc
            base = (
                "docker", "exec", "-i", "-w", "/workspace",
                "blockchain-node-benchmark-bench-1",
            )
            worker_root = "/workspace"
            python = ".venv-adk/bin/python"
        else:
            target_for_worker = target
            base = ()
            worker_root = str(repo_root)
            python = str(repo_root / ".venv-adk" / "bin" / "python")
        if lane == "edge":
            return (*base,
                python, "-m", "tests.agent_live.codex_simulator_bridge",
                "--targets", str(target_for_worker), "--repo-root", worker_root,
                "--seed", str(seed), "--session-id", session, "--runtime", "linux",
            )
        _, registry_import = _journey_target(payload)
        return (*base,
            python, "-m", "tests.agent_live.journey_simulator_bridge",
            "--journey-definition", str(target_for_worker),
            "--verifier-registry", registry_import,
            "--repo-root", worker_root, "--seed", str(seed),
            "--expected-schedule-id", schedule_id,
            "--session-id", session, "--runtime", "linux",
        )
    return factory


def _require_linux_control_plane() -> None:
    if sys.platform != "linux" or not Path("/proc/self/stat").is_file():
        raise RuntimeError(
            "batch execution requires a Linux control plane with /proc; "
            "macOS can freeze or inspect manifests but cannot produce qualifying evidence"
        )


def _target_rows(payload: Any) -> list[Mapping[str, Any]]:
    rows = payload.get("targets") if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list):
        raise ValueError("target file must be a list or contain a targets list")
    if not all(isinstance(item, Mapping) for item in rows):
        raise ValueError("target rows must be JSON objects")
    allowed = {
        "target_id", "edge_key", "persona", "goal", "sequence_id",
        "tuple_ids", "scenario_id",
    }
    for index, row in enumerate(rows):
        unknown = sorted(set(row) - allowed)
        if unknown:
            raise ValueError(
                f"target row {index} contains unsupported or future-dialogue fields: "
                + ", ".join(unknown)
            )
    return list(rows)


def _target_lane(payload: Any) -> str:
    if isinstance(payload, Mapping) and "lane" in payload:
        lane = str(payload.get("lane") or "").strip()
    else:
        lane = "edge"
    if lane not in SHARD_LANES:
        raise ValueError(f"unsupported batch target lane: {lane or '<empty>'}")
    return lane


def _journey_target(payload: Any) -> tuple[Mapping[str, Any], str]:
    if not isinstance(payload, Mapping) or _target_lane(payload) != "journey":
        raise ValueError("Journey target must be an object with lane=journey")
    allowed = {"lane", "journey", "verifier_registry"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(
            "Journey target contains unsupported or future-dialogue fields: "
            + ", ".join(unknown)
        )
    journey = payload.get("journey")
    if not isinstance(journey, Mapping):
        raise ValueError("Journey target requires a journey object")
    registry_import = str(payload.get("verifier_registry") or "").strip()
    if not registry_import:
        raise ValueError("Journey target requires verifier_registry")
    return journey, registry_import


def _reject_secret_bearing_command(command: Sequence[str]) -> None:
    markers = ("api_key=", "token=", "password=", "authorization=", "bearer ")
    lowered = " ".join(command).lower()
    if any(marker in lowered for marker in markers):
        raise ValueError("worker command must reference environment names, not secret values")


def _parse_frame(line: str, prefix: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(line[len(prefix):])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid {prefix.strip()} frame") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{prefix.strip()} frame is not an object")
    return payload


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65_536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, content: bytes, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, mode)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"immutable artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"immutable artifact already exists: {path}") from exc
        os.chmod(path, 0o444)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)

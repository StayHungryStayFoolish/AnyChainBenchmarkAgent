from __future__ import annotations

import asyncio
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent.harness.runtime_identity import repository_revision
from tests.agent_live.chaos_scheduler import (
    build_chaos_schedule,
    build_journey_schedule,
    journey_schedule_payload,
    schedule_payload,
)
from tests.agent_live.coverage_evidence import content_hash
from tests.agent_live.generate_harness_coverage_ledger import build_ledger

from tests.agent_live.batch_orchestrator import (
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_SHARD_COUNT,
    ExternalDecisionBlocked,
    ShardResult,
    TimeoutPolicy,
    _RunState,
    _append_discovery_results,
    _classify,
    _scan_batch_execution_ids,
    _validate_result_frame,
    freeze_batch_manifest,
    load_frozen_manifest,
    run_batch,
    validate_frozen_manifest,
)
from tests.agent_live.formal_journey_catalog import formal_journey_definitions


FAKE_WORKER = r'''
import argparse
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path

CONTEXT = "CODEX_SIMULATOR_CONTEXT "
DECISION = "CODEX_SIMULATOR_DECISION "
RESULT = "CODEX_SIMULATOR_RESULT "
JOURNEY_CONTEXT = "CODEX_JOURNEY_SIMULATOR_CONTEXT "
JOURNEY_DECISION = "CODEX_JOURNEY_SIMULATOR_DECISION "
JOURNEY_RESULT = "CODEX_JOURNEY_SIMULATOR_RESULT "

parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True)
parser.add_argument("--runtime", required=True)
parser.add_argument("--marker", required=True)
parser.add_argument("--index", type=int, required=True)
parser.add_argument("--target", required=True)
parser.add_argument("--schedule", required=True)
parser.add_argument("--session", required=True)
args = parser.parse_args()
if args.mode in {"stubborn", "stubborn-after-result"}:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
runtime = Path(args.runtime)
runtime.mkdir(parents=True, exist_ok=True)
Path(args.marker).write_text("started\n", encoding="utf-8")
execution_id = os.environ["ANYCHAIN_CHAOS_EXECUTION_ID"]
receipt_dir = Path(os.environ["ANYCHAIN_CHAOS_INNER_CLEANUP_RECEIPT_DIR"])
namespace_inode = Path("/proc/self/ns/pid").stat().st_ino
stat_fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
bridge_identity = {
    "pid": os.getpid(), "pgid": int(stat_fields[2]),
    "start_ticks": int(stat_fields[19]), "pid_namespace_inode": namespace_inode,
}
agent_identity = {
    "pid": os.getpid() + 100000, "pgid": int(stat_fields[2]),
    "start_ticks": int(stat_fields[19]) + 1, "pid_namespace_inode": namespace_inode,
}
scan = {
    "scan_index": 1, "purpose": "zero_survivor_verification",
    "scanned_at_ns": time.time_ns(), "survivors": [], "scan_errors": [],
}
scan_two = {**scan, "scan_index": 2, "scanned_at_ns": scan["scanned_at_ns"] + 1}
inner_unsigned = {
    "schema_version": 1,
    "execution_id": ("wrong-execution" if args.mode == "wrong-inner-execution" else execution_id),
    "container_id": "fake-container",
    "pid_namespace_inode": namespace_inode,
    "bridge_identity": bridge_identity,
    "registered_processes": [
        {**bridge_identity, "roles": ["container_bridge", "execution_id_match"]},
        {**agent_identity, "roles": ["agent_process_group_leader", "execution_id_match"]},
    ],
    "term_decisions": [{
        "identity": agent_identity, "signal": "SIGTERM", "sent": True,
        "outcome": "sent", "decided_at_ns": time.time_ns(),
    }],
    "kill_decisions": [],
    "reap_results": [{
        "pid": agent_identity["pid"], "reaped": True, "return_code": -15, "error": "",
    }],
    "survivor_scans": [scan, scan_two],
    "zero_survivor_scans": [scan, scan_two],
    "errors": (["injected cleanup uncertainty"] if args.mode == "inner-not-cleaned" else []),
    "cleaned": args.mode != "inner-not-cleaned",
    "started_at_ns": time.time_ns(),
    "finished_at_ns": time.time_ns(),
}
if args.mode == "inner-survivor":
    inner_unsigned["zero_survivor_scans"][1]["survivors"] = [agent_identity]
if args.mode != "missing-inner":
    receipt_dir.mkdir(parents=True, exist_ok=True)
    inner_id = hashlib.sha256(json.dumps(
        inner_unsigned, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    inner_receipt = {"receipt_id": inner_id, **inner_unsigned}
    if args.mode == "tampered-inner-hash":
        inner_receipt["receipt_id"] = "0" * 64
        inner_id = "0" * 64
    inner_path = receipt_dir / f"container-cleanup-receipt-{inner_id}.json"
    inner_path.write_text(json.dumps(inner_receipt, sort_keys=True) + "\n", encoding="utf-8")
    if args.mode == "duplicate-inner":
        (receipt_dir / ("container-cleanup-receipt-" + "f" * 64 + ".json")).write_text(
            json.dumps(inner_receipt), encoding="utf-8"
        )
response_hash = hashlib.sha256(f"response-{args.index}".encode()).hexdigest()
target_payload = json.loads(Path(args.target).read_text(encoding="utf-8"))
lane = target_payload.get("lane", "edge")
target = (
    target_payload["journey"]
    if lane == "journey"
    else target_payload["targets"][0]
)
schedule_payload = json.loads(Path(args.schedule).read_text(encoding="utf-8"))
context = {
    "session_id": args.session,
    "turn_index": 1,
    "previous_response_hash": response_hash,
    "previous_agent_response": f"agent response {args.index}",
}
if lane == "journey":
    context["schedule"] = schedule_payload
else:
    context["scheduled_target"] = target
context_frame = (JOURNEY_CONTEXT if lane == "journey" else CONTEXT) + json.dumps(context) + "\n"
if args.mode == "fragmented":
    midpoint = len(context_frame) // 2
    sys.stdout.write(context_frame[:midpoint])
    sys.stdout.flush()
    time.sleep(0.01)
    sys.stdout.write(context_frame[midpoint:])
    sys.stdout.flush()
else:
    print(context_frame, end="", flush=True)
line = sys.stdin.readline()
decision_prefix = JOURNEY_DECISION if lane == "journey" else DECISION
if not line.startswith(decision_prefix):
    sys.exit(10)
json.loads(line[len(decision_prefix):])
if args.mode == "stubborn":
    while True:
        time.sleep(1)
if args.mode == "pass-slow":
    time.sleep(0.25)
if args.mode == "mutate-target":
    Path(args.target).write_text('{"targets": []}', encoding="utf-8")
transcript = runtime / "transcript.txt"
transcript.write_text(f"complete transcript {args.index}\n", encoding="utf-8")
schedule = runtime / "schedule.json"
schedule.write_text(json.dumps(schedule_payload), encoding="utf-8")
evidence_dir = runtime / "evidence"
evidence_dir.mkdir()
evidence = evidence_dir / "evidence.json"
evidence.write_text(json.dumps({
    "evidence_id": hashlib.sha256(f"evidence-{args.index}".encode()).hexdigest(),
    "edge_key": target.get("edge_key", ""),
}), encoding="utf-8")
if lane == "journey":
    journey_result = runtime / "journey-result.json"
    evidence_id = json.loads(evidence.read_text())["evidence_id"]
    journey_result.write_text(json.dumps({
        "terminal_classification": "passed",
        "evidence_id": evidence_id,
        "failure_reason": "",
    }), encoding="utf-8")
    result_frame = JOURNEY_RESULT + json.dumps({
        "session_id": args.session,
        "terminal_classification": "passed",
        "schedule_id": schedule_payload["schedule_id"],
        "schedule_path": str(schedule),
        "journey_result_path": str(journey_result),
        "transcript_path": str(transcript),
        "evidence_id": evidence_id,
        "evidence_path": str(evidence),
    }) + "\n"
    print(result_frame, end="", flush=True)
    sys.exit(0)
if args.mode == "product":
    diagnostic_dir = runtime / "diagnostics"
    diagnostic_dir.mkdir()
    diagnostic = diagnostic_dir / "diagnostic.json"
    diagnostic.write_text(json.dumps({
        "diagnostic_id": hashlib.sha256(f"diagnostic-{args.index}".encode()).hexdigest(),
        "verification_status": "postcondition_failed",
        "revision": schedule_payload["revision"],
        "target_id": target["target_id"],
        "target_edge_key": target["edge_key"],
    }), encoding="utf-8")
    targets = [{
        "target_id": target["target_id"], "edge_key": target["edge_key"],
        "status": "failed", "reason": "real product postcondition failed",
        "diagnostic_path": str(diagnostic),
    }]
    execution_status = "incomplete"
    exit_code = 1
elif args.mode == "infrastructure":
    sys.stderr.write("Authorization: Bearer super-secret-token\n" + "x" * 10000)
    sys.stderr.flush()
    sys.exit(2)
else:
    targets = [{
        "target_id": target["target_id"], "edge_key": target["edge_key"],
        "status": "passed",
        "lane_evidence": {"dynamic_dual_ai": {
            "evidence_id": json.loads(evidence.read_text())["evidence_id"],
            "evidence_path": str(evidence),
        }},
    }]
    execution_status = "complete"
    exit_code = 0
schedule_result = runtime / "schedule-result.json"
schedule_result.write_text(json.dumps({
    "schema_version": 1,
    "schedule_id": schedule_payload["schedule_id"],
    "revision": ({"commit": "forged", "worktree_hash": "0" * 64}
                 if args.mode == "forged-revision" else schedule_payload["revision"]),
    "execution_status": execution_status,
    "required_target_count": 1,
    "passed_target_count": int(exit_code == 0),
    "targets": targets,
}), encoding="utf-8")
if args.mode == "forged-schedule":
    data = json.loads(schedule_result.read_text())
    data["schedule_id"] = "0" * 64
    schedule_result.write_text(json.dumps(data))
if args.mode == "forged-target":
    data = json.loads(schedule_result.read_text())
    data["targets"][0]["target_id"] = "another-target"
    schedule_result.write_text(json.dumps(data))
if args.mode == "product":
    sys.exit(exit_code)
result_transcript = transcript
if args.mode == "path-escape":
    result_transcript = runtime.parent / "escaped.txt"
    result_transcript.write_text("escaped")
result_frame = RESULT + json.dumps({
    "session_id": ("another-session" if args.mode == "forged-session" else args.session),
    "execution_status": execution_status,
    "terminal_classification": "passed",
    "failure_reason": "",
    "schedule_path": str(schedule),
    "transcript_path": str(result_transcript),
    "schedule_result_path": str(schedule_result),
    "evidence_paths": [str(evidence)],
}) + "\n"
if args.mode == "fragmented":
    for offset in range(0, len(result_frame), 7):
        sys.stdout.write(result_frame[offset:offset + 7])
        sys.stdout.flush()
else:
    print(result_frame, end="", flush=True)
if args.mode == "stubborn-after-result":
    while True:
        time.sleep(1)
sys.exit(exit_code)
'''


class BatchOrchestratorTests(unittest.TestCase):
    def test_typed_worker_result_classifies_simulator_invalid_without_stderr(self) -> None:
        classification, reason = _classify(
            None,
            SimpleNamespace(lane="edge", target_ids=("target-1",)),
            _RunState(),
            0,
            True,
            {
                "validation_error": "",
                "product_failure": "",
                "terminal_classification": "simulator_invalid",
                "failure_reason": "SimulatorDecisionInvalid: wrong input shape",
            },
        )

        self.assertEqual(classification, "simulator_invalid")
        self.assertEqual(reason, "SimulatorDecisionInvalid: wrong input shape")

    def test_simulator_invalid_terminal_frame_is_schema_valid(self) -> None:
        self._write_targets(1)
        manifest = freeze_batch_manifest(
            repo_root=self.root,
            targets_dir=self.targets,
            manifest_path=self.root / ".agent" / "typed-invalid-manifest.json",
            runtime_base=self.root / ".agent" / "typed-invalid-runtime",
            shard_count=1,
            command_factory=self._factory(
                ["pass"], self.root / ".agent" / "typed-invalid-markers"
            ),
        )
        shard = manifest.shards[0]
        runtime = Path(shard.runtime_root)
        runtime.mkdir(parents=True, exist_ok=True)
        for filename in ("schedule.json", "schedule-result.json", "transcript.txt"):
            (runtime / filename).write_text("{}\n", encoding="utf-8")
        payload = {
            "session_id": shard.session_id,
            "execution_status": "incomplete",
            "terminal_classification": "simulator_invalid",
            "failure_reason": "SimulatorDecisionInvalid: wrong input shape",
            "schedule_path": str(runtime / "schedule.json"),
            "schedule_result_path": str(runtime / "schedule-result.json"),
            "transcript_path": str(runtime / "transcript.txt"),
            "evidence_paths": [],
        }

        _validate_result_frame(manifest, shard, payload)

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        subprocess.run(("git", "init", "-q"), cwd=self.root, check=True)
        subprocess.run(("git", "config", "user.email", "test@example.com"), cwd=self.root, check=True)
        subprocess.run(("git", "config", "user.name", "Test"), cwd=self.root, check=True)
        (self.root / ".gitignore").write_text(".agent/\n", encoding="utf-8")
        (self.root / "tracked.txt").write_text("frozen\n", encoding="utf-8")
        subprocess.run(("git", "add", "."), cwd=self.root, check=True)
        subprocess.run(("git", "commit", "-qm", "fixture"), cwd=self.root, check=True)
        self.revision = repository_revision(self.root)
        self.ledger = build_ledger(revision=self.revision)
        self.edge = next(
            edge for edge in self.ledger["edges"]
            if bool(((edge.get("evidence") or {}).get("dynamic_dual_ai") or {}).get("required"))
            and edge.get("executable_scenario_ids")
        )
        self.targets = self.root / ".agent" / "targets"
        self.targets.mkdir(parents=True)
        self.worker = self.root / ".agent" / "fake_worker.py"
        self.worker.write_text(textwrap.dedent(FAKE_WORKER), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_targets(self, count: int) -> None:
        for index in range(1, count + 1):
            (self.targets / f"{index:02d}.json").write_text(json.dumps({
                "targets": [{
                    "target_id": f"target-{index}",
                    "edge_key": self.edge["edge_key"],
                    "persona": f"persona-{index}",
                    "goal": f"goal-{index}",
                    "scenario_id": self.edge["executable_scenario_ids"][0],
                    "sequence_id": f"sequence-{index}",
                    "tuple_ids": [f"factor-{index}"],
                }]
            }), encoding="utf-8")

    def _factory(self, modes: list[str], markers: Path):
        markers.mkdir(parents=True, exist_ok=True)
        def factory(index, target, runtime, session, seed, schedule_id):
            schedule = build_chaos_schedule(
                self.ledger,
                revision=self.revision,
                seed=seed,
                targets=json.loads(target.read_text())["targets"],
            )
            self.assertEqual(schedule.schedule_id, schedule_id)
            schedule_path = markers / f"schedule-{index:02d}.json"
            schedule_path.write_text(json.dumps(schedule_payload(schedule)), encoding="utf-8")
            return (
                sys.executable, str(self.worker), "--mode", modes[index - 1],
                "--runtime", str(runtime), "--marker", str(markers / f"{index:02d}"),
                "--index", str(index), "--target", str(target),
                "--schedule", str(schedule_path), "--session", session,
            )
        return factory

    def _write_mixed_targets(self) -> None:
        self._write_targets(24)
        journeys = formal_journey_definitions()
        self.assertEqual(len(journeys), 8)
        for offset, journey in enumerate(journeys, start=25):
            (self.targets / f"{offset:02d}.json").write_text(
                json.dumps(journey), encoding="utf-8"
            )

    def _mixed_factory(self, modes: list[str], markers: Path):
        markers.mkdir(parents=True, exist_ok=True)

        def factory(index, target, runtime, session, seed, schedule_id):
            payload = json.loads(target.read_text())
            if payload.get("lane") == "journey":
                schedule = build_journey_schedule(
                    revision=self.revision,
                    seed=seed,
                    journey=payload["journey"],
                )
                frozen_schedule = journey_schedule_payload(schedule)
            else:
                schedule = build_chaos_schedule(
                    self.ledger,
                    revision=self.revision,
                    seed=seed,
                    targets=payload["targets"],
                )
                frozen_schedule = schedule_payload(schedule)
            self.assertEqual(schedule.schedule_id, schedule_id)
            schedule_path = markers / f"schedule-{index:02d}.json"
            schedule_path.write_text(json.dumps(frozen_schedule), encoding="utf-8")
            return (
                sys.executable, str(self.worker), "--mode", modes[index - 1],
                "--runtime", str(runtime), "--marker", str(markers / f"{index:02d}"),
                "--index", str(index), "--target", str(target),
                "--schedule", str(schedule_path), "--session", session,
            )

        return factory

    @staticmethod
    def _accept_fake_evidence(reference, *, edge, revision):
        del edge, revision
        return json.loads(Path(reference).read_text(encoding="utf-8")), ""

    def test_default_shard_count_and_manifest_tamper_detection(self) -> None:
        self._write_targets(DEFAULT_SHARD_COUNT)
        manifest_path = self.root / ".agent" / "manifest.json"
        manifest = freeze_batch_manifest(
            repo_root=self.root,
            targets_dir=self.targets,
            manifest_path=manifest_path,
            runtime_base=self.root / ".agent" / "runtime",
            command_factory=self._factory(["pass"] * DEFAULT_SHARD_COUNT, self.root / ".agent" / "markers"),
        )
        self.assertEqual(manifest.shard_count, 32)
        self.assertEqual(manifest.max_concurrency, DEFAULT_MAX_CONCURRENCY)
        self.assertLess(manifest.max_concurrency, manifest.shard_count)
        self.assertFalse(manifest_path.stat().st_mode & stat.S_IWUSR)
        self.assertNotIn("super-secret", manifest_path.read_text(encoding="utf-8"))
        loaded = load_frozen_manifest(manifest_path)
        self.assertEqual(loaded.manifest_id, manifest.manifest_id)
        os.chmod(self.targets / "01.json", 0o644)
        (self.targets / "01.json").write_text('{"targets": []}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "frozen target changed"):
            validate_frozen_manifest(manifest)

    def test_max_concurrency_is_frozen_validated_and_tamper_evident(self) -> None:
        self._write_targets(4)
        common = {
            "repo_root": self.root,
            "targets_dir": self.targets,
            "runtime_base": self.root / ".agent" / "bounded-runtime",
            "shard_count": 4,
            "command_factory": self._factory(
                ["pass"] * 4, self.root / ".agent" / "bounded-markers"
            ),
        }
        with self.assertRaisesRegex(ValueError, "positive integer"):
            freeze_batch_manifest(
                **common,
                manifest_path=self.root / ".agent" / "zero-concurrency.json",
                max_concurrency=0,
            )
        with self.assertRaisesRegex(ValueError, "cannot exceed shard_count"):
            freeze_batch_manifest(
                **common,
                manifest_path=self.root / ".agent" / "excess-concurrency.json",
                max_concurrency=5,
            )

        manifest_path = self.root / ".agent" / "bounded-manifest.json"
        manifest = freeze_batch_manifest(
            **common,
            manifest_path=manifest_path,
            max_concurrency=2,
        )
        self.assertEqual(manifest.max_concurrency, 2)
        self.assertEqual(
            json.loads(manifest_path.read_text())["max_concurrency"],
            2,
        )

        os.chmod(manifest_path, 0o644)
        payload = json.loads(manifest_path.read_text())
        payload["max_concurrency"] = 3
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        os.chmod(manifest_path, 0o444)
        with self.assertRaisesRegex(ValueError, "content hash is stale"):
            load_frozen_manifest(manifest_path)

    def test_run_batch_never_exceeds_frozen_max_concurrency(self) -> None:
        shard_count = 12
        max_concurrency = 3
        self._write_targets(shard_count)
        manifest = freeze_batch_manifest(
            repo_root=self.root,
            targets_dir=self.targets,
            manifest_path=self.root / ".agent" / "concurrency-manifest.json",
            runtime_base=self.root / ".agent" / "concurrency-runtime",
            shard_count=shard_count,
            max_concurrency=max_concurrency,
            required_env_names=(),
            command_factory=self._factory(
                ["pass"] * shard_count,
                self.root / ".agent" / "concurrency-markers",
            ),
        )
        active = 0
        peak = 0
        first_wave_ready = asyncio.Event()
        release_first_wave = asyncio.Event()

        async def controlled_run(_manifest, shard, _broker):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if active == max_concurrency:
                first_wave_ready.set()
            await release_first_wave.wait()
            active -= 1
            runtime = Path(shard.runtime_root)
            runtime.mkdir(parents=True, exist_ok=True)
            stderr_path = runtime / "worker.stderr.redacted.txt"
            stderr_path.write_bytes(b"")
            receipt_path = runtime / "cleanup-receipt.json"
            receipt_path.write_text('{"cleaned": true}\n', encoding="utf-8")
            empty_hash = hashlib.sha256(b"").hexdigest()
            return ShardResult(
                shard_id=shard.shard_id,
                classification="passed",
                attempt_count=1,
                started_at_ns=1,
                finished_at_ns=2,
                exit_code=0,
                target_hash=shard.target_hash,
                response_hashes=(),
                decision_hashes=(),
                transcript_hash=empty_hash,
                schedule_result_hash=empty_hash,
                evidence_hashes=(),
                diagnostic_hashes=(),
                evidence_ids=(),
                diagnostic_ids=(),
                stderr_hash=empty_hash,
                stderr_path=str(stderr_path),
                stderr_truncated=False,
                cleanup_receipt_path=str(receipt_path),
                cleanup_receipt_hash=hashlib.sha256(
                    receipt_path.read_bytes()
                ).hexdigest(),
                reason="",
            )

        async def clean_batch(_manifest):
            return {"cleaned": True, "scans": [], "errors": []}

        async def scenario():
            task = asyncio.create_task(run_batch(
                manifest,
                broker=lambda _shard_id, _context: {},
                result_index_path=self.root / ".agent" / "concurrency-index.json",
            ))
            await asyncio.wait_for(first_wave_ready.wait(), timeout=1)
            await asyncio.sleep(0)
            self.assertEqual(active, max_concurrency)
            self.assertEqual(peak, max_concurrency)
            release_first_wave.set()
            return await task

        with patch(
            "tests.agent_live.batch_orchestrator._run_shard",
            side_effect=controlled_run,
        ), patch(
            "tests.agent_live.batch_orchestrator._validate_composite_cleanup_receipt"
        ), patch(
            "tests.agent_live.batch_orchestrator._final_batch_survivor_proof",
            side_effect=clean_batch,
        ), patch(
            "tests.agent_live.batch_orchestrator._append_discovery_results",
            return_value=(),
        ):
            index = asyncio.run(scenario())

        self.assertEqual(peak, max_concurrency)
        self.assertEqual(active, 0)
        self.assertEqual(index.completed, shard_count)

    def test_journey_discovery_attempt_uses_journey_identity(self) -> None:
        target = self.targets / "01.json"
        target.write_text(
            json.dumps(formal_journey_definitions()[0]), encoding="utf-8"
        )
        manifest = freeze_batch_manifest(
            repo_root=self.root,
            targets_dir=self.targets,
            manifest_path=self.root / ".agent" / "journey-manifest.json",
            runtime_base=self.root / ".agent" / "journey-runtime",
            shard_count=1,
            required_env_names=(),
            command_factory=lambda *_args: (sys.executable, "-c", "pass"),
        )
        shard = manifest.shards[0]
        empty_hash = hashlib.sha256(b"").hexdigest()
        result = ShardResult(
            shard_id=shard.shard_id,
            classification="externally_blocked",
            attempt_count=1,
            started_at_ns=1,
            finished_at_ns=2,
            exit_code=None,
            target_hash=shard.target_hash,
            response_hashes=("a" * 64,),
            decision_hashes=(),
            transcript_hash=empty_hash,
            schedule_result_hash="",
            evidence_hashes=(),
            diagnostic_hashes=(),
            evidence_ids=(),
            diagnostic_ids=("broker-unavailable",),
            stderr_hash=empty_hash,
            stderr_path="",
            stderr_truncated=False,
            cleanup_receipt_path="",
            cleanup_receipt_hash="b" * 64,
            reason="external simulator unavailable",
        )

        attempt_ids = _append_discovery_results(manifest, (result,))

        self.assertEqual(len(attempt_ids), 1)
        row = json.loads(
            Path(manifest.discovery_ledger_path).read_text(encoding="utf-8")
        )
        self.assertEqual(row["stable_coverage_ids"], [])
        self.assertEqual(row["journey_ids"], [shard.target_ids[0]])

    def test_product_journey_preserves_frozen_seed_and_schedule_identity(self) -> None:
        target = self.targets / "01.json"
        payload = formal_journey_definitions()[0]
        journey = dict(payload["journey"])
        journey["journey_id"] = "g4-obligation-1"
        seed = 20260724
        schedule = build_journey_schedule(
            revision=self.revision,
            seed=seed,
            journey=journey,
        )
        schedule_payload_value = journey_schedule_payload(schedule)
        payload = {
            **payload,
            "journey": journey,
            "frozen_execution": {
                "obligation_id": journey["journey_id"],
                "seed": seed,
                "schedule_id": schedule.schedule_id,
                "schedule_hash": content_hash(schedule_payload_value),
                "subject_group": schedule.subject_group,
                "revision_binding": self.revision,
            },
            "simulator_attestation_contract": {
                "required": True,
                "identity_strength": "auditable_declaration_only",
                "scripted_actor_qualifies": False,
                "cryptographic_identity_claimed": False,
            },
        }
        target.write_text(json.dumps(payload), encoding="utf-8")
        observed = {}

        def factory(index, target_path, runtime, session, observed_seed, schedule_id):
            del index, target_path, runtime, session
            observed.update(seed=observed_seed, schedule_id=schedule_id)
            return (sys.executable, "-c", "pass")

        manifest = freeze_batch_manifest(
            repo_root=self.root,
            targets_dir=self.targets,
            manifest_path=self.root / ".agent" / "product-manifest.json",
            runtime_base=self.root / ".agent" / "product-runtime",
            shard_count=1,
            required_env_names=(),
            command_factory=factory,
        )
        shard = manifest.shards[0]
        self.assertEqual(shard.seed, seed)
        self.assertEqual(shard.schedule_id, schedule.schedule_id)
        self.assertEqual(shard.obligation_id, journey["journey_id"])
        self.assertTrue(shard.simulator_attestation_required)
        self.assertEqual(observed, {
            "seed": seed,
            "schedule_id": schedule.schedule_id,
        })

    def test_product_journey_rejects_frozen_schedule_hash_drift(self) -> None:
        target = self.targets / "01.json"
        payload = formal_journey_definitions()[0]
        journey = dict(payload["journey"])
        journey["journey_id"] = "g4-obligation-stale-schedule"
        seed = 20260724
        schedule = build_journey_schedule(
            revision=self.revision,
            seed=seed,
            journey=journey,
        )
        target.write_text(json.dumps({
            **payload,
            "journey": journey,
            "frozen_execution": {
                "obligation_id": journey["journey_id"],
                "seed": seed,
                "schedule_id": schedule.schedule_id,
                "schedule_hash": "0" * 64,
                "subject_group": schedule.subject_group,
                "revision_binding": self.revision,
            },
            "simulator_attestation_contract": {
                "required": True,
                "identity_strength": "auditable_declaration_only",
                "scripted_actor_qualifies": False,
                "cryptographic_identity_claimed": False,
            },
        }), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "schedule hash drifted"):
            freeze_batch_manifest(
                repo_root=self.root,
                targets_dir=self.targets,
                manifest_path=self.root / ".agent" / "product-manifest.json",
                runtime_base=self.root / ".agent" / "product-runtime",
                shard_count=1,
                required_env_names=(),
                command_factory=lambda *_args: (sys.executable, "-c", "pass"),
            )

    def test_all_shards_finish_once_with_five_terminal_classifications(self) -> None:
        modes = ["pass-slow", "product", "pass", "pass", "infrastructure"]
        self._write_targets(len(modes))
        markers = self.root / ".agent" / "markers"
        markers.mkdir()
        manifest_path = self.root / ".agent" / "manifest.json"
        manifest = freeze_batch_manifest(
            repo_root=self.root,
            targets_dir=self.targets,
            manifest_path=manifest_path,
            runtime_base=self.root / ".agent" / "runtime",
            shard_count=len(modes),
            command_factory=self._factory(modes, markers),
            timeout_policy=TimeoutPolicy(shard_seconds=5, decision_seconds=1, cleanup_seconds=1),
            stderr_cap_bytes=128,
        )

        async def broker(shard_id, context):
            index = int(shard_id.rsplit("-", 1)[1])
            if index == 3:
                return self._decision(context, response_hash="stale")
            if index == 4:
                raise ExternalDecisionBlocked("external simulator unavailable")
            return self._decision(context)

        result_path = self.root / ".agent" / "result-index.json"
        with patch(
            "tests.agent_live.batch_orchestrator.load_valid_evidence_reference",
            side_effect=self._accept_fake_evidence,
        ), patch(
            "tests.agent_live.batch_orchestrator.validate_pty_diagnostic_artifact",
            return_value=(True, ""),
        ):
            index = asyncio.run(run_batch(
                manifest, broker=broker, result_index_path=result_path
            ))
        self.assertEqual(index.execution_status, "discovery_complete")
        self.assertEqual(index.release_status, "not_evaluated")
        self.assertEqual(index.scheduled, 5)
        self.assertEqual(index.completed, 5)
        self.assertEqual(index.classification_counts, {
            "externally_blocked": 1,
            "infrastructure_interrupted": 1,
            "passed": 1,
            "product_failed": 1,
            "simulator_invalid": 1,
        })
        self.assertEqual({row.attempt_count for row in index.shards}, {1})
        self.assertEqual(len([path for path in markers.iterdir() if path.name.isdigit()]), 5)
        self.assertEqual(len(index.discovery_attempt_ids), 5)
        discovery_rows = Path(manifest.discovery_ledger_path).read_text().splitlines()
        self.assertEqual(len(discovery_rows), 5)
        for row in index.shards:
            self.assertTrue(Path(row.cleanup_receipt_path).is_file())
            receipt = json.loads(Path(row.cleanup_receipt_path).read_text(encoding="utf-8"))
            self.assertTrue(receipt["cleaned"])
            self.assertEqual(receipt["execution_id"], next(
                shard.execution_id for shard in manifest.shards if shard.shard_id == row.shard_id
            ))
            self.assertTrue(receipt["host_proof"]["cleaned"])
            self.assertTrue(receipt["container_proof"]["cleaned"])
        self.assertTrue(index.batch_survivor_proof["cleaned"])
        self.assertEqual(len(index.batch_survivor_proof["scans"]), 2)
        infrastructure = next(
            row for row in index.shards if row.classification == "infrastructure_interrupted"
        )
        stderr = Path(infrastructure.stderr_path).read_text(encoding="utf-8")
        self.assertNotIn("super-secret-token", stderr)
        self.assertIn("***REDACTED***", stderr)
        self.assertLessEqual(len(Path(infrastructure.stderr_path).read_bytes()), 128)
        self.assertTrue(infrastructure.stderr_truncated)
        self.assertFalse(result_path.stat().st_mode & stat.S_IWUSR)
        with self.assertRaises(FileExistsError):
            asyncio.run(run_batch(manifest, broker=broker, result_index_path=result_path))

    def test_complete_32_shard_mixed_lane_infrastructure_stress_wave(self) -> None:
        self._write_mixed_targets()
        modes = ["fragmented", "infrastructure", "stubborn-after-result", "pass"]
        modes.extend(["pass"] * (DEFAULT_SHARD_COUNT - len(modes)))
        manifest = freeze_batch_manifest(
            repo_root=self.root,
            targets_dir=self.targets,
            manifest_path=self.root / ".agent" / "stress-manifest.json",
            runtime_base=self.root / ".agent" / "stress-runtime",
            shard_count=DEFAULT_SHARD_COUNT,
            max_concurrency=DEFAULT_MAX_CONCURRENCY,
            required_env_names=(),
            command_factory=self._mixed_factory(
                modes, self.root / ".agent" / "stress-markers"
            ),
            timeout_policy=TimeoutPolicy(
                shard_seconds=10, decision_seconds=1, cleanup_seconds=3
            ),
            stderr_cap_bytes=128,
        )
        self.assertEqual(
            {lane: sum(shard.lane == lane for shard in manifest.shards) for lane in ("edge", "journey")},
            {"edge": 24, "journey": 8},
        )

        def broker(shard_id, context):
            index = int(shard_id.rsplit("-", 1)[1])
            if index == 4:
                raise ExternalDecisionBlocked("synthetic blocked decision")
            if "schedule" in context:
                schedule = context["schedule"]
                return {
                    "previous_response_hash": context["previous_response_hash"],
                    "user_message": "A response-driven Journey turn",
                    "persona": schedule["persona"],
                    "mission": schedule["mission"],
                    "rationale": "Selected after the complete Journey response",
                    "risk_factor_ids": [],
                }
            return self._decision(context)

        with patch(
            "tests.agent_live.batch_orchestrator.load_valid_evidence_reference",
            side_effect=self._accept_fake_evidence,
        ), patch(
            "tests.agent_live.batch_orchestrator.validate_journey_evidence_artifact"
        ):
            index = asyncio.run(run_batch(
                manifest,
                broker=broker,
                result_index_path=self.root / ".agent" / "stress-index.json",
            ))

        self.assertEqual(index.scheduled, DEFAULT_SHARD_COUNT)
        self.assertEqual(index.completed, DEFAULT_SHARD_COUNT)
        self.assertEqual(index.classification_counts["externally_blocked"], 1)
        self.assertEqual(
            index.classification_counts["infrastructure_interrupted"],
            2,
            [(row.shard_id, row.classification, row.reason) for row in index.shards],
        )
        self.assertEqual(index.classification_counts["passed"], 29)
        self.assertTrue(index.batch_survivor_proof["cleaned"])
        passed = [row for row in index.shards if row.classification == "passed"]
        evidence_hashes = [item for row in passed for item in row.evidence_hashes]
        self.assertEqual(len(evidence_hashes), len(set(evidence_hashes)))
        self.assertEqual(len(index.discovery_attempt_ids), DEFAULT_SHARD_COUNT)
        for row in index.shards:
            receipt = json.loads(Path(row.cleanup_receipt_path).read_text())
            self.assertTrue(receipt["cleaned"])

    def test_adversarial_result_artifacts_are_never_counted_as_passed(self) -> None:
        modes = [
            "forged-session", "forged-revision", "forged-target",
            "forged-schedule", "path-escape", "pass",
        ]
        self._write_targets(len(modes))
        manifest = freeze_batch_manifest(
            repo_root=self.root,
            targets_dir=self.targets,
            manifest_path=self.root / ".agent" / "adversarial-manifest.json",
            runtime_base=self.root / ".agent" / "adversarial-runtime",
            shard_count=len(modes),
            command_factory=self._factory(modes, self.root / ".agent" / "adversarial-markers"),
            timeout_policy=TimeoutPolicy(shard_seconds=5, decision_seconds=1, cleanup_seconds=1),
        )
        index = asyncio.run(run_batch(
            manifest,
            broker=lambda shard_id, context: self._decision(context),
            result_index_path=self.root / ".agent" / "adversarial-index.json",
        ))
        self.assertEqual(index.classification_counts["passed"], 0)
        self.assertEqual(index.classification_counts["infrastructure_interrupted"], len(modes))
        reasons = "\n".join(row.reason for row in index.shards)
        self.assertIn("another session", reasons)
        self.assertIn("repository revision mismatch", reasons)
        self.assertIn("targets differ", reasons)
        self.assertIn("schedule result identity mismatch", reasons)
        self.assertIn("escapes the shard runtime roots", reasons)
        self.assertIn("invalid evidence artifact", reasons)

    def test_sync_broker_runs_off_loop_and_times_out(self) -> None:
        self._write_targets(1)
        manifest = freeze_batch_manifest(
            repo_root=self.root,
            targets_dir=self.targets,
            manifest_path=self.root / ".agent" / "broker-manifest.json",
            runtime_base=self.root / ".agent" / "broker-runtime",
            shard_count=1,
            command_factory=self._factory(["pass"], self.root / ".agent" / "broker-markers"),
            timeout_policy=TimeoutPolicy(shard_seconds=2, decision_seconds=0.05, cleanup_seconds=1),
        )
        broker_threads: list[int] = []
        main_thread = threading.get_ident()

        def blocking_broker(shard_id, context):
            del shard_id
            broker_threads.append(threading.get_ident())
            time.sleep(0.2)
            return self._decision(context)

        index = asyncio.run(run_batch(
            manifest,
            broker=blocking_broker,
            result_index_path=self.root / ".agent" / "broker-index.json",
        ))
        self.assertEqual(index.classification_counts["externally_blocked"], 1)
        self.assertEqual(len(broker_threads), 1)
        self.assertNotEqual(broker_threads[0], main_thread)

    def test_batch_cancellation_awaits_shard_cleanup_and_writes_interrupted_index(self) -> None:
        self._write_targets(1)
        markers = self.root / ".agent" / "cancel-markers"
        manifest = freeze_batch_manifest(
            repo_root=self.root,
            targets_dir=self.targets,
            manifest_path=self.root / ".agent" / "cancel-manifest.json",
            runtime_base=self.root / ".agent" / "cancel-runtime",
            shard_count=1,
            command_factory=self._factory(["pass"], markers),
            timeout_policy=TimeoutPolicy(
                shard_seconds=5, decision_seconds=5, cleanup_seconds=1
            ),
        )
        result_path = self.root / ".agent" / "cancel-index.json"

        async def blocked_broker(_shard_id, _context):
            await asyncio.sleep(60)
            raise AssertionError("cancelled broker resumed")

        async def scenario():
            task = asyncio.create_task(run_batch(
                manifest,
                broker=blocked_broker,
                result_index_path=result_path,
            ))
            marker = markers / "01"
            deadline = asyncio.get_running_loop().time() + 2
            while not marker.exists():
                if asyncio.get_running_loop().time() >= deadline:
                    self.fail("worker did not start before cancellation")
                await asyncio.sleep(0.01)
            task.cancel()
            return await task

        index = asyncio.run(scenario())

        self.assertEqual(index.execution_status, "infrastructure_interrupted")
        self.assertEqual(index.classification_counts["infrastructure_interrupted"], 1)
        self.assertTrue(result_path.is_file())
        self.assertTrue(index.batch_survivor_proof["cleaned"])
        receipt = json.loads(Path(index.shards[0].cleanup_receipt_path).read_text())
        self.assertTrue(receipt["cleaned"])
        self.assertIn("batch cancellation requested", index.shards[0].reason)

    def test_batch_interruption_event_converges_without_cancelling_batch_owner(self) -> None:
        self._write_targets(1)
        markers = self.root / ".agent" / "interrupt-markers"
        manifest = freeze_batch_manifest(
            repo_root=self.root,
            targets_dir=self.targets,
            manifest_path=self.root / ".agent" / "interrupt-manifest.json",
            runtime_base=self.root / ".agent" / "interrupt-runtime",
            shard_count=1,
            command_factory=self._factory(["pass"], markers),
            timeout_policy=TimeoutPolicy(
                shard_seconds=5, decision_seconds=5, cleanup_seconds=2
            ),
        )
        result_path = self.root / ".agent" / "interrupt-index.json"

        async def blocked_broker(_shard_id, _context):
            await asyncio.sleep(60)
            raise AssertionError("interrupted broker resumed")

        async def scenario():
            interruption_event = asyncio.Event()
            task = asyncio.create_task(run_batch(
                manifest,
                broker=blocked_broker,
                result_index_path=result_path,
                interruption_event=interruption_event,
            ))
            marker = markers / "01"
            deadline = asyncio.get_running_loop().time() + 2
            while not marker.exists():
                if asyncio.get_running_loop().time() >= deadline:
                    self.fail("worker did not start before interruption")
                await asyncio.sleep(0.01)
            interruption_event.set()
            self.assertFalse(task.cancelled())
            return await task

        index = asyncio.run(scenario())

        self.assertEqual(index.execution_status, "infrastructure_interrupted")
        self.assertEqual(index.classification_counts["infrastructure_interrupted"], 1)
        self.assertTrue(index.batch_survivor_proof["cleaned"])
        self.assertTrue(result_path.is_file())
        receipt = json.loads(Path(index.shards[0].cleanup_receipt_path).read_text())
        self.assertTrue(receipt["cleaned"])

    def test_bounded_interruption_cleans_active_and_not_started_shards(self) -> None:
        self._write_targets(3)
        markers = self.root / ".agent" / "bounded-interrupt-markers"
        manifest = freeze_batch_manifest(
            repo_root=self.root,
            targets_dir=self.targets,
            manifest_path=self.root / ".agent" / "bounded-interrupt-manifest.json",
            runtime_base=self.root / ".agent" / "bounded-interrupt-runtime",
            shard_count=3,
            max_concurrency=1,
            command_factory=self._factory(["pass"] * 3, markers),
            timeout_policy=TimeoutPolicy(
                shard_seconds=5, decision_seconds=5, cleanup_seconds=1
            ),
        )

        async def blocked_broker(_shard_id, _context):
            await asyncio.sleep(60)
            raise AssertionError("interrupted broker resumed")

        async def scenario():
            interruption_event = asyncio.Event()
            task = asyncio.create_task(run_batch(
                manifest,
                broker=blocked_broker,
                result_index_path=(
                    self.root / ".agent" / "bounded-interrupt-index.json"
                ),
                interruption_event=interruption_event,
            ))
            deadline = asyncio.get_running_loop().time() + 2
            while not (markers / "01").exists():
                if asyncio.get_running_loop().time() >= deadline:
                    self.fail("bounded worker did not start before interruption")
                await asyncio.sleep(0.01)
            self.assertFalse((markers / "02").exists())
            self.assertFalse((markers / "03").exists())
            interruption_event.set()
            return await task

        index = asyncio.run(scenario())

        self.assertEqual(index.execution_status, "infrastructure_interrupted")
        self.assertEqual(
            index.classification_counts["infrastructure_interrupted"], 3
        )
        receipts = [
            json.loads(Path(row.cleanup_receipt_path).read_text())
            for row in index.shards
        ]
        self.assertTrue(all(receipt["cleaned"] for receipt in receipts))
        self.assertEqual(
            sum(receipt["actions"] == ["worker_not_started"] for receipt in receipts),
            2,
        )
        self.assertTrue(index.batch_survivor_proof["cleaned"])

    @unittest.skipUnless(sys.platform.startswith("linux"), "formal control plane is Linux-only")
    def test_filesystem_broker_process_sigterm_persists_truthful_cleanup(self) -> None:
        self._write_targets(1)
        markers = self.root / ".agent" / "signal-markers"
        manifest_path = self.root / ".agent" / "signal-manifest.json"
        result_path = self.root / ".agent" / "signal-index.json"
        broker_root = self.root / ".agent" / "signal-broker"
        manifest = freeze_batch_manifest(
            repo_root=self.root,
            targets_dir=self.targets,
            manifest_path=manifest_path,
            runtime_base=self.root / ".agent" / "signal-runtime",
            shard_count=1,
            command_factory=self._factory(["pass"], markers),
            timeout_policy=TimeoutPolicy(
                shard_seconds=10, decision_seconds=10, cleanup_seconds=1
            ),
        )
        source_root = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(source_root)
        process = subprocess.Popen(
            (
                sys.executable,
                "-m",
                "tests.agent_live.filesystem_decision_broker",
                "run",
                "--manifest",
                str(manifest_path),
                "--result-index",
                str(result_path),
                "--broker-root",
                str(broker_root),
                "--decision-timeout",
                "10",
            ),
            cwd=source_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            deadline = time.monotonic() + 8
            request_dir = broker_root / "requests"
            while not tuple(request_dir.glob("*.json")):
                if process.poll() is not None:
                    stdout, stderr = process.communicate()
                    self.fail(f"broker exited before signal: {stdout}\n{stderr}")
                if time.monotonic() >= deadline:
                    process.kill()
                    stdout, stderr = process.communicate(timeout=5)
                    self.fail(
                        "broker did not publish a response-bound request: "
                        f"{stdout}\n{stderr}"
                    )
                time.sleep(0.01)
            process.send_signal(signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=5)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2)

        self.assertEqual(process.returncode, 128 + signal.SIGTERM, (stdout, stderr))
        self.assertTrue(result_path.is_file())
        index = json.loads(result_path.read_text())
        self.assertEqual(index["execution_status"], "infrastructure_interrupted")
        self.assertTrue(index["batch_survivor_proof"]["cleaned"])
        shard = index["shards"][0]
        self.assertEqual(shard["classification"], "infrastructure_interrupted")
        receipt = json.loads(Path(shard["cleanup_receipt_path"]).read_text())
        self.assertTrue(receipt["cleaned"])
        self.assertEqual(receipt["execution_id"], manifest.shards[0].execution_id)

    def test_run_revalidates_frozen_target_before_discovery_append(self) -> None:
        self._write_targets(1)
        manifest = freeze_batch_manifest(
            repo_root=self.root,
            targets_dir=self.targets,
            manifest_path=self.root / ".agent" / "mutation-manifest.json",
            runtime_base=self.root / ".agent" / "mutation-runtime",
            shard_count=1,
            command_factory=self._factory(
                ["mutate-target"], self.root / ".agent" / "mutation-markers"
            ),
        )
        result_path = self.root / ".agent" / "mutation-index.json"
        with self.assertRaisesRegex(ValueError, "frozen target changed"):
            asyncio.run(run_batch(
                manifest,
                broker=lambda shard_id, context: self._decision(context),
                result_index_path=result_path,
            ))
        self.assertFalse(result_path.exists())
        self.assertFalse(Path(manifest.discovery_ledger_path).exists())

    def test_freeze_rejects_future_dialogue_fields_and_unknown_edges(self) -> None:
        self._write_targets(1)
        path = self.targets / "01.json"
        payload = json.loads(path.read_text())
        payload["targets"][0]["user_message"] = "prewritten future turn"
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "future-dialogue fields"):
            freeze_batch_manifest(
                repo_root=self.root,
                targets_dir=self.targets,
                manifest_path=self.root / ".agent" / "future-manifest.json",
                runtime_base=self.root / ".agent" / "future-runtime",
                shard_count=1,
                command_factory=self._factory(["pass"], self.root / ".agent" / "future-markers"),
            )

        payload["targets"][0].pop("user_message")
        payload["targets"][0]["edge_key"] = "not-an-authoritative-edge"
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "unknown authoritative ledger edge"):
            freeze_batch_manifest(
                repo_root=self.root,
                targets_dir=self.targets,
                manifest_path=self.root / ".agent" / "unknown-manifest.json",
                runtime_base=self.root / ".agent" / "unknown-runtime",
                shard_count=1,
                command_factory=self._factory(["pass"], self.root / ".agent" / "unknown-markers"),
            )

    def test_default_worker_command_uses_docker_linux_boundary(self) -> None:
        self._write_targets(1)
        timeout_policy = TimeoutPolicy(
            shard_seconds=20,
            decision_seconds=7,
            cleanup_seconds=2,
        )
        manifest = freeze_batch_manifest(
            repo_root=self.root,
            targets_dir=self.targets,
            manifest_path=self.root / ".agent" / "docker-manifest.json",
            runtime_base=self.root / ".agent" / "docker-runtime",
            shard_count=1,
            timeout_policy=timeout_policy,
        )
        command = manifest.shards[0].command
        self.assertEqual(command[:2], ("docker", "exec"))
        self.assertIn(
            f"ANYCHAIN_CHAOS_EXECUTION_ID={manifest.shards[0].execution_id}",
            command,
        )
        self.assertTrue(any(
            item.startswith("ANYCHAIN_CHAOS_INNER_CLEANUP_RECEIPT_DIR=/workspace/")
            for item in command
        ))
        self.assertIn("blockchain-node-benchmark-bench-1", command)
        self.assertIn("/workspace/.agent/targets/01.json", command)
        timeout_index = command.index("--decision-timeout-seconds")
        self.assertGreater(
            float(command[timeout_index + 1]),
            timeout_policy.decision_seconds,
        )

    def test_batch_execution_rejects_non_linux_control_plane(self) -> None:
        self._write_targets(1)
        manifest = freeze_batch_manifest(
            repo_root=self.root,
            targets_dir=self.targets,
            manifest_path=self.root / ".agent" / "host-boundary-manifest.json",
            runtime_base=self.root / ".agent" / "host-boundary-runtime",
            shard_count=1,
            command_factory=self._factory(
                ["pass"], self.root / ".agent" / "host-boundary-markers"
            ),
        )
        with patch("tests.agent_live.batch_orchestrator.sys.platform", "darwin"):
            with self.assertRaisesRegex(RuntimeError, "Linux control plane"):
                asyncio.run(run_batch(
                    manifest,
                    broker=lambda shard_id, context: self._decision(context),
                    result_index_path=self.root / ".agent" / "host-boundary-index.json",
                ))
        self.assertFalse((self.root / ".agent" / "host-boundary-runtime").exists())

    def test_execution_ids_are_unique_and_inner_receipts_fail_closed(self) -> None:
        modes = [
            "pass", "missing-inner", "duplicate-inner", "tampered-inner-hash",
            "wrong-inner-execution", "inner-survivor", "inner-not-cleaned",
        ]
        self._write_targets(len(modes))
        manifest = freeze_batch_manifest(
            repo_root=self.root,
            targets_dir=self.targets,
            manifest_path=self.root / ".agent" / "inner-manifest.json",
            runtime_base=self.root / ".agent" / "inner-runtime",
            shard_count=len(modes),
            command_factory=self._factory(modes, self.root / ".agent" / "inner-markers"),
            timeout_policy=TimeoutPolicy(shard_seconds=2, decision_seconds=1, cleanup_seconds=0.2),
        )
        self.assertEqual(
            len({shard.execution_id for shard in manifest.shards}), len(modes)
        )

        with patch(
            "tests.agent_live.batch_orchestrator.load_valid_evidence_reference",
            side_effect=self._accept_fake_evidence,
        ):
            index = asyncio.run(run_batch(
                manifest,
                broker=lambda shard_id, context: self._decision(context),
                result_index_path=self.root / ".agent" / "inner-index.json",
            ))

        self.assertEqual(index.classification_counts["passed"], 1)
        self.assertEqual(index.classification_counts["infrastructure_interrupted"], 6)
        self.assertEqual(index.execution_status, "infrastructure_interrupted")
        failures = "\n".join(row.reason for row in index.shards[1:])
        self.assertIn("found 0", failures)
        self.assertIn("found 2", failures)
        self.assertIn("content hash is stale", failures)
        self.assertIn("execution id mismatch", failures)
        self.assertIn("zero-survivor scans are not clean", failures)
        self.assertIn("container proof", failures)

    def test_stubborn_worker_uses_registered_term_then_kill(self) -> None:
        self._write_targets(1)
        manifest = freeze_batch_manifest(
            repo_root=self.root,
            targets_dir=self.targets,
            manifest_path=self.root / ".agent" / "stubborn-manifest.json",
            runtime_base=self.root / ".agent" / "stubborn-runtime",
            shard_count=1,
            command_factory=self._factory(
                ["stubborn-after-result"], self.root / ".agent" / "stubborn-markers"
            ),
            timeout_policy=TimeoutPolicy(shard_seconds=0.1, decision_seconds=1, cleanup_seconds=0.2),
        )
        index = asyncio.run(run_batch(
            manifest,
            broker=lambda shard_id, context: self._decision(context),
            result_index_path=self.root / ".agent" / "stubborn-index.json",
        ))

        receipt = json.loads(Path(index.shards[0].cleanup_receipt_path).read_text())
        host = receipt["host_proof"]
        term = {row["identity"]["pid"] for row in host["term_decisions"]}
        killed = {row["identity"]["pid"] for row in host["kill_decisions"]}
        self.assertTrue(killed)
        self.assertTrue(killed.issubset(term))
        registered = {row["pid"] for row in host["registered_processes"]}
        self.assertTrue(killed.issubset(registered))
        self.assertTrue(receipt["cleaned"])

    def test_final_batch_survivor_scan_and_synthetic_results_fail_closed(self) -> None:
        self._write_targets(1)
        manifest = freeze_batch_manifest(
            repo_root=self.root,
            targets_dir=self.targets,
            manifest_path=self.root / ".agent" / "final-scan-manifest.json",
            runtime_base=self.root / ".agent" / "final-scan-runtime",
            shard_count=1,
            command_factory=self._factory(["pass"], self.root / ".agent" / "final-scan-markers"),
        )
        survivor = {
            "execution_id": manifest.shards[0].execution_id,
            "identity": {
                "pid": 99999, "pgid": 99999, "start_ticks": 1,
                "pid_namespace_inode": 1,
            },
        }
        scan = {"scanned_at_ns": 1, "survivors": [survivor], "scan_errors": []}
        with patch(
            "tests.agent_live.batch_orchestrator.load_valid_evidence_reference",
            side_effect=self._accept_fake_evidence,
        ), patch(
            "tests.agent_live.batch_orchestrator._scan_batch_execution_ids",
            return_value=scan,
        ):
            index = asyncio.run(run_batch(
                manifest,
                broker=lambda shard_id, context: self._decision(context),
                result_index_path=self.root / ".agent" / "final-scan-index.json",
            ))
        self.assertEqual(index.execution_status, "infrastructure_interrupted")
        self.assertFalse(index.batch_survivor_proof["cleaned"])
        self.assertEqual(index.shards[0].classification, "infrastructure_interrupted")

        self._write_targets(1)
        synthetic_manifest = freeze_batch_manifest(
            repo_root=self.root,
            targets_dir=self.targets,
            manifest_path=self.root / ".agent" / "synthetic-manifest.json",
            runtime_base=self.root / ".agent" / "synthetic-runtime",
            shard_count=1,
            command_factory=self._factory(["pass"], self.root / ".agent" / "synthetic-markers"),
        )

        async def explode(*_args, **_kwargs):
            raise RuntimeError("orchestrator task exploded")

        with patch("tests.agent_live.batch_orchestrator._run_shard", side_effect=explode):
            synthetic = asyncio.run(run_batch(
                synthetic_manifest,
                broker=lambda shard_id, context: self._decision(context),
                result_index_path=self.root / ".agent" / "synthetic-index.json",
            ))
        synthetic_receipt = json.loads(
            Path(synthetic.shards[0].cleanup_receipt_path).read_text()
        )
        self.assertFalse(synthetic_receipt["cleaned"])
        self.assertEqual(synthetic.shards[0].classification, "infrastructure_interrupted")

    def test_batch_survivor_scan_matches_only_the_exact_execution_id(self) -> None:
        execution_id = f"batch-exact-{os.getpid()}-{time.time_ns()}"
        exact_env = os.environ.copy()
        exact_env["ANYCHAIN_CHAOS_EXECUTION_ID"] = execution_id
        near_env = os.environ.copy()
        near_env["ANYCHAIN_CHAOS_EXECUTION_ID"] = execution_id + "-near"
        exact = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            env=exact_env,
            start_new_session=True,
        )
        near = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            env=near_env,
            start_new_session=True,
        )
        try:
            scan = _scan_batch_execution_ids((execution_id,))
            survivor_pids = {row["identity"]["pid"] for row in scan["survivors"]}
            self.assertIn(exact.pid, survivor_pids)
            self.assertNotIn(near.pid, survivor_pids)
            self.assertEqual(scan["scan_errors"], [])
        finally:
            for process in (exact, near):
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)

    @staticmethod
    def _decision(context, *, response_hash=None):
        target = context["scheduled_target"]
        return {
            "previous_response_hash": response_hash or context["previous_response_hash"],
            "user_message": "A response-driven user turn",
            "persona": target["persona"],
            "goal": target["goal"],
            "rationale": "Generated only after reading the complete response",
            "target_coverage_ids": [target["edge_key"]],
        }


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.agent_live.batch_orchestrator import (
    ExternalDecisionBlocked,
    _SimulatorInvalid,
    _validate_decision,
)
from tests.agent_live.filesystem_decision_broker import (
    FilesystemDecisionBroker,
    _run_with_signal_cleanup,
    _write_channel_payload_descriptor,
    _write_immutable_json,
    pending_requests,
    submit_decision,
)
from tests.agent_live.coverage_evidence import content_hash
from tests.agent_live.retained_regression_attestations import (
    build_source_contract,
    build_variant_contract,
)


class FilesystemDecisionBrokerTest(unittest.TestCase):
    def test_close_releases_every_pending_decision_wait(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            broker = FilesystemDecisionBroker(
                root,
                batch_id="batch-close",
                timeout_seconds=30,
                poll_seconds=0.01,
            )
            observed: list[BaseException] = []

            def run() -> None:
                try:
                    asyncio.run(broker("closing-shard", self._context()))
                except BaseException as exc:
                    observed.append(exc)

            thread = threading.Thread(target=run)
            thread.start()
            deadline = time.monotonic() + 1
            while not pending_requests(root) and time.monotonic() < deadline:
                time.sleep(0.01)
            broker.close()
            thread.join(timeout=1)

            self.assertFalse(thread.is_alive())
            self.assertEqual(len(observed), 1)
            self.assertIsInstance(observed[0], ExternalDecisionBlocked)
            self.assertIn("shutting down", str(observed[0]))
            self.assertEqual(pending_requests(root), ())

    def test_expired_request_is_not_pending_and_cannot_leave_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            broker = FilesystemDecisionBroker(
                root, batch_id="batch-expired", timeout_seconds=0.05, poll_seconds=0.01
            )
            context = self._context()
            observed: list[BaseException] = []

            def run() -> None:
                try:
                    asyncio.run(broker("expired-shard", context))
                except BaseException as exc:
                    observed.append(exc)

            thread = threading.Thread(target=run)
            thread.start()
            deadline = time.monotonic() + 1
            while not pending_requests(root) and time.monotonic() < deadline:
                time.sleep(0.01)
            request_id = pending_requests(root)[0]["request_id"]
            thread.join(timeout=1)

            self.assertFalse(thread.is_alive())
            self.assertTrue(observed)
            self.assertEqual(pending_requests(root), ())
            with self.assertRaises(OSError):
                submit_decision(
                    root,
                    request_id=request_id,
                    user_message="late",
                    rationale="expired",
                )
            self.assertFalse((root / "committed" / f"{request_id}.json").exists())

    def test_failed_channel_delivery_rolls_back_decision_and_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            broker = FilesystemDecisionBroker(
                root, batch_id="batch-write-failure", timeout_seconds=0.1, poll_seconds=0.01
            )
            observed: list[BaseException] = []

            def run() -> None:
                try:
                    asyncio.run(broker("failed-delivery", self._context()))
                except BaseException as exc:
                    observed.append(exc)

            thread = threading.Thread(target=run)
            thread.start()
            deadline = time.monotonic() + 1
            while not pending_requests(root) and time.monotonic() < deadline:
                time.sleep(0.01)
            request_id = pending_requests(root)[0]["request_id"]

            with patch(
                "tests.agent_live.filesystem_decision_broker."
                "_write_channel_payload_descriptor",
                side_effect=BrokenPipeError("reader disappeared"),
            ):
                with self.assertRaises(BrokenPipeError):
                    submit_decision(
                        root,
                        request_id=request_id,
                        user_message="continue",
                        rationale="exercise transaction rollback",
                        actor_kind="codex",
                        actor_task_id="task-write-failure",
                        actor_model="gpt-test",
                    )

            self.assertFalse((root / "committed" / f"{request_id}.json").exists())
            thread.join(timeout=1)
            self.assertFalse(thread.is_alive())
            self.assertTrue(observed)

    def test_concurrent_submit_delivers_exactly_one_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            broker = FilesystemDecisionBroker(
                root, batch_id="batch-concurrent", timeout_seconds=2, poll_seconds=0.01
            )
            observed: list[dict] = []
            worker = threading.Thread(
                target=lambda: observed.append(
                    dict(asyncio.run(broker("concurrent-shard", self._context())))
                )
            )
            worker.start()
            deadline = time.monotonic() + 1
            while not pending_requests(root) and time.monotonic() < deadline:
                time.sleep(0.01)
            request_id = pending_requests(root)[0]["request_id"]
            barrier = threading.Barrier(3)
            successes: list[str] = []
            failures: list[BaseException] = []

            def submit(label: str) -> None:
                barrier.wait()
                try:
                    submit_decision(
                        root,
                        request_id=request_id,
                        user_message=label,
                        rationale=f"concurrent submit {label}",
                    )
                    successes.append(label)
                except BaseException as exc:
                    failures.append(exc)

            submitters = [
                threading.Thread(target=submit, args=(label,))
                for label in ("first", "second")
            ]
            for submitter in submitters:
                submitter.start()
            barrier.wait()
            for submitter in submitters:
                submitter.join(timeout=2)
            worker.join(timeout=2)

            self.assertFalse(worker.is_alive())
            self.assertEqual(len(successes), 1)
            self.assertEqual(len(failures), 1)
            self.assertIn(observed[0]["user_message"], {"first", "second"})
            self.assertEqual(len(list((root / "committed").glob("*.json"))), 1)

    def test_large_multiline_decision_is_delivered_without_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            broker = FilesystemDecisionBroker(
                root, batch_id="batch-large", timeout_seconds=2, poll_seconds=0.01
            )
            observed: list[dict] = []
            worker = threading.Thread(
                target=lambda: observed.append(
                    dict(asyncio.run(broker("large-shard", self._context())))
                )
            )
            worker.start()
            deadline = time.monotonic() + 1
            while not pending_requests(root) and time.monotonic() < deadline:
                time.sleep(0.01)
            request_id = pending_requests(root)[0]["request_id"]
            message = "\n".join(
                json.dumps({"line": index, "payload": "x" * 256})
                for index in range(64)
            )

            submit_decision(
                root,
                request_id=request_id,
                user_message=message,
                rationale="Exercise realistic pasted JSON evidence.",
            )
            worker.join(timeout=2)

            self.assertFalse(worker.is_alive())
            self.assertEqual(observed[0]["user_message"], message)
            self.assertGreater(len(message.encode("utf-8")), 4096)

    def test_partial_failed_frame_is_discarded_before_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            broker = FilesystemDecisionBroker(
                root, batch_id="batch-retry", timeout_seconds=2, poll_seconds=0.01
            )
            observed: list[dict] = []
            worker = threading.Thread(
                target=lambda: observed.append(
                    dict(asyncio.run(broker("retry-shard", self._context())))
                )
            )
            worker.start()
            deadline = time.monotonic() + 1
            while not pending_requests(root) and time.monotonic() < deadline:
                time.sleep(0.01)
            request_id = pending_requests(root)[0]["request_id"]

            def write_partial_then_fail(descriptor: int, payload: bytes) -> None:
                os.write(descriptor, payload[:128])
                raise BrokenPipeError("simulated interrupted submitter")

            with patch(
                "tests.agent_live.filesystem_decision_broker."
                "_write_channel_payload_descriptor",
                side_effect=write_partial_then_fail,
            ):
                with self.assertRaises(BrokenPipeError):
                    submit_decision(
                        root,
                        request_id=request_id,
                        user_message="partial",
                        rationale="simulate a failed writer",
                    )

            submit_decision(
                root,
                request_id=request_id,
                user_message="complete retry",
                rationale="retry after the interrupted frame",
            )
            worker.join(timeout=2)
            self.assertFalse(worker.is_alive())
            self.assertEqual(observed[0]["user_message"], "complete retry")

    def test_acknowledgement_is_bound_to_the_current_submitter_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            broker = FilesystemDecisionBroker(
                root, batch_id="batch-ack", timeout_seconds=2, poll_seconds=0.01
            )
            observed: list[dict] = []
            worker = threading.Thread(
                target=lambda: observed.append(
                    dict(asyncio.run(broker("ack-shard", self._context())))
                )
            )
            worker.start()
            deadline = time.monotonic() + 1
            while not pending_requests(root) and time.monotonic() < deadline:
                time.sleep(0.01)
            request_id = pending_requests(root)[0]["request_id"]
            allow_commit = threading.Event()
            second_written = threading.Event()
            write_count = 0
            real_immutable_write = _write_immutable_json

            def controlled_wire_write(descriptor: int, payload: bytes) -> None:
                nonlocal write_count
                write_count += 1
                _write_channel_payload_descriptor(descriptor, payload)
                if write_count == 1:
                    raise RuntimeError("first submitter crashed after delivery")
                second_written.set()

            def delayed_commit(path: Path, value: dict) -> None:
                if path.parent.name == "committed":
                    self.assertTrue(allow_commit.wait(timeout=1))
                real_immutable_write(path, value)

            second_failures: list[BaseException] = []

            def submit_second() -> None:
                try:
                    submit_decision(
                        root,
                        request_id=request_id,
                        user_message="second decision",
                        rationale="must not claim the first decision acknowledgement",
                    )
                except BaseException as exc:
                    second_failures.append(exc)

            with (
                patch(
                    "tests.agent_live.filesystem_decision_broker."
                    "_write_channel_payload_descriptor",
                    side_effect=controlled_wire_write,
                ),
                patch(
                    "tests.agent_live.filesystem_decision_broker."
                    "_write_immutable_json",
                    side_effect=delayed_commit,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "first submitter crashed"):
                    submit_decision(
                        root,
                        request_id=request_id,
                        user_message="first decision",
                        rationale="delivered before the submitter crashes",
                    )
                second = threading.Thread(target=submit_second)
                second.start()
                self.assertTrue(second_written.wait(timeout=1))
                allow_commit.set()
                second.join(timeout=2)

            worker.join(timeout=2)
            self.assertFalse(worker.is_alive())
            self.assertEqual(observed[0]["user_message"], "first decision")
            self.assertEqual(len(second_failures), 1)
            self.assertIsInstance(second_failures[0], ExternalDecisionBlocked)
            self.assertIn("another external decision", str(second_failures[0]))

    def test_immutable_write_failure_leaves_no_final_or_temporary_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "records" / "record.json"
            real_fsync = os.fsync
            directory_syncs = 0

            def fail_first_directory_sync(descriptor: int) -> None:
                nonlocal directory_syncs
                if os.path.isdir(f"/proc/self/fd/{descriptor}"):
                    directory_syncs += 1
                    if directory_syncs == 1:
                        raise OSError("directory fsync failed")
                real_fsync(descriptor)

            with patch(
                "tests.agent_live.filesystem_decision_broker.os.fsync",
                side_effect=fail_first_directory_sync,
            ):
                with self.assertRaisesRegex(OSError, "directory fsync failed"):
                    _write_immutable_json(path, {"value": 1})

            self.assertFalse(path.exists())
            self.assertEqual(list(path.parent.glob(".*.tmp")), [])
            _write_immutable_json(path, {"value": 1})
            self.assertEqual(json.loads(path.read_text()), {"value": 1})

    def _context(self):
        return {
            "session_id": "broker-session",
            "turn_index": 1,
            "previous_response_hash": "a" * 64,
            "previous_response_received_at_ns": 1,
            "observed_edge_keys": [],
            "previous_agent_response": (
                "Choose a mode. endpoint=https://node.example/"
                "abcdefghijklmnopqrstuvwxyz123456"
            ),
            "scheduled_target": {
                "persona": "returning operator",
                "goal": "change configuration safely",
                "edge_key": "coverage:resume:modify",
            },
        }

    def _verifier_input_contract(self):
        source = build_source_contract({
            "case_id": "broker-retained-case",
            "source_turns_hash": content_hash(("source turn",)),
            "source_steps": [{
                "step_id": "source-1",
                "turn_index": 1,
                "semantic_role": "capability_consultation",
            }],
        })
        variant = build_variant_contract(
            variant="isomorphic",
            source_contract=source,
            declaration={
                "relation": "isomorphic_meaning",
                "minimum_attestations": 1,
            },
        )
        return {
            "mode": "response_driven_isomorphic",
            "source_contract": source,
            "source_contract_hash": content_hash(source),
            "variant_contract": variant,
            "variant_contract_hash": content_hash(variant),
        }

    def test_response_driven_round_trip_is_bound_and_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            broker = FilesystemDecisionBroker(
                root, batch_id="batch-1", timeout_seconds=2, poll_seconds=0.01
            )
            observed = []
            thread = threading.Thread(
                target=lambda: observed.append(asyncio.run(broker("shard-1", self._context())))
            )
            thread.start()
            deadline = time.monotonic() + 1
            while not pending_requests(root) and time.monotonic() < deadline:
                time.sleep(0.01)
            pending = pending_requests(root)
            self.assertEqual(len(pending), 1)
            encoded = json.dumps(pending[0])
            self.assertNotIn("abcdefghijklmnopqrstuvwxyz123456", encoded)
            submit_decision(
                root,
                request_id=pending[0]["request_id"],
                user_message="Go back and change the region.",
                rationale="Selected after reading the complete response.",
            )
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(observed[0]["user_message"], "Go back and change the region.")
            self.assertEqual(pending_requests(root), ())

    def test_secret_execution_value_is_not_persisted_and_control_id_is_immutable(self) -> None:
        secret = "runtime-secret-4821"
        context = self._context()
        context["scheduled_target"] = {
            **context["scheduled_target"],
            "edge_key": "chain_auxiliary_endpoints::RPC_API_KEY::contract-hash",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            broker = FilesystemDecisionBroker(
                root, batch_id="batch-1", timeout_seconds=2, poll_seconds=0.01
            )
            observed = []
            thread = threading.Thread(
                target=lambda: observed.append(asyncio.run(broker("shard-12", context)))
            )
            thread.start()
            deadline = time.monotonic() + 1
            while not pending_requests(root) and time.monotonic() < deadline:
                time.sleep(0.01)
            request = pending_requests(root)[0]
            self.assertEqual(
                request["control_identity"]["scheduled_target"]["edge_key"],
                context["scheduled_target"]["edge_key"],
            )

            submit_decision(
                root,
                request_id=request["request_id"],
                user_message=f"RPC_API_KEY={secret}",
                rationale="Use the supplied runtime credential.",
            )
            thread.join(timeout=2)

            self.assertFalse(thread.is_alive())
            self.assertEqual(observed[0]["user_message"], f"RPC_API_KEY={secret}")
            self.assertEqual(
                observed[0]["target_coverage_ids"],
                [context["scheduled_target"]["edge_key"]],
            )
            persisted = "\n".join(
                path.read_text(encoding="utf-8")
                for path in root.rglob("*.json")
            )
            self.assertNotIn(secret, persisted)
            self.assertIn("***REDACTED***", persisted)
            self.assertEqual(list((root / "channels").glob("*.fifo")), [])

    def test_stale_decision_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            broker = FilesystemDecisionBroker(
                root, batch_id="batch-1", timeout_seconds=2, poll_seconds=0.01
            )
            failure = []

            def invoke():
                try:
                    asyncio.run(broker("shard-1", self._context()))
                except Exception as exc:  # asserted below
                    failure.append(exc)

            thread = threading.Thread(target=invoke)
            thread.start()
            deadline = time.monotonic() + 1
            while not pending_requests(root) and time.monotonic() < deadline:
                time.sleep(0.01)
            request = pending_requests(root)[0]
            def write_forged_record(descriptor: int, payload: bytes) -> None:
                wire = json.loads(payload.strip())
                wire["record"]["previous_response_hash"] = "b" * 64
                forged = b"\n" + (
                    json.dumps(wire, sort_keys=True, separators=(",", ":")) + "\n"
                ).encode()
                _write_channel_payload_descriptor(descriptor, forged)

            with patch(
                "tests.agent_live.filesystem_decision_broker."
                "_write_channel_payload_descriptor",
                side_effect=write_forged_record,
            ):
                with self.assertRaises(ExternalDecisionBlocked):
                    submit_decision(
                        root,
                        request_id=request["request_id"],
                        user_message="Continue.",
                        rationale="Bound to the response.",
                    )
            thread.join(timeout=2)
            self.assertEqual(len(failure), 1)
            self.assertIsInstance(failure[0], ExternalDecisionBlocked)
            self.assertIn("identity is stale", str(failure[0]))
            self.assertFalse(
                (root / "committed" / f"{request['request_id']}.json").exists()
            )

    def test_product_decision_requires_auditable_codex_attestation(self) -> None:
        context = self._context()
        context["schedule"] = {
            "schedule_id": "schedule-1",
            "persona": "operator",
            "mission": "exercise frozen factors",
            "allowed_risk_factors": ["language:en"],
        }
        context.pop("scheduled_target")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            broker = FilesystemDecisionBroker(
                root, batch_id="batch-1", timeout_seconds=2, poll_seconds=0.01
            )
            observed = []
            thread = threading.Thread(
                target=lambda: observed.append(
                    asyncio.run(broker("shard-1", context))
                )
            )
            thread.start()
            deadline = time.monotonic() + 1
            while not pending_requests(root) and time.monotonic() < deadline:
                time.sleep(0.01)
            request = pending_requests(root)[0]
            submit_decision(
                root,
                request_id=request["request_id"],
                user_message="Continue after reading the response.",
                rationale="The live response requests the next configuration choice.",
                risk_factor_ids=("language:en",),
                actor_kind="codex",
                actor_task_id="task-123",
                actor_model="gpt-test",
            )
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
            decision = observed[0]
            normalized = _validate_decision(
                context,
                decision,
                lane="journey",
                simulator_attestation_required=True,
            )
            self.assertEqual(
                normalized["simulator_attestation"]["actor"],
                {
                    "actor_kind": "codex",
                    "task_id": "task-123",
                    "model": "gpt-test",
                },
            )
            self.assertFalse(
                normalized["simulator_attestation"][
                    "cryptographic_identity_claimed"
                ]
            )
            committed = json.loads(
                next((root / "committed").glob("*.json")).read_text()
            )
            self.assertIsInstance(committed["simulator_attestation"], dict)

            without_attestation = {
                key: value for key, value in decision.items()
                if key != "simulator_attestation"
            }
            with self.assertRaisesRegex(_SimulatorInvalid, "no simulator attestation"):
                _validate_decision(
                    context,
                    without_attestation,
                    lane="journey",
                    simulator_attestation_required=True,
                )

    def test_retained_journey_requires_exact_semantic_binding(self) -> None:
        context = self._context()
        context["schedule"] = {
            "schedule_id": "retained-schedule-1",
            "persona": "operator",
            "mission": "exercise retained semantics",
            "allowed_risk_factors": [],
            "verifier_input_contract": self._verifier_input_contract(),
        }
        context.pop("scheduled_target")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            broker = FilesystemDecisionBroker(
                root, batch_id="batch-1", timeout_seconds=2, poll_seconds=0.01
            )
            observed = []
            thread = threading.Thread(
                target=lambda: observed.append(
                    asyncio.run(broker("retained-shard", context))
                )
            )
            thread.start()
            deadline = time.monotonic() + 1
            while not pending_requests(root) and time.monotonic() < deadline:
                time.sleep(0.01)
            request = pending_requests(root)[0]
            with self.assertRaisesRegex(ValueError, "requires source step"):
                submit_decision(
                    root,
                    request_id=request["request_id"],
                    user_message="What can this Agent do?",
                    rationale="Isomorphic capability request.",
                    actor_kind="codex",
                    actor_task_id="task-retained",
                    actor_model="gpt-test",
                )
            submit_decision(
                root,
                request_id=request["request_id"],
                user_message="What can this Agent do?",
                rationale="Isomorphic capability request.",
                actor_kind="codex",
                actor_task_id="task-retained",
                actor_model="gpt-test",
                source_step_id="source-1",
                semantic_role="capability_consultation",
            )
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
            normalized = _validate_decision(
                context,
                observed[0],
                lane="journey",
                simulator_attestation_required=True,
            )
            self.assertEqual(
                normalized["variant_binding"],
                {
                    "source_step_id": "source-1",
                    "semantic_role": "capability_consultation",
                },
            )
            wrong = dict(normalized)
            wrong["variant_binding"] = {
                "source_step_id": "source-1",
                "semantic_role": "wrong",
            }
            with self.assertRaisesRegex(_SimulatorInvalid, "outside the frozen"):
                _validate_decision(
                    context,
                    wrong,
                    lane="journey",
                    simulator_attestation_required=False,
                )

    @unittest.skipUnless(sys.platform.startswith("linux"), "formal control plane is Linux-only")
    def test_signals_are_translated_into_awaited_batch_interruption(self) -> None:
        for signum in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signum=signum):
                lifecycle = {
                    "started": False,
                    "cleaned": False,
                    "interrupted": False,
                    "broker_closed": False,
                }

                class Broker:
                    def close(self) -> None:
                        lifecycle["broker_closed"] = True

                async def fake_run_batch(*_args, interruption_event, **_kwargs):
                    lifecycle["started"] = True
                    try:
                        await interruption_event.wait()
                        lifecycle["interrupted"] = True
                    finally:
                        lifecycle["cleaned"] = True

                async def scenario():
                    async def terminate():
                        while not lifecycle["started"]:
                            await asyncio.sleep(0)
                        os.kill(os.getpid(), signum)

                    task = asyncio.create_task(terminate())
                    with patch(
                        "tests.agent_live.filesystem_decision_broker.run_batch",
                        side_effect=fake_run_batch,
                    ):
                        interrupted_by = await _run_with_signal_cleanup(
                            object(),
                            broker=Broker(),
                            result_index_path="unused.json",
                            authority_signer=object(),
                        )
                    await task
                    return interrupted_by

                interrupted_by = asyncio.run(scenario())

                self.assertEqual(interrupted_by, signum)
                self.assertTrue(lifecycle["interrupted"])
                self.assertTrue(lifecycle["cleaned"])
                self.assertTrue(lifecycle["broker_closed"])


if __name__ == "__main__":
    unittest.main()

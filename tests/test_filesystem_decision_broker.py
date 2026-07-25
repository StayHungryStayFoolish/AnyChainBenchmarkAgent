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
    pending_requests,
    submit_decision,
)
from tests.agent_live.coverage_evidence import content_hash
from tests.agent_live.retained_regression_attestations import (
    build_source_contract,
    build_variant_contract,
)


class FilesystemDecisionBrokerTest(unittest.TestCase):
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
                root, batch_id="batch-1", timeout_seconds=2, poll_seconds=0.5
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
            path = submit_decision(
                root,
                request_id=request["request_id"],
                user_message="Continue.",
                rationale="Bound to the response.",
            )
            path.chmod(0o644)
            payload = json.loads(path.read_text())
            payload["previous_response_hash"] = "b" * 64
            path.write_text(json.dumps(payload), encoding="utf-8")
            thread.join(timeout=2)
            self.assertEqual(len(failure), 1)
            self.assertIsInstance(failure[0], ExternalDecisionBlocked)
            self.assertIn("identity is stale", str(failure[0]))

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
            self.assertEqual(len(list((root / "attestations").glob("*.json"))), 1)

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
                }

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
                            broker=object(),
                            result_index_path="unused.json",
                        )
                    await task
                    return interrupted_by

                interrupted_by = asyncio.run(scenario())

                self.assertEqual(interrupted_by, signum)
                self.assertTrue(lifecycle["interrupted"])
                self.assertTrue(lifecycle["cleaned"])


if __name__ == "__main__":
    unittest.main()

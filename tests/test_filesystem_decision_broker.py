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

from tests.agent_live.batch_orchestrator import ExternalDecisionBlocked
from tests.agent_live.filesystem_decision_broker import (
    FilesystemDecisionBroker,
    _run_with_signal_cleanup,
    pending_requests,
    submit_decision,
)


class FilesystemDecisionBrokerTest(unittest.TestCase):
    def _context(self):
        return {
            "previous_response_hash": "a" * 64,
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

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from tests.agent_live.batch_orchestrator import ExternalDecisionBlocked
from tests.agent_live.filesystem_decision_broker import (
    FilesystemDecisionBroker,
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
                target=lambda: observed.append(broker("shard-1", self._context()))
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
                    broker("shard-1", self._context())
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


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import os
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]


class AgentRuntimeIsolationTests(unittest.TestCase):
    def test_jobs_directory_can_be_isolated_for_cli_and_chaos_processes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            isolated = Path(tmpdir, "jobs").resolve()
            env = dict(os.environ)
            env["ANYCHAIN_AGENT_JOBS_DIR"] = str(isolated)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from agent.runners.job_manager import DEFAULT_JOBS_DIR; "
                        "from agent.terminal.startup_state import load_startup_state; "
                        "print(DEFAULT_JOBS_DIR); "
                        "print(load_startup_state()['jobs_dir'])"
                    ),
                ],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertEqual(completed.stdout.splitlines(), [str(isolated), str(isolated)])

    def test_runtime_emits_committed_turn_fingerprint_without_exposing_config(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime
        from tests.agent_live.graph_turn import (
            reviewed_action_plan,
            reviewed_stage_planner,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            event_file = root / "turn-events.jsonl"
            with (
                patch.dict(os.environ, {"ANYCHAIN_AGENT_TURN_EVENT_FILE": str(event_file)}),
                reviewed_stage_planner(
                    lambda state, text: reviewed_action_plan(
                        state,
                        text,
                        [{"type": "greeting", "confidence": "high"}],
                    )
                ),
                AnyChainGraphRuntime(
                    "event-runtime",
                    checkpoint_path=root / "checkpoints.sqlite",
                    session_purpose="chaos",
                ) as runtime,
            ):
                result = runtime.invoke("hello", language="en")

            event = json.loads(event_file.read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(event["event_type"], "turn_committed")
            self.assertEqual(event["schema_version"], 6)
            self.assertEqual(event["terminal_outcome"], "committed")
            self.assertTrue(event["runtime_event_id"])
            self.assertTrue(event["terminal_event_id"])
            self.assertTrue(event["transaction_id"])
            self.assertEqual(
                event["product_authority_id"],
                "chaos:event-runtime",
            )
            self.assertEqual(
                event["physical_thread_id"],
                event["product_checkpoint_thread_id"],
            )
            self.assertEqual(
                event["attempt_checkpoint_id"],
                event["product_checkpoint_id"],
            )
            self.assertEqual(len(event["render_hash"]), 64)
            self.assertEqual(event["thread_id"], "event-runtime")
            self.assertEqual(event["turn_index"], result["turn_index"])
            self.assertEqual(len(event["before_fingerprint"]), 64)
            self.assertEqual(len(event["after_fingerprint"]), 64)
            self.assertIn("revision", event)
            self.assertIn("pending_contract", event)
            self.assertIn("admitted_action_types", event)
            self.assertTrue(event["state_diff_hashes"])
            self.assertTrue(event["next_result"])
            self.assertEqual(
                [item["type"] for item in event["admitted_action_provenance"]],
                ["greeting"],
            )
            action_receipt = event["admitted_action_provenance"][0]
            self.assertEqual(len(action_receipt["arguments_hash"]), 64)
            self.assertEqual(len(action_receipt["source_hash"]), 64)
            self.assertIn(
                event["turn_receipt_summary"]["status"],
                {"blocked", "committed"},
            )
            self.assertEqual(
                event["pending_transition"]["after_id"],
                result["pending_question"]["id"],
            )
            self.assertEqual(
                event["render_manifest"]["fragment_count"],
                len(result["visible_response"]),
            )
            self.assertIn("execution_receipt_summary", event)
            self.assertTrue(event["control_receipts"])
            self.assertTrue(
                all(
                    len(item["receipt_id"]) == 64
                    for item in event["control_receipts"]
                )
            )
            self.assertIn("material_state_diff_hashes", event)
            self.assertNotIn("confirmed_config", event)
            self.assertNotIn(
                "hello",
                json.dumps(event, ensure_ascii=False),
            )

    def test_invariant_recovery_emits_one_committed_turn_observation(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime
        from agent.harness.state import new_state

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            event_file = root / "turn-events.jsonl"
            invalid = new_state("recovered-runtime", language="en", session_purpose="chaos")
            invalid["turn_index"] = 1
            invalid["active_group"] = "not-a-real-group"
            with (
                patch.dict(os.environ, {"ANYCHAIN_AGENT_TURN_EVENT_FILE": str(event_file)}),
                AnyChainGraphRuntime(
                    "recovered-runtime",
                    checkpoint_path=root / "checkpoints.sqlite",
                    session_purpose="chaos",
                ) as runtime,
                patch.object(runtime.graph, "invoke", return_value=invalid),
            ):
                result = runtime.invoke("trigger invalid transition", language="en")

            events = [json.loads(line) for line in event_file.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([event["event_type"] for event in events], ["turn_committed"])
            self.assertEqual([event["observation"] for event in events], ["turn_recovered"])
            self.assertEqual(events[0]["turn_index"], result["turn_index"])
            self.assertEqual(result["active_group"], "failure_recovery")
            self.assertEqual(events[0]["pending_question_id"], "failure_recovery_action")

    def test_invariant_raised_inside_graph_is_recovered_at_transaction_boundary(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime
        from agent.harness.invariants import StateInvariantError

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            event_file = root / "turn-events.jsonl"
            with (
                patch.dict(os.environ, {"ANYCHAIN_AGENT_TURN_EVENT_FILE": str(event_file)}),
                AnyChainGraphRuntime(
                    "raised-invariant-runtime",
                    checkpoint_path=root / "checkpoints.sqlite",
                    session_purpose="chaos",
                ) as runtime,
                patch.object(
                    runtime.graph,
                    "invoke",
                    side_effect=StateInvariantError("declared option postcondition failed"),
                ),
            ):
                result = runtime.invoke("confirm", language="en")

            events = [json.loads(line) for line in event_file.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([event["event_type"] for event in events], ["turn_committed"])
            self.assertEqual([event["observation"] for event in events], ["turn_recovered"])
            self.assertEqual(result["active_group"], "failure_recovery")
            self.assertEqual(result["turn_index"], 1)
            record = result["failure_recovery"]["record"]
            self.assertIn("declared option postcondition failed", str(record))


if __name__ == "__main__":
    unittest.main()

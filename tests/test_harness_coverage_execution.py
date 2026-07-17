"""Execution tests for every cataloged deterministic Harness edge."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.agent_live.execute_harness_contract_ledger import execute_ledger
from tests.agent_live.coverage_evidence import validate_evidence_artifact


class HarnessCoverageExecutionTest(unittest.TestCase):
    def test_runner_reports_only_edges_with_executable_artifacts(self) -> None:
        revision = {"commit": "test-commit", "worktree_hash": "a" * 64}
        with TemporaryDirectory() as tmpdir:
            ledger = execute_ledger(artifact_dir=tmpdir, revision=revision)
            required = [
                edge for edge in ledger["edges"]
                if edge["evidence"]["deterministic"]["required"]
            ]
            self.assertTrue(required)
            passed = [edge for edge in required if edge["evidence"]["deterministic"]["status"] == "passed"]
            failed = [edge for edge in required if edge["evidence"]["deterministic"]["status"] == "failed"]
            not_run = [edge for edge in required if edge["evidence"]["deterministic"]["status"] == "not_run"]
            self.assertTrue(passed)
            self.assertEqual(failed, [], "deterministic required rows must all verify")
            self.assertEqual(not_run, [], "the deterministic runner must close its denominator")
            self.assertEqual(len(passed) + len(failed), len(required))
            rejected = [edge for edge in required if edge.get("expected_admitted") is False]
            self.assertTrue(rejected, "the deterministic denominator must include rejection rows")
            self.assertTrue(all(
                edge["evidence"]["deterministic"]["status"] == "passed"
                and len(edge["evidence"]["deterministic"]["evidence_ids"]) == 1
                for edge in rejected
            ))

            for edge in passed + failed:
                references = edge["evidence"]["deterministic"]["evidence_ids"]
                self.assertEqual(len(references), 1)
                artifact = json.loads(Path(references[0]).read_text(encoding="utf-8"))
                valid, reason = validate_evidence_artifact(artifact, edge=edge, revision=revision)
                self.assertTrue(valid, reason)
                self.assertEqual(
                    artifact["runner_type"],
                    "tests.agent_live.graph_turn.invoke_product_graph_turn",
                )
                turn = artifact["turn_evidence"]
                self.assertEqual(turn["before"]["last_user_input"], turn["input"])
                self.assertEqual(turn["before"]["pending_question"], turn["question"])
                self.assertEqual(turn["after"]["visible_response"], turn["response"])
                self.assertTrue(turn["admitted_action"])
                graph_returned = "compiled_graph_turn_returned" in {
                    event["event_type"] for event in artifact["event_trace"]
                }
                self.assertEqual(
                    turn["admitted_action"]["admitted"],
                    (
                        True if edge.get("expected_admitted") is None
                        else bool(edge.get("expected_admitted"))
                    ),
                )
                event_types = {event["event_type"] for event in artifact["event_trace"]}
                if graph_returned:
                    self.assertNotEqual(turn["before"], turn["after"])
                else:
                    self.assertEqual(artifact["outcome"], "failed")
                self.assertNotIn("edge_executed", event_types)
                self.assertNotIn("edge_execution_failed", event_types)
                if artifact["outcome"] == "failed":
                    self.assertTrue(artifact["error"])
                self.assertEqual(
                    turn["compiled_graph_helper"],
                    "tests.agent_live.graph_turn.invoke_product_graph_turn",
                )
                self.assertEqual(
                    turn["next_question"],
                    turn["after"].get("pending_question") or {},
                )

            summary = ledger["summary"]
            self.assertEqual(summary["edges"], len(ledger["edges"]))
            self.assertEqual(summary["uncataloged_questions"], 0)
            self.assertEqual(summary["execution_closure"]["deterministic"]["not_run"], 0)

            action_only = [edge for edge in ledger["edges"] if edge["edge_type"] == "action_transition"]
            self.assertTrue(action_only)
            self.assertTrue(all(
                edge["evidence"]["deterministic"]["status"] == "not_applicable"
                for edge in action_only
            ))
            real_execution = [edge for edge in ledger["edges"] if edge["real_execution_required"]]
            self.assertTrue(real_execution)
            self.assertTrue(all(
                edge["evidence"]["deterministic"]["status"] == "not_applicable"
                for edge in real_execution
            ))

    def test_natural_language_and_unimplemented_manual_classes_are_not_mass_passed(self) -> None:
        with TemporaryDirectory() as tmpdir:
            ledger = execute_ledger(
                artifact_dir=tmpdir,
                revision={"commit": "test-commit", "worktree_hash": "a" * 64},
            )
        unsupported = [
            edge for edge in ledger["edges"]
            if edge.get("applicable") and edge.get("input_class") in {
                "natural_language_option",
                "natural_language_answer",
                "multiline_prose",
                "partial_request_response_evidence",
            }
        ]
        self.assertTrue(unsupported)
        self.assertTrue(all(
            edge["evidence"]["deterministic"]["status"] == "not_applicable"
            and not edge["evidence"]["deterministic"]["required"]
            for edge in unsupported
        ))

    def test_executor_runs_as_a_script_and_emits_only_executed_artifacts(self) -> None:
        script = Path(__file__).resolve().parent / "agent_live" / "execute_harness_contract_ledger.py"
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "ledger.json"
            artifacts = Path(tmpdir) / "artifacts"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--output",
                    str(output),
                    "--artifact-dir",
                    str(artifacts),
                ],
                cwd=tmpdir,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            ledger = json.loads(output.read_text(encoding="utf-8"))
            passed = [
                edge for edge in ledger["edges"]
                if edge["evidence"]["deterministic"]["status"] == "passed"
            ]
            failed = [
                edge for edge in ledger["edges"]
                if edge["evidence"]["deterministic"]["status"] == "failed"
            ]
            artifact_files = list(artifacts.glob("*.json"))
        self.assertEqual(completed.returncode, 1 if failed else 0)
        self.assertEqual(len(artifact_files), len(passed) + len(failed))
        self.assertGreater(len(passed), 0)
        self.assertEqual(ledger["summary"]["execution_closure"]["deterministic"]["not_run"], 0)
        closure = ledger["summary"]["execution_closure"]["deterministic"]
        self.assertEqual(closure["status"], "failed" if failed else "complete")
        self.assertEqual(closure["open_required"], len(failed))


if __name__ == "__main__":
    unittest.main()

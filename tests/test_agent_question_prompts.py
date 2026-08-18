"""Regression tests for removal of the obsolete planner question protocol."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]


class QuestionProtocolRemovalTest(unittest.TestCase):
    def test_obsolete_question_modules_are_deleted(self) -> None:
        self.assertFalse((REPO_ROOT / "agent/planners/config_questions.py").exists())
        self.assertFalse((REPO_ROOT / "agent/validators/config_contract.py").exists())
        self.assertFalse((REPO_ROOT / "agent/workflows/requirements.py").exists())

    def test_incomplete_plan_has_no_question_payload_and_fails_preflight(self) -> None:
        from agent.planners.preflight import run_preflight
        from agent.planners.strategy_planner import generate_plan

        plan = generate_plan({"chain": "bsc", "use_fake_node": True, "goal": "smoke"})

        self.assertNotIn("required_questions", plan)
        self.assertTrue(plan["required_inputs"])
        self.assertTrue(plan["configuration_checklist"]["missing_blockers"])

        preflight = run_preflight(plan)
        self.assertFalse(preflight["passed"])
        failed = {item["name"] for item in preflight["checks"] if not item["passed"]}
        self.assertIn("required_inputs_present", failed)
        self.assertIn("configuration_checklist_complete", failed)

    def test_prepare_result_does_not_publish_a_second_question_protocol(self) -> None:
        from agent.runners.benchmark_pipeline import prepare_benchmark_run

        discovery = {
            "source": "test",
            "warnings": [],
            "deployment": {"type": "unknown"},
            "cloud": {},
            "disks": {},
            "network": {},
            "dependencies": {},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch("agent.runners.benchmark_pipeline._discover_environment", return_value=discovery),
                patch("agent.runners.benchmark_pipeline._run_doctor", return_value={"warnings": []}),
            ):
                result = prepare_benchmark_run(
                    chain="bsc",
                    goal="smoke",
                    use_fake_node=True,
                    output_dir=tmpdir,
                )

        data = result["data"]
        self.assertEqual(result["status"], "blocked")
        self.assertNotIn("required_questions", data["plan"])
        for legacy_key in ("questions", "next_question", "configuration_questions"):
            self.assertNotIn(legacy_key, data)
        self.assertTrue(data["missing_required"])
        self.assertFalse(data["preflight"]["passed"])

    def test_runbook_ignores_legacy_required_questions(self) -> None:
        from agent.runners.runbook import render_runbook

        text = render_runbook({
            "plan_id": "test-plan",
            "chain": "bsc",
            "required_questions": [
                {"id": "chain_template_reviewed", "severity": "blocker", "prompt": "Legacy prompt."},
            ],
        })

        self.assertNotIn("Required Questions", text)
        self.assertNotIn("Legacy prompt.", text)

    def test_fake_node_smoke_uses_required_inputs_and_checklist_only(self) -> None:
        from agent.runners.benchmark_pipeline import _fake_node_smoke_plan

        plan = {
            "plan_id": "test-plan",
            "required_inputs": ["local_rpc_url", "chain_template_reviewed"],
            "configuration_checklist": {
                "missing_blockers": ["local_rpc_url", "chain_template_reviewed"],
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_file = Path(tmpdir) / "plan.json"
            plan_file.write_text(json.dumps(plan), encoding="utf-8")
            smoke_plan = _fake_node_smoke_plan(plan_file, Path(tmpdir) / "smoke")

        self.assertNotIn("required_questions", smoke_plan)
        self.assertNotIn("local_rpc_url", smoke_plan["required_inputs"])
        self.assertIn("chain_template_reviewed", smoke_plan["required_inputs"])
        self.assertNotIn("local_rpc_url", smoke_plan["configuration_checklist"]["missing_blockers"])

    def test_live_harness_remains_question_authority(self) -> None:
        from agent.harness.coordinator import _ask_next_blocking_question
        from agent.harness.questions import render_question
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed"}
        result = _ask_next_blocking_question(state)

        self.assertEqual(result["pending_question"]["group"], "provider_deployment")
        self.assertEqual(
            render_question(result["pending_question"], "en").splitlines()[0],
            "Confirm CLOUD_REGION; use the detected value or enter a custom region.",
        )

    def test_qps_defaults_have_one_shared_authority(self) -> None:
        from agent.knowledge.qps_profiles import (
            STRATEGY_BENCHMARK_MODE,
            qps_profile_defaults,
            strategy_qps_defaults,
        )

        for strategy, mode in STRATEGY_BENCHMARK_MODE.items():
            runtime = qps_profile_defaults(mode)
            planner = strategy_qps_defaults(strategy)
            self.assertEqual(planner["initial"], int(runtime["INITIAL_QPS"]))
            self.assertEqual(planner["max"], int(runtime["MAX_QPS"]))
            self.assertEqual(planner["step"], int(runtime["QPS_STEP"]))
            self.assertEqual(
                planner["duration_seconds"],
                int(runtime["DURATION"]),
            )


if __name__ == "__main__":
    unittest.main()

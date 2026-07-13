"""Regression tests for Phase 3: consolidating the two question-generation

engines (architecture audit Finding C1/C2). Before this refactor,
`agent.validators.config_contract.build_missing_config_questions` and
`agent.planners.config_questions.required_questions` independently authored
prompt text for the same missing fields and could silently disagree; five
separate "required fields" catalogs existed with mismatched member sets.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class QuestionPromptsConsolidationTest(unittest.TestCase):
    def test_config_contract_and_config_questions_agree_on_shared_field_wording(self) -> None:
        from agent.planners.config_questions import required_questions
        from agent.planners.strategy_planner import generate_plan
        from agent.validators.config_contract import build_missing_config_questions

        plan = generate_plan({"chain": "bsc", "use_fake_node": True, "goal": "smoke"})
        plan_questions = {q["id"]: q["prompt"] for q in required_questions(plan)}
        contract_result = build_missing_config_questions("fake-node", {})
        contract_questions = {q["id"]: q["prompt"] for q in contract_result["questions"]}

        shared_ids = set(plan_questions) & set(contract_questions)
        self.assertTrue(shared_ids, "expected at least one field covered by both engines")
        for field_id in shared_ids:
            self.assertEqual(
                plan_questions[field_id],
                contract_questions[field_id],
                f"prompt wording diverged for shared field {field_id!r}",
            )

    def test_question_prompts_registry_covers_all_field_catalog_keys(self) -> None:
        from agent.knowledge.entry_contract import field_specs_for
        from agent.planners import question_prompts

        for mode in ("fake_node", "real_node", "sync_observe"):
            for field in field_specs_for(mode):
                text = question_prompts.text_for(field.key)
                self.assertNotEqual(
                    text,
                    f"Provide required value: {field.key}",
                    f"no registered prompt wording for catalog key {field.key!r} (mode={mode})",
                )

    def test_prepare_benchmark_run_returns_reconciled_questions(self) -> None:
        from agent.planners.strategy_planner import generate_plan
        from agent.validators.config_contract import build_missing_config_questions

        plan = generate_plan({"chain": "bsc", "use_fake_node": True, "goal": "smoke"})
        contract_result = build_missing_config_questions("fake-node", {})
        plan_by_id = {q["id"]: q["prompt"] for q in plan.get("required_questions", [])}
        contract_by_id = {q["id"]: q["prompt"] for q in contract_result["questions"]}
        next_question = contract_result.get("next_question") or {}
        if next_question.get("id") in plan_by_id and next_question.get("field") in plan_by_id:
            self.assertEqual(plan_by_id[next_question["field"]], next_question.get("prompt"))
        for field_id in set(plan_by_id) & set(contract_by_id):
            self.assertEqual(plan_by_id[field_id], contract_by_id[field_id])

    def test_entry_contract_is_single_source_for_required_fields(self) -> None:
        from agent.knowledge.entry_contract import WORKFLOW_CONFIRMATION_FIELDS
        from agent.planners.config_checklist import COMMON_REQUIRED
        from agent.workflows.requirements import COMMON_BLOCKERS

        canonical_keys = {field.key for field in WORKFLOW_CONFIRMATION_FIELDS}
        self.assertEqual(set(COMMON_BLOCKERS), canonical_keys)
        self.assertEqual(set(COMMON_REQUIRED.keys()), canonical_keys)

    def test_runbook_render_runbook_still_renders_required_questions(self) -> None:
        from agent.runners.runbook import render_runbook

        plan = {
            "plan_id": "test-plan",
            "chain": "bsc",
            "required_questions": [
                {"id": "chain_template_reviewed", "severity": "blocker", "prompt": "Review the chain template."},
            ],
        }
        text = render_runbook(plan)
        self.assertIn("chain_template_reviewed", text)
        self.assertIn("Review the chain template.", text)

    def test_fake_node_smoke_plan_filters_required_questions_by_id(self) -> None:
        from agent.runners.benchmark_pipeline import _fake_node_smoke_plan

        plan = {
            "plan_id": "test-plan",
            "required_inputs": ["local_rpc_url", "chain_template_reviewed"],
            "required_questions": [
                {"id": "local_rpc_url", "prompt": "Provide LOCAL_RPC_URL."},
                {"id": "chain_template_reviewed", "prompt": "Review the chain template."},
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_file = Path(tmpdir) / "plan.json"
            plan_file.write_text(json.dumps(plan), encoding="utf-8")
            smoke_root = Path(tmpdir) / "smoke"
            smoke_plan = _fake_node_smoke_plan(plan_file, smoke_root)

        remaining_ids = {q["id"] for q in smoke_plan["required_questions"]}
        self.assertNotIn("local_rpc_url", remaining_ids)
        self.assertIn("chain_template_reviewed", remaining_ids)

    def test_groups_cloud_region_wording_is_pinned(self) -> None:
        """Pin `groups.py`'s live conversational wording for one representative

        field so a future edit to `question_prompts.py` cannot silently change
        what the LangGraph Harness actually says without this test noticing.
        """

        from agent.harness.groups import _ask_next_blocking_question
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed"}
        result = _ask_next_blocking_question(state)

        self.assertEqual(result["pending_question"]["group"], "provider_deployment")
        self.assertEqual(
            result["pending_question"]["prompt"],
            "Confirm CLOUD_REGION; use the detected value or enter a custom region.",
        )

    def test_chain_prompt_reflects_already_known_target_mode(self) -> None:
        """Code review found `_prompt_for_key`/`_required_prompt` never

        forwarded the real target mode, so the "chain" question always
        claimed "Current target mode: not selected" even when a mode had
        already been confirmed (real-node, fake-node, or sync-observe).
        """

        from agent.planners.config_questions import required_questions
        from agent.validators.config_contract import build_missing_config_questions

        real_node_result = build_missing_config_questions("real-node", {"use_fake_node": False})
        chain_question = next(q for q in real_node_result["questions"] if q["id"] == "chain")
        self.assertIn("real-node", chain_question["prompt"])
        self.assertNotIn("not selected", chain_question["prompt"])

        sync_observe_result = build_missing_config_questions(None, {"workflow_type": "sync_observe"})
        chain_question = next(q for q in sync_observe_result["questions"] if q["id"] == "chain")
        self.assertIn("sync-observe", chain_question["prompt"])

        plan = {"required_inputs": ["chain"], "confirmed_inputs": [], "use_fake_node": True, "chain_template_requirements": {}}
        plan_questions = {q["id"]: q["prompt"] for q in required_questions(plan)}
        self.assertIn("fake-node", plan_questions["chain"])

    def test_qps_profile_prompt_does_not_echo_an_unrecognized_mode(self) -> None:
        """Code review found the shared `qps_profile_prompt` dropped

        `config_contract.py`'s old sanitization: an unrecognized
        `benchmark_mode_confirmed` value used to render as "selected"
        instead of being echoed back next to numbers that were never
        actually chosen for it.
        """

        from agent.planners.question_prompts import qps_profile_prompt

        garbage = qps_profile_prompt("bogus_mode", fake_node=False)
        self.assertIn("`selected`", garbage)
        self.assertNotIn("bogus_mode", garbage)

        empty = qps_profile_prompt("", fake_node=False)
        self.assertIn("`quick`", empty)

        known = qps_profile_prompt("intensive", fake_node=False)
        self.assertIn("`intensive`", known)
        self.assertIn("INITIAL_QPS=50000", known)


if __name__ == "__main__":
    unittest.main()

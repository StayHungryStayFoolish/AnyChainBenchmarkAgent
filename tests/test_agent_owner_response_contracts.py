from __future__ import annotations

import ast
import unittest
from pathlib import Path

from agent.harness.contracts import ActionProposal, FailureDescriptor, ResponseFragment
from agent.harness.response_catalog import render_failure, render_fragment
from agent.harness.state import new_state


ROOT = Path(__file__).resolve().parents[1]
OWNER_DOMAINS = tuple(
    ROOT / "agent" / "harness" / "domains" / f"{owner}.py"
    for owner in ("orientation", "environment", "performance", "sync_observe")
)


class OwnerResponseContractTests(unittest.TestCase):
    def test_owner_domains_do_not_use_legacy_fragment_or_string_blocker_contracts(
        self,
    ) -> None:
        violations: list[str] = []
        for path in OWNER_DOMAINS:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if any(alias.name == "message_fragments" for alias in node.names):
                        violations.append(f"{path.name}:{node.lineno}:message_fragments")
                if not isinstance(node, ast.Call):
                    continue
                called = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else ""
                )
                if called != "HandlerResult":
                    continue
                for keyword in node.keywords:
                    if keyword.arg != "blocker":
                        continue
                    if isinstance(
                        keyword.value,
                        (ast.Constant, ast.JoinedStr),
                    ) or (
                        isinstance(keyword.value, ast.Call)
                        and isinstance(keyword.value.func, ast.Name)
                        and keyword.value.func.id == "localized"
                    ):
                        violations.append(f"{path.name}:{node.lineno}:string blocker")
        self.assertEqual(violations, [])

    def test_orientation_greeting_is_catalog_backed_and_bilingual(self) -> None:
        fragment = ResponseFragment(
            kind="message",
            message_id="harness.orientation.greeting",
            source="agent.harness.domains.orientation",
        )
        self.assertIn("AnyChain Benchmark Agent", render_fragment(fragment, "en").text)
        self.assertIn("AnyChain Benchmark Agent", render_fragment(fragment, "zh").text)

    def test_sync_observe_behavior_consultation_is_read_only_and_bilingual(self) -> None:
        from agent.harness.domains.orientation import consultation_fragment

        state = new_state("sync-observe-behavior", language="en")
        state["target_mode"] = "sync-observe"
        state["workflow_mode"] = "sync_observe"
        state["pending_question"] = {
            "id": "sync_observe_stop_condition",
            "group": "sync_observe",
        }
        fragment = consultation_fragment(
            state,
            {
                "type": "answer_opening_question",
                "topic": "sync_observe_behavior",
            },
        )

        self.assertEqual(
            fragment.message_id,
            "harness.orientation.consultation.sync_observe_behavior",
        )
        self.assertIn("until the user stops", render_fragment(fragment, "en").text)
        self.assertIn("手动停止", render_fragment(fragment, "zh").text)

    def test_recommendation_completion_does_not_prescribe_stale_next_state(
        self,
    ) -> None:
        from agent.harness.domains.orientation import apply_orientation_answer

        state = new_state("recommendation-local-effect", language="en")
        question = {
            "id": "accept_recommendation",
            "recommended_setup": {"target_mode": "fake-node"},
        }

        declined = apply_orientation_answer(state, question, False, "N")
        accepted = apply_orientation_answer(state, question, True, "Y")

        for result in (declined, accepted):
            for language in ("en", "zh"):
                text = render_fragment(result.response_fragments[0], language).text
                self.assertNotIn("Next", text)
                self.assertNotIn("下一步", text)
                self.assertNotIn("Tell me which", text)
                self.assertNotIn("请告诉我", text)

    def test_environment_invalid_value_and_failure_are_typed(self) -> None:
        from agent.harness.domains.environment import (
            apply_environment_action,
            apply_environment_answer,
        )

        state = new_state("owner-response-environment", language="en")
        question = {
            "id": "DATA_VOL_SIZE",
            "group": "ledger_disk",
            "field": "DATA_VOL_SIZE",
        }
        invalid = apply_environment_answer(state, question, "0")
        self.assertEqual(
            invalid.response_fragments[0].message_id,
            "harness.environment.positive_number_required",
        )
        failure = apply_environment_action(
            state,
            ActionProposal("unsupported", "not_an_environment_action", {}, "high"),
        ).blocker
        self.assertIsInstance(failure, FailureDescriptor)
        self.assertIn(
            "not_an_environment_action",
            render_failure(failure, "en").text,
        )

    def test_environment_proposals_are_idempotent_against_confirmed_values(
        self,
    ) -> None:
        from agent.harness.domains.environment import apply_environment_action

        state = new_state("owner-environment-idempotency", language="en")
        state["confirmed_config"] = {
            "LEDGER_DEVICE": "vda",
            "DATA_VOL_TYPE": "hyperdisk-balanced",
        }

        equal = apply_environment_action(
            state,
            ActionProposal(
                "equal",
                "propose_config_values",
                {"config_values": {"LEDGER_DEVICE": "vda"}},
                "high",
            ),
        )
        self.assertIsNone(equal.pending_question)
        self.assertTrue(equal.delta.is_empty())

        conflict = apply_environment_action(
            state,
            ActionProposal(
                "conflict",
                "propose_config_values",
                {"config_values": {"LEDGER_DEVICE": "vdb"}},
                "high",
            ),
        )
        self.assertEqual(conflict.pending_question["id"], "inferred_config_review")

        mixed = apply_environment_action(
            state,
            ActionProposal(
                "mixed",
                "propose_config_values",
                {
                    "config_values": {
                        "LEDGER_DEVICE": "vda",
                        "DATA_VOL_TYPE": "nvme",
                        "DATA_VOL_MAX_IOPS": "20000",
                    }
                },
                "high",
            ),
        )
        pending_review = {
            write.path[-1]: write.value
            for write in mixed.delta.writes
            if write.path[:3]
            == ("inferred_config", "pending_review", "config_values")
        }
        self.assertEqual(
            pending_review,
            {
                "DATA_VOL_TYPE": "nvme",
                "DATA_VOL_MAX_IOPS": "20000",
            },
        )

    def test_performance_responses_and_failures_use_registered_messages(self) -> None:
        from agent.harness.domains.performance import (
            apply_performance_action,
            apply_performance_answer,
        )

        state = new_state("owner-response-performance", language="en")
        selected = apply_performance_action(
            state,
            ActionProposal(
                "observability",
                "set_observability",
                {"observability_mode": "exporter"},
                "high",
            ),
        )
        self.assertIn(
            "9108",
            render_fragment(selected.response_fragments[0], "en").text,
        )
        qps_state = new_state("owner-response-qps", language="en")
        qps_state["qps_profile"] = {
            "mode": "quick",
            "confirmed": False,
            "default_decision_made": True,
        }
        qps_action = apply_performance_action(
            qps_state,
            ActionProposal(
                "qps-override",
                "set_qps_override",
                {"qps_overrides": {"INITIAL_QPS": 50}},
                "high",
            ),
        )
        self.assertEqual(
            render_fragment(qps_action.response_fragments[0], "en").text,
            "Applied QPS profile overrides: INITIAL_QPS=50",
        )
        qps_state["qps_profile"]["adjust_field"] = "MAX_QPS"
        qps_answer = apply_performance_answer(
            qps_state,
            {"group": "qps_profile", "field": "qps_adjust_value"},
            "1200",
        )
        self.assertEqual(
            render_fragment(qps_answer.response_fragments[0], "zh").text,
            "已应用 QPS profile 覆盖值：MAX_QPS=1200",
        )
        failure = apply_performance_action(
            state,
            ActionProposal(
                "bad-qps",
                "set_qps_mode",
                {"qps_mode": "turbo"},
                "high",
            ),
        ).blocker
        self.assertIsInstance(failure, FailureDescriptor)
        self.assertIn("quick", render_failure(failure, "en").text)

    def test_sync_observe_guidance_and_duration_failure_are_typed(self) -> None:
        from agent.harness.domains.sync_observe import (
            apply_sync_observe_action,
            apply_sync_observe_answer,
        )

        state = new_state("owner-response-sync", language="en")
        state["workflow_mode"] = "sync_observe"
        state["web_research"] = {"google_search_available": False}
        setup = apply_sync_observe_action(
            state,
            ActionProposal(
                "setup",
                "set_sync_observe_source",
                {"sync_observe_source": "client_setup"},
                "high",
            ),
        )
        self.assertIn(
            "does not auto-download",
            render_fragment(setup.response_fragments[0], "en").text,
        )
        invalid = apply_sync_observe_answer(
            state,
            "0",
            {
                "id": "sync_observe_duration_seconds",
                "group": "sync_observe",
                "field": "sync_observe_duration_seconds",
            },
        )
        self.assertIsInstance(invalid.blocker, FailureDescriptor)
        self.assertIn("positive integer", render_failure(invalid.blocker, "en").text)


if __name__ == "__main__":
    unittest.main()

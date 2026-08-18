"""Owner-receipt tests for evidence collection and report analysis."""

from __future__ import annotations

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from agent.harness.contracts import ActionProposal
from agent.harness.domains.analysis import (
    apply_analysis_action,
    continue_evidence_collection,
    finish_evidence_collection,
    prompt_evidence_collection_waiting,
    report_artifact_entry_result,
    start_evidence_collection,
)
from agent.harness.domains.analysis_receipts import (
    ANALYSIS_RECEIPT_VERSION,
    analysis_hash,
    evidence_block_id,
    validate_analysis_receipt,
)
from agent.harness.response_catalog import render_fragment
from agent.harness.state import new_state


class AnalysisOwnerReceiptTest(unittest.TestCase):
    def _state(self, name: str = "analysis-receipt"):
        state = new_state(name, language="en")
        state["turn_index"] = 7
        state["turn_context"] = {"turn_id": f"{name}-turn", "control_receipts": []}
        return state

    @staticmethod
    def _receipts(state, receipt_type: str):
        return [
            item
            for item in (state.get("turn_context") or {}).get("control_receipts") or []
            if item.get("receipt_type") == receipt_type
        ]

    def _invocation_receipt(self, state):
        action = ActionProposal(
            action_id="analysis-evidence",
            action_type="analyze_evidence",
            arguments={
                "evidence": "ERROR connection refused",
                "question": "What failed?",
            },
            confidence="high",
        )
        with patch(
            "agent.harness.domains.analysis.analyze_evidence_with_model",
            return_value="The endpoint refused the connection.",
        ):
            apply_analysis_action(state, action)
        return deepcopy(self._receipts(state, "analysis_invocation")[-1])

    @staticmethod
    def _verified_job(job_id: str, status: str = "completed") -> dict:
        from agent.runners.job_manager import _job_read_receipt

        job = {
            "job_id": job_id,
            "status": status,
            "artifacts": {},
            "execution_receipts": {},
        }
        receipt = _job_read_receipt(
            job=job,
            plan={},
            source_sha256="a" * 64,
            persisted_status=status,
        )
        job["execution_receipts"] = {"last_read": receipt}
        return job

    def test_analysis_receipt_survives_atomic_graph_commit(self) -> None:
        from tests.agent_live.graph_turn import invoke_actions

        state = self._state("analysis-graph-commit")
        with patch(
            "agent.harness.domains.analysis.analyze_evidence_with_model",
            return_value="The log shows an endpoint timeout.",
        ):
            result = invoke_actions(
                state,
                [
                    {
                        "type": "analyze_evidence",
                        "evidence": "connection timed out",
                        "question": "What failed?",
                        "confidence": "high",
                    }
                ],
                "analyze this log",
            )

        receipts = self._receipts(result, "analysis_invocation")
        self.assertEqual(len(receipts), 1)
        self.assertTrue(validate_analysis_receipt(receipts[0])[0])
        response_manifest = (
            (result.get("turn_context") or {}).get("response_manifest") or []
        )
        self.assertEqual(len(response_manifest), 1)
        self.assertEqual(
            receipts[0]["response_semantic_hash"],
            response_manifest[0]["semantic_hash"],
        )
        self.assertEqual(
            receipts[0]["response_render_hash"],
            response_manifest[0]["render_hash"],
        )
        self.assertTrue(
            any(
                item.get("receipt_type") == "domain_commit"
                and item.get("owner") == "analysis"
                for item in (result.get("turn_context") or {}).get(
                    "control_receipts"
                )
                or ()
            )
        )

    def test_multiline_block_receipts_frame_content_without_retaining_secrets(self) -> None:
        state = self._state()
        secret = "https://rpc.example/v1/super-secret-api-key"
        question = {"id": "freeform_evidence", "kind": "log_evidence"}
        text = f"Traceback (most recent call last):\n\nendpoint={secret}"

        started = start_evidence_collection(state, text, question)

        start_receipt = self._receipts(state, "analysis_evidence_block")[-1]
        valid, reason = validate_analysis_receipt(start_receipt)
        self.assertTrue(valid, reason)
        self.assertEqual(start_receipt["operation"], "start")
        self.assertEqual(start_receipt["line_count"], 2)
        self.assertEqual(start_receipt["block_content_hash"], analysis_hash(
            f"Traceback (most recent call last):\nendpoint={secret}"
        ))
        self.assertNotIn(secret, json.dumps(start_receipt, ensure_ascii=False))

        block_id = start_receipt["block_id"]
        collecting = {
            "question": question,
            "lines": ["Traceback (most recent call last):", f"endpoint={secret}"],
            "language": "en",
            "status": "active",
            "block_id": block_id,
        }
        finished = finish_evidence_collection(state, collecting)
        finish_receipt = self._receipts(state, "analysis_evidence_block")[-1]

        self.assertEqual(started.disposition, "collecting")
        self.assertEqual(finished.disposition, "saved")
        self.assertEqual(finish_receipt["operation"], "finish")
        self.assertEqual(finish_receipt["status"], "saved")
        self.assertEqual(finish_receipt["block_id"], block_id)
        rendered = render_fragment(finished.result.response_fragments[0], "en")
        self.assertEqual(
            finish_receipt["response_semantic_hash"],
            rendered.semantic_hash,
        )
        self.assertEqual(
            finish_receipt["response_render_hash"],
            rendered.render_hash,
        )
        self.assertTrue(validate_analysis_receipt(finish_receipt)[0])
        self.assertNotIn(secret, json.dumps(finish_receipt, ensure_ascii=False))

    def test_blank_collection_input_is_ignored_and_bound_to_response_identity(self) -> None:
        state = self._state("blank-evidence")
        question = {"id": "freeform_evidence", "kind": "log_evidence"}
        lines = ["Traceback (most recent call last):"]
        block_id = evidence_block_id(question, lines)
        collecting = {
            "question": question,
            "lines": list(lines),
            "language": "en",
            "status": "active",
            "block_id": block_id,
        }

        outcome = continue_evidence_collection(state, " \t ", collecting)

        writes = {write.path: write.value for write in outcome.result.delta.writes}
        receipt = self._receipts(state, "analysis_evidence_block")[-1]
        self.assertEqual(writes[("evidence_collection", "lines")], lines)
        self.assertEqual(receipt["operation"], "ignore")
        self.assertEqual(receipt["input_disposition"], "blank_ignored")
        self.assertEqual(receipt["input_non_empty_line_count"], 0)
        self.assertEqual(receipt["block_id"], block_id)
        rendered = render_fragment(outcome.result.response_fragments[0], "en")
        self.assertEqual(
            receipt["response_semantic_hash"],
            rendered.semantic_hash,
        )
        self.assertEqual(
            receipt["response_render_hash"],
            rendered.render_hash,
        )
        self.assertTrue(validate_analysis_receipt(receipt)[0])

    def test_analysis_invocation_binds_block_question_and_response_identity(self) -> None:
        state = self._state("analysis-call")
        question = {"id": "freeform_evidence", "kind": "log_evidence"}
        lines = ["RuntimeError: endpoint probe failed"]
        block_id = evidence_block_id(question, lines)
        state["evidence_collection"] = {
            "question": question,
            "lines": list(lines),
            "language": "en",
            "status": "active",
            "block_id": block_id,
        }
        action = ActionProposal(
            action_id="analyze-current-block",
            action_type="analyze_evidence",
            arguments={"question": "What failed?"},
            confidence="high",
        )

        with patch(
            "agent.harness.domains.analysis.analyze_evidence_with_model",
            return_value="The endpoint probe failed.",
        ) as analyze:
            result = apply_analysis_action(state, action)

        analyze.assert_called_once_with(
            state,
            "\n".join(lines),
            "What failed?",
        )
        receipt = self._receipts(state, "analysis_invocation")[-1]
        self.assertEqual(receipt["source_kind"], "active_block")
        self.assertEqual(receipt["block_id"], block_id)
        self.assertTrue(receipt["invoked"])
        self.assertEqual(receipt["evidence_hash"], analysis_hash("\n".join(lines)))
        self.assertEqual(receipt["question_hash"], analysis_hash("What failed?"))
        rendered = render_fragment(result.response_fragments[0], "en")
        self.assertEqual(
            receipt["response_message_ids"],
            ["analysis.model_document"],
        )
        self.assertEqual(
            receipt["response_semantic_hash"],
            rendered.semantic_hash,
        )
        self.assertEqual(
            receipt["response_render_hash"],
            rendered.render_hash,
        )
        self.assertTrue(validate_analysis_receipt(receipt)[0])
        self.assertNotIn("endpoint probe failed", json.dumps(receipt))

    def test_missing_evidence_records_no_model_invocation(self) -> None:
        state = self._state("analysis-help")
        action = ActionProposal(
            action_id="analysis-help",
            action_type="request_evidence_analysis",
            arguments={"question": "Can you analyze a log?"},
            confidence="high",
        )

        with patch("agent.harness.domains.analysis.analyze_evidence_with_model") as analyze:
            result = apply_analysis_action(state, action)

        analyze.assert_not_called()
        self.assertEqual(self._receipts(state, "analysis_invocation"), [])
        receipt = self._receipts(state, "analysis_evidence_block")[-1]
        self.assertEqual(receipt["operation"], "start")
        self.assertEqual(receipt["input_disposition"], "request_only")
        self.assertEqual(receipt["line_count"], 0)
        rendered = render_fragment(result.response_fragments[0], "en")
        self.assertEqual(
            receipt["response_semantic_hash"],
            rendered.semantic_hash,
        )
        self.assertEqual(
            receipt["response_render_hash"],
            rendered.render_hash,
        )
        self.assertTrue(validate_analysis_receipt(receipt)[0])

    def test_every_collection_transition_binds_its_admitted_action(self) -> None:
        question = {"id": "freeform_evidence", "kind": "log_evidence"}
        cases = (
            (
                "start_evidence_collection",
                {"evidence": "ERROR connection refused"},
                {"pending_question": {
                    "id": "new_chain_schema_evidence",
                    "kind": "evidence",
                }},
            ),
            (
                "append_evidence_collection",
                {"evidence": "retry exhausted"},
                {"evidence_collection": {
                    "question": question,
                    "lines": ["ERROR connection refused"],
                    "language": "en",
                    "status": "active",
                }},
            ),
            (
                "finish_evidence_collection",
                {},
                {"evidence_collection": {
                    "question": question,
                    "lines": ["ERROR connection refused"],
                    "language": "en",
                    "status": "active",
                }},
            ),
            (
                "pause_evidence_collection",
                {},
                {"evidence_collection": {
                    "question": question,
                    "lines": ["ERROR connection refused"],
                    "language": "en",
                    "status": "active",
                }},
            ),
            (
                "resume_evidence_collection",
                {},
                {"evidence_collection": {
                    "question": question,
                    "lines": ["ERROR connection refused"],
                    "language": "en",
                    "status": "paused",
                }},
            ),
            (
                "cancel_evidence_collection",
                {},
                {"evidence_collection": {
                    "question": question,
                    "lines": ["ERROR connection refused"],
                    "language": "en",
                    "status": "active",
                }},
            ),
        )
        for action_type, arguments, state_values in cases:
            with self.subTest(action_type=action_type):
                state = self._state(f"collection-{action_type}")
                state.update(deepcopy(state_values))
                action_id = f"action-{action_type}"
                action = ActionProposal(
                    action_id=action_id,
                    action_type=action_type,
                    arguments=arguments,
                    confidence="high",
                )
                result = apply_analysis_action(state, action)
                self.assertIsNone(result.blocker)
                self.assertEqual(result.consumed_action_ids, (action_id,))

    def test_inline_collection_completion_keeps_append_action_identity(self) -> None:
        state = self._state("collection-inline-finish")
        state["evidence_collection"] = {
            "question": {"id": "freeform_evidence", "kind": "log_evidence"},
            "lines": ["ERROR connection refused"],
            "language": "en",
            "status": "active",
        }
        action = ActionProposal(
            action_id="append-and-finish",
            action_type="append_evidence_collection",
            arguments={"evidence": "retry exhausted\nEND"},
            confidence="high",
        )

        result = apply_analysis_action(state, action)

        self.assertEqual(result.consumed_action_ids, ("append-and-finish",))
        self.assertEqual(result.completion, "completed")
        self.assertTrue(any(
            write.path == ("evidence_buffer",)
            for write in result.delta.writes
        ))

    def test_rpc_collection_handoff_keeps_finish_action_identity(self) -> None:
        state = self._state("collection-pending-answer")
        state["evidence_collection"] = {
            "question": {"id": "new_chain_schema_evidence", "kind": "evidence"},
            "lines": ['{"jsonrpc":"2.0","method":"eth_blockNumber","params":[]}'],
            "language": "en",
            "status": "active",
        }
        action = ActionProposal(
            action_id="finish-rpc-evidence",
            action_type="finish_evidence_collection",
            arguments={},
            confidence="high",
        )

        result = apply_analysis_action(state, action)

        self.assertEqual(result.consumed_action_ids, ("finish-rpc-evidence",))
        self.assertEqual(result.pending_question["id"], "new_chain_schema_evidence")
        self.assertEqual(result.followup_actions[0]["type"], "answer_pending")

    def test_collection_blocker_does_not_claim_action_consumption(self) -> None:
        state = self._state("collection-invalid-pause")
        action = ActionProposal(
            action_id="invalid-pause",
            action_type="pause_evidence_collection",
            arguments={},
            confidence="high",
        )

        result = apply_analysis_action(state, action)

        self.assertIsNotNone(result.blocker)
        self.assertEqual(result.consumed_action_ids, ())

    def test_empty_collection_waiting_is_read_only(self) -> None:
        state = self._state("collection-empty-wait")
        collecting = {
            "question": {"id": "freeform_evidence", "kind": "log_evidence"},
            "lines": ["ERROR connection refused"],
            "language": "en",
            "status": "active",
            "block_id": "existing-block",
        }
        state["evidence_collection"] = deepcopy(collecting)

        result = prompt_evidence_collection_waiting(state, collecting)

        self.assertEqual(result.delta.writes, ())
        self.assertEqual(result.delta.deletes, ())
        self.assertEqual(state["evidence_collection"], collecting)

    def test_report_receipt_binds_requested_job_resolved_job_and_response_identity(self) -> None:
        state = self._state("report-analysis")
        state["report_context"] = {"requested_job_id": "job_20260724000000_deadbeef"}
        persisted = self._verified_job("job_20260724000000_deadbeef")

        with patch(
            "agent.harness.domains.analysis._report_artifact_entry",
            return_value=("Persisted report facts.", persisted, "report", {}),
        ):
            result = report_artifact_entry_result(state)

        receipt = self._receipts(state, "analysis_report")[-1]
        self.assertTrue(receipt["invoked"])
        self.assertTrue(receipt["evidence_verified"])
        self.assertEqual(
            receipt["job_read_receipt_id"],
            persisted["execution_receipts"]["last_read"]["receipt_id"],
        )
        self.assertEqual(
            receipt["requested_job_hash"],
            analysis_hash("job_20260724000000_deadbeef"),
        )
        self.assertEqual(receipt["resolved_job_hash"], receipt["requested_job_hash"])
        self.assertEqual(receipt["resolved_status_hash"], analysis_hash("completed"))
        rendered = render_fragment(result.response_fragments[0], "en")
        self.assertEqual(
            receipt["response_message_ids"],
            ["analysis.model_document"],
        )
        self.assertEqual(
            receipt["response_semantic_hash"],
            rendered.semantic_hash,
        )
        self.assertEqual(
            receipt["response_render_hash"],
            rendered.render_hash,
        )
        self.assertTrue(validate_analysis_receipt(receipt)[0])

    def test_report_action_emits_receipt_on_authoritative_state_not_temporary_view(self) -> None:
        state = self._state("report-action")
        action = ActionProposal(
            action_id="analyze-report",
            action_type="analyze_report",
            arguments={"job_id": "job_20260724000000_cafebabe"},
            confidence="high",
        )
        persisted = self._verified_job("job_20260724000000_cafebabe")

        with patch(
            "agent.harness.domains.analysis._report_artifact_entry",
            return_value=("Persisted report facts.", persisted, "report", {}),
        ):
            result = apply_analysis_action(state, action)

        receipt = self._receipts(state, "analysis_report")[-1]
        self.assertEqual(result.consumed_action_ids, ("analyze-report",))
        self.assertEqual(
            receipt["resolved_job_hash"],
            analysis_hash("job_20260724000000_cafebabe"),
        )
        rendered = render_fragment(result.response_fragments[0], "en")
        self.assertEqual(
            receipt["response_semantic_hash"],
            rendered.semantic_hash,
        )
        self.assertEqual(
            receipt["response_render_hash"],
            rendered.render_hash,
        )
        self.assertTrue(validate_analysis_receipt(receipt)[0])

    def test_static_response_semantic_identity_is_language_independent(self) -> None:
        question = {"id": "freeform_evidence", "kind": "log_evidence"}
        receipts = []
        for language in ("en", "zh"):
            state = new_state(f"language-{language}", language=language)
            state["turn_index"] = 7
            state["turn_context"] = {
                "turn_id": f"language-{language}-turn",
                "control_receipts": [],
            }
            collecting = {
                "question": question,
                "lines": ["Traceback"],
                "language": language,
                "status": "active",
                "block_id": evidence_block_id(question, ["Traceback"]),
            }
            continue_evidence_collection(state, "", collecting)
            receipts.append(self._receipts(state, "analysis_evidence_block")[-1])

        self.assertEqual(
            receipts[0]["response_message_ids"],
            ["analysis.response.waiting_logs"],
        )
        self.assertEqual(
            receipts[0]["response_semantic_hash"],
            receipts[1]["response_semantic_hash"],
        )
        self.assertNotEqual(
            receipts[0]["response_render_hash"],
            receipts[1]["response_render_hash"],
        )

    def test_version_one_receipt_fails_closed(self) -> None:
        state = self._state("old-analysis-receipt")
        receipt = self._invocation_receipt(state)
        receipt["receipt_version"] = ANALYSIS_RECEIPT_VERSION - 1
        receipt["receipt_id"] = analysis_hash({
            key: value for key, value in receipt.items() if key != "receipt_id"
        })

        valid, reason = validate_analysis_receipt(receipt)

        self.assertFalse(valid)
        self.assertEqual(reason, "unsupported receipt version")

    def test_rehashed_render_identity_mismatch_fails_closed(self) -> None:
        state = self._state("render-identity-mismatch")
        receipt = self._invocation_receipt(state)
        receipt["response_rendered"] = False
        receipt["receipt_id"] = analysis_hash({
            key: value for key, value in receipt.items() if key != "receipt_id"
        })

        valid, reason = validate_analysis_receipt(receipt)

        self.assertFalse(valid)
        self.assertEqual(reason, "inconsistent response render identity")

    def test_report_fallback_is_marked_unverified_without_manager_read_receipt(
        self,
    ) -> None:
        state = self._state("unverified-report")
        state["report_context"] = {
            "requested_job_id": "job_20260724000000_unverified"
        }
        fallback = {
            "job_id": "job_20260724000000_unverified",
            "status": "completed",
        }
        with patch(
            "agent.harness.domains.analysis._report_artifact_entry",
            return_value=("Last-known report summary.", fallback, "report", {}),
        ):
            report_artifact_entry_result(state)

        receipt = self._receipts(state, "analysis_report")[-1]
        self.assertFalse(receipt["invoked"])
        self.assertFalse(receipt["evidence_verified"])
        self.assertEqual(receipt["job_read_receipt_id"], "")
        self.assertTrue(validate_analysis_receipt(receipt)[0])

    def test_tampered_receipt_fails_closed(self) -> None:
        state = self._state("tampered-analysis-receipt")
        receipt = self._invocation_receipt(state)
        receipt["invoked"] = False

        valid, reason = validate_analysis_receipt(receipt)

        self.assertFalse(valid)
        self.assertEqual(reason, "receipt hash mismatch")

    def test_rehashed_semantically_inconsistent_receipt_fails_closed(self) -> None:
        state = self._state("inconsistent-analysis-receipt")
        receipt = self._invocation_receipt(state)
        receipt["source_kind"] = "missing"
        receipt["receipt_id"] = analysis_hash({
            key: value for key, value in receipt.items() if key != "receipt_id"
        })

        valid, reason = validate_analysis_receipt(receipt)

        self.assertFalse(valid)
        self.assertEqual(reason, "inconsistent invocation state")

    def test_non_analysis_consultation_does_not_emit_analysis_receipt_or_change_block(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn

        state = new_state("evidence-consultation", language="en")
        lines = ["Traceback (most recent call last):"]
        state["evidence_collection"] = {
            "question": {"id": "freeform_evidence", "kind": "log_evidence"},
            "lines": list(lines),
            "language": "en",
            "status": "active",
        }
        state["last_user_input"] = "Hello, who are you?"
        with (
            patch(
                "tests.agent_live.graph_turn.TEST_SEMANTIC_PLANNER",
                return_value={"actions": [{
                    "type": "greeting",
                    "source_evidence": state["last_user_input"],
                    "confidence": "high",
                }]},
            ),
            patch(
                "agent.harness.domains.orientation.opening_question",
                return_value=None,
            ),
        ):
            result = invoke_product_graph_turn(state)

        self.assertEqual(result["evidence_collection"]["lines"], lines)
        self.assertFalse([
            item
            for item in (result.get("turn_context") or {}).get("control_receipts") or []
            if str(item.get("receipt_type") or "").startswith("analysis_")
        ])


if __name__ == "__main__":
    unittest.main()

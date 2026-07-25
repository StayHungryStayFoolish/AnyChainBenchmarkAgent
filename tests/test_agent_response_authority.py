from __future__ import annotations

import ast
import unittest
from pathlib import Path

from agent.harness.contracts import (
    FailureDescriptor,
    HandlerResult,
    ResponseFragment,
    TextRef,
    handler_result_from_dict,
    handler_result_to_dict,
)
from agent.harness.questions import choice_question, question_text, render_question
from agent.harness.response import finalize_turn_response
from agent.harness.response_catalog import (
    MESSAGE_CATALOG,
    ResponseCatalogError,
    render_failure,
    render_fragment,
)
from agent.harness.state import new_state


ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ROOT / "agent" / "harness" / "domains"
COORDINATOR = ROOT / "agent" / "harness" / "coordinator.py"


class ResponseAuthorityTests(unittest.TestCase):
    def test_typed_fragments_and_failures_survive_serialization(self) -> None:
        result = HandlerResult(
            response_fragments=(
                ResponseFragment(
                    kind="evidence",
                    message_id="harness.response.probe_evidence_saved",
                    arguments={"receipt": "r1"},
                    payload={"artifact": "probe.json"},
                    source="agent.harness.domains.rpc_endpoint",
                ),
            ),
            blocker=FailureDescriptor(
                code="harness.failure.internal_contract_violation",
                arguments={"reason": "bad owner"},
                payload={"owner": "rpc_endpoint"},
                source="test",
                retryable=False,
                severity="critical",
            ),
        )

        serialized = handler_result_to_dict(result)
        restored = handler_result_from_dict(serialized)

        self.assertEqual(restored, result)
        self.assertNotIn("content", serialized["response_fragments"][0])
        self.assertEqual(
            serialized["blocker"]["code"],
            "harness.failure.internal_contract_violation",
        )
        self.assertEqual(serialized["blocker"]["severity"], "critical")

    def test_content_compatibility_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            ResponseFragment(  # type: ignore[call-arg]
                kind="message",
                content="legacy prose",
                source="test",
            )
        with self.assertRaises(ValueError):
            finalize_turn_response(
                self._state_with_fragments(
                    [
                        {
                            "kind": "message",
                            "content": "legacy prose",
                            "source": "test",
                            "metadata": {},
                        }
                    ]
                )
            )
        with self.assertRaises(ValueError):
            handler_result_from_dict(
                {
                    "response_fragments": [
                        {
                            "kind": "message",
                            "content": "legacy prose",
                            "source": "test",
                            "metadata": {},
                        }
                    ]
                }
            )
        with self.assertRaises(TypeError):
            handler_result_from_dict({"blocker": "legacy blocker prose"})
        with self.assertRaises(TypeError):
            handler_result_from_dict({"response_fragments": ["legacy prose"]})
        with self.assertRaises(TypeError):
            HandlerResult(blocker="legacy blocker prose")  # type: ignore[arg-type]

    def test_contract_rejects_untyped_arguments(self) -> None:
        with self.assertRaises(TypeError):
            TextRef(  # type: ignore[arg-type]
                "harness.response.configuration_loaded",
                {"bad": ["value"]},
            )
        with self.assertRaises(ValueError):
            ResponseFragment(  # type: ignore[arg-type]
                kind="question",
                message_id="harness.response.configuration_loaded",
            )

    def test_catalog_fails_closed_for_unknown_id_arguments_kind_and_language(self) -> None:
        cases = (
            ResponseFragment(kind="message", message_id="unknown.message"),
            ResponseFragment(
                kind="evidence",
                message_id="harness.response.probe_evidence_saved",
            ),
            ResponseFragment(
                kind="status",
                message_id="harness.response.probe_evidence_saved",
                arguments={"receipt": "r1"},
            ),
        )
        for fragment in cases:
            with self.subTest(fragment=fragment):
                with self.assertRaises(ResponseCatalogError):
                    render_fragment(fragment, "en")
        with self.assertRaises(ResponseCatalogError):
            render_fragment(
                ResponseFragment(
                    kind="status",
                    message_id="harness.response.configuration_loaded",
                ),
                "fr",
            )
        with self.assertRaises(TypeError):
            MESSAGE_CATALOG["runtime.injection"] = MESSAGE_CATALOG[  # type: ignore[index]
                "harness.response.configuration_loaded"
            ]

    def test_semantic_hash_is_language_independent_and_render_hash_is_not(self) -> None:
        fragment = ResponseFragment(
            kind="evidence",
            message_id="harness.response.probe_evidence_saved",
            arguments={"receipt": "r1"},
            payload={"artifact": "probe.json"},
        )

        english = render_fragment(fragment, "en")
        chinese = render_fragment(fragment, "zh")

        self.assertEqual(english.semantic_hash, chinese.semantic_hash)
        self.assertNotEqual(english.render_hash, chinese.render_hash)
        self.assertNotEqual(english.text, chinese.text)

    def test_payload_contributes_to_semantic_hash_only(self) -> None:
        first = render_fragment(
            ResponseFragment(
                kind="status",
                message_id="harness.response.configuration_loaded",
                payload={"checkpoint": "a"},
            ),
            "en",
        )
        second = render_fragment(
            ResponseFragment(
                kind="status",
                message_id="harness.response.configuration_loaded",
                payload={"checkpoint": "b"},
            ),
            "en",
        )

        self.assertNotEqual(first.semantic_hash, second.semantic_hash)
        self.assertEqual(first.render_hash, second.render_hash)
        with self.assertRaises(ResponseCatalogError):
            render_fragment(
                ResponseFragment(
                    kind="status",
                    message_id="harness.response.configuration_loaded",
                    payload={"invalid": object()},
                ),
                "en",
            )

    def test_structured_analysis_payload_cannot_bypass_catalog_contract(self) -> None:
        valid = ResponseFragment(
            kind="evidence",
            message_id="analysis.model_document",
            payload={
                "text": "Observed result.",
                "source_kind": "persisted_job",
                "language": "en",
                "evidence_hash": "a" * 64,
                "evidence_paths": ["report.json"],
            },
        )
        self.assertEqual(render_fragment(valid, "en").text, "Observed result.")
        with self.assertRaises(ResponseCatalogError):
            render_fragment(
                ResponseFragment(
                    kind="status",
                    message_id="analysis.model_document",
                    payload={
                        "text": "Observed result.",
                        "source_kind": "inline",
                        "language": "en",
                        "evidence_hash": "a" * 64,
                        "evidence_paths": [],
                    },
                ),
                "en",
            )
        with self.assertRaises(ResponseCatalogError):
            render_fragment(
                ResponseFragment(
                    kind="evidence",
                    message_id="analysis.model_document",
                    arguments={"unregistered": "value"},
                    payload={
                        "text": "Observed result.",
                        "source_kind": "inline",
                        "language": "en",
                        "evidence_hash": "a" * 64,
                        "evidence_paths": [],
                    },
                ),
                "en",
            )
        with self.assertRaises(ResponseCatalogError):
            render_fragment(
                ResponseFragment(
                    kind="evidence",
                    message_id="analysis.model_document",
                    payload={
                        "text": "Observed result.",
                        "source_kind": "inline",
                        "language": "en",
                        "evidence_hash": "",
                        "evidence_paths": [],
                    },
                ),
                "en",
            )

    def test_failure_uses_the_same_fail_closed_catalog(self) -> None:
        rendered = render_failure(
            FailureDescriptor(
                code="harness.failure.internal_contract_violation",
                arguments={"reason": "bad owner"},
                payload={"owner": "rpc"},
            ),
            "en",
        )
        self.assertEqual(rendered.kind, "error")
        self.assertIn("bad owner", rendered.text)
        with self.assertRaises(ResponseCatalogError):
            render_failure(FailureDescriptor(code="unknown.failure"), "en")

    def test_response_composer_deduplicates_semantics_and_renders_question_once(self) -> None:
        state = new_state("response-authority", language="en")
        question = choice_question(
            "opening",
            "opening_next_action",
            question_text("question.orientation.opening.prompt"),
            owner="orientation",
            field="opening_next_action",
            options=[
                {
                    "label": question_text(
                        "question.orientation.resume.option.continue"
                    ),
                    "value": "continue",
                    "action": {"type": "answer_opening_question", "topic": "start"},
                    "expected_patch": {},
                    "return_policy": "stop_after_response",
                }
            ],
        )
        state["pending_question"] = question
        fragment = {
            "kind": "status",
            "message_id": "harness.response.configuration_loaded",
            "arguments": {},
            "payload": {},
            "source": "test",
        }
        state["response_fragments"] = [fragment, dict(fragment)]

        finalized = finalize_turn_response(state)

        rendered_question = render_question(question, "en")
        terminal_response = finalized["visible_response"][0]
        self.assertEqual(terminal_response.count("Configuration loaded."), 1)
        self.assertEqual(terminal_response.count(rendered_question), 1)
        manifest = finalized["turn_context"]["response_manifest"]
        self.assertEqual(len(manifest), 2)
        self.assertEqual(
            set(manifest[0]),
            {"semantic_hash", "render_hash", "role", "message_id"},
        )
        self.assertEqual(manifest[1]["message_id"], "opening_next_action")
        self.assertNotEqual(
            finalized["turn_context"]["terminal_semantic_hash"],
            finalized["turn_context"]["terminal_response_hash"],
        )
        self.assertEqual(finalized["response_fragments"], [])

    def test_domains_cannot_render_questions_or_write_terminal_output(self) -> None:
        violations: list[str] = []
        for path in sorted(DOMAINS.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = {alias.name for alias in node.names}
                    if "render_question" in names:
                        violations.append(f"{path.name}:{node.lineno}:import render_question")
                if isinstance(node, ast.Call):
                    called = (
                        node.func.id
                        if isinstance(node.func, ast.Name)
                        else node.func.attr
                        if isinstance(node.func, ast.Attribute)
                        else ""
                    )
                    if called == "render_question":
                        violations.append(f"{path.name}:{node.lineno}:call render_question")
                    if called == "HandlerResult":
                        for keyword in node.keywords:
                            if keyword.arg in {"visible_result", "visible_results"}:
                                violations.append(
                                    f"{path.name}:{node.lineno}:HandlerResult {keyword.arg}"
                                )
                if isinstance(node, ast.Subscript):
                    key = node.slice.value if isinstance(node.slice, ast.Constant) else None
                    if key == "visible_response":
                        violations.append(f"{path.name}:{node.lineno}:visible_response access")
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if (
                        node.func.attr == "get"
                        and node.args
                        and isinstance(node.args[0], ast.Constant)
                        and node.args[0].value == "visible_response"
                    ):
                        violations.append(f"{path.name}:{node.lineno}:visible_response read")

        self.assertEqual(violations, [])

    def test_coordinator_cannot_render_or_compose_terminal_output(self) -> None:
        tree = ast.parse(COORDINATOR.read_text(encoding="utf-8"), filename=str(COORDINATOR))
        violations: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = {alias.name for alias in node.names}
                if "render_question" in names:
                    violations.append(f"{node.lineno}:import render_question")
            if isinstance(node, ast.Call):
                called = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else ""
                )
                if called == "render_question":
                    violations.append(f"{node.lineno}:call render_question")
            if isinstance(node, ast.Subscript):
                key = node.slice.value if isinstance(node.slice, ast.Constant) else None
                if key == "visible_response":
                    violations.append(f"{node.lineno}:visible_response access")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if (
                    node.func.attr == "get"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == "visible_response"
                ):
                    violations.append(f"{node.lineno}:visible_response read")

        self.assertEqual(violations, [])
        self.assertNotIn("visible_result", HandlerResult.__dataclass_fields__)
        self.assertNotIn("visible_results", HandlerResult.__dataclass_fields__)
        self.assertNotIn("content", ResponseFragment.__dataclass_fields__)

    @staticmethod
    def _state_with_fragments(fragments: list[dict[str, object]]):
        state = new_state("response-authority", language="en")
        state["response_fragments"] = fragments
        return state


if __name__ == "__main__":
    unittest.main()

"""R29 regression tests for Agent turn deadlines, cancellation, and commits."""

from __future__ import annotations

import json
import os
import pty
import select
import signal
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]


class TurnBudgetContractTest(unittest.TestCase):
    def test_copied_worker_context_shares_deadline_and_cancellation(self) -> None:
        from agent.llm.types import (
            LLMTurnCancelledError,
            cancel_active_llm_turn,
            copy_llm_turn_context,
            ensure_turn_active,
            llm_turn_scope,
            remaining_turn_seconds,
            run_in_llm_turn_context,
        )

        entered = threading.Event()
        proceed = threading.Event()

        def worker() -> float:
            remaining = remaining_turn_seconds()
            entered.set()
            self.assertTrue(proceed.wait(timeout=1))
            ensure_turn_active()
            return remaining

        with llm_turn_scope(1):
            time.sleep(0.03)
            context = copy_llm_turn_context()
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    run_in_llm_turn_context,
                    context,
                    worker,
                )
                self.assertTrue(entered.wait(timeout=1))
                cancel_active_llm_turn()
                proceed.set()
                with self.assertRaises(LLMTurnCancelledError):
                    future.result(timeout=1)


    def test_deepseek_has_explicit_transport_limits_and_bounded_retries(self) -> None:
        from agent.llm.config import LLMConfig
        from agent.llm.providers import DeepSeekProvider
        from agent.llm.types import LLMMessage, LLMRequest, llm_turn_scope

        captured: dict[str, object] = {}

        class Response:
            choices = [types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))]

            def model_dump(self):
                return {}

        class Completions:
            def create(self, **kwargs):
                captured["request"] = kwargs
                return Response()

        class OpenAI:
            def __init__(self, **kwargs):
                captured["client"] = kwargs
                self.chat = types.SimpleNamespace(completions=Completions())

        module = types.ModuleType("openai")
        module.OpenAI = OpenAI
        httpx_module = types.ModuleType("httpx")
        httpx_module.Timeout = lambda **values: types.SimpleNamespace(**values)
        config = LLMConfig(
            provider="deepseek",
            model="deepseek-chat",
            deepseek_api_key="secret",
            deepseek_api_key_present=True,
            turn_timeout_seconds=7,
            connect_timeout_seconds=2,
            read_timeout_seconds=4,
            max_retries=1,
        )
        with patch.dict(sys.modules, {"openai": module, "httpx": httpx_module}):
            with llm_turn_scope(7):
                DeepSeekProvider(config).complete(LLMRequest(messages=[LLMMessage(role="user", content="hello")]))

        client = captured["client"]
        self.assertEqual(client["max_retries"], 1)
        self.assertLessEqual(client["timeout"].connect, 2)
        self.assertLessEqual(client["timeout"].read, 4)
        self.assertNotIn("extra_body", captured["request"])

    def test_deepseek_maps_bounded_reasoning_to_official_non_thinking_mode(
        self,
    ) -> None:
        from agent.llm.config import LLMConfig
        from agent.llm.providers import DeepSeekProvider
        from agent.llm.types import LLMMessage, LLMRequest, llm_turn_scope

        captured: dict[str, object] = {}

        class Response:
            choices = [
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content='{"ok":true}')
                )
            ]

            def model_dump(self):
                return {}

        class OpenAI:
            def __init__(self, **_kwargs):
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(
                        create=lambda **kwargs: (
                            captured.setdefault("request", kwargs),
                            Response(),
                        )[1]
                    )
                )

        module = types.ModuleType("openai")
        module.OpenAI = OpenAI
        httpx_module = types.ModuleType("httpx")
        httpx_module.Timeout = lambda **values: types.SimpleNamespace(**values)
        config = LLMConfig(
            provider="deepseek",
            model="deepseek-v4-pro",
            deepseek_api_key="secret",
            deepseek_api_key_present=True,
        )
        with patch.dict(sys.modules, {"openai": module, "httpx": httpx_module}):
            with llm_turn_scope(2):
                DeepSeekProvider(config).complete(
                    LLMRequest(
                        messages=[LLMMessage(role="user", content="map value")],
                        reasoning_mode="disabled",
                    )
                )

        self.assertEqual(
            captured["request"]["extra_body"],
            {"thinking": {"type": "disabled"}},
        )

    def test_provider_timeout_is_typed(self) -> None:
        from agent.llm.config import LLMConfig
        from agent.llm.providers import DeepSeekProvider
        from agent.llm.types import LLMTurnTimeoutError, LLMMessage, LLMRequest, llm_turn_scope

        class APITimeoutError(Exception):
            __module__ = "openai"

        class OpenAI:
            def __init__(self, **_kwargs):
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(create=lambda **_call: (_ for _ in ()).throw(APITimeoutError("slow")))
                )

        module = types.ModuleType("openai")
        module.OpenAI = OpenAI
        httpx_module = types.ModuleType("httpx")
        httpx_module.Timeout = lambda **values: types.SimpleNamespace(**values)
        config = LLMConfig(
            provider="deepseek",
            model="deepseek-chat",
            deepseek_api_key="secret",
            deepseek_api_key_present=True,
        )
        with patch.dict(sys.modules, {"openai": module, "httpx": httpx_module}):
            with llm_turn_scope(2):
                with self.assertRaises(LLMTurnTimeoutError):
                    DeepSeekProvider(config).complete(LLMRequest(messages=[LLMMessage(role="user", content="hello")]))

    def test_deepseek_provider_configuration_failure_is_typed(self) -> None:
        from agent.llm.config import LLMConfig
        from agent.llm.providers import DeepSeekProvider
        from agent.llm.types import (
            LLMMessage,
            LLMProviderError,
            LLMRequest,
            llm_turn_scope,
        )

        class BadRequestError(Exception):
            __module__ = "openai"
            status_code = 400

        class OpenAI:
            def __init__(self, **_kwargs):
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(
                        create=lambda **_call: (_ for _ in ()).throw(
                            BadRequestError("unsupported model")
                        )
                    )
                )

        module = types.ModuleType("openai")
        module.OpenAI = OpenAI
        httpx_module = types.ModuleType("httpx")
        httpx_module.Timeout = lambda **values: types.SimpleNamespace(**values)
        config = LLMConfig(
            provider="deepseek",
            model="invalid-model",
            deepseek_api_key="secret",
            deepseek_api_key_present=True,
        )
        with patch.dict(sys.modules, {"openai": module, "httpx": httpx_module}):
            with llm_turn_scope(2):
                with self.assertRaises(LLMProviderError) as raised:
                    DeepSeekProvider(config).complete(
                        LLMRequest(messages=[LLMMessage(role="user", content="hello")])
                    )

        self.assertEqual(raised.exception.category, "configuration")
        self.assertEqual(raised.exception.status_code, 400)

    def test_malformed_openai_success_response_is_typed_at_provider_boundary(
        self,
    ) -> None:
        from agent.llm.config import LLMConfig
        from agent.llm.providers import DeepSeekProvider
        from agent.llm.types import (
            LLMMessage,
            LLMProviderError,
            LLMRequest,
            llm_turn_scope,
        )

        malformed_responses = (
            types.SimpleNamespace(choices=[]),
            types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace())]
            ),
            types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content="")
                    )
                ]
            ),
        )
        config = LLMConfig(
            provider="deepseek",
            model="deepseek-chat",
            deepseek_api_key="secret",
            deepseek_api_key_present=True,
        )
        httpx_module = types.ModuleType("httpx")
        httpx_module.Timeout = lambda **values: types.SimpleNamespace(**values)

        for response in malformed_responses:
            with self.subTest(response=response):
                class OpenAI:
                    def __init__(self, **_kwargs):
                        self.chat = types.SimpleNamespace(
                            completions=types.SimpleNamespace(
                                create=lambda **_call: response
                            )
                        )

                module = types.ModuleType("openai")
                module.OpenAI = OpenAI
                with patch.dict(
                    sys.modules,
                    {"openai": module, "httpx": httpx_module},
                ):
                    with llm_turn_scope(2):
                        with self.assertRaises(LLMProviderError) as raised:
                            DeepSeekProvider(config).complete(
                                LLMRequest(
                                    messages=[
                                        LLMMessage(role="user", content="hello")
                                    ]
                                )
                            )

                self.assertEqual(raised.exception.category, "response")
                self.assertEqual(
                    raised.exception.stage,
                    "provider_response",
                )

    def test_malformed_native_success_responses_are_typed_at_provider_boundary(
        self,
    ) -> None:
        from agent.llm.config import LLMConfig
        from agent.llm.providers import _anthropic_text, _gemini_text
        from agent.llm.types import LLMProviderError

        cases = (
            (
                _gemini_text,
                LLMConfig(
                    provider="gemini",
                    model="gemini-test",
                    gemini_api_key="secret",
                    gemini_api_key_present=True,
                ),
                {"candidates": []},
            ),
            (
                _anthropic_text,
                LLMConfig(
                    provider="claude",
                    model="claude-test",
                    anthropic_api_key="secret",
                    anthropic_api_key_present=True,
                ),
                {"content": [{"type": "text"}]},
            ),
        )
        for extractor, config, response in cases:
            with self.subTest(provider=config.provider):
                with self.assertRaises(LLMProviderError) as raised:
                    extractor(config, response)
                self.assertEqual(raised.exception.category, "response")
                self.assertEqual(
                    raised.exception.stage,
                    "provider_response",
                )


class TurnCheckpointContractTest(unittest.TestCase):
    def test_turn_receipt_binds_every_semantic_unit_to_its_admitted_action(self) -> None:
        from agent.harness.state import new_state
        from tests.agent_live.graph_turn import invoke_product_graph_turn

        state = new_state("receipt-unit-binding", language="en")
        state["last_user_input"] = "Who are you?"
        semantic_plan = {
            "actions": [{
                "type": "answer_opening_question",
                "topic": "identity",
                "source_evidence": "Who are you?",
                "confidence": "high",
            }],
            "semantic_units": [{
                "unit_id": "identity-unit",
                "clause_id": "clause-1",
                "source_text": "Who are you?",
                "disposition": "action",
                "action_indexes": [0],
            }],
        }

        with patch(
            "tests.agent_live.graph_turn.TEST_SEMANTIC_PLANNER",
            return_value=semantic_plan,
        ):
            result = invoke_product_graph_turn(state)

        receipt = result["turn_receipt"]
        action_id = receipt["admitted_action_ids"][0]
        self.assertEqual(
            receipt["action_unit_bindings"][action_id],
            ["identity-unit"],
        )
        self.assertEqual(
            receipt["unit_action_bindings"]["identity-unit"],
            [action_id],
        )
        self.assertEqual(
            receipt["sibling_omission_checks"][0]["verdict"],
            "covered",
        )

    def test_prepare_resumes_the_first_incomplete_side_effect_phase(self) -> None:
        from agent.harness.contracts import ActionEnvelope, action_envelope_to_dict
        from agent.harness.coordinator import prepare_turn_step
        from agent.harness.state import new_state

        envelope = ActionEnvelope(
            action_id="execute-1",
            action_type="approve_preflight_smoke",
            owner="execution",
            target_group="preflight_smoke_execution",
            effect_kind="external",
            status="admitted",
        )
        for status, expected_phase in (
            ("prepared", "invoke_effect"),
            ("invoking", "perform_effect"),
        ):
            with self.subTest(status=status):
                state = new_state(f"resume-{status}", language="en")
                state["action_queue"] = [action_envelope_to_dict(envelope)]
                state["selected_action"] = action_envelope_to_dict(
                    ActionEnvelope(
                        **{
                            **envelope.__dict__,
                            "status": "selected",
                        }
                    )
                )
                state["current_action"] = {
                    "type": envelope.action_type,
                    "action_id": envelope.action_id,
                }
                state["control"] = {"selected_owner": "execution"}
                state["side_effect_intent"] = {
                    "intent_id": "intent-1",
                    "turn_id": "turn-1",
                    "action_id": envelope.action_id,
                    "operation": envelope.action_type,
                    "idempotency_key": "harness:request-1",
                    "request": {},
                    "request_fingerprint": "fingerprint",
                    "expected_receipt_kind": "execution_handler_result",
                    "status": status,
                    "attempt_count": 1,
                }
                before_turn = state["turn_index"]

                resumed = prepare_turn_step(state)

                self.assertEqual(resumed["control"]["phase"], expected_phase)
                self.assertEqual(resumed["turn_index"], before_turn)

    def test_fresh_startup_persists_the_opening_contract_as_next_turn_baseline(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime
        from tests.agent_live.graph_turn import (
            reviewed_action_plan,
            reviewed_stage_planner,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            event_file = root / "turn-events.jsonl"
            with patch.dict(os.environ, {"ANYCHAIN_AGENT_TURN_EVENT_FILE": str(event_file)}):
                runtime = AnyChainGraphRuntime(
                    thread_id="fresh-startup-chain",
                    checkpoint_path=root / "checkpoint.sqlite",
                    session_purpose="dynamic-dual-ai-chaos",
                )
                offered = runtime.prepare_resume_offer("en")
                with reviewed_stage_planner(
                    lambda state, text: reviewed_action_plan(
                        state,
                        text,
                        [{"type": "greeting", "confidence": "high"}],
                    )
                ):
                    runtime.invoke("Hi", language="en")
                runtime.close()

            events = [json.loads(line) for line in event_file.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(offered["pending_question"]["id"], "opening_next_action")
        self.assertIn("Start a fake-node benchmark", "\n".join(offered["visible_response"]))
        self.assertEqual(events[0]["event_type"], "turn_committed")
        self.assertEqual(events[0]["observation"], "startup_snapshot")
        self.assertEqual(events[0]["pending_question_id"], "opening_next_action")
        self.assertEqual(events[1]["before_fingerprint"], events[0]["after_fingerprint"])

    def test_turn_event_reports_only_current_pending_answer_admission(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime
        from agent.harness.questions import manual_question, question_text
        from agent.harness.state import new_state
        from tests.agent_live.graph_turn import reviewed_execution_planner

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            event_file = root / "turn-events.jsonl"
            runtime = AnyChainGraphRuntime(
                thread_id="current-turn-admission",
                checkpoint_path=root / "checkpoint.sqlite",
                session_purpose="dynamic-dual-ai-chaos",
            )
            state = new_state(
                "current-turn-admission",
                language="en",
                session_purpose="dynamic-dual-ai-chaos",
            )
            state["active_group"] = "provider_deployment"
            state["pending_question"] = manual_question(
                "provider_deployment",
                "CLOUD_REGION",
                question_text("question.environment.cloud_region.prompt"),
                owner="environment",
                field="CLOUD_REGION",
            )
            state["completed_actions"] = [{
                "type": "choose_target_mode",
                "_submitted_turn_index": 0,
            }]
            runtime._persist_state(state)
            with patch.dict(os.environ, {"ANYCHAIN_AGENT_TURN_EVENT_FILE": str(event_file)}):
                with reviewed_execution_planner(
                    expected_input="asia-east1",
                    expected_admitted=True,
                ):
                    result = runtime.invoke("asia-east1", language="en")
            runtime.close()

            event = json.loads(event_file.read_text(encoding="utf-8").splitlines()[-1])

        self.assertEqual(result["confirmed_config"]["CLOUD_REGION"], "asia-east1")
        self.assertEqual(event["admitted_action_types"], ["answer_pending"])
        self.assertNotIn("choose_target_mode", event["admitted_action_types"])
        pending_receipts = [
            item
            for item in event["control_receipts"]
            if item["receipt_type"] == "pending_resolution"
        ]
        self.assertEqual(len(pending_receipts), 1)
        self.assertEqual(pending_receipts[0]["pending_id"], "CLOUD_REGION")
        self.assertEqual(pending_receipts[0]["verdict"], "accepted")
        self.assertIn(
            pending_receipts[0]["resolved_action_id"],
            event["turn_receipt_summary"]["execution_order"],
        )
        semantic_unit = event["turn_receipt_summary"]["semantic_units"][0]
        self.assertEqual(
            (semantic_unit["start"], semantic_unit["end"]),
            (0, len("asia-east1")),
        )
        response_receipts = [
            item
            for item in event["control_receipts"]
            if item["receipt_type"] == "response_composition"
        ]
        self.assertEqual(len(response_receipts), 1)
        self.assertEqual(
            len(response_receipts[0]["fragments"]),
            len(result["visible_response"]),
        )
        domain_receipts = [
            item
            for item in event["control_receipts"]
            if item["receipt_type"] == "domain_commit"
        ]
        self.assertEqual(len(domain_receipts), 1)
        self.assertEqual(
            domain_receipts[0]["group_state_transitions"],
            [
                {
                    "group": "job_monitoring",
                    "before": "",
                    "after": "invalidated",
                },
                {
                    "group": "preflight_smoke_execution",
                    "before": "",
                    "after": "invalidated",
                },
            ],
        )

    def test_cancelled_turn_does_not_commit_partial_checkpoint(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime
        from agent.llm.types import LLMTurnCancelledError

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AnyChainGraphRuntime(thread_id="rollback-cancel", checkpoint_path=Path(tmpdir) / "checkpoint.sqlite")
            runtime._persist_state(
                {
                    "target_mode": "fake-node",
                    "active_group": "target_mode",
                    "confirmed_config": {"CLOUD_REGION": "asia-east1"},
                }
            )
            before = runtime.snapshot()

            class CancelledGraph:
                def invoke(self, state, **kwargs):
                    state["target_mode"] = "real-node"
                    state["confirmed_config"]["CLOUD_REGION"] = "corrupt"
                    runtime.graph.update_state(
                        kwargs["config"],
                        state,
                        as_node="partition",
                    )
                    raise LLMTurnCancelledError("cancelled")

            with (
                patch.object(runtime.graph, "invoke", side_effect=CancelledGraph().invoke),
                self.assertRaises(LLMTurnCancelledError),
            ):
                runtime.invoke("change everything", language="en")
            attempts = runtime.turn_transactions.list_attempts(
                runtime.transaction_authority_id
            )
            runtime.close()
            restarted = AnyChainGraphRuntime(
                thread_id="rollback-cancel",
                checkpoint_path=Path(tmpdir) / "checkpoint.sqlite",
            )
            after = restarted.snapshot()
            restarted.close()

        self.assertEqual(after, before)
        self.assertGreaterEqual(len(attempts), 2)
        failed = attempts[-1]
        self.assertEqual(failed.status, "aborted")
        self.assertNotEqual(failed.physical_thread_id, "rollback-cancel")
        self.assertTrue(failed.diagnostic_hash)

    def test_timeout_turn_does_not_commit_partial_checkpoint(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime
        from agent.llm.types import LLMTurnTimeoutError

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AnyChainGraphRuntime(thread_id="rollback-timeout", checkpoint_path=Path(tmpdir) / "checkpoint.sqlite")
            runtime._persist_state({"target_mode": "fake-node", "active_group": "target_mode"})
            before = runtime.snapshot()

            class TimedOutGraph:
                def invoke(self, state, **kwargs):
                    state["target_mode"] = "real-node"
                    runtime.graph.update_state(
                        kwargs["config"],
                        state,
                        as_node="partition",
                    )
                    raise LLMTurnTimeoutError("deadline")

            with (
                patch.object(runtime.graph, "invoke", side_effect=TimedOutGraph().invoke),
                self.assertRaises(LLMTurnTimeoutError),
            ):
                runtime.invoke("change everything", language="en")
            attempt = runtime.turn_transactions.list_attempts(
                runtime.transaction_authority_id
            )[-1]
            runtime.close()
            restarted = AnyChainGraphRuntime(
                thread_id="rollback-timeout",
                checkpoint_path=Path(tmpdir) / "checkpoint.sqlite",
            )
            after = restarted.snapshot()
            restarted.close()

        self.assertEqual(after, before)
        self.assertEqual(attempt.status, "aborted")
        self.assertTrue(attempt.attempt_checkpoint_id is None)

    def test_provider_failure_does_not_commit_partial_checkpoint(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime
        from agent.llm.types import LLMProviderError

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AnyChainGraphRuntime(
                thread_id="rollback-provider",
                checkpoint_path=Path(tmpdir) / "checkpoint.sqlite",
            )
            runtime._persist_state({
                "target_mode": "fake-node",
                "active_group": "target_mode",
            })
            before = runtime.snapshot()

            class FailedGraph:
                def invoke(self, state, **kwargs):
                    state["target_mode"] = "real-node"
                    runtime.graph.update_state(
                        kwargs["config"],
                        state,
                        as_node="partition",
                    )
                    raise LLMProviderError(
                        "provider unavailable",
                        provider="deepseek",
                        model="invalid-model",
                        category="configuration",
                    )

            with (
                patch.object(runtime.graph, "invoke", side_effect=FailedGraph().invoke),
                self.assertRaises(LLMProviderError),
            ):
                runtime.invoke("change everything", language="en")
            attempt = runtime.turn_transactions.list_attempts(
                runtime.transaction_authority_id
            )[-1]
            runtime.close()
            restarted = AnyChainGraphRuntime(
                thread_id="rollback-provider",
                checkpoint_path=Path(tmpdir) / "checkpoint.sqlite",
            )
            after = restarted.snapshot()
            restarted.close()

        self.assertEqual(after, before)
        self.assertEqual(attempt.status, "aborted")

    def test_uncertain_external_effect_requires_reconciliation_without_moving_head(
        self,
    ) -> None:
        from agent.harness.graph import AnyChainGraphRuntime

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.sqlite"
            runtime = AnyChainGraphRuntime(
                thread_id="reconcile-effect",
                checkpoint_path=path,
            )
            runtime._persist_state({
                "target_mode": "fake-node",
                "active_group": "target_mode",
            })
            before = runtime.snapshot()
            head_before = runtime.turn_transactions.get_product_head(
                runtime.transaction_authority_id
            )

            class UncertainGraph:
                def invoke(self, state, **kwargs):
                    state["side_effect_intent"] = {
                        "status": "invoking",
                        "intent_id": "intent-1",
                    }
                    runtime.graph.update_state(
                        kwargs["config"],
                        state,
                        as_node="perform_effect",
                    )
                    raise RuntimeError("connection lost after invocation")

            with (
                patch.object(runtime.graph, "invoke", side_effect=UncertainGraph().invoke),
                self.assertRaisesRegex(RuntimeError, "connection lost"),
            ):
                runtime.invoke("approve execution", language="en")

            outcome = runtime.last_terminal_outcome
            head_after = runtime.turn_transactions.get_product_head(
                runtime.transaction_authority_id
            )
            runtime.close()
            restarted = AnyChainGraphRuntime(
                thread_id="reconcile-effect",
                checkpoint_path=path,
            )
            after = restarted.snapshot()
            restarted.close()

        self.assertIsNotNone(outcome)
        assert outcome is not None
        self.assertEqual(outcome.outcome, "reconciliation_required")
        self.assertTrue(outcome.attempt_checkpoint_id)
        self.assertEqual(head_after, head_before)
        self.assertEqual(after, before)

    def test_restart_closes_interrupted_attempt_without_advancing_product_head(
        self,
    ) -> None:
        from agent.harness.graph import AnyChainGraphRuntime

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.sqlite"
            runtime = AnyChainGraphRuntime(
                thread_id="restart-interrupted",
                checkpoint_path=path,
            )
            runtime._persist_state({
                "target_mode": "fake-node",
                "active_group": "target_mode",
            })
            before = runtime.snapshot()
            attempt = runtime.turn_transactions.begin_attempt(
                logical_thread_id=runtime.transaction_authority_id,
            )
            interrupted = dict(before)
            interrupted["target_mode"] = "real-node"
            runtime.graph.update_state(
                {"configurable": {"thread_id": attempt.physical_thread_id}},
                interrupted,
                as_node="partition",
            )
            runtime.close()

            restarted = AnyChainGraphRuntime(
                thread_id="restart-interrupted",
                checkpoint_path=path,
            )
            recovered_attempt = restarted.turn_transactions.get_attempt(
                attempt.transaction_id
            )
            after = restarted.snapshot()
            restarted.close()

        self.assertEqual(after, before)
        self.assertIsNotNone(recovered_attempt)
        assert recovered_attempt is not None
        self.assertEqual(recovered_attempt.status, "aborted")

    def test_restart_marks_interrupted_invocation_for_reconciliation(
        self,
    ) -> None:
        from agent.harness.graph import AnyChainGraphRuntime

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.sqlite"
            runtime = AnyChainGraphRuntime(
                thread_id="restart-reconcile",
                checkpoint_path=path,
            )
            runtime._persist_state({
                "target_mode": "fake-node",
                "active_group": "target_mode",
            })
            before = runtime.snapshot()
            attempt = runtime.turn_transactions.begin_attempt(
                logical_thread_id=runtime.transaction_authority_id,
            )
            interrupted = dict(before)
            interrupted["side_effect_intent"] = {
                "status": "invoking",
                "intent_id": "intent-restart",
            }
            runtime.graph.update_state(
                {"configurable": {"thread_id": attempt.physical_thread_id}},
                interrupted,
                as_node="perform_effect",
            )
            runtime.close()

            restarted = AnyChainGraphRuntime(
                thread_id="restart-reconcile",
                checkpoint_path=path,
            )
            recovered_attempt = restarted.turn_transactions.get_attempt(
                attempt.transaction_id
            )
            outcome = restarted.turn_transactions.get_terminal_outcome(
                attempt.transaction_id
            )
            after = restarted.snapshot()
            restarted.close()

        self.assertEqual(after, before)
        self.assertIsNotNone(recovered_attempt)
        self.assertEqual(recovered_attempt.status, "reconciliation_required")
        self.assertIsNotNone(outcome)
        self.assertEqual(outcome.outcome, "reconciliation_required")

    def test_unreadable_effect_evidence_fails_closed_to_reconciliation(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AnyChainGraphRuntime(
                thread_id="unreadable-effect-evidence",
                checkpoint_path=Path(tmpdir) / "checkpoint.sqlite",
            )
            before = runtime.snapshot()
            attempt = runtime._begin_turn_attempt(before)
            with patch.object(
                runtime.graph,
                "get_state",
                side_effect=RuntimeError("checkpoint read unavailable"),
            ):
                outcome = runtime._finish_failed_turn_attempt(
                    attempt,
                    RuntimeError("turn failed"),
                )
            runtime.close()

        self.assertEqual(outcome.outcome, "reconciliation_required")
        self.assertEqual(outcome.failure_category, "unexpected_failure")

    def test_commit_window_failure_leaves_no_active_attempt(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime
        from tests.agent_live.graph_turn import (
            reviewed_action_plan,
            reviewed_stage_planner,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AnyChainGraphRuntime(
                thread_id="commit-window-failure",
                checkpoint_path=Path(tmpdir) / "checkpoint.sqlite",
            )
            with reviewed_stage_planner(
                lambda state, text: reviewed_action_plan(
                    state,
                    text,
                    [{"type": "greeting", "confidence": "high"}],
                )
            ), patch.object(
                runtime.turn_transactions,
                "commit_attempt",
                side_effect=RuntimeError("commit window interrupted"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "commit window interrupted",
                ):
                    runtime.invoke("Hi", language="en")

            attempts = runtime.turn_transactions.list_attempts(
                runtime.transaction_authority_id
            )
            outcomes = runtime.turn_transactions.list_terminal_outcomes(
                runtime.transaction_authority_id
            )
            runtime.close()

        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0].status, "aborted")
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0].outcome, "aborted")

    def test_committed_outbox_replays_exact_response_once_after_restart(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime
        from agent.terminal.repl import AnyChainTerminal, TerminalSession
        from tests.agent_live.graph_turn import (
            reviewed_action_plan,
            reviewed_stage_planner,
        )

        messages: list[str] = []

        class IO:
            def agent(self, _language: str, message: str) -> None:
                messages.append(message)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "checkpoint.sqlite"
            runtime = AnyChainGraphRuntime(
                thread_id="terminal-outbox-replay",
                checkpoint_path=checkpoint,
            )
            with reviewed_stage_planner(
                lambda state, text: reviewed_action_plan(
                    state,
                    text,
                    [{"type": "greeting", "confidence": "high"}],
                )
            ):
                committed = runtime.invoke("Hi", language="en")
            expected = tuple(committed["visible_response"])
            self.assertEqual(len(runtime.pending_terminal_outcomes()), 1)
            runtime.close()

            restarted = AnyChainGraphRuntime(
                thread_id="terminal-outbox-replay",
                checkpoint_path=checkpoint,
            )
            app = AnyChainTerminal(
                state=TerminalSession(language="en"),
                io=IO(),
                session_id="terminal-outbox-replay",
                checkpoint_path=checkpoint,
            )
            app._harness = restarted
            app._deliver_pending_terminal_outcomes()
            app._deliver_pending_terminal_outcomes()
            restarted.close()

        self.assertEqual(tuple(messages), expected)

    def test_restart_rebuilds_runtime_event_after_post_commit_crash(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime
        from tests.agent_live.graph_turn import (
            reviewed_action_plan,
            reviewed_stage_planner,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = root / "checkpoint.sqlite"
            event_file = root / "turn-events.jsonl"
            runtime = AnyChainGraphRuntime(
                thread_id="runtime-event-recovery",
                checkpoint_path=checkpoint,
            )
            with reviewed_stage_planner(
                lambda state, text: reviewed_action_plan(
                    state,
                    text,
                    [{"type": "greeting", "confidence": "high"}],
                )
            ), patch.object(
                runtime,
                "_write_turn_observation",
                side_effect=RuntimeError("crash after commit"),
            ):
                with self.assertRaisesRegex(RuntimeError, "crash after commit"):
                    runtime.invoke("Hi", language="en")
            outcome = runtime.last_terminal_outcome
            self.assertIsNotNone(outcome)
            self.assertEqual(outcome.outcome, "committed")
            outcome = runtime.mark_terminal_delivered(outcome)
            self.assertIsNotNone(outcome.delivered_at)
            runtime.close()

            with patch.dict(
                os.environ,
                {"ANYCHAIN_AGENT_TURN_EVENT_FILE": str(event_file)},
            ):
                restarted = AnyChainGraphRuntime(
                    thread_id="runtime-event-recovery",
                    checkpoint_path=checkpoint,
                )
                rebuilt = json.loads(
                    event_file.read_text(encoding="utf-8").splitlines()[-1]
                )
                restarted.close()

        self.assertEqual(rebuilt["event_type"], "turn_committed")
        self.assertEqual(rebuilt["observation"], "turn_recovered")
        self.assertEqual(rebuilt["transaction_id"], outcome.transaction_id)
        self.assertEqual(rebuilt["terminal_event_id"], outcome.event_id)
        self.assertEqual(
            rebuilt["after_fingerprint"],
            outcome.product_fingerprint,
        )

    def test_restart_acknowledges_durable_runtime_event_without_duplicate(
        self,
    ) -> None:
        from agent.harness.graph import AnyChainGraphRuntime
        from tests.agent_live.graph_turn import (
            reviewed_action_plan,
            reviewed_stage_planner,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = root / "checkpoint.sqlite"
            event_file = root / "turn-events.jsonl"
            with patch.dict(
                os.environ,
                {"ANYCHAIN_AGENT_TURN_EVENT_FILE": str(event_file)},
            ):
                runtime = AnyChainGraphRuntime(
                    thread_id="runtime-event-ack-recovery",
                    checkpoint_path=checkpoint,
                )
                with reviewed_stage_planner(
                    lambda state, text: reviewed_action_plan(
                        state,
                        text,
                        [{"type": "greeting", "confidence": "high"}],
                    )
                ), patch.object(
                    runtime.turn_transactions,
                    "mark_runtime_event_published",
                    side_effect=RuntimeError("crash before publication ack"),
                ):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "crash before publication ack",
                    ):
                        runtime.invoke("Hi", language="en")
                pending = runtime.last_terminal_outcome
                self.assertIsNotNone(pending)
                self.assertEqual(pending.runtime_event_status, "pending")
                runtime.close()

                before_restart = event_file.read_text(
                    encoding="utf-8"
                ).splitlines()
                restarted = AnyChainGraphRuntime(
                    thread_id="runtime-event-ack-recovery",
                    checkpoint_path=checkpoint,
                )
                after_restart = event_file.read_text(
                    encoding="utf-8"
                ).splitlines()
                recovered = restarted.turn_transactions.get_terminal_outcome(
                    pending.transaction_id
                )
                restarted.close()

        self.assertEqual(len(before_restart), 1)
        self.assertEqual(after_restart, before_restart)
        self.assertEqual(recovered.runtime_event_status, "published")

    def test_v11_store_and_v5_runtime_log_are_quarantined_and_rebuilt(
        self,
    ) -> None:
        from agent.harness.graph import AnyChainGraphRuntime
        from tests.agent_live.graph_turn import (
            reviewed_action_plan,
            reviewed_stage_planner,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = root / "checkpoint.sqlite"
            event_file = root / "turn-events.jsonl"
            with patch.dict(
                os.environ,
                {"ANYCHAIN_AGENT_TURN_EVENT_FILE": str(event_file)},
            ):
                runtime = AnyChainGraphRuntime(
                    thread_id="legacy-runtime-event-recovery",
                    checkpoint_path=checkpoint,
                )
                with reviewed_stage_planner(
                    lambda state, text: reviewed_action_plan(
                        state,
                        text,
                        [{"type": "greeting", "confidence": "high"}],
                    )
                ):
                    runtime.invoke("Hi", language="en")
                original = runtime.last_terminal_outcome
                self.assertIsNotNone(original)
                runtime.close()

                legacy_payload = json.loads(
                    event_file.read_text(encoding="utf-8")
                )
                legacy_payload["schema_version"] = 5
                event_file.write_text(
                    json.dumps(legacy_payload, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                connection = sqlite3.connect(checkpoint)
                connection.execute("DROP INDEX anychain_runtime_event_id")
                connection.execute(
                    "DROP INDEX anychain_runtime_event_sequence_per_thread"
                )
                for column in (
                    "runtime_event_id",
                    "runtime_event_sequence",
                    "runtime_event_status",
                    "runtime_event_payload_hash",
                    "runtime_event_published_at",
                ):
                    connection.execute(
                        "ALTER TABLE anychain_terminal_outbox "
                        f"DROP COLUMN {column}"
                    )
                for column in (
                    "runtime_event_fence_sequence",
                    "runtime_event_fence_terminal_event_id",
                    "runtime_event_fence_id",
                    "runtime_event_fence_hash",
                ):
                    connection.execute(
                        "ALTER TABLE anychain_terminal_detours "
                        f"DROP COLUMN {column}"
                    )
                connection.execute(
                    """
                    UPDATE anychain_turn_transaction_meta
                    SET schema_version = 11
                    WHERE singleton = 1
                    """
                )
                connection.commit()
                connection.close()

                restarted = AnyChainGraphRuntime(
                    thread_id="legacy-runtime-event-recovery",
                    checkpoint_path=checkpoint,
                )
                recovered = restarted.turn_transactions.get_terminal_outcome(
                    original.transaction_id
                )
                rebuilt_records = event_file.read_text(
                    encoding="utf-8"
                ).splitlines()
                rebuilt = json.loads(rebuilt_records[0])
                quarantines = tuple(
                    root.glob(
                        "turn-events.jsonl.legacy-runtime-schema.*.quarantine"
                    )
                )
                quarantined = json.loads(
                    quarantines[0].read_text(encoding="utf-8")
                )
                restarted.close()

        self.assertEqual(len(quarantines), 1)
        self.assertEqual(quarantined["schema_version"], 5)
        self.assertEqual(len(rebuilt_records), 1)
        self.assertEqual(rebuilt["schema_version"], 6)
        self.assertEqual(rebuilt["observation"], "turn_recovered")
        self.assertEqual(rebuilt["terminal_event_id"], original.event_id)
        self.assertEqual(
            rebuilt["runtime_event_id"],
            f"migrated-runtime:{original.event_id}",
        )
        self.assertEqual(recovered.runtime_event_status, "published")
        self.assertEqual(
            recovered.runtime_event_payload_hash,
            rebuilt["runtime_event_payload_hash"],
        )

    def test_legacy_runtime_quarantine_recovery_is_reentrant_after_crash(
        self,
    ) -> None:
        from agent.harness.graph import AnyChainGraphRuntime
        from agent.harness.turn_transactions import TurnTransactionStore
        from tests.agent_live.graph_turn import (
            reviewed_action_plan,
            reviewed_stage_planner,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = root / "checkpoint.sqlite"
            event_file = root / "turn-events.jsonl"
            with patch.dict(
                os.environ,
                {"ANYCHAIN_AGENT_TURN_EVENT_FILE": str(event_file)},
            ):
                runtime = AnyChainGraphRuntime(
                    thread_id="legacy-quarantine-crash",
                    checkpoint_path=checkpoint,
                )
                with reviewed_stage_planner(
                    lambda state, text: reviewed_action_plan(
                        state,
                        text,
                        [{"type": "greeting", "confidence": "high"}],
                    )
                ):
                    runtime.invoke("Hi", language="en")
                original = runtime.last_terminal_outcome
                self.assertIsNotNone(original)
                runtime.close()

                legacy = json.loads(event_file.read_text(encoding="utf-8"))
                legacy["schema_version"] = 5
                event_file.write_text(
                    json.dumps(legacy, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                with patch(
                    "agent.harness.graph.quarantine_jsonl_file",
                    side_effect=RuntimeError("crash after SQLite requeue"),
                ), self.assertRaisesRegex(
                    RuntimeError,
                    "crash after SQLite requeue",
                ):
                    AnyChainGraphRuntime(
                        thread_id="legacy-quarantine-crash",
                        checkpoint_path=checkpoint,
                    )

                pending_store = TurnTransactionStore(checkpoint)
                pending = pending_store.get_terminal_outcome(
                    original.transaction_id
                )
                self.assertEqual(pending.runtime_event_status, "pending")
                self.assertTrue(event_file.exists())

                restarted = AnyChainGraphRuntime(
                    thread_id="legacy-quarantine-crash",
                    checkpoint_path=checkpoint,
                )
                recovered = restarted.turn_transactions.get_terminal_outcome(
                    original.transaction_id
                )
                rebuilt = json.loads(
                    event_file.read_text(encoding="utf-8")
                )
                restarted.close()

        self.assertEqual(rebuilt["schema_version"], 6)
        self.assertEqual(recovered.runtime_event_status, "published")

    def test_legacy_runtime_rebuild_uses_contiguous_product_revision_order(
        self,
    ) -> None:
        from agent.harness.graph import AnyChainGraphRuntime
        from tests.agent_live.graph_turn import (
            reviewed_action_plan,
            reviewed_stage_planner,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = root / "checkpoint.sqlite"
            event_file = root / "turn-events.jsonl"
            with patch.dict(
                os.environ,
                {"ANYCHAIN_AGENT_TURN_EVENT_FILE": str(event_file)},
            ):
                runtime = AnyChainGraphRuntime(
                    thread_id="legacy-sequence-rebuild",
                    checkpoint_path=checkpoint,
                )
                with reviewed_stage_planner(
                    lambda state, text: reviewed_action_plan(
                        state,
                        text,
                        [{"type": "greeting", "confidence": "high"}],
                    )
                ):
                    runtime.invoke("Hi", language="en")
                    runtime.invoke("Hello", language="en")
                runtime.close()

                legacy_records = [
                    json.loads(line)
                    for line in event_file.read_text(
                        encoding="utf-8"
                    ).splitlines()
                ]
                for record in legacy_records:
                    record["schema_version"] = 5
                event_file.write_text(
                    "".join(
                        json.dumps(record, sort_keys=True) + "\n"
                        for record in legacy_records
                    ),
                    encoding="utf-8",
                )

                restarted = AnyChainGraphRuntime(
                    thread_id="legacy-sequence-rebuild",
                    checkpoint_path=checkpoint,
                )
                rebuilt = [
                    json.loads(line)
                    for line in event_file.read_text(
                        encoding="utf-8"
                    ).splitlines()
                ]
                restarted.close()

        self.assertEqual(
            [record["runtime_event_sequence"] for record in rebuilt],
            [1, 2],
        )
        self.assertEqual(
            [record["schema_version"] for record in rebuilt],
            [6, 6],
        )

    def test_restart_rejects_corrupt_durable_runtime_event(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime
        from tests.agent_live.graph_turn import (
            reviewed_action_plan,
            reviewed_stage_planner,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = root / "checkpoint.sqlite"
            event_file = root / "turn-events.jsonl"
            with patch.dict(
                os.environ,
                {"ANYCHAIN_AGENT_TURN_EVENT_FILE": str(event_file)},
            ):
                runtime = AnyChainGraphRuntime(
                    thread_id="runtime-event-corruption",
                    checkpoint_path=checkpoint,
                )
                with reviewed_stage_planner(
                    lambda state, text: reviewed_action_plan(
                        state,
                        text,
                        [{"type": "greeting", "confidence": "high"}],
                    )
                ):
                    runtime.invoke("Hi", language="en")
                runtime.close()

                payload = json.loads(event_file.read_text(encoding="utf-8"))
                payload["attempt_checkpoint_id"] = "wrong-checkpoint"
                event_file.write_text(
                    json.dumps(payload, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

                with self.assertRaisesRegex(
                    RuntimeError,
                    "runtime observation does not match committed terminal facts",
                ):
                    AnyChainGraphRuntime(
                        thread_id="runtime-event-corruption",
                        checkpoint_path=checkpoint,
                    )


class TerminalDeadlineContractTest(unittest.TestCase):
    def test_terminal_surfaces_reconciliation_instead_of_generic_retry(self) -> None:
        from agent.terminal.language import t
        from agent.terminal.repl import AnyChainTerminal, TerminalSession
        from agent.harness.turn_transactions import TerminalOutcome

        messages: list[str] = []

        class IO:
            def agent(self, _language: str, message: str) -> None:
                messages.append(message)

        class UncertainHarness:
            last_terminal_outcome = TerminalOutcome(
                event_id="event-reconciliation",
                transaction_id="00000000-0000-0000-0000-000000000001",
                logical_thread_id="terminal-test:user",
                physical_thread_id="attempt:00000000-0000-0000-0000-000000000001",
                outcome="reconciliation_required",
                base_revision=0,
                base_checkpoint_thread_id="terminal-test",
                base_checkpoint_id="checkpoint-0",
                base_fingerprint="0" * 64,
                origin_revision_commit="test-revision",
                origin_revision_worktree_hash="2" * 64,
                attempt_checkpoint_id=None,
                attempt_fingerprint=None,
                product_revision=0,
                product_checkpoint_thread_id="terminal-test",
                product_checkpoint_id="checkpoint-0",
                product_fingerprint="0" * 64,
                diagnostic_hash="1" * 64,
                render_hash="",
                failure_category="external_effect_uncertain",
                runtime_event_id="",
                runtime_event_sequence=0,
                runtime_event_status="not_applicable",
                runtime_event_payload_hash=None,
                runtime_event_published_at=None,
                created_at="2026-07-27T00:00:00+00:00",
                delivered_at=None,
            )

            def invoke(self, *_args, **_kwargs):
                raise RuntimeError("external result unknown")

            def ensure_runtime_observation(self, _outcome):
                return None

            def mark_terminal_delivered(self, outcome):
                return outcome

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {
                "ANYCHAIN_AGENT_TERMINAL_OUTCOME_FILE": str(
                    Path(tmpdir) / "terminal.jsonl"
                )
            },
        ):
            app = AnyChainTerminal(state=TerminalSession(language="en"), io=IO())
            app._llm_runtime_available = True
            app._harness = UncertainHarness()
            app.handle_user_text("approve")

        self.assertIn(t("en", "turn_reconciliation_required"), messages)
        self.assertNotIn(t("en", "harness_runtime_error"), messages)

    def test_terminal_timeout_returns_control_without_applying_response(self) -> None:
        from dataclasses import replace

        from agent.terminal.repl import AnyChainTerminal, TerminalSession

        messages: list[str] = []

        class IO:
            def agent(self, _language: str, message: str) -> None:
                messages.append(message)

        class SlowHarness:
            def invoke(self, *_args, **_kwargs):
                time.sleep(5)
                return {"visible_response": ["must-not-render"]}

        app = AnyChainTerminal(state=TerminalSession(language="en"), io=IO())
        app._llm_runtime_available = True
        app._llm_config = replace(app._llm_config, turn_timeout_seconds=0.05)
        app._harness = SlowHarness()

        started = time.monotonic()
        app.handle_user_text("hello")

        self.assertLess(time.monotonic() - started, 1)
        self.assertFalse(app._turn_active)
        self.assertIn("deadline", "\n".join(messages))
        self.assertNotIn("must-not-render", "\n".join(messages))


class TerminalRuntimeLifecycleTest(unittest.TestCase):
    def test_interactive_run_closes_harness_on_normal_exit(self) -> None:
        from agent.terminal.repl import AnyChainTerminal, TerminalSession

        class Harness:
            def __init__(self) -> None:
                self.closed = 0

            def close(self) -> None:
                self.closed += 1

        class IO:
            def input(self, _language: str) -> str:
                raise EOFError

        harness = Harness()

        class App(AnyChainTerminal):
            def startup(self) -> None:
                self._harness = harness

            def _record_terminal_termination(self, **_kwargs) -> bool:
                return True

        app = App(state=TerminalSession(language="en"), io=IO())

        self.assertEqual(app.run(), 0)
        self.assertEqual(harness.closed, 1)
        app.close()
        self.assertEqual(harness.closed, 1)

    def test_interactive_run_closes_harness_when_startup_raises(self) -> None:
        from agent.terminal.repl import AnyChainTerminal, TerminalSession

        class Harness:
            def __init__(self) -> None:
                self.closed = 0

            def close(self) -> None:
                self.closed += 1

        harness = Harness()

        class App(AnyChainTerminal):
            def startup(self) -> None:
                self._harness = harness
                raise RuntimeError("startup failed")

        app = App(state=TerminalSession(language="en"))

        with self.assertRaisesRegex(RuntimeError, "startup failed"):
            app.run()
        self.assertEqual(harness.closed, 1)

    def test_prompt_entrypoint_closes_harness(self) -> None:
        from agent.terminal import repl

        class App:
            def __init__(self) -> None:
                self._requested_exit_code = None
                self.closed = 0

            def startup(self) -> None:
                return None

            def handle_user_text(self, _text: str) -> None:
                return None

            def close(self) -> None:
                self.closed += 1

        app = App()
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.object(repl, "AnyChainTerminal", return_value=app),
        ):
            result = repl.main([
                "--prompt",
                "hello",
                "--state-file",
                str(Path(tmpdir) / "session.json"),
            ])

        self.assertEqual(result, 0)
        self.assertEqual(app.closed, 1)


@unittest.skipUnless(os.name == "posix", "PTY/SIGINT coverage requires POSIX")
class InteractiveSignalContractTest(unittest.TestCase):
    def test_whitespace_bracketed_paste_is_a_terminal_noop(self) -> None:
        script = textwrap.dedent(
            """
            import tempfile
            from pathlib import Path
            from agent.terminal.repl import AnyChainTerminal, TerminalSession, TerminalSessionStore
            from agent.terminal.io import TerminalIO

            class CountingHarness:
                def __init__(self):
                    self.calls = 0

                def invoke(self, *args, **kwargs):
                    self.calls += 1
                    return {"visible_response": [f"HARNESS_CALLS={self.calls}"]}

            class App(AnyChainTerminal):
                def startup(self):
                    self._llm_runtime_available = True
                    self.harness = CountingHarness()
                    self.io.agent(self.state.language, "READY")

                def _ensure_harness(self):
                    return self.harness

            with tempfile.TemporaryDirectory() as tmpdir:
                raise SystemExit(App(
                    state=TerminalSession(language="en"),
                    store=TerminalSessionStore(Path(tmpdir) / "session.json"),
                    io=TerminalIO(),
                ).run())
            """
        )
        pid, fd = pty.fork()
        if pid == 0:  # pragma: no cover - child process
            env = os.environ.copy()
            env["PYTHONPATH"] = str(REPO_ROOT)
            os.execve(sys.executable, [sys.executable, "-c", script], env)

        transcript = bytearray()
        try:
            self._read_until(fd, transcript, b"User>", timeout=10)
            transcript.clear()
            os.write(fd, b"\x1b[200~   \x1b[201~\r")
            self._read_until(fd, transcript, b"User>", timeout=10)
            self.assertNotIn(b"HARNESS_CALLS", transcript)

            transcript.clear()
            os.write(fd, b"hello\r")
            self._read_until(fd, transcript, b"HARNESS_CALLS=1", timeout=10)
            self.assertNotIn(b"HARNESS_CALLS=2", transcript)
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def test_busy_sigint_cancels_turn_then_idle_sigint_exits(self) -> None:
        contract_dir = tempfile.TemporaryDirectory()
        self.addCleanup(contract_dir.cleanup)
        checkpoint_path = Path(contract_dir.name) / "turns.sqlite"
        projection_path = Path(contract_dir.name) / "terminal.jsonl"
        runtime_event_path = Path(contract_dir.name) / "runtime.jsonl"
        script = textwrap.dedent(
            """
            import os
            import tempfile
            import time
            from pathlib import Path
            from agent.terminal.repl import AnyChainTerminal, TerminalSession, TerminalSessionStore
            from agent.terminal.io import TerminalIO

            class BlockingHarness:
                def __init__(self):
                    self.calls = 0

                def invoke(self, *args, **kwargs):
                    self.calls += 1
                    if self.calls == 1:
                        time.sleep(30)
                        return {"visible_response": ["unexpected"]}
                    return {"visible_response": ["RECOVERED"]}

            class App(AnyChainTerminal):
                def startup(self):
                    self._llm_runtime_available = True
                    self.harness = BlockingHarness()
                    self.io.agent(self.state.language, "READY")

                def _ensure_harness(self):
                    return self.harness

            with tempfile.TemporaryDirectory() as tmpdir:
                raise SystemExit(App(
                    state=TerminalSession(language="en"),
                    store=TerminalSessionStore(Path(tmpdir) / "session.json"),
                    io=TerminalIO(),
                    checkpoint_path=Path(os.environ["TEST_CHECKPOINT"]),
                    session_id="pty-signal",
                    session_purpose="chaos",
                ).run())
            """
        )
        pid, fd = pty.fork()
        if pid == 0:  # pragma: no cover - child process
            env = os.environ.copy()
            env["PYTHONPATH"] = str(REPO_ROOT)
            env["TEST_CHECKPOINT"] = str(checkpoint_path)
            env["ANYCHAIN_AGENT_TERMINAL_OUTCOME_FILE"] = str(projection_path)
            env["ANYCHAIN_AGENT_TURN_EVENT_FILE"] = str(runtime_event_path)
            os.execve(sys.executable, [sys.executable, "-c", script], env)

        transcript = bytearray()
        try:
            self._read_until(fd, transcript, b"User>", timeout=10)
            transcript.clear()
            os.write(fd, b"hello\n")
            self._read_until(fd, transcript, b"Press Ctrl+C to cancel this turn", timeout=10)
            transcript.clear()
            os.kill(pid, signal.SIGINT)
            self._read_until(fd, transcript, b"Agent session is still active", timeout=10)
            transcript.clear()
            self._read_until(fd, transcript, b"User>", timeout=10)
            transcript.clear()
            os.write(fd, b"again\n")
            self._read_until(fd, transcript, b"RECOVERED", timeout=10)
            transcript.clear()
            self._read_until(fd, transcript, b"User>", timeout=10)
            os.kill(pid, signal.SIGINT)
            waited_pid, status = os.waitpid(pid, 0)
            self.assertEqual(waited_pid, pid)
            self.assertEqual(os.waitstatus_to_exitcode(status), 130)
            from agent.harness.terminal_protocol import read_jsonl_records

            projections = read_jsonl_records(projection_path)
            termination = projections[-1]
            self.assertEqual(
                termination["record_type"],
                "terminal_detour_projection",
            )
            self.assertEqual(termination["result_kind"], "session_termination")
            self.assertEqual(termination["termination_reason"], "outer_ctrl_c")
            self.assertEqual(termination["exit_code"], 130)
            self.assertFalse(runtime_event_path.exists())
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    @staticmethod
    def _read_until(fd: int, transcript: bytearray, marker: bytes, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while marker not in transcript:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(f"PTY marker {marker!r} not found in {transcript.decode(errors='replace')!r}")
            readable, _, _ = select.select([fd], [], [], remaining)
            if not readable:
                continue
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            transcript.extend(chunk)


if __name__ == "__main__":
    unittest.main()

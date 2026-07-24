"""R29 regression tests for Agent turn deadlines, cancellation, and commits."""

from __future__ import annotations

import json
import os
import pty
import select
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]


class TurnBudgetContractTest(unittest.TestCase):
    def test_bounded_recovery_and_repair_use_remaining_whole_turn_budget(self) -> None:
        from agent.harness.intent import resolve_action_queue
        from agent.llm.types import LLMResponse, llm_turn_scope, remaining_turn_seconds

        observed: list[float] = []

        class Provider:
            def complete(self, _request):
                observed.append(remaining_turn_seconds())
                if len(observed) == 1:
                    time.sleep(0.03)
                    return LLMResponse(text="not-json", model="test", provider="test")
                return LLMResponse(text='{"actions": [], "conflicts": [], "reason": "done"}', model="test", provider="test")

        with patch("agent.harness.intent.provider_from_config", return_value=Provider()):
            with llm_turn_scope(0.25):
                resolve_action_queue({}, "hello")

        self.assertGreaterEqual(len(observed), 2)
        self.assertLessEqual(len(observed), 8)
        self.assertLess(observed[1], observed[0] - 0.02)
        self.assertTrue(all(later <= earlier for earlier, later in zip(observed, observed[1:])))

    def test_distinct_resolvers_share_the_same_turn_deadline(self) -> None:
        from agent.harness.intent import resolve_action_queue, resolve_unknown_chain_identity
        from agent.llm.types import LLMResponse, llm_turn_scope, remaining_turn_seconds

        observed: list[float] = []

        class Provider:
            def complete(self, _request):
                observed.append(remaining_turn_seconds())
                if len(observed) == 1:
                    return LLMResponse(
                        text=(
                            '{"actions":[{"type":"answer_opening_question","topic":"identity"}],'
                            '"semantic_units":[{"unit_id":"unit-1","clause_id":"clause-1","start":0,"end":5,'
                            '"source_text":"hello","disposition":"action","action_indexes":[0],'
                            '"reason":"identity greeting"}],'
                            '"conflicts":[],"reason":"done"}'
                        ),
                        model="test",
                        provider="test",
                    )
                if len(observed) == 2:
                    return LLMResponse(
                        text='{"reviews":[{"action_index":0,"supported":true,"reason":"identity question"}],"unit_reviews":[]}',
                        model="test",
                        provider="test",
                    )
                return LLMResponse(
                    text='{"chain_exists": true, "confidence": "medium"}',
                    model="test",
                    provider="test",
                )

        with patch("agent.harness.intent.provider_from_config", return_value=Provider()):
            with llm_turn_scope(0.25):
                resolve_action_queue({}, "hello")
                time.sleep(0.03)
                resolve_unknown_chain_identity({}, "example-chain")

        self.assertGreaterEqual(len(observed), 3)
        self.assertTrue(all(later <= earlier for earlier, later in zip(observed, observed[1:])))
        self.assertLess(observed[-1], observed[0] - 0.02)

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
            "agent.harness.coordinator.resolve_action_queue",
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
                with patch(
                    "agent.harness.coordinator.resolve_action_queue",
                    return_value={"actions": [{"type": "greeting", "confidence": "high"}]},
                ):
                    runtime.invoke("Hi", language="en")
                runtime.close()

            events = [json.loads(line) for line in event_file.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(offered["pending_question"]["id"], "opening_next_action")
        self.assertIn("Start a fake-node benchmark", "\n".join(offered["visible_response"]))
        self.assertEqual(events[0]["event_type"], "startup_snapshot")
        self.assertEqual(events[0]["pending_question_id"], "opening_next_action")
        self.assertEqual(events[1]["before_fingerprint"], events[0]["after_fingerprint"])

    def test_turn_event_reports_only_current_pending_answer_admission(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime
        from agent.harness.questions import manual_question
        from agent.harness.state import new_state

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
                "Enter CLOUD_REGION.",
                field="CLOUD_REGION",
            )
            state["completed_actions"] = [{
                "type": "choose_target_mode",
                "_submitted_turn_index": 0,
            }]
            runtime._persist_state(state)
            with patch.dict(os.environ, {"ANYCHAIN_AGENT_TURN_EVENT_FILE": str(event_file)}):
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
                def invoke(self, state, **_kwargs):
                    state["target_mode"] = "real-node"
                    state["confirmed_config"]["CLOUD_REGION"] = "corrupt"
                    raise LLMTurnCancelledError("cancelled")

            with (
                patch.object(runtime.graph, "invoke", side_effect=CancelledGraph().invoke),
                self.assertRaises(LLMTurnCancelledError),
            ):
                runtime.invoke("change everything", language="en")
            after = runtime.snapshot()
            runtime.close()

        self.assertEqual(after, before)

    def test_timeout_turn_does_not_commit_partial_checkpoint(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime
        from agent.llm.types import LLMTurnTimeoutError

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AnyChainGraphRuntime(thread_id="rollback-timeout", checkpoint_path=Path(tmpdir) / "checkpoint.sqlite")
            runtime._persist_state({"target_mode": "fake-node", "active_group": "target_mode"})
            before = runtime.snapshot()

            class TimedOutGraph:
                def invoke(self, state, **_kwargs):
                    state["target_mode"] = "real-node"
                    raise LLMTurnTimeoutError("deadline")

            with (
                patch.object(runtime.graph, "invoke", side_effect=TimedOutGraph().invoke),
                self.assertRaises(LLMTurnTimeoutError),
            ):
                runtime.invoke("change everything", language="en")
            after = runtime.snapshot()
            runtime.close()

        self.assertEqual(after, before)


class TerminalDeadlineContractTest(unittest.TestCase):
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
        script = textwrap.dedent(
            """
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

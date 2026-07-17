"""Product terminal contract tests for the LangGraph Harness entrypoint."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class ProductTerminalHarnessContractTest(unittest.TestCase):
    def test_product_launchers_use_canonical_package_modules(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        launcher = (repo / "bin" / "anychain-agent").read_text(encoding="utf-8")
        manager = (repo / "agent" / "runners" / "job_manager.py").read_text(encoding="utf-8")

        self.assertIn('exec "$AGENT_PYTHON" -m agent.terminal.repl "$@"', launcher)
        self.assertNotIn("agent/terminal/repl.py", launcher)
        self.assertIn('"agent.runners.job_worker"', manager)
        self.assertNotIn('"agent" / "runners" / "job_worker.py"', manager)

    def test_terminal_uses_langgraph_harness_not_retired_workflow_runtime(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        text = (repo / "agent" / "terminal" / "repl.py").read_text(encoding="utf-8")

        self.assertIn("AnyChainGraphRuntime", text)
        self.assertNotIn("ADKRunnerBridge", text)
        self.assertNotIn("workflows.conversation_state", text)
        self.assertNotIn("workflows.transition_executor", text)
        self.assertNotIn("terminal.pending_answers", text)
        self.assertNotIn("terminal.input_classifier", text)

    def test_startup_preserves_dependency_offer_so_yes_installs(self) -> None:
        """The startup "[Y/n]" dependency-install offer must stay actionable.

        Regression for a real first-interaction bug: `_startup_doctor` set
        current_question_id="install_dependencies" (missing vegeta), but the
        ADK-available branch in `startup()` then reset current_question_id="",
        so the user's "y" never reached `_install_dependencies` and fell through
        to the harness (greeting/capabilities).
        """

        import types

        from agent.terminal import repl as repl_mod
        from agent.terminal.io import OutputOnlyIO
        from agent.terminal.repl import AnyChainTerminal, TerminalSession

        with tempfile.TemporaryDirectory() as tmpdir:
            app = AnyChainTerminal(
                state=TerminalSession(language="en"),
                io=OutputOnlyIO(),
                session_id="dep-thread",
                checkpoint_path=Path(tmpdir) / "cp.sqlite",
                session_purpose="chaos",
            )
            installed = {"called": False}
            app._load_framework_context = lambda: None  # type: ignore[method-assign]
            app._startup_doctor = lambda: setattr(app.state, "current_question_id", "install_dependencies")  # type: ignore[method-assign]
            app._install_dependencies = lambda: installed.__setitem__("called", True)  # type: ignore[method-assign]
            app._offer_harness_resume_if_needed = lambda: None  # type: ignore[method-assign]

            with patch.object(
                repl_mod, "adk_status", return_value=types.SimpleNamespace(as_dict=lambda: {"available": True, "reason": "t"})
            ), patch.object(repl_mod, "provider_runtime_errors", return_value=[]):
                app.startup()
                # The offer must survive startup, not be clobbered.
                self.assertEqual(app.state.current_question_id, "install_dependencies")
                # "y" must route to the installer.
                app.handle_user_text("y")

        self.assertTrue(installed["called"])

    def test_deepseek_runtime_does_not_require_google_adk(self) -> None:
        from agent.llm.config import LLMConfig
        from agent.llm import providers

        config = LLMConfig(
            provider="deepseek",
            model="deepseek-chat",
            auth_mode="api_key",
            deepseek_api_key="secret",
            deepseek_api_key_present=True,
        )

        def find_spec(name: str):
            return object() if name == "openai" else None

        with patch.object(providers.importlib.util, "find_spec", side_effect=find_spec):
            self.assertEqual(providers.provider_runtime_errors(config), [])

    def test_logs_command_reports_clean_error_for_missing_job(self) -> None:
        """`logs <bad-id>` must emit a clean "job not found" message, not a raw

        FileNotFoundError. Regression for a live chaos finding: `_logs` called
        tail_job_log without a guard, leaking the exception to the user, while
        `_follow_logs`/`_status` reported cleanly.
        """

        from agent.terminal.job_commands import JobCommandHandler

        class _State:
            language = "en"
            latest_job_id = ""

        messages: list[str] = []

        class _IO:
            def agent(self, language: str, message: str) -> None:
                messages.append(message)

        handler = JobCommandHandler(_State(), _IO())
        handler._logs("totally-bogus-xyz")  # must not raise
        joined = "\n".join(messages)
        self.assertNotIn("FileNotFoundError", joined)
        self.assertNotIn("Traceback", joined)
        self.assertIn("totally-bogus-xyz", joined)

    def test_technical_scalar_keeps_existing_chinese_language(self) -> None:
        from agent.terminal.language import detect_language

        self.assertEqual(detect_language("eth_chainId=70", default="zh"), "zh")
        self.assertEqual(detect_language("eth_blockNumber=70,eth_getBalance=30", default="zh"), "zh")
        self.assertEqual(detect_language('  File "agent/terminal/repl.py", line 123, in run', default="zh"), "zh")
        self.assertEqual(detect_language("machine_type: n2-standard-16", default="zh"), "zh")
        self.assertEqual(detect_language("data_vol_type: hyperdisk-balanced,", default="zh"), "zh")
        self.assertEqual(detect_language("throughput: 1000 MiB/s", default="zh"), "zh")
        self.assertEqual(
            detect_language("region=asia-east1 zone=asia-east1-c machine=n2-standard-16 ledger=vda", default="zh"),
            "zh",
        )

    def test_complete_english_technical_prose_switches_from_chinese(self) -> None:
        from agent.terminal.language import detect_language

        self.assertEqual(
            detect_language(
                "I am not sure what I can test here. Please explain the supported chains, "
                "RPC methods, and extension options first.",
                default="zh",
            ),
            "en",
        )
        self.assertEqual(
            detect_language("Can you validate this endpoint and explain the response?", default="zh"),
            "en",
        )

    def test_structured_and_pasted_inputs_preserve_chinese_language(self) -> None:
        from agent.terminal.language import detect_language

        self.assertEqual(detect_language("Y", default="zh"), "zh")
        self.assertEqual(detect_language("eth0", default="zh"), "zh")
        self.assertEqual(detect_language('{"jsonrpc":"2.0","method":"eth_chainId"}', default="zh"), "zh")
        self.assertEqual(
            detect_language("machine_type: n2-standard-16\nzone: asia-east1-c", default="zh"),
            "zh",
        )

    def test_terminal_entrypoint_routes_greeting_to_harness_opening_group(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AnyChainGraphRuntime(thread_id="terminal-contract", checkpoint_path=Path(tmpdir) / "checkpoint.sqlite")
            with patch("agent.harness.coordinator.resolve_action_queue", return_value={"actions": [{"type": "greeting", "confidence": "high"}]}):
                state = runtime.invoke("Hi", language="en")

        self.assertEqual(state["active_group"], "opening")
        self.assertEqual(state["pending_question"]["id"], "opening_next_action")
        self.assertIn("AnyChain Benchmark Agent", state["visible_response"][0])

    def test_harness_snapshot_and_reset_support_terminal_resume_gate(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AnyChainGraphRuntime(thread_id="resume-contract", checkpoint_path=Path(tmpdir) / "checkpoint.sqlite")
            runtime._persist_state(
                {
                    "target_mode": "fake-node",
                    "workflow_mode": "rpc_benchmark",
                    "chain_identity": {"canonical": "bsc", "status": "confirmed"},
                    "confirmed_config": {"CLOUD_REGION": "asia-east1"},
                    "pending_question": {"id": "CLOUD_ZONE", "group": "provider_deployment", "kind": "manual_value", "prompt": "Confirm CLOUD_ZONE."},
                    "active_group": "provider_deployment",
                }
            )

            snapshot = runtime.snapshot()
            self.assertEqual(snapshot["target_mode"], "fake-node")
            self.assertEqual(snapshot["confirmed_config"]["CLOUD_REGION"], "asia-east1")
            self.assertEqual(snapshot["pending_question"]["id"], "CLOUD_ZONE")

            fresh = runtime.reset(language="en")
            self.assertEqual(fresh["target_mode"], "")
            self.assertEqual(fresh["confirmed_config"], {})
            self.assertEqual(fresh["pending_question"], {})
            self.assertEqual(fresh["session"]["purpose"], "user")

    def test_terminal_passes_explicit_checkpoint_and_session_purpose_to_harness(self) -> None:
        from agent.terminal.io import OutputOnlyIO
        from agent.terminal.repl import AnyChainTerminal, TerminalSession

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "isolated.sqlite"
            app = AnyChainTerminal(
                state=TerminalSession(language="en"),
                io=OutputOnlyIO(),
                session_id="isolated-thread",
                checkpoint_path=checkpoint,
                session_purpose="chaos",
            )
            runtime = app._ensure_harness()
            state = runtime.invoke("Hi", language="en")

        self.assertEqual(state["session"]["id"], "isolated-thread")
        self.assertEqual(state["session"]["purpose"], "chaos")

    def test_terminal_user_session_does_not_resume_live_matrix_checkpoint(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime
        from agent.terminal.io import TerminalIO
        from agent.terminal.repl import AnyChainTerminal, TerminalSession

        class CapturingIO(TerminalIO):
            def __init__(self) -> None:
                self.messages: list[str] = []

            def input(self, language: str) -> str:
                raise EOFError()

            def agent(self, language: str, message: str) -> None:
                self.messages.append(message)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "matrix.sqlite"
            matrix_runtime = AnyChainGraphRuntime(
                thread_id="shared-thread",
                checkpoint_path=checkpoint,
                session_purpose="live-matrix",
            )
            matrix_runtime._persist_state(
                {
                    "target_mode": "fake-node",
                    "workflow_mode": "rpc_benchmark",
                    "chain_identity": {"canonical": "bsc", "status": "confirmed"},
                    "confirmed_config": {"CLOUD_REGION": "asia-east1"},
                    "pending_question": {"id": "CLOUD_ZONE", "prompt": "Confirm CLOUD_ZONE."},
                }
            )
            session = TerminalSession(language="zh")
            app = AnyChainTerminal(
                state=session,
                io=CapturingIO(),
                session_id="shared-thread",
                checkpoint_path=checkpoint,
                session_purpose="user",
            )
            app._offer_harness_resume_if_needed()

        self.assertEqual(session.current_question_id, "")

    def test_terminal_resume_modify_clears_pending_but_keeps_confirmed_config(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "checkpoint.sqlite"
            runtime = AnyChainGraphRuntime(thread_id="resume-modify", checkpoint_path=checkpoint)
            runtime._persist_state(
                {
                    "target_mode": "fake-node",
                    "workflow_mode": "rpc_benchmark",
                    "chain_identity": {"canonical": "bsc", "status": "confirmed"},
                    "confirmed_config": {"CLOUD_REGION": "asia-east1"},
                    "pending_question": {"id": "CLOUD_ZONE", "group": "provider_deployment", "kind": "manual_value", "prompt": "Confirm CLOUD_ZONE."},
                    "active_group": "provider_deployment",
                }
            )
            runtime.prepare_resume_offer(language="en")
            runtime.invoke("2", language="en")

            snapshot = runtime.snapshot()
            self.assertEqual(snapshot["confirmed_config"]["CLOUD_REGION"], "asia-east1")
            self.assertEqual(snapshot["pending_question"], {})
            self.assertEqual(snapshot["active_group"], "opening")

    def test_terminal_resume_continue_restores_exact_pending_contract(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "checkpoint.sqlite"
            runtime = AnyChainGraphRuntime(thread_id="resume-continue", checkpoint_path=checkpoint)
            original = {
                "id": "unknown_chain_identity_confirm",
                "group": "chain_identity",
                "kind": "numbered_choice",
                "field": "unknown_chain_decision",
                "prompt": "Confirm the researched chain identity.",
                "manual_input_allowed": False,
                "options": [
                    {"label": "Continue", "value": "confirm", "action": {"type": "confirm_proposed_protocol"}},
                    {"label": "Choose protocol", "value": "protocol", "action": {"type": "choose_protocol"}},
                ],
                "accepted_action_types": ["answer_pending", "choose_chain", "change_chain"],
            }
            runtime._persist_state(
                {
                    "target_mode": "fake-node",
                    "workflow_mode": "rpc_benchmark",
                    "chain_identity": {"raw": "Flow", "canonical": "flow", "status": "needs_identity_confirmation"},
                    "pending_question": original,
                    "active_group": "chain_identity",
                }
            )
            runtime.prepare_resume_offer(language="en")
            state = runtime.invoke("1", language="en")

        for key, value in original.items():
            self.assertEqual(state["pending_question"].get(key), value)
        self.assertEqual(state["active_group"], "chain_identity")
        self.assertIn("Confirm the researched chain identity", "\n".join(state["visible_response"]))
        self.assertEqual(state.get("resume_context"), {})

    def test_terminal_resume_continue_preserves_deferred_action_queue(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "checkpoint.sqlite"
            runtime = AnyChainGraphRuntime(thread_id="resume-queue", checkpoint_path=checkpoint)
            original = {
                "id": "custom_rpc_method",
                "group": "endpoint_process",
                "kind": "manual_value",
                "field": "custom_rpc_method",
                "prompt": "Enter the custom RPC method.",
                "manual_input_allowed": True,
                "queue_barrier": True,
                "resume_action_queue": True,
            }
            runtime._persist_state(
                {
                    "target_mode": "fake-node",
                    "workflow_mode": "rpc_benchmark",
                    "chain_identity": {"canonical": "bsc", "status": "confirmed"},
                    "pending_question": original,
                    "active_group": "endpoint_process",
                    "action_queue": [
                        {
                            "action_id": "obs-deferred",
                            "type": "set_observability",
                            "observability_mode": "disabled",
                            "confidence": "high",
                        }
                    ],
                }
            )
            runtime.prepare_resume_offer(language="en")
            state = runtime.invoke("1", language="en")

        for key, value in original.items():
            self.assertEqual(state["pending_question"].get(key), value)
        self.assertEqual(state["active_group"], "endpoint_process")
        self.assertEqual([item["action_id"] for item in state["action_queue"]], ["obs-deferred"])

    def test_terminal_resume_reset_discards_deferred_action_queue(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "checkpoint.sqlite"
            runtime = AnyChainGraphRuntime(thread_id="reset-queue", checkpoint_path=checkpoint)
            runtime._persist_state(
                {
                    "target_mode": "fake-node",
                    "workflow_mode": "rpc_benchmark",
                    "pending_question": {
                        "id": "custom_rpc_method",
                        "group": "endpoint_process",
                        "kind": "manual_value",
                        "prompt": "Enter the custom RPC method.",
                    },
                    "active_group": "endpoint_process",
                    "action_queue": [
                        {
                            "action_id": "obs-deferred",
                            "type": "set_observability",
                            "observability_mode": "disabled",
                            "confidence": "high",
                        }
                    ],
                }
            )
            runtime.prepare_resume_offer(language="en")
            state = runtime.invoke("3", language="en")

        self.assertEqual(state["pending_question"], {})
        self.assertEqual(state["action_queue"], [])
        self.assertEqual(state["target_mode"], "")

    def test_harness_resume_menu_allows_a_natural_language_consultation(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AnyChainGraphRuntime(thread_id="resume-natural-language", checkpoint_path=Path(tmpdir) / "checkpoint.sqlite")
            runtime._persist_state({"target_mode": "fake-node", "workflow_mode": "rpc_benchmark", "active_group": "target_mode"})
            runtime.prepare_resume_offer(language="zh")
            with patch(
                "agent.harness.coordinator.resolve_action_queue",
                return_value={"actions": [{"type": "answer_opening_question", "topic": "identity", "confidence": "high"}]},
            ):
                state = runtime.invoke("你是谁", language="zh")

        self.assertEqual(state["pending_question"]["id"], "resume_harness_session")
        self.assertEqual(state["target_mode"], "fake-node")
        self.assertIn("AnyChain Benchmark Agent", "\n".join(state["visible_response"]))

    def test_invalid_checkpoint_is_quarantined_instead_of_silently_resumed(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AnyChainGraphRuntime(thread_id="resume-quarantine", checkpoint_path=Path(tmpdir) / "checkpoint.sqlite")
            runtime._persist_state(
                {
                    "active_group": "provider_deployment",
                    "confirmed_config": {"CLOUD_REGION": "asia-east1"},
                    "pending_question": {"id": "CLOUD_ZONE", "prompt": "missing owner"},
                }
            )
            state = runtime.snapshot()

        self.assertEqual(state["checkpoint_recovery"]["status"], "quarantined")
        self.assertEqual(state["checkpoint_recovery"]["safe_confirmed_config"]["CLOUD_REGION"], "asia-east1")

    def test_terminal_has_no_resume_workflow_branch(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        text = (repo / "agent" / "terminal" / "repl.py").read_text(encoding="utf-8")

        self.assertNotIn('current_question_id == "resume_harness_session"', text)
        self.assertNotIn("_is_resume_session_explanation_request", text)

    def test_harness_invariant_error_is_not_reported_as_model_failure(self) -> None:
        from agent.harness.invariants import StateInvariantError
        from agent.terminal.repl import _adk_error_message

        message = _adk_error_message("en", StateInvariantError("invalid pending owner"))

        self.assertIn("workflow state validation failed", message)
        self.assertNotIn("underlying model", message)

    def test_harness_programming_error_is_not_reported_as_model_failure(self) -> None:
        from agent.terminal.repl import _adk_error_message

        message = _adk_error_message("en", NameError("missing runtime collaborator"))

        self.assertIn("workflow state validation failed", message)
        self.assertNotIn("underlying model", message)


if __name__ == "__main__":
    unittest.main()

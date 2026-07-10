"""Product terminal contract tests for the LangGraph Harness entrypoint."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class ProductTerminalHarnessContractTest(unittest.TestCase):
    def test_terminal_uses_langgraph_harness_not_retired_workflow_runtime(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        text = (repo / "agent" / "terminal" / "repl.py").read_text(encoding="utf-8")

        self.assertIn("AnyChainGraphRuntime", text)
        self.assertNotIn("ADKRunnerBridge", text)
        self.assertNotIn("workflows.conversation_state", text)
        self.assertNotIn("workflows.transition_executor", text)
        self.assertNotIn("terminal.pending_answers", text)
        self.assertNotIn("terminal.input_classifier", text)

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

    def test_terminal_entrypoint_routes_greeting_to_harness_opening_group(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AnyChainGraphRuntime(thread_id="terminal-contract", checkpoint_path=Path(tmpdir) / "checkpoint.sqlite")
            with patch("agent.harness.groups.resolve_intent_action", return_value={"intent": "greeting", "confidence": "high"}):
                state = runtime.invoke("Hi", language="en")

        self.assertEqual(state["active_group"], "opening")
        self.assertEqual(state["pending_question"]["id"], "opening_next_action")
        self.assertIn("AnyChain Benchmark Agent", state["visible_response"][0])

    def test_harness_snapshot_and_reset_support_terminal_resume_gate(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AnyChainGraphRuntime(thread_id="resume-contract", checkpoint_path=Path(tmpdir) / "checkpoint.sqlite")
            runtime.update(
                {
                    "target_mode": "fake-node",
                    "workflow_mode": "rpc_benchmark",
                    "chain_identity": {"canonical": "bsc", "status": "confirmed"},
                    "confirmed_config": {"CLOUD_REGION": "asia-east1"},
                    "pending_question": {"id": "CLOUD_ZONE", "prompt": "Confirm CLOUD_ZONE."},
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
        import sys

        repo = Path(__file__).resolve().parents[1]
        agent_root = str(repo / "agent")
        if agent_root not in sys.path:
            sys.path.insert(0, agent_root)
        sys.modules.pop("utils", None)
        sys.modules.pop("utils.redaction", None)

        from terminal.io import OutputOnlyIO
        from terminal.repl import AnyChainTerminal, TerminalSession

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
        import sys

        repo = Path(__file__).resolve().parents[1]
        agent_root = str(repo / "agent")
        if agent_root not in sys.path:
            sys.path.insert(0, agent_root)
        sys.modules.pop("utils", None)
        sys.modules.pop("utils.redaction", None)

        from agent.harness.graph import AnyChainGraphRuntime
        from terminal.io import TerminalIO
        from terminal.repl import AnyChainTerminal, TerminalSession

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
            matrix_runtime.update(
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
            runtime.update(
                {
                    "target_mode": "fake-node",
                    "workflow_mode": "rpc_benchmark",
                    "chain_identity": {"canonical": "bsc", "status": "confirmed"},
                    "confirmed_config": {"CLOUD_REGION": "asia-east1"},
                    "pending_question": {"id": "CLOUD_ZONE", "prompt": "Confirm CLOUD_ZONE."},
                    "active_group": "provider_deployment",
                }
            )
            runtime.update({"pending_question": {}, "active_group": "", "visible_response": []})

            snapshot = runtime.snapshot()
            self.assertEqual(snapshot["confirmed_config"]["CLOUD_REGION"], "asia-east1")
            self.assertEqual(snapshot["pending_question"], {})
            self.assertEqual(snapshot["active_group"], "")

    def test_terminal_resume_gate_allows_natural_language_goal(self) -> None:
        import sys

        repo = Path(__file__).resolve().parents[1]
        agent_root = str(repo / "agent")
        if agent_root not in sys.path:
            sys.path.insert(0, agent_root)
        sys.modules.pop("utils", None)
        sys.modules.pop("utils.redaction", None)

        from terminal.io import OutputOnlyIO
        from terminal.repl import AnyChainTerminal, TerminalSession

        session = TerminalSession(language="zh", current_question_id="resume_harness_session")
        app = AnyChainTerminal(state=session, io=OutputOnlyIO(), session_id="resume-natural-language")

        handled = app._handle_pending_confirmation("先别管之前配置，帮我分析最近一次 job 的报告和日志")

        self.assertFalse(handled)
        self.assertEqual(session.current_question_id, "")

    def test_terminal_resume_gate_clears_old_pending_before_natural_language(self) -> None:
        import sys

        repo = Path(__file__).resolve().parents[1]
        agent_root = str(repo / "agent")
        if agent_root not in sys.path:
            sys.path.insert(0, agent_root)
        sys.modules.pop("utils", None)
        sys.modules.pop("utils.redaction", None)

        from agent.harness.graph import AnyChainGraphRuntime
        from terminal.io import OutputOnlyIO
        from terminal.repl import AnyChainTerminal, TerminalSession

        with tempfile.TemporaryDirectory() as tmpdir:
            session = TerminalSession(language="zh", current_question_id="resume_harness_session")
            app = AnyChainTerminal(state=session, io=OutputOnlyIO(), session_id="resume-natural-language-clear")
            runtime = AnyChainGraphRuntime(thread_id="resume-natural-language-clear", checkpoint_path=Path(tmpdir) / "checkpoint.sqlite")
            runtime.update(
                {
                    "target_mode": "sync-observe",
                    "workflow_mode": "sync_observe",
                    "chain_identity": {"canonical": "bsc", "status": "confirmed"},
                    "confirmed_config": {"BLOCKCHAIN_NODE": "bsc"},
                    "pending_question": {"id": "CLOUD_REGION", "group": "provider_deployment", "kind": "manual_value", "prompt": "请输入 CLOUD_REGION"},
                    "evidence_collection": {
                        "question": {"id": "freeform_evidence", "kind": "log_evidence"},
                        "lines": ["Traceback (most recent call last):"],
                    },
                    "active_group": "provider_deployment",
                }
            )
            app._harness = runtime

            handled = app._handle_pending_confirmation("你是谁啊")
            snapshot = runtime.snapshot()

        self.assertFalse(handled)
        self.assertEqual(session.current_question_id, "")
        self.assertEqual(snapshot["pending_question"], {})
        self.assertEqual(snapshot["evidence_collection"], {})
        self.assertEqual(snapshot["active_group"], "")

    def test_terminal_resume_gate_explains_current_menu_without_losing_context(self) -> None:
        import sys

        repo = Path(__file__).resolve().parents[1]
        agent_root = str(repo / "agent")
        if agent_root not in sys.path:
            sys.path.insert(0, agent_root)
        sys.modules.pop("utils", None)
        sys.modules.pop("utils.redaction", None)

        from terminal.repl import AnyChainTerminal, TerminalSession

        class CapturingIO:
            def __init__(self) -> None:
                self.messages: list[str] = []

            def input(self, language: str) -> str:
                raise EOFError()

            def agent(self, language: str, message: str) -> None:
                self.messages.append(message)

        io = CapturingIO()
        session = TerminalSession(language="zh", current_question_id="resume_harness_session")
        app = AnyChainTerminal(state=session, io=io, session_id="resume-explain")

        handled = app._handle_pending_confirmation("这个是做什么的")

        self.assertTrue(handled)
        self.assertEqual(session.current_question_id, "resume_harness_session")
        self.assertIn("上次未完成", io.messages[-1])
        self.assertIn("清空之前的配置", io.messages[-1])


if __name__ == "__main__":
    unittest.main()

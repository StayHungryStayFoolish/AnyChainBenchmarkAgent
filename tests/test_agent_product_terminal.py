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


if __name__ == "__main__":
    unittest.main()

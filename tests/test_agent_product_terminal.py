import builtins
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
AGENT_BIN = REPO / "bin" / "anychain-agent"

import sys

sys.path.insert(0, str(REPO / "agent"))

from terminal.io import TerminalIO  # noqa: E402
from terminal.repl import AnyChainTerminal, TerminalSession, TerminalSessionStore  # noqa: E402
from adk_app.runner_bridge import sanitize_adk_text  # noqa: E402
from validators.config_contract import build_missing_config_questions, validate_required_config  # noqa: E402
from validators.execution_gate import validate_execution_gate  # noqa: E402
from validators.rpc_workload import default_workload, validate_rpc_workload  # noqa: E402
from workflows.conversation_state import load_workflow_state, revert_workflow_state, update_workflow_state  # noqa: E402


class CapturingIO:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.inputs: list[str | BaseException] = []

    def input(self, language: str) -> str:
        if not self.inputs:
            raise EOFError()
        item = self.inputs.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def agent(self, language: str, message: str) -> None:
        self.messages.append(message)


class FakeBridge:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def run_text(self, text: str, state_delta=None) -> str:
        self.calls.append((text, state_delta or {}))
        return f"bridge handled: {text}"


class AgentDependencyTests(unittest.TestCase):
    def test_adk_requirements_include_prompt_toolkit(self):
        requirements = (REPO / "requirements-adk.txt").read_text(encoding="utf-8")
        self.assertIn("prompt-toolkit", requirements)
        self.assertIn("litellm", requirements)

    def test_adk_text_sanitizer_removes_internal_implementation_leakage_only(self):
        text = "\n".join([
            "我会确认你的目标，然后给出下一步。",
            "不要向用户展示 prepare_benchmark_run。",
            "不要向用户展示 knowledge_search。",
            "不要向用户展示 chain_rpc_onboarding_agent。",
            "不要告诉用户转给子代理。",
            "我能帮助你完成区块链节点性能基准测试。",
            "```",
            "internal fenced debug",
            "```",
        ])
        self.assertEqual(
            sanitize_adk_text(text),
            "我会确认你的目标，然后给出下一步。\n我能帮助你完成区块链节点性能基准测试。",
        )

    def test_adk_text_sanitizer_does_not_patch_model_style(self):
        text = "Let me check the framework capabilities. 我先确认环境。Please provide LOCAL_RPC_URL."
        sanitized = sanitize_adk_text(text)
        self.assertIn("Let me check the framework capabilities.", sanitized)
        self.assertIn("我先确认环境。", sanitized)
        self.assertIn("Please provide LOCAL_RPC_URL.", sanitized)

    def test_adk_text_sanitizer_removes_inline_internal_leakage_only(self):
        text = "可以，当前支持 36 条链。不要向用户展示 prepare_benchmark_run。请提供 LOCAL_RPC_URL。"
        self.assertEqual(
            sanitize_adk_text(text),
            "可以，当前支持 36 条链。 请提供 LOCAL_RPC_URL。",
        )

    def test_terminal_io_requires_prompt_toolkit(self):
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "prompt_toolkit":
                raise ImportError("missing prompt_toolkit")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            with self.assertRaisesRegex(RuntimeError, "prompt-toolkit is required"):
                TerminalIO()

    def test_entrypoint_bootstraps_interactive_terminal_dependencies(self):
        entrypoint = AGENT_BIN.read_text(encoding="utf-8")
        self.assertIn("has_prompt_toolkit", entrypoint)
        self.assertIn("scripts/install_agent_deps.sh", entrypoint)
        self.assertIn("Install Agent dependencies into the isolated environment now?", entrypoint)
        self.assertIn('uses_scripted_prompt" == "0"', entrypoint)


class ADKNativeTerminalContractTest(unittest.TestCase):
    def test_product_terminal_no_longer_imports_old_wizard_or_responder(self):
        source = (REPO / "agent" / "terminal" / "repl.py").read_text(encoding="utf-8")
        forbidden = [
            "BenchmarkWizard",
            "terminal.responder",
            "answer_conversation",
            "planning_bridge",
            "WorkflowState",
        ]
        for marker in forbidden:
            self.assertNotIn(marker, source)

    def test_product_terminal_must_not_be_keyword_intent_router(self):
        source = (REPO / "agent" / "terminal" / "repl.py").read_text(encoding="utf-8")
        forbidden = [
            "def _looks_like_",
            "_looks_like_benchmark_request",
            "_looks_like_onboarding_request",
            "_looks_like_custom_rpc_request",
            "_looks_like_unknown_chain_request",
            "\"压测\"",
            "\"测一下\"",
            "\"我要测\"",
            "\"新链\"",
            "\"自定义 rpc\"",
            "\"二次开发\"",
        ]
        for marker in forbidden:
            self.assertNotIn(marker, source)

    def test_benchmark_turn_delegates_to_adk_bridge(self):
        fake_bridge = FakeBridge()
        io = CapturingIO()
        with tempfile.TemporaryDirectory() as tmp:
            store = TerminalSessionStore(Path(tmp) / "session.json")
            update_workflow_state(
                {"active_intent": "START_BENCHMARK", "chain": "solana"},
                session_id="terminal-session",
                state_root=Path(tmp) / "workflow",
            )
            app = AnyChainTerminal(
                state=TerminalSession(language="zh"),
                store=store,
                io=io,
                bridge_factory=lambda: fake_bridge,
            )
            with patch("terminal.repl.adk_status") as adk_status_mock, \
                patch("terminal.repl.runner_bridge_status") as bridge_status_mock, \
                patch("terminal.repl.web_research_status") as web_research_mock, \
                patch("terminal.repl.run_doctor") as doctor_mock, \
                patch("terminal.repl.load_framework_context") as context_mock, \
                patch("terminal.repl.load_framework_capabilities") as capabilities_mock:
                adk_status_mock.return_value.as_dict.return_value = {"available": True, "reason": "ok"}
                bridge_status_mock.return_value.as_dict.return_value = {"available": True, "reason": "ok"}
                web_research_mock.return_value.as_dict.return_value = {
                    "enabled": False,
                    "mode": "disabled",
                    "reason": "unavailable for current provider",
                }
                doctor_mock.return_value = {
                    "status": "ready",
                    "capabilities": {"chain_count": 36, "unique_rpc_method_count": 184},
                    "environment": {
                        "cloud": {"provider": "gcp", "platform": "gce"},
                        "deployment": {"type": "vm"},
                        "host": {"cpu_count": 8, "memory_gib": 32},
                        "network": {"default_interface": "eth0"},
                        "disks": {"candidates": [], "proposed_ledger_device": "sdb"},
                        "dependencies": {"missing_required": []},
                    },
                }
                context_mock.return_value = {"capability_summary": {"chain_count": 36, "family_count": 6, "unique_rpc_method_count": 184}}
                capabilities_mock.return_value = {"chain_count": 36, "family_count": 6, "unique_rpc_method_count": 184}
                with patch("terminal.repl.load_workflow_state") as workflow_state_mock:
                    workflow_state_mock.return_value = load_workflow_state(
                        session_id="terminal-session",
                        state_root=Path(tmp) / "workflow",
                    )
                    app.startup()
                    app.handle_user_text("我想测试 solana fake-node")

        self.assertEqual(len(fake_bridge.calls), 1)
        self.assertEqual(fake_bridge.calls[0][0], "我想测试 solana fake-node")
        self.assertEqual(fake_bridge.calls[0][1]["workflow_state"]["chain"], "solana")
        self.assertEqual(fake_bridge.calls[0][1]["workflow_state"]["active_intent"], "START_BENCHMARK")
        self.assertFalse(fake_bridge.calls[0][1]["web_research"]["enabled"])
        self.assertEqual(app.state.discovery["cloud"]["provider"], "gcp")
        self.assertEqual(app.state.framework_summary["chain_count"], 36)
        self.assertTrue(any("Web research" in item and "unavailable for current provider" in item for item in io.messages))
        self.assertFalse(any("benchmark 计划" in item for item in io.messages))

    def test_missing_dependencies_do_not_block_non_confirmation_adk_turns(self):
        fake_bridge = FakeBridge()
        io = CapturingIO()
        with tempfile.TemporaryDirectory() as tmp:
            store = TerminalSessionStore(Path(tmp) / "session.json")
            app = AnyChainTerminal(
                state=TerminalSession(language="zh"),
                store=store,
                io=io,
                bridge_factory=lambda: fake_bridge,
            )
            with patch("terminal.repl.adk_status") as adk_status_mock, \
                patch("terminal.repl.runner_bridge_status") as bridge_status_mock, \
                patch("terminal.repl.web_research_status") as web_research_mock, \
                patch("terminal.repl.run_doctor") as doctor_mock, \
                patch("terminal.repl.load_framework_context") as context_mock, \
                patch("terminal.repl.load_framework_capabilities") as capabilities_mock:
                adk_status_mock.return_value.as_dict.return_value = {"available": True, "reason": "ok"}
                bridge_status_mock.return_value.as_dict.return_value = {"available": True, "reason": "ok"}
                web_research_mock.return_value.as_dict.return_value = {
                    "enabled": False,
                    "mode": "disabled",
                    "reason": "unavailable for current provider",
                }
                doctor_mock.return_value = {
                    "status": "needs_dependencies",
                    "capabilities": {"chain_count": 36, "unique_rpc_method_count": 184},
                    "environment": {
                        "cloud": {"provider": "gcp", "platform": "gce"},
                        "deployment": {"type": "vm"},
                        "host": {"cpu_count": 8, "memory_gib": 32},
                        "network": {"default_interface": "eth0"},
                        "disks": {"candidates": [], "proposed_ledger_device": "sdb"},
                        "dependencies": {"missing_required": ["vegeta"]},
                    },
                }
                context_mock.return_value = {"capability_summary": {"chain_count": 36, "family_count": 6, "unique_rpc_method_count": 184}}
                capabilities_mock.return_value = {"chain_count": 36, "family_count": 6, "unique_rpc_method_count": 184}
                app.startup()
                app.handle_user_text("我要测试 solana，使用 fake-node smoke")

        self.assertEqual(len(fake_bridge.calls), 1)
        self.assertEqual(fake_bridge.calls[0][0], "我要测试 solana，使用 fake-node smoke")
        self.assertEqual(app.state.current_question_id, "install_dependencies")
        self.assertTrue(any("缺失依赖" in item and "scripts/install_deps.sh --yes" in item for item in io.messages))

    def test_ready_startup_clears_stale_pending_dependencies(self):
        fake_bridge = FakeBridge()
        io = CapturingIO()
        with tempfile.TemporaryDirectory() as tmp:
            store = TerminalSessionStore(Path(tmp) / "session.json")
            app = AnyChainTerminal(
                state=TerminalSession(
                    language="zh",
                    current_question_id="install_dependencies",
                    pending_missing_dependencies=["vegeta"],
                ),
                store=store,
                io=io,
                bridge_factory=lambda: fake_bridge,
            )
            with patch("terminal.repl.adk_status") as adk_status_mock, \
                patch("terminal.repl.runner_bridge_status") as bridge_status_mock, \
                patch("terminal.repl.web_research_status") as web_research_mock, \
                patch("terminal.repl.run_doctor") as doctor_mock, \
                patch("terminal.repl.load_framework_context") as context_mock, \
                patch("terminal.repl.load_framework_capabilities") as capabilities_mock:
                adk_status_mock.return_value.as_dict.return_value = {"available": True, "reason": "ok"}
                bridge_status_mock.return_value.as_dict.return_value = {"available": True, "reason": "ok"}
                web_research_mock.return_value.as_dict.return_value = {
                    "enabled": True,
                    "mode": "adk_google_search",
                    "reason": "enabled via ADK google_search",
                }
                doctor_mock.return_value = {
                    "status": "ready",
                    "capabilities": {"chain_count": 36, "unique_rpc_method_count": 184},
                    "environment": {
                        "cloud": {"provider": "other", "platform": "container"},
                        "deployment": {"type": "container"},
                        "host": {"cpu_count": 8, "memory_gib": 32},
                        "network": {"default_interface": "eth0"},
                        "disks": {"candidates": [], "proposed_ledger_device": "vda1"},
                        "dependencies": {"missing_required": []},
                    },
                }
                context_mock.return_value = {"capability_summary": {"chain_count": 36, "family_count": 6, "unique_rpc_method_count": 184}}
                capabilities_mock.return_value = {"chain_count": 36, "family_count": 6, "unique_rpc_method_count": 184}
                app.startup()
                app.handle_user_text("我要测试 solana fake-node smoke")

        self.assertEqual(app.state.pending_missing_dependencies, [])
        self.assertNotEqual(app.state.current_question_id, "install_dependencies")
        self.assertEqual(len(fake_bridge.calls), 1)
        self.assertEqual(fake_bridge.calls[0][0], "我要测试 solana fake-node smoke")
        self.assertFalse(any("benchmark 计划" in item for item in io.messages))

    def test_capability_question_is_answered_from_repo_without_adk(self):
        fake_bridge = FakeBridge()
        io = CapturingIO()
        app = AnyChainTerminal(
            state=TerminalSession(language="zh"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=io,
            bridge_factory=lambda: fake_bridge,
        )
        app._adk_available = True
        app.handle_user_text("当前支持多少个链和 RPC method？如果增加自定义 RPC method 怎么做？")

        self.assertEqual(len(fake_bridge.calls), 1)
        self.assertEqual(fake_bridge.calls[0][0], "当前支持多少个链和 RPC method？如果增加自定义 RPC method 怎么做？")

    def test_onboarding_question_is_answered_from_local_handoff_without_adk(self):
        fake_bridge = FakeBridge()
        io = CapturingIO()
        app = AnyChainTerminal(
            state=TerminalSession(language="zh"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=io,
            bridge_factory=lambda: fake_bridge,
        )
        app._adk_available = True
        app.handle_user_text("我想添加一个不在 36 个链里的 FooChain，它是 EVM JSON-RPC 兼容链")

        self.assertEqual(len(fake_bridge.calls), 1)
        self.assertEqual(fake_bridge.calls[0][0], "我想添加一个不在 36 个链里的 FooChain，它是 EVM JSON-RPC 兼容链")
        self.assertFalse(any("接入 handoff" in item for item in io.messages))

    def test_negative_custom_rpc_benchmark_request_does_not_route_to_onboarding(self):
        fake_bridge = FakeBridge()
        io = CapturingIO()
        app = AnyChainTerminal(
            state=TerminalSession(language="zh"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=io,
            bridge_factory=lambda: fake_bridge,
        )
        app._adk_available = True
        app.handle_user_text("确认，不需要自定义 RPC，请运行 fake-node smoke")

        self.assertEqual(len(fake_bridge.calls), 1)
        self.assertEqual(fake_bridge.calls[0][0], "确认，不需要自定义 RPC，请运行 fake-node smoke")
        self.assertFalse(any("benchmark 计划" in item for item in io.messages))
        self.assertFalse(any("接入 handoff" in item for item in io.messages))

    def test_dependency_question_does_not_generate_benchmark_plan(self):
        fake_bridge = FakeBridge()
        io = CapturingIO()
        app = AnyChainTerminal(
            state=TerminalSession(language="zh"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=io,
            bridge_factory=lambda: fake_bridge,
        )
        app._adk_available = True
        app.handle_user_text("如果缺少 vegeta，你会让我自己安装，还是你帮我安装？")

        self.assertEqual(len(fake_bridge.calls), 1)
        self.assertEqual(fake_bridge.calls[0][0], "如果缺少 vegeta，你会让我自己安装，还是你帮我安装？")
        self.assertFalse(any("benchmark 计划" in item for item in io.messages))

    def test_natural_language_environment_request_delegates_to_adk(self):
        fake_bridge = FakeBridge()
        io = CapturingIO()
        app = AnyChainTerminal(
            state=TerminalSession(language="zh"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=io,
            bridge_factory=lambda: fake_bridge,
        )
        app._adk_available = True
        app.handle_user_text("请帮我先做环境检查，然后配置 solana fake-node")

        self.assertEqual(len(fake_bridge.calls), 1)
        self.assertEqual(fake_bridge.calls[0][0], "请帮我先做环境检查，然后配置 solana fake-node")

    def test_exact_chinese_doctor_command_stays_terminal_command(self):
        fake_bridge = FakeBridge()
        io = CapturingIO()
        app = AnyChainTerminal(
            state=TerminalSession(language="zh"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=io,
            bridge_factory=lambda: fake_bridge,
        )
        app._adk_available = True
        with patch("terminal.repl.run_doctor") as doctor_mock:
            doctor_mock.return_value = {
                "status": "ready",
                "capabilities": {"chain_count": 36, "unique_rpc_method_count": 184},
                "environment": {
                    "cloud": {"provider": "gcp", "platform": "gce"},
                    "deployment": {"type": "vm"},
                    "host": {"cpu_count": 8, "memory_gib": 32},
                    "network": {"default_interface": "eth0"},
                    "disks": {"candidates": [], "proposed_ledger_device": "sdb"},
                    "dependencies": {"missing_required": []},
                },
            }
            app.handle_user_text("环境检查")

        self.assertEqual(fake_bridge.calls, [])
        self.assertTrue(any("status=ready" in item for item in io.messages))

    def test_prometheus_grafana_request_enters_benchmark_plan(self):
        fake_bridge = FakeBridge()
        io = CapturingIO()
        app = AnyChainTerminal(
            state=TerminalSession(language="zh"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=io,
            bridge_factory=lambda: fake_bridge,
        )
        app._adk_available = True
        app.handle_user_text("我要用 solana fake-node smoke，并开启本地 Prometheus/Grafana，Grafana 端口 3001")

        self.assertEqual(len(fake_bridge.calls), 1)
        self.assertEqual(fake_bridge.calls[0][0], "我要用 solana fake-node smoke，并开启本地 Prometheus/Grafana，Grafana 端口 3001")
        self.assertFalse(any("observability=local" in item for item in io.messages))

    def test_fake_node_plan_with_missing_environment_cannot_enter_smoke_on_yes(self):
        fake_bridge = FakeBridge()
        io = CapturingIO()
        app = AnyChainTerminal(
            state=TerminalSession(language="zh"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=io,
            bridge_factory=lambda: fake_bridge,
        )
        app._adk_available = True
        app.handle_user_text("我要用 solana fake-node smoke，1 QPS，持续 3 秒")

        self.assertEqual(len(fake_bridge.calls), 1)
        self.assertEqual(app.state.current_question_id, "")
        self.assertFalse(any("当前计划还不能执行" in item for item in io.messages))
        self.assertFalse(any("是否现在执行隔离的 fake-node smoke" in item for item in io.messages))

    def test_existing_prometheus_request_uses_exporter_mode(self):
        fake_bridge = FakeBridge()
        io = CapturingIO()
        app = AnyChainTerminal(
            state=TerminalSession(language="en"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=io,
            bridge_factory=lambda: fake_bridge,
        )
        app._adk_available = True
        app.handle_user_text("Benchmark solana with fake-node smoke and use existing Prometheus via exporter port 9200")

        self.assertEqual(len(fake_bridge.calls), 1)
        self.assertEqual(fake_bridge.calls[0][0], "Benchmark solana with fake-node smoke and use existing Prometheus via exporter port 9200")
        self.assertFalse(any("observability=exporter" in item for item in io.messages))

    def test_logs_command_uses_terminal_control_not_adk_business_routing(self):
        fake_bridge = FakeBridge()
        io = CapturingIO()
        app = AnyChainTerminal(
            state=TerminalSession(language="en", latest_job_id="job_1"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=io,
            bridge_factory=lambda: fake_bridge,
        )
        app._adk_available = True
        with patch("terminal.repl.tail_job_log") as tail_mock:
            tail_mock.return_value = {
                "job_id": "job_1",
                "log_file": "/tmp/job_1/benchmark.log",
                "exists": True,
                "lines": ["line one", "line two"],
            }
            app.handle_user_text("logs")

        self.assertEqual(fake_bridge.calls, [])
        self.assertTrue(any("/tmp/job_1/benchmark.log" in item for item in io.messages))
        self.assertTrue(any("line one\nline two" in item for item in io.messages))

    def test_missing_adk_does_not_fallback_to_custom_brain(self):
        io = CapturingIO()
        app = AnyChainTerminal(state=TerminalSession(language="en"), store=TerminalSessionStore(Path("/tmp/unused-session.json")), io=io)
        with patch("terminal.repl.adk_status") as adk_status_mock, \
            patch("terminal.repl.runner_bridge_status") as bridge_status_mock, \
            patch("terminal.repl.run_doctor") as doctor_mock, \
            patch("terminal.repl.load_framework_context") as context_mock, \
            patch("terminal.repl.load_framework_capabilities") as capabilities_mock:
            adk_status_mock.return_value.as_dict.return_value = {"available": False, "reason": "missing"}
            adk_status_mock.return_value.available = False
            bridge_status_mock.return_value.as_dict.return_value = {"available": False, "reason": "missing"}
            doctor_mock.return_value = {
                "status": "ready_without_llm",
                "capabilities": {"chain_count": 36, "unique_rpc_method_count": 184},
                "environment": {"cloud": {}, "deployment": {}, "host": {}, "network": {}, "disks": {}, "dependencies": {"missing_required": []}},
            }
            context_mock.return_value = {"capability_summary": {"chain_count": 36, "family_count": 6, "unique_rpc_method_count": 184}}
            capabilities_mock.return_value = {"chain_count": 36, "family_count": 6, "unique_rpc_method_count": 184}
            app.startup()
            app.handle_user_text("benchmark solana")
        self.assertTrue(any("Agent runtime dependency is missing" in item for item in io.messages))
        self.assertFalse(any("Selected fake-node" in item for item in io.messages))


class DeterministicValidatorContractTest(unittest.TestCase):
    def test_workflow_state_persists_structured_updates_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            result = update_workflow_state(
                {
                    "active_intent": "START_BENCHMARK",
                    "active_workflow": "benchmark",
                    "workflow_step": "chain_selected",
                    "chain": "solana",
                    "confirmed_config": {"NETWORK_INTERFACE": "eth0"},
                    "unsupported_key": "ignored",
                },
                reason="unit test",
                session_id="test-session",
                state_root=root,
            )
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertEqual(loaded["active_intent"], "START_BENCHMARK")
        self.assertEqual(loaded["active_workflow"], "benchmark")
        self.assertEqual(loaded["workflow_step"], "chain_selected")
        self.assertEqual(loaded["chain"], "solana")
        self.assertEqual(loaded["confirmed_config"]["NETWORK_INTERFACE"], "eth0")
        self.assertIn("unsupported_key", result["ignored_keys"])

    def test_config_validator_distinguishes_fake_node_and_real_node(self):
        fake = validate_required_config("fake-node", {"chain": "solana", "rpc_mode": "single"})
        real = validate_required_config("real-node", {"chain": "solana", "rpc_mode": "single"})
        self.assertIn("local_rpc_url", real["missing"])
        self.assertNotIn("local_rpc_url", fake["missing"])

    def test_config_questions_include_disk_inventory(self):
        questions = build_missing_config_questions(
            "real-node",
            {"chain": "solana", "rpc_mode": "single"},
            {"disks": {"candidates": [{"name": "sdb", "type": "disk", "size": "2T", "mountpoint": "/ledger"}]}},
        )
        ledger = next(item for item in questions["questions"] if item["id"] == "ledger_device")
        self.assertEqual(ledger["candidates"][0]["name"], "sdb")
        self.assertTrue(ledger["manual_input_allowed"])
        self.assertIn("benchmark_mode_confirmed", questions["missing"])
        self.assertIn("qps_profile_confirmed", questions["missing"])
        self.assertIn("observability_choice_confirmed", questions["missing"])
        qps = next(item for item in questions["questions"] if item["id"] == "qps_profile_confirmed")
        self.assertEqual(qps["interaction_mode"], "accept_defaults_or_adjust_item")
        self.assertIn("initial_qps", qps["parameter_descriptions"])
        self.assertIn("duration_seconds", {item["id"] for item in qps["adjustable_items"]})

    def test_workflow_state_can_revert_previous_user_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            update_workflow_state(
                {"confirmed_config": {"LEDGER_DEVICE": "sdb"}},
                reason="first disk",
                session_id="test-session",
                state_root=root,
            )
            update_workflow_state(
                {"confirmed_config": {"LEDGER_DEVICE": "sdc"}},
                reason="wrong disk",
                session_id="test-session",
                state_root=root,
            )
            reverted = revert_workflow_state(
                steps=1,
                reason="user went back",
                session_id="test-session",
                state_root=root,
            )

        self.assertTrue(reverted["reverted"])
        self.assertEqual(reverted["state"]["confirmed_config"]["LEDGER_DEVICE"], "sdb")

    def test_rpc_workload_validator_accepts_custom_methods_but_requires_review(self):
        custom = validate_rpc_workload("solana", "mixed", mixed_weights={"getSlot": 70, "customMethod": 30})
        self.assertTrue(custom["ready"])
        self.assertIn("customMethod", custom["custom_methods"])
        self.assertTrue(custom["requires_fixture_review"])

    def test_rpc_workload_validator_blocks_bad_weight_total(self):
        result = validate_rpc_workload("solana", "mixed", mixed_weights={"getSlot": 80, "getBlockHeight": 10})
        self.assertFalse(result["ready"])
        self.assertIn("mixed_weights total must be 100, got 90", result["errors"])

    def test_default_workload_is_grounded_in_chain_template(self):
        workload = default_workload("solana")
        self.assertTrue(workload["exists"])
        self.assertTrue(workload["single"])
        self.assertTrue(workload["mixed_weighted"])

    def test_execution_gate_blocks_without_approval_for_real_run(self):
        result = validate_execution_gate(
            plan={"id": "plan-1"},
            preflight={"passed": True},
            smoke={"status": "completed"},
            approved=False,
            real_execution=True,
        )
        self.assertFalse(result["ready"])
        self.assertIn("explicit user approval is required", result["blockers"])


if __name__ == "__main__":
    unittest.main()

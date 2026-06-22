import subprocess
import sys
import tempfile
import unittest
import os
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
AGENT_BIN = REPO / "bin" / "anychain-agent"
sys.path.insert(0, str(REPO / "agent"))

from terminal.repl import AnyChainTerminal  # noqa: E402
from terminal.io import TerminalIO  # noqa: E402
from analyzers.execution_artifacts import diagnose_execution_artifacts  # noqa: E402
from workflows.benchmark_wizard import BenchmarkWizard  # noqa: E402
from workflows.planning_bridge import prepare_plan_from_state, submit_mock_smoke_from_plan  # noqa: E402
from workflows.state import WorkflowState, WorkflowStateStore  # noqa: E402


class CapturingIO(TerminalIO):
    def __init__(self):
        self.messages = []

    def input(self, language: str) -> str:  # pragma: no cover - not used by these tests
        raise EOFError()

    def agent(self, language: str, message: str) -> None:
        self.messages.append(message)


class AgentProductTerminalTest(unittest.TestCase):
    def run_agent(self, *args: str) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            completed = subprocess.run(
                [str(AGENT_BIN), "--state-file", str(Path(tmp) / "state.json"), *args],
                cwd=REPO,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
        self.assertEqual(completed.stderr, "")
        return completed.stdout

    def test_chinese_scripted_turns_do_not_use_adk_run_prompt(self):
        output = self.run_agent(
            "--language",
            "zh",
            "--prompt",
            "我现在需要测试 solana",
            "--prompt",
            "fake-node",
            "--prompt",
            "你帮我执行，不要让我 export PATH",
        )
        self.assertNotIn("[user]", output)
        self.assertIn("我理解你要测试 solana", output)
        self.assertIn("不需要真实 LOCAL_RPC_URL", output)
        self.assertIn("不应该要求你手动 export PATH", output)

    def test_english_single_turn_can_capture_chain_and_fake_node(self):
        output = self.run_agent("--prompt", "I want to benchmark Solana with fake-node")
        self.assertNotIn("[user]", output)
        self.assertIn("Selected fake-node", output)
        self.assertIn("LOCAL_RPC_URL", output)

    def test_workflow_state_tracks_confirmed_values_and_gate(self):
        state = WorkflowState(language="zh")
        wizard = BenchmarkWizard(state)

        response = wizard.handle("我现在需要测试 solana")
        self.assertTrue(response.handled)
        self.assertEqual(state.confirmed_values["chain"], "solana")
        self.assertEqual(state.current_question_id, "target_type")

        response = wizard.handle("fake-node")
        self.assertTrue(response.handled)
        self.assertTrue(state.confirmed_values["use_fake_node"])
        self.assertEqual(state.current_question_id, "rpc_mode")

        ok, missing = wizard.can_run_smoke()
        self.assertFalse(ok)
        self.assertIn("rpc_mode", missing)

        response = wizard.handle("single")
        self.assertTrue(response.handled)
        self.assertEqual(state.confirmed_values["rpc_mode"], "single")
        ok, missing = wizard.can_run_smoke()
        self.assertFalse(ok)
        self.assertIn("rpc_workload_confirmed", missing)

        response = wizard.handle("确认")
        self.assertTrue(response.handled)
        self.assertTrue(state.confirmed_values["rpc_workload_confirmed"])
        ok, missing = wizard.can_run_smoke()
        self.assertFalse(ok)
        self.assertIn("rpc_param_samples_confirmed", missing)

        response = wizard.handle("确认")
        self.assertTrue(response.handled)
        self.assertTrue(state.confirmed_values["rpc_param_samples_confirmed"])
        ok, missing = wizard.can_run_smoke()
        self.assertTrue(ok)
        self.assertEqual(missing, [])

    def test_confirmed_fake_node_workflow_generates_plan_and_mock_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = WorkflowState(language="zh", stage="ready_for_smoke", current_question_id="smoke_confirmation")
            state.confirmed_values.update({
                "chain": "solana",
                "use_fake_node": True,
                "rpc_mode": "single",
                "rpc_workload_confirmed": True,
                "rpc_param_samples_confirmed": True,
            })
            prepared = prepare_plan_from_state(state, output_dir=Path(tmp) / "prepared")
            self.assertTrue(Path(prepared["plan_file"]).is_file())
            self.assertTrue(prepared["preflight"]["passed"])
            self.assertEqual(prepared["plan"]["chain"], "solana")
            self.assertTrue(prepared["plan"]["use_fake_node"])
            self.assertIn("--fake-node", prepared["plan"]["execution"]["command"])

            job = submit_mock_smoke_from_plan(prepared["plan_file"], jobs_dir=Path(tmp) / "jobs")
            self.assertEqual(job["status"], "completed")
            self.assertTrue(Path(job["runtime_env_file"]).is_file())
            self.assertTrue(Path(job["artifact_index"]).is_file())
            runtime_env = Path(job["runtime_env_file"]).read_text(encoding="utf-8")
            self.assertIn("export BLOCKCHAIN_NODE='solana'", runtime_env)
            self.assertIn("export RPC_MODE='single'", runtime_env)

    def test_scripted_terminal_can_prepare_and_run_mock_smoke(self):
        output = self.run_agent(
            "--language",
            "zh",
            "--prompt",
            "我现在需要测试 solana",
            "--prompt",
            "fake-node",
            "--prompt",
            "single",
            "--prompt",
            "确认",
            "--prompt",
            "确认",
            "--prompt",
            "确认",
            "--prompt",
            "确认",
        )
        self.assertIn("preflight 通过", output)
        self.assertIn("mock smoke 完成", output)
        self.assertIn("runtime_env=", output)
        self.assertIn("artifact_index=", output)

    def test_real_node_flow_collects_required_values_one_at_a_time(self):
        state = WorkflowState(language="en")
        wizard = BenchmarkWizard(state)

        self.assertTrue(wizard.handle("benchmark solana").handled)
        response = wizard.handle("real-node")
        self.assertTrue(response.handled)
        self.assertEqual(state.current_question_id, "local_rpc_url")

        answers = [
            ("http://127.0.0.1:8899", "blockchain_process_names"),
            ("agave-validator", "ledger_device"),
            ("sdb", "data_vol_type"),
            ("pd-ssd", "data_vol_size"),
            ("2048", "data_vol_max_iops"),
            ("12000", "data_vol_max_throughput"),
            ("500", "network_interface"),
            ("eth0", "network_max_bandwidth_gbps"),
            ("16", "rpc_mode"),
        ]
        for answer, next_question in answers:
            response = wizard.handle(answer)
            self.assertTrue(response.handled)
            self.assertEqual(state.current_question_id, next_question)
        self.assertEqual(state.confirmed_values["blockchain_process_names"], ["agave-validator"])
        self.assertEqual(state.confirmed_values["ledger_device"], "sdb")
        self.assertEqual(state.confirmed_values["network_max_bandwidth_gbps"], "16")

    def test_missing_blockers_accept_list_values(self):
        state = WorkflowState(language="en")
        state.confirmed_values.update({
            "chain": "solana",
            "use_fake_node": False,
            "rpc_mode": "single",
            "rpc_workload_confirmed": True,
            "rpc_param_samples_confirmed": True,
            "local_rpc_url": "http://127.0.0.1:8899",
            "blockchain_process_names": ["agave-validator"],
            "ledger_device": "sdb",
            "data_vol_type": "pd-ssd",
            "data_vol_size": "2048",
            "data_vol_max_iops": "12000",
            "data_vol_max_throughput": "500",
            "network_interface": "eth0",
            "network_max_bandwidth_gbps": "16",
        })
        ok, missing = BenchmarkWizard(state).can_run_smoke()
        self.assertTrue(ok)
        self.assertEqual(missing, [])

    def test_state_store_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = WorkflowStateStore(Path(tmp) / "state.json")
            state = WorkflowState(language="zh", stage="select_rpc_mode")
            state.confirmed_values["chain"] = "solana"
            store.save(state)
            loaded = store.load()
            self.assertEqual(loaded.language, "zh")
            self.assertEqual(loaded.stage, "select_rpc_mode")
            self.assertEqual(loaded.confirmed_values["chain"], "solana")

    def test_dependency_confirmation_decline_is_stateful(self):
        io = CapturingIO()
        state = WorkflowState(language="zh", stage="dependency_install_confirmation", current_question_id="install_dependencies")
        app = AnyChainTerminal(state=state, io=io)
        app.handle_user_text("n")
        self.assertEqual(state.stage, "dependency_install_declined")
        self.assertEqual(state.current_question_id, "")
        self.assertTrue(any("已跳过依赖安装" in message for message in io.messages))

    def test_general_question_does_not_fill_pending_configuration_value(self):
        old_env = os.environ.copy()
        try:
            os.environ["LLM_PROVIDER"] = "unsupported_provider_for_test"
            io = CapturingIO()
            state = WorkflowState(language="zh", stage="collect_data_vol_max_throughput", current_question_id="data_vol_max_throughput")
            state.confirmed_values.update({
                "chain": "solana",
                "use_fake_node": False,
                "rpc_mode": "single",
            })
            app = AnyChainTerminal(state=state, io=io)
            app.handle_user_text("你是 AI 么？")
            self.assertNotIn("data_vol_max_throughput", state.confirmed_values)
            self.assertEqual(state.current_question_id, "data_vol_max_throughput")
            self.assertTrue(any("AnyChain Benchmark Agent" in message for message in io.messages))
        finally:
            os.environ.clear()
            os.environ.update(old_env)

    def test_reset_clears_pending_workflow_but_keeps_language(self):
        io = CapturingIO()
        state = WorkflowState(language="zh", stage="collect_network_interface", current_question_id="network_interface", job_id="job_1")
        state.confirmed_values["chain"] = "solana"
        app = AnyChainTerminal(state=state, io=io)
        app.handle_user_text("重新开始")
        self.assertEqual(state.stage, "start")
        self.assertEqual(state.current_question_id, "")
        self.assertEqual(state.confirmed_values, {})
        self.assertEqual(state.job_id, "job_1")
        self.assertTrue(any("已重置" in message for message in io.messages))

    def test_execution_artifact_diagnosis_does_not_treat_missing_performance_as_no_traffic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proxy = root / "proxy_method.csv"
            proxy.write_text("timestamp,method,status\n1,getAccountInfo,200\n", encoding="utf-8")
            index = root / "artifact_index.json"
            index.write_text(
                '{"run_dir": "%s", "evidence": {"proxy_method_csv": "%s", "performance_csv": "%s"}}'
                % (root, proxy, root / "performance_latest.csv"),
                encoding="utf-8",
            )
            diagnosis = diagnose_execution_artifacts(artifact_index=index)
            self.assertEqual(diagnosis["conclusion"], "traffic_ok_monitor_sample_missing")
            self.assertEqual(diagnosis["signals"]["proxy_method_rows"], 1)
            self.assertEqual(diagnosis["signals"]["performance_rows"], -1)


if __name__ == "__main__":
    unittest.main()

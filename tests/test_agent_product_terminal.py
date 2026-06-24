import subprocess
import sys
import tempfile
import unittest
import os
import builtins
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
AGENT_BIN = REPO / "bin" / "anychain-agent"
sys.path.insert(0, str(REPO / "agent"))

from terminal.repl import AnyChainTerminal  # noqa: E402
from terminal.io import TerminalIO  # noqa: E402
from terminal.responder import answer_conversation  # noqa: E402
from analyzers.execution_artifacts import diagnose_execution_artifacts  # noqa: E402
from planners.preflight import run_preflight  # noqa: E402
from workflows.benchmark_wizard import BenchmarkWizard, known_chains  # noqa: E402
from workflows.planning_bridge import prepare_plan_from_state, submit_mock_smoke_from_plan  # noqa: E402
from workflows.state import WorkflowState, WorkflowStateStore  # noqa: E402


class CapturingIO(TerminalIO):
    def __init__(self):
        self.messages = []
        self.inputs = []

    def input(self, language: str) -> str:  # pragma: no cover - not used by these tests
        if self.inputs:
            item = self.inputs.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item
        raise EOFError()

    def agent(self, language: str, message: str) -> None:
        self.messages.append(message)


class FakeADKStatus:
    def __init__(self, available: bool):
        self.available = available

    def as_dict(self):
        return {"available": self.available, "reason": "available" if self.available else "missing"}


class AgentDependencyTests(unittest.TestCase):
    def test_adk_requirements_include_prompt_toolkit(self):
        requirements = (REPO / "requirements-adk.txt").read_text(encoding="utf-8")
        self.assertIn("prompt-toolkit", requirements)

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


def _complete_environment(wizard: BenchmarkWizard, state: WorkflowState) -> None:
    answers = [
        ("us-central1", "cloud_zone"),
        ("us-central1-a", "machine_type"),
        ("c3-standard-22", "blockchain_process_names"),
        ("agave-validator", "ledger_device"),
        ("sdb", "data_vol_type"),
        ("pd-ssd", "data_vol_size"),
        ("2048", "data_vol_max_iops"),
        ("12000", "data_vol_max_throughput"),
        ("500", "network_interface"),
        ("eth0", "network_max_bandwidth_gbps"),
        ("16", "has_accounts_device"),
        ("n", "advanced_config_review"),
        ("n", "rpc_mode"),
    ]
    for answer, expected_next in answers:
        response = wizard.handle(answer)
        assert response.handled
        assert state.current_question_id == expected_next


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
        wizard = BenchmarkWizard(state, discovery={"cloud": {"provider": "gcp", "platform": "gce"}, "deployment": {"type": "vm"}, "network": {}, "disks": {"candidates": []}})

        response = wizard.handle("我现在需要测试 solana")
        self.assertTrue(response.handled)
        self.assertEqual(state.confirmed_values["chain"], "solana")
        self.assertEqual(state.current_question_id, "target_type")

        response = wizard.handle("fake-node")
        self.assertTrue(response.handled)
        self.assertTrue(state.confirmed_values["use_fake_node"])
        self.assertEqual(state.current_question_id, "cloud_region")

        _complete_environment(wizard, state)
        self.assertEqual(state.current_question_id, "rpc_mode")

        ok, missing = wizard.can_run_smoke()
        self.assertFalse(ok)
        self.assertIn("rpc_mode", missing)

        response = wizard.handle("single")
        self.assertTrue(response.handled)
        self.assertEqual(state.confirmed_values["rpc_mode"], "single")
        self.assertEqual(state.current_question_id, "single_method_confirm")
        ok, missing = wizard.can_run_smoke()
        self.assertFalse(ok)
        self.assertIn("rpc_workload_confirmed", missing)

        response = wizard.handle("n")
        self.assertTrue(response.handled)
        self.assertEqual(state.current_question_id, "rpc_workload")

        response = wizard.handle("getSlot")
        self.assertTrue(response.handled)
        self.assertEqual(state.confirmed_values["single_method"], "getSlot")
        self.assertEqual(state.confirmed_values["rpc_methods"], ["getSlot"])
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

    def test_benchmark_intent_without_chain_asks_for_chain_first(self):
        state = WorkflowState(language="zh")
        wizard = BenchmarkWizard(state, discovery={"cloud": {"provider": "gcp", "platform": "gce"}, "deployment": {"type": "vm"}, "network": {}, "disks": {"candidates": []}})

        response = wizard.handle("我要压测一个区块链节点")
        self.assertTrue(response.handled)
        self.assertEqual(state.intent, "benchmark")
        self.assertEqual(state.stage, "select_chain")
        self.assertEqual(state.current_question_id, "chain_choice")
        self.assertIn("solana", "\n".join(response.messages))

        response = wizard.handle("solana")
        self.assertTrue(response.handled)
        self.assertEqual(state.confirmed_values["chain"], "solana")
        self.assertEqual(state.current_question_id, "target_type")

    def test_target_before_chain_is_remembered_until_chain_is_confirmed(self):
        state = WorkflowState(language="zh")
        wizard = BenchmarkWizard(state, discovery={"cloud": {"provider": "gcp", "platform": "gce"}, "deployment": {"type": "vm"}, "network": {}, "disks": {"candidates": []}})

        response = wizard.handle("fake-node")
        self.assertTrue(response.handled)
        self.assertEqual(state.current_question_id, "chain_choice")
        self.assertEqual(state.defaulted_values["pending_target"], "fake-node")
        self.assertNotIn("use_fake_node", state.confirmed_values)

        response = wizard.handle("solana")
        self.assertTrue(response.handled)
        self.assertEqual(state.confirmed_values["chain"], "solana")
        self.assertTrue(state.confirmed_values["use_fake_node"])
        self.assertEqual(state.current_question_id, "cloud_region")

    def test_numbered_target_and_rpc_mode_choices_are_supported(self):
        state = WorkflowState(language="zh")
        wizard = BenchmarkWizard(state, discovery={"cloud": {"provider": "gcp", "platform": "gce"}, "deployment": {"type": "vm"}, "network": {}, "disks": {"candidates": []}})

        self.assertTrue(wizard.handle("测试 solana").handled)
        response = wizard.handle("1")
        self.assertTrue(response.handled)
        self.assertTrue(state.confirmed_values["use_fake_node"])
        self.assertEqual(state.current_question_id, "cloud_region")

        _complete_environment(wizard, state)
        response = wizard.handle("2")
        self.assertTrue(response.handled)
        self.assertEqual(state.confirmed_values["rpc_mode"], "mixed")
        self.assertEqual(state.current_question_id, "mixed_weights_confirm")

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

    def test_user_supplied_target_and_chain_endpoint_values_reach_runtime_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = WorkflowState(language="zh", stage="ready_for_smoke", current_question_id="smoke_confirmation")
            state.confirmed_values.update({
                "chain": "aptos",
                "use_fake_node": True,
                "rpc_mode": "single",
                "rpc_workload_confirmed": True,
                "rpc_param_samples_confirmed": True,
                "cloud_provider": "gcp",
                "cloud_region": "us-central1",
                "cloud_zone": "us-central1-a",
                "machine_type": "c3-standard-22",
                "blockchain_process_names": ["aptos-node"],
                "ledger_device": "sdb",
                "data_vol_type": "pd-ssd",
                "data_vol_size": "2048",
                "data_vol_max_iops": "12000",
                "data_vol_max_throughput": "500",
                "network_interface": "eth0",
                "network_max_bandwidth_gbps": "16",
                "target_address": "0xabc",
                "target_tx_hash": "0xtx",
                "chain_rest_url": "http://127.0.0.1:8080",
            })
            prepared = prepare_plan_from_state(state, output_dir=Path(tmp) / "prepared")
            job = submit_mock_smoke_from_plan(prepared["plan_file"], jobs_dir=Path(tmp) / "jobs")
            runtime_env = Path(job["runtime_env_file"]).read_text(encoding="utf-8")
            self.assertIn("export TARGET_ADDRESS='0xabc'", runtime_env)
            self.assertIn("export TARGET_TX_HASH='0xtx'", runtime_env)
            self.assertIn("export CHAIN_REST_URL='http://127.0.0.1:8080'", runtime_env)

    def test_scripted_terminal_can_prepare_and_run_mock_smoke(self):
        output = self.run_agent(
            "--language",
            "zh",
            "--prompt",
            "我现在需要测试 solana",
            "--prompt",
            "fake-node",
            "--prompt",
            "us-central1",
            "--prompt",
            "us-central1-a",
            "--prompt",
            "c3-standard-22",
            "--prompt",
            "agave-validator",
            "--prompt",
            "sdb",
            "--prompt",
            "pd-ssd",
            "--prompt",
            "2048",
            "--prompt",
            "12000",
            "--prompt",
            "500",
            "--prompt",
            "eth0",
            "--prompt",
            "16",
            "--prompt",
            "n",
            "--prompt",
            "n",
            "--prompt",
            "single",
            "--prompt",
            "n",
            "--prompt",
            "getSlot",
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

    def test_single_mode_can_accept_chain_template_default_method(self):
        state = WorkflowState(language="zh")
        wizard = BenchmarkWizard(state, discovery={"cloud": {"provider": "gcp", "platform": "gce"}, "deployment": {"type": "vm"}, "network": {}, "disks": {"candidates": []}})
        self.assertTrue(wizard.handle("测试 solana").handled)
        self.assertTrue(wizard.handle("fake-node").handled)
        _complete_environment(wizard, state)
        response = wizard.handle("single")
        self.assertTrue(response.handled)
        self.assertEqual(state.current_question_id, "single_method_confirm")
        self.assertIn("getAccountInfo", "\n".join(response.messages))

        response = wizard.handle("y")
        self.assertTrue(response.handled)
        self.assertEqual(state.confirmed_values["single_method"], "getAccountInfo")
        self.assertEqual(state.confirmed_values["rpc_methods"], ["getAccountInfo"])
        self.assertTrue(state.confirmed_values["rpc_workload_confirmed"])
        self.assertEqual(state.current_question_id, "rpc_param_samples")

    def test_mixed_weights_are_parsed_and_must_total_100(self):
        state = WorkflowState(language="zh")
        wizard = BenchmarkWizard(state, discovery={"cloud": {"provider": "gcp", "platform": "gce"}, "deployment": {"type": "vm"}, "network": {}, "disks": {"candidates": []}})
        self.assertTrue(wizard.handle("测试 solana").handled)
        self.assertTrue(wizard.handle("fake-node").handled)
        _complete_environment(wizard, state)
        self.assertTrue(wizard.handle("mixed").handled)

        response = wizard.handle("getSlot=60,getBlockHeight=20")
        self.assertTrue(response.handled)
        self.assertFalse(state.confirmed_values.get("mixed_weights_confirmed"))
        self.assertEqual(state.current_question_id, "rpc_workload")

        response = wizard.handle("getSlot=70,getBlockHeight=30")
        self.assertTrue(response.handled)
        self.assertEqual(state.confirmed_values["mixed_weights"], {"getSlot": 70, "getBlockHeight": 30})
        self.assertTrue(state.confirmed_values["mixed_weights_confirmed"])
        self.assertTrue(state.confirmed_values["rpc_workload_confirmed"])
        self.assertEqual(state.current_question_id, "rpc_param_samples")

    def test_real_node_flow_collects_required_values_one_at_a_time(self):
        state = WorkflowState(language="en")
        discovery = {
            "cloud": {"provider": "gcp", "platform": "gce"},
            "deployment": {"type": "vm"},
            "network": {},
            "disks": {"candidates": []},
        }
        wizard = BenchmarkWizard(state, discovery=discovery)

        self.assertTrue(wizard.handle("benchmark solana").handled)
        response = wizard.handle("real-node")
        self.assertTrue(response.handled)
        self.assertEqual(state.current_question_id, "local_rpc_url")

        answers = [
            ("http://127.0.0.1:8899", "mainnet_rpc_url_reviewed"),
            ("default", "cloud_region"),
            ("us-central1", "cloud_zone"),
            ("us-central1-a", "machine_type"),
            ("c3-standard-22", "blockchain_process_names"),
            ("agave-validator", "ledger_device"),
            ("sdb", "data_vol_type"),
            ("pd-ssd", "data_vol_size"),
            ("2048", "data_vol_max_iops"),
            ("12000", "data_vol_max_throughput"),
            ("500", "network_interface"),
            ("eth0", "network_max_bandwidth_gbps"),
            ("16", "has_accounts_device"),
            ("n", "advanced_config_review"),
            ("n", "rpc_mode"),
        ]
        for answer, next_question in answers:
            response = wizard.handle(answer)
            self.assertTrue(response.handled)
            self.assertEqual(state.current_question_id, next_question)
        self.assertEqual(state.confirmed_values["blockchain_process_names"], ["agave-validator"])
        self.assertEqual(state.confirmed_values["ledger_device"], "sdb")
        self.assertEqual(state.confirmed_values["network_max_bandwidth_gbps"], "16")
        self.assertFalse(state.confirmed_values["has_accounts_device"])

    def test_detected_cloud_values_are_confirmed_before_use(self):
        state = WorkflowState(language="zh")
        discovery = {
            "cloud": {
                "provider": "gcp",
                "platform": "gce",
                "region": "us-central1",
                "zone": "us-central1-a",
                "machine_type": "c3-standard-22",
            },
            "deployment": {"type": "vm"},
            "network": {},
            "disks": {"candidates": []},
        }
        wizard = BenchmarkWizard(state, discovery=discovery)

        self.assertTrue(wizard.handle("测试 solana").handled)
        response = wizard.handle("fake-node")
        self.assertTrue(response.handled)
        self.assertEqual(state.current_question_id, "cloud_region_confirm")
        self.assertIn("CLOUD_REGION=us-central1", "\n".join(response.messages))

        response = wizard.handle("y")
        self.assertTrue(response.handled)
        self.assertEqual(state.confirmed_values["cloud_region"], "us-central1")
        self.assertEqual(state.current_question_id, "cloud_zone_confirm")

        response = wizard.handle("asia-east1-a")
        self.assertTrue(response.handled)
        self.assertEqual(state.confirmed_values["cloud_zone"], "asia-east1-a")
        self.assertEqual(state.current_question_id, "machine_type_confirm")

        response = wizard.handle("y")
        self.assertTrue(response.handled)
        self.assertEqual(state.confirmed_values["machine_type"], "c3-standard-22")
        self.assertEqual(state.current_question_id, "blockchain_process_names")

    def test_real_node_disk_candidates_are_confirmed_by_number(self):
        state = WorkflowState(language="zh")
        discovery = {
            "cloud": {"provider": "gcp", "platform": "gce"},
            "deployment": {"type": "vm"},
            "network": {"default_interface": "eth0"},
            "disks": {
                "candidates": [
                    {"name": "sda", "type": "disk", "size": "100G", "mountpoint": "/", "fstype": "ext4", "label": ""},
                    {"name": "sdb", "type": "disk", "size": "2T", "mountpoint": "/ledger", "fstype": "xfs", "label": "ledger"},
                    {"name": "sdc", "type": "disk", "size": "1T", "mountpoint": "/accounts", "fstype": "xfs", "label": "accounts"},
                ]
            },
        }
        wizard = BenchmarkWizard(state, discovery=discovery)
        self.assertTrue(wizard.handle("测试 solana").handled)
        response = wizard.handle("真实节点")
        self.assertTrue(response.handled)
        self.assertEqual(state.current_question_id, "local_rpc_url")
        for answer in ["http://127.0.0.1:8899", "default", "us-central1", "us-central1-a", "c3-standard-22", "agave-validator"]:
            self.assertTrue(wizard.handle(answer).handled)
        self.assertEqual(state.current_question_id, "ledger_device_choice")
        response = wizard.handle("2")
        self.assertTrue(response.handled)
        self.assertEqual(state.confirmed_values["ledger_device"], "sdb")
        for answer in ["pd-ssd", "2048", "12000", "500", "eth0", "16"]:
            self.assertTrue(wizard.handle(answer).handled)
        self.assertEqual(state.current_question_id, "has_accounts_device")
        self.assertTrue(wizard.handle("y").handled)
        self.assertEqual(state.current_question_id, "accounts_device_choice")
        self.assertTrue(wizard.handle("3").handled)
        self.assertEqual(state.confirmed_values["accounts_device"], "sdc")
        self.assertEqual(state.current_question_id, "accounts_vol_type")
        for answer in ["pd-ssd", "1024", "10000", "300"]:
            self.assertTrue(wizard.handle(answer).handled)
        self.assertEqual(state.current_question_id, "advanced_config_review")
        self.assertTrue(wizard.handle("n").handled)
        self.assertEqual(state.current_question_id, "rpc_mode")

    def test_fake_node_to_real_node_keeps_workload_context_but_collects_real_fields(self):
        state = WorkflowState(language="zh")
        discovery = {
            "cloud": {"provider": "gcp", "platform": "gce"},
            "deployment": {"type": "vm"},
            "network": {},
            "disks": {"candidates": []},
        }
        wizard = BenchmarkWizard(state, discovery=discovery)
        for prompt in ["测试 solana", "fake-node"]:
            self.assertTrue(wizard.handle(prompt).handled)
        _complete_environment(wizard, state)
        for prompt in ["single", "确认", "确认"]:
            self.assertTrue(wizard.handle(prompt).handled)
        ok, missing = wizard.can_run_smoke()
        self.assertTrue(ok)

        response = wizard.handle("切换到真实节点")
        self.assertTrue(response.handled)
        self.assertEqual(state.current_question_id, "local_rpc_url")
        self.assertEqual(state.confirmed_values["chain"], "solana")
        self.assertEqual(state.confirmed_values["rpc_mode"], "single")
        self.assertTrue(state.confirmed_values["rpc_workload_confirmed"])
        self.assertTrue(state.confirmed_values["rpc_param_samples_confirmed"])
        self.assertFalse(state.confirmed_values["use_fake_node"])
        ok, missing = wizard.can_run_smoke()
        self.assertFalse(ok)
        self.assertIn("local_rpc_url", missing)
        self.assertNotIn("rpc_mode", missing)

    def test_invalid_real_node_values_are_rejected_before_advancing(self):
        state = WorkflowState(language="en")
        discovery = {
            "cloud": {"provider": "gcp", "platform": "gce"},
            "deployment": {"type": "vm"},
            "network": {},
            "disks": {"candidates": []},
        }
        wizard = BenchmarkWizard(state, discovery=discovery)
        self.assertTrue(wizard.handle("benchmark solana").handled)
        self.assertTrue(wizard.handle("real-node").handled)
        response = wizard.handle("not-a-url")
        self.assertTrue(response.handled)
        self.assertEqual(state.current_question_id, "local_rpc_url")
        self.assertNotIn("local_rpc_url", state.confirmed_values)
        self.assertTrue(wizard.handle("http://127.0.0.1:8899").handled)
        self.assertEqual(state.current_question_id, "mainnet_rpc_url_reviewed")

    def test_chain_detection_uses_all_chain_templates(self):
        chains = known_chains()
        self.assertIn("avalanche-c", chains)
        state = WorkflowState(language="en")
        wizard = BenchmarkWizard(state, discovery={"cloud": {}, "deployment": {}, "network": {}, "disks": {"candidates": []}})
        response = wizard.handle("benchmark avalanche-c")
        self.assertTrue(response.handled)
        self.assertEqual(state.confirmed_values["chain"], "avalanche-c")

    def test_missing_blockers_accept_list_values(self):
        state = WorkflowState(language="en")
        state.confirmed_values.update({
            "chain": "solana",
            "use_fake_node": False,
            "rpc_mode": "single",
            "rpc_workload_confirmed": True,
            "rpc_param_samples_confirmed": True,
            "local_rpc_url": "http://127.0.0.1:8899",
            "mainnet_rpc_url_reviewed": True,
            "cloud_provider": "gcp",
            "cloud_region": "us-central1",
            "cloud_zone": "us-central1-a",
            "machine_type": "c3-standard-22",
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

    def test_startup_runs_read_only_doctor_and_sets_dependency_confirmation(self):
        io = CapturingIO()
        state = WorkflowState(language="zh")
        app = AnyChainTerminal(state=state, io=io)
        report = {
            "status": "needs_dependencies",
            "environment": {
                "cloud": {"provider": "gcp"},
                "deployment": {"type": "vm"},
                "dependencies": {"missing_required": ["jq"], "missing_optional": []},
            },
            "capabilities": {"chain_count": 36, "unique_rpc_method_count": 184},
        }
        with patch("terminal.repl.adk_status", return_value=FakeADKStatus(True)), patch("terminal.repl.run_doctor", return_value=report):
            app._startup()
        self.assertEqual(state.current_question_id, "install_dependencies")
        self.assertIn("jq", state.missing_blockers)
        self.assertTrue(state.defaulted_values["framework_context_loaded"])
        self.assertEqual(state.defaulted_values["framework_context_summary"]["chain_count"], 36)
        self.assertTrue(any("启动检查完成" in message for message in io.messages))
        self.assertTrue(any("已加载框架事实" in message for message in io.messages))
        self.assertTrue(any("是否允许" in message for message in io.messages))

    def test_startup_prioritizes_agent_runtime_install_when_adk_is_missing(self):
        io = CapturingIO()
        state = WorkflowState(language="zh")
        app = AnyChainTerminal(state=state, io=io)
        report = {
            "status": "needs_dependencies",
            "environment": {
                "cloud": {"provider": "gcp"},
                "deployment": {"type": "vm"},
                "dependencies": {"missing_required": ["jq"], "missing_optional": []},
            },
            "capabilities": {"chain_count": 36, "unique_rpc_method_count": 184},
        }
        with patch("terminal.repl.adk_status", return_value=FakeADKStatus(False)), patch("terminal.repl.run_doctor", return_value=report):
            app._startup()
        self.assertEqual(state.current_question_id, "install_agent_runtime")
        self.assertEqual(state.missing_blockers, ["google-adk"])
        self.assertTrue(any("install_agent_deps.sh" in message for message in io.messages))

    def test_agent_runtime_install_adds_gcloud_for_google_adc_when_missing(self):
        io = CapturingIO()
        state = WorkflowState(language="zh", current_question_id="install_agent_runtime")
        app = AnyChainTerminal(state=state, io=io)
        fake_config = SimpleNamespace(provider="gemini", auth_mode="google_adc")
        completed = SimpleNamespace(returncode=0, stdout="")
        with (
            patch("terminal.repl.load_llm_config", return_value=fake_config),
            patch("terminal.repl.shutil.which", return_value=""),
            patch("terminal.repl.subprocess.run", return_value=completed) as run_mock,
            patch.object(app, "_startup_doctor"),
        ):
            app._install_agent_runtime()
        command = run_mock.call_args.args[0]
        self.assertEqual(command, ["bash", "scripts/install_agent_deps.sh", "--yes", "--with-gcloud"])

    def test_ctrl_c_exits_instead_of_being_swallowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            io = CapturingIO()
            io.inputs = [KeyboardInterrupt()]
            store = WorkflowStateStore(Path(tmp) / "state.json")
            app = AnyChainTerminal(state=WorkflowState(language="zh"), store=store, io=io)
            report = {
                "status": "ready",
                "environment": {
                    "cloud": {"provider": "gcp"},
                    "deployment": {"type": "vm"},
                    "dependencies": {"missing_required": [], "missing_optional": []},
                },
                "capabilities": {"chain_count": 36, "unique_rpc_method_count": 184},
            }
            with patch("terminal.repl.adk_status", return_value=FakeADKStatus(True)), patch("terminal.repl.run_doctor", return_value=report):
                exit_code = app.run()
            self.assertEqual(exit_code, 130)
            self.assertTrue((Path(tmp) / "state.json").is_file())
            self.assertTrue(any("Ctrl+C" in message for message in io.messages))

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

    def test_conversation_responder_prefers_adk_runner_when_config_is_valid(self):
        fake_config = SimpleNamespace(
            provider="gemini",
            model="gemini-3.1-pro",
            auth_mode="api_key",
            validate=lambda: [],
        )
        state = WorkflowState(language="zh", stage="collect_network_interface", current_question_id="network_interface")
        with patch("terminal.responder.run_text_once", return_value="ADK answer") as run_mock:
            answer = answer_conversation("你支持多少链？", state, fake_config)
        self.assertEqual(answer, "ADK answer")
        self.assertTrue(run_mock.called)

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

    def test_preflight_reports_entrypoint_dependency_warnings_in_audit_mode(self):
        plan = {
            "chain": "solana",
            "rpc_mode": "single",
            "use_fake_node": True,
            "dependency_mode": "audit",
            "required_inputs": [],
            "configuration_checklist": {"missing_blockers": []},
            "chain_template_requirements": {"mixed_weighted": []},
            "execution": {"environment": {"LOCAL_RPC_URL": ""}},
            "materialized_config": {
                "LEDGER_DEVICE": "sdb",
                "NETWORK_INTERFACE": "eth0",
            },
            "discovery": {
                "dependencies": {
                    "tools": {
                        "bash": {"available": True},
                        "python3": {"available": True},
                        "jq": {"available": True},
                        "curl": {"available": True},
                        "vegeta": {"available": False},
                        "go": {"available": False},
                        "iostat": {"available": True},
                        "ip": {"available": True},
                        "lsblk": {"available": True},
                    }
                },
                "disks": {"candidates": [{"name": "sdb"}]},
                "network": {"default_interface": "eth0"},
            },
        }
        preflight = run_preflight(plan)
        self.assertTrue(preflight["passed"])
        self.assertTrue(any("vegeta" in item for item in preflight["warnings"]))
        self.assertTrue(any("go" in item for item in preflight["warnings"]))

    def test_preflight_blocks_invalid_mixed_weight_total(self):
        plan = {
            "chain": "solana",
            "rpc_mode": "mixed",
            "use_fake_node": True,
            "dependency_mode": "audit",
            "required_inputs": [],
            "configuration_checklist": {"missing_blockers": []},
            "chain_template_requirements": {
                "mixed_weighted": [
                    {"method": "getSlot", "weight": 60},
                    {"method": "getBlockHeight", "weight": 20},
                ]
            },
            "execution": {"environment": {"LOCAL_RPC_URL": ""}},
            "materialized_config": {
                "LEDGER_DEVICE": "sdb",
                "NETWORK_INTERFACE": "eth0",
            },
            "discovery": {
                "dependencies": {"tools": {}},
                "disks": {"candidates": [{"name": "sdb"}]},
                "network": {"default_interface": "eth0"},
            },
        }
        preflight = run_preflight(plan)
        self.assertFalse(preflight["passed"])
        self.assertTrue(any("mixed_weighted_total_valid" in item for item in preflight["blockers"]))


if __name__ == "__main__":
    unittest.main()

"""Tests for the unexposed LangGraph Harness skeleton."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


LANGGRAPH_AVAILABLE = importlib.util.find_spec("langgraph") is not None


@unittest.skipUnless(LANGGRAPH_AVAILABLE, "langgraph is not installed in this Python environment")
class LangGraphHarnessSkeletonTest(unittest.TestCase):
    def test_opening_turn_returns_typed_pending_question(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoints.sqlite"
            runtime = AnyChainGraphRuntime(thread_id="unit-thread", checkpoint_path=checkpoint_path)
            with patch("agent.harness.groups.resolve_intent_action") as resolver:
                resolver.return_value = {"intent": "greeting", "confidence": "high"}
                state = runtime.invoke("Hi", language="en")

        self.assertEqual(state["active_group"], "opening")
        self.assertEqual(state["pending_question"]["id"], "opening_next_action")
        self.assertEqual(state["pending_question"]["kind"], "numbered_choice")
        self.assertFalse(state["pending_question"]["manual_input_allowed"])
        self.assertIn("What would you like", state["visible_response"][0])

    def test_group_order_prioritizes_chain_after_target_mode(self) -> None:
        from agent.harness.state import DEFAULT_GROUP_ORDER

        self.assertLess(DEFAULT_GROUP_ORDER.index("chain_identity"), DEFAULT_GROUP_ORDER.index("provider_deployment"))

    def test_target_mode_choice_asks_chain_before_provider_values(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoints.sqlite"
            runtime = AnyChainGraphRuntime(thread_id="unit-thread", checkpoint_path=checkpoint_path)
            with patch("agent.harness.groups.resolve_intent_action") as resolver:
                resolver.return_value = {"intent": "greeting", "confidence": "high"}
                runtime.invoke("Hi", language="en")
            state = runtime.invoke("1", language="en")

        self.assertEqual(state["target_mode"], "fake-node")
        self.assertEqual(state["active_group"], "chain_identity")
        self.assertEqual(state["pending_question"]["id"], "chain")

    def test_chain_turn_can_set_target_mode_in_same_intent(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["active_group"] = "opening"
        state["last_user_input"] = "我想观察 BNB 节点同步，不想打 RPC 压测流量"

        with patch("agent.harness.groups.resolve_intent_action") as resolver:
            resolver.return_value = {
                "intent": "choose_chain",
                "target_mode": "sync-observe",
                "chain_text": "BNB",
                "confidence": "high",
            }
            result = process_turn(state)

        self.assertEqual(result["target_mode"], "sync-observe")
        self.assertEqual(result["workflow_mode"], "sync_observe")
        self.assertEqual(result["chain_identity"]["canonical"], "bsc")
        self.assertEqual(result["confirmed_config"]["BLOCKCHAIN_NODE"], "bsc")
        self.assertEqual(result["active_group"], "provider_deployment")

    def test_target_mode_turn_uses_chain_mention_extractor_when_primary_intent_drops_chain(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["active_group"] = "opening"
        state["last_user_input"] = "我想观察 BNB 节点同步，不想打 RPC 压测流量"

        with (
            patch(
                "agent.harness.groups.resolve_intent_action",
                return_value={"intent": "choose_target_mode", "target_mode": "sync-observe", "confidence": "high"},
            ),
            patch(
                "agent.harness.groups.extract_chain_mention",
                return_value={"found": True, "chain_text": "BNB", "confidence": "high"},
            ),
        ):
            result = process_turn(state)

        self.assertEqual(result["target_mode"], "sync-observe")
        self.assertEqual(result["workflow_mode"], "sync_observe")
        self.assertEqual(result["chain_identity"]["canonical"], "bsc")
        self.assertEqual(result["pending_question"]["id"], "CLOUD_REGION")

    def test_multi_action_turn_sets_nonblocking_groups_then_falls_back(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["active_group"] = "opening"
        state["last_user_input"] = "我要用 fake-node 测试 BNB，用 mixed，quick QPS，并开启本地 Grafana"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "choose_target_mode", "target_mode": "fake-node", "target_mode_explicit": True, "confidence": "high"},
                    {"type": "choose_chain", "chain_text": "BNB", "confidence": "high"},
                    {"type": "set_rpc_mode", "rpc_mode": "mixed", "confidence": "high"},
                    {"type": "set_qps_mode", "qps_mode": "quick", "confidence": "high"},
                    {"type": "set_observability", "observability_mode": "local", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["target_mode"], "fake-node")
        self.assertEqual(result["chain_identity"]["canonical"], "bsc")
        self.assertEqual(result["rpc_mode"], "mixed")
        self.assertEqual(result["qps_profile"]["mode"], "quick")
        self.assertEqual(result["observability"]["mode"], "local")
        self.assertEqual(result["pending_question"]["id"], "qps_profile_confirm")
        self.assertEqual(len(result["completed_actions"]), 5)

    def test_capability_action_does_not_swallow_benchmark_goal_in_queue(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["active_group"] = "opening"
        state["last_user_input"] = "我想先随便跑通一下框架，但不确定 fake-node 还是 real-node；可能测 BNB，也想看看支持哪些链"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "ask_capabilities", "confidence": "high"},
                    {"type": "choose_chain", "chain_text": "BNB", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        visible = "\n".join(result.get("visible_response") or [])
        self.assertIn("当前框架", visible)
        self.assertNotIn("{'chain':", visible)
        self.assertIn("已确认链为 `bsc`", visible)
        self.assertEqual(result["chain_identity"]["canonical"], "bsc")
        self.assertEqual(result.get("target_mode"), "")
        self.assertEqual(result["pending_question"]["id"], "opening_next_action")
        self.assertEqual(len(result["completed_actions"]), 2)

    def test_analyze_report_action_returns_visible_job_entrypoint(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["active_group"] = "opening"
        state["latest_job_id"] = "job_demo"
        state["last_user_input"] = "看最近任务报告"

        with (
            patch("agent.harness.groups.resolve_action_queue") as resolver,
            patch("agent.harness.groups.resume_job") as resume,
        ):
            resolver.return_value = {"actions": [{"type": "analyze_report", "confidence": "high"}]}
            resume.return_value = {
                "job_id": "job_demo",
                "status": "completed",
                "run_dir": ".agent/jobs/job_demo",
                "runtime_env_file": ".agent/jobs/job_demo/runtime.env",
                "artifact_index": ".agent/jobs/job_demo/artifact_index.json",
                "next_actions": ["status", "analyze", "artifact-qa"],
            }
            result = process_turn(state)

        visible = "\n".join(result.get("visible_response") or [])
        self.assertIn("job_demo", visible)
        self.assertIn("artifact_index", visible)
        self.assertEqual(result.get("pending_question"), {})

    def test_endpoint_pending_treats_error_line_as_evidence_not_url_answer(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "sync-observe"
        state["workflow_mode"] = "sync_observe"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["active_group"] = "endpoint_process"
        state["pending_question"] = {
            "group": "endpoint_process",
            "id": "SYNC_OBSERVE_RPC_URL",
            "field": "SYNC_OBSERVE_RPC_URL",
            "kind": "url",
            "manual_input_allowed": True,
            "prompt": "Provide endpoint.",
        }
        state["last_user_input"] = "RuntimeError: prometheus exporter port 9108 already in use"

        with patch("agent.harness.groups.validate_rpc_endpoint") as probe:
            result = process_turn(state)

        probe.assert_not_called()
        self.assertIn("evidence_buffer", result)
        self.assertEqual(result["pending_question"], {})
        self.assertIn("证据", "\n".join(result.get("visible_response") or []))

    def test_final_endpoint_pending_requires_bare_endpoint_not_complex_intent(self) -> None:
        from agent.harness.groups import _answer_fits_pending

        pending = {
            "id": "LOCAL_RPC_URL",
            "group": "endpoint_process",
            "kind": "url",
            "field": "LOCAL_RPC_URL",
            "manual_input_allowed": True,
        }

        self.assertTrue(_answer_fits_pending("https://example.invalid/rpc", pending))
        self.assertFalse(_answer_fits_pending(
            "我要改测 Flow，endpoint 用 https://example.invalid/rpc，这只是验证 method，不是最终压测 endpoint",
            pending,
        ))

    def test_multiline_config_proposal_requires_confirmation_before_apply(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "solana", "canonical": "solana", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "solana"}
        state["active_group"] = "provider_deployment"
        state["last_user_input"] = """这是从环境里复制出来的：
cloud:
  region: asia-east1
  zone: asia-east1-c
machine: n2-standard-16
disk:
  ledger: vda
  size_gib: 926
"""

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {
                        "type": "propose_config_values",
                        "source_format": "yaml",
                        "config_values": {
                            "CLOUD_REGION": "asia-east1",
                            "CLOUD_ZONE": "asia-east1-c",
                            "MACHINE_TYPE": "n2-standard-16",
                            "LEDGER_DEVICE": "vda",
                            "DATA_VOL_SIZE": "926",
                        },
                        "unmapped_values": {"disk.ledger": "vda"},
                        "confidence": "high",
                    }
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "inferred_config_review")
        self.assertNotIn("CLOUD_REGION", result["confirmed_config"])
        self.assertNotIn("LEDGER_DEVICE", result["confirmed_config"])
        self.assertIn("CLOUD_REGION", result["visible_response"][0])
        self.assertIn("disk.ledger", result["visible_response"][0])

        result["last_user_input"] = "Y"
        result = process_turn(result)

        self.assertEqual(result["confirmed_config"]["CLOUD_REGION"], "asia-east1")
        self.assertEqual(result["confirmed_config"]["CLOUD_ZONE"], "asia-east1-c")
        self.assertEqual(result["confirmed_config"]["MACHINE_TYPE"], "n2-standard-16")
        self.assertEqual(result["confirmed_config"]["LEDGER_DEVICE"], "vda")
        self.assertEqual(result["confirmed_config"]["DATA_VOL_SIZE"], "926")
        self.assertEqual(result["pending_question"]["id"], "DATA_VOL_TYPE")
        self.assertIn("accepted_reviews", result["inferred_config"])

    def test_multiline_config_paste_is_preserved_when_action_queue_only_extracts_chain(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["active_group"] = "opening"
        state["last_user_input"] = """我要用 fake-node 测试 BNB，用 mixed，QPS quick，并开启本地 Grafana。下面是从环境里复制的配置：
cloud:
  region: asia-east1
  zone: asia-east1-c
  machine_type: n2-standard-16
disk:
  ledger_device: vda
  data_vol_type: hyperdisk-balanced
  data_vol_size: 926
  data_vol_max_iops: 20000
  data_vol_max_throughput: 1000
network:
  interface: eth0
  bandwidth_gbps: 100"""

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "choose_target_mode", "target_mode": "fake-node", "target_mode_explicit": True, "confidence": "high"},
                    {"type": "choose_chain", "chain_text": "BNB", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["target_mode"], "fake-node")
        self.assertEqual(result["chain_identity"]["canonical"], "bsc")
        self.assertEqual(result["pending_question"]["id"], "inferred_config_review")
        self.assertNotIn("CLOUD_REGION", result["confirmed_config"])
        self.assertIn("CLOUD_REGION", result["visible_response"][0])
        self.assertIn("DATA_VOL_MAX_IOPS", result["visible_response"][0])

    def test_qps_override_and_accounts_presence_can_be_set_from_one_freeform_turn(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {
            "BLOCKCHAIN_NODE": "bsc",
            "CLOUD_REGION": "asia-east1",
            "CLOUD_ZONE": "asia-east1-c",
            "MACHINE_TYPE": "n2-standard-16",
            "LEDGER_DEVICE": "vda",
            "DATA_VOL_TYPE": "hyperdisk-balanced",
            "DATA_VOL_MAX_IOPS": "20000",
            "DATA_VOL_MAX_THROUGHPUT": "1000",
        }
        state["qps_profile"] = {"mode": "quick", "confirmed": False, "default_decision_made": True}
        state["active_group"] = "qps_profile"
        state["pending_question"] = {
            "id": "qps_adjust_field",
            "group": "qps_profile",
            "kind": "numbered_choice",
            "field": "qps_adjust_field",
            "options": [
                {"label": "INITIAL_QPS / 起始 QPS", "value": "INITIAL_QPS"},
                {"label": "完成调整", "value": "done"},
            ],
            "manual_input_allowed": False,
        }
        state["last_user_input"] = "把 initial qps 设成 5，然后我没有 accounts 盘，回去继续磁盘"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "set_qps_override", "qps_overrides": {"INITIAL_QPS": 5}, "confidence": "high"},
                    {"type": "set_accounts_presence", "has_accounts_device": False, "confidence": "high"},
                    {"type": "change_group", "group": "ledger_disk", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["qps_profile"]["overrides"]["INITIAL_QPS"], "5")
        self.assertTrue(result["qps_profile"]["confirmed"])
        self.assertFalse(result["confirmed_config"]["has_accounts_device"])
        self.assertEqual(result["pending_question"]["id"], "DATA_VOL_SIZE")

    def test_jump_to_completed_group_continues_to_next_fallback_question(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {
            "BLOCKCHAIN_NODE": "bsc",
            "CLOUD_REGION": "asia-east1",
            "CLOUD_ZONE": "asia-east1-c",
            "MACHINE_TYPE": "n2-standard-16",
            "LEDGER_DEVICE": "vda",
            "DATA_VOL_TYPE": "hyperdisk-balanced",
            "has_accounts_device": False,
        }
        state["active_group"] = "qps_profile"
        state["last_user_input"] = "回到 accounts 配置"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "change_group", "group": "accounts_disk", "confidence": "high"}]}
            result = process_turn(state)

        self.assertIn("没有阻塞项", "\n".join(result.get("visible_response") or []))
        self.assertEqual(result["pending_question"]["id"], "DATA_VOL_SIZE")

    def test_config_proposal_rejection_does_not_mutate_confirmed_config(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["active_group"] = "provider_deployment"
        state["last_user_input"] = "CLOUD_REGION=us-1\nCLOUD_ZONE=us-1-z\nunknown_flag=true"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {
                        "type": "propose_config_values",
                        "source_format": "env",
                        "config_values": {"CLOUD_REGION": "us-1", "CLOUD_ZONE": "us-1-z"},
                        "unmapped_values": {"unknown_flag": True},
                        "confidence": "high",
                    }
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "inferred_config_review")
        result["last_user_input"] = "N"
        result = process_turn(result)

        self.assertEqual(result["confirmed_config"], {})
        self.assertNotIn("pending_review", result.get("inferred_config", {}))
        self.assertIn("Discarded", result["visible_response"][0])

    def test_inferred_config_review_is_blocking_until_user_confirms(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["inferred_config"] = {
            "config_values": {"CLOUD_REGION": "asia-east1"},
            "unmapped_values": {},
            "source_format": "mixed",
            "reason": "",
        }
        state["pending_question"] = {
            "id": "inferred_config_review",
            "group": "provider_deployment",
            "kind": "yes_no",
            "field": "inferred_config_review",
            "options": [{"label": "Y", "value": True}, {"label": "N", "value": False}],
            "manual_input_allowed": False,
        }
        state["last_user_input"] = 'endpoint 是 http://fake-node:19000，method 是 eth_chainId，请求 {"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}'

        result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "inferred_config_review")
        self.assertNotIn("CLOUD_REGION", result["confirmed_config"])
        self.assertIn("请先确认", "\n".join(result["visible_response"]))

    def test_config_proposal_saves_endpoint_as_candidate_not_validated_config(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["target_mode"] = "real-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "ethereum", "canonical": "ethereum", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "ethereum"}
        state["active_group"] = "endpoint_process"
        state["last_user_input"] = '{"LOCAL_RPC_URL":"https://node.example","RPC_MODE":"mixed"}'

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {
                        "type": "propose_config_values",
                        "source_format": "json",
                        "config_values": {"LOCAL_RPC_URL": "https://node.example", "RPC_MODE": "mixed"},
                        "confidence": "high",
                    }
                ]
            }
            result = process_turn(state)

        result["last_user_input"] = "Y"
        result = process_turn(result)

        self.assertEqual(result["rpc_mode"], "mixed")
        self.assertEqual(result["endpoint_evidence"]["proposed_values"]["LOCAL_RPC_URL"], "https://node.example")
        self.assertNotIn("LOCAL_RPC_URL", result["confirmed_config"])
        self.assertNotEqual(result.get("endpoint_evidence", {}).get("local_rpc_url_ready"), True)

    def test_config_proposal_normalizes_units_and_accounts_absence(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["active_group"] = "provider_deployment"
        state["last_user_input"] = "pasted environment facts"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {
                        "type": "propose_config_values",
                        "source_format": "mixed",
                        "config_values": {
                            "CLOUD_REGION": "asia-east1",
                            "DATA_VOL_TYPE": "hyperdisk-balanced,",
                            "DATA_VOL_SIZE": "926GiB",
                            "DATA_VOL_MAX_IOPS": "20000 IOPS",
                            "DATA_VOL_MAX_THROUGHPUT": "1000 MiB/s",
                            "ACCOUNTS_DEVICE": "None",
                            "NETWORK_MAX_BANDWIDTH_GBPS": "100Gbps",
                        },
                        "confidence": "high",
                    }
                ]
            }
            result = process_turn(state)

        prompt = "\n".join(result.get("visible_response") or [])
        self.assertIn("DATA_VOL_SIZE: `926`", prompt)
        self.assertIn("HAS_ACCOUNTS_DEVICE", prompt)
        self.assertNotIn("ACCOUNTS_DEVICE: `None`", prompt)

        result["last_user_input"] = "Y"
        result = process_turn(result)

        self.assertEqual(result["confirmed_config"]["DATA_VOL_TYPE"], "hyperdisk-balanced")
        self.assertEqual(result["confirmed_config"]["DATA_VOL_SIZE"], "926")
        self.assertEqual(result["confirmed_config"]["DATA_VOL_MAX_IOPS"], "20000")
        self.assertEqual(result["confirmed_config"]["DATA_VOL_MAX_THROUGHPUT"], "1000")
        self.assertEqual(result["confirmed_config"]["NETWORK_MAX_BANDWIDTH_GBPS"], "100")
        self.assertIs(result["confirmed_config"]["has_accounts_device"], False)
        self.assertNotIn("ACCOUNTS_DEVICE", result["confirmed_config"])

    def test_config_proposal_derives_accounts_absence_when_model_outputs_empty_device(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["active_group"] = "provider_deployment"
        state["last_user_input"] = "环境信息：没有 accounts，ledger=vda"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {
                        "type": "propose_config_values",
                        "source_format": "mixed",
                        "config_values": {"ACCOUNTS_DEVICE": "", "LEDGER_DEVICE": "vda"},
                        "confidence": "high",
                    }
                ]
            }
            result = process_turn(state)

        prompt = "\n".join(result.get("visible_response") or [])
        self.assertIn("HAS_ACCOUNTS_DEVICE: `False`", prompt)
        self.assertNotIn("ACCOUNTS_DEVICE: ``", prompt)

    def test_config_review_then_continues_to_custom_rpc_action(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["active_group"] = "provider_deployment"
        state["last_user_input"] = "我要加自定义 RPC method eth_chainId，endpoint http://fake-node:19000，region asia-east1"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {
                        "type": "propose_config_values",
                        "source_format": "mixed",
                        "config_values": {"CLOUD_REGION": "asia-east1"},
                        "confidence": "high",
                    },
                    {
                        "type": "start_custom_rpc",
                        "rpc_method": "eth_chainId",
                        "rpc_endpoint": "http://fake-node:19000",
                        "confidence": "high",
                    },
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "inferred_config_review")

        ok_probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}
        with patch("agent.harness.groups.validate_rpc_endpoint", return_value=ok_probe):
            result["last_user_input"] = "Y"
            result = process_turn(result)

        self.assertEqual(result["confirmed_config"]["CLOUD_REGION"], "asia-east1")
        self.assertNotIn("LOCAL_RPC_URL", result["confirmed_config"])
        self.assertEqual(result["custom_rpc"]["method"], "eth_chainId")
        self.assertTrue(result["custom_rpc"]["endpoint_ready"])
        self.assertEqual(result["custom_rpc"]["status"], "needs_schema_evidence")
        self.assertEqual(result["pending_question"]["id"], "custom_rpc_schema_evidence")
        self.assertIn("custom_rpc_endpoint_probe", result["endpoint_evidence"])

    def test_config_review_is_prioritized_before_custom_rpc_even_if_model_orders_it_later(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["active_group"] = "provider_deployment"
        state["last_user_input"] = "我要加自定义 RPC method eth_chainId，endpoint http://fake-node:19000，region asia-east1"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {
                        "type": "start_custom_rpc",
                        "rpc_method": "eth_chainId",
                        "rpc_endpoint": "http://fake-node:19000",
                        "confidence": "high",
                    },
                    {
                        "type": "propose_config_values",
                        "source_format": "mixed",
                        "config_values": {"CLOUD_REGION": "asia-east1"},
                        "confidence": "high",
                    },
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "inferred_config_review")
        self.assertEqual(result["action_queue"][0]["type"], "start_custom_rpc")
        self.assertNotIn("endpoint_ready", result.get("custom_rpc", {}))

    def test_custom_rpc_schema_answer_clears_stale_queue_control(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["active_group"] = "endpoint_process"
        state["custom_rpc"] = {
            "status": "needs_schema_evidence",
            "endpoint": "http://fake-node:19000",
            "endpoint_ready": True,
            "method": "eth_chainId",
        }
        state["pending_question"] = {
            "group": "endpoint_process",
            "id": "custom_rpc_schema_evidence",
            "field": "custom_rpc_schema_evidence",
            "kind": "evidence",
            "manual_input_allowed": True,
        }
        state["action_queue"] = [
            {
                "type": "propose_config_values",
                "config_values": {"CLOUD_REGION": "asia-east1"},
                "confidence": "high",
            }
        ]
        state["last_user_input"] = "[]"

        ok_probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}
        with patch("agent.harness.groups.validate_rpc_endpoint", return_value=ok_probe):
            result = process_turn(state)

        self.assertEqual(result["custom_rpc"]["status"], "method_validated_next")
        self.assertEqual(result["pending_question"]["id"], "custom_rpc_continue")
        self.assertEqual(result.get("action_queue"), [])
        self.assertNotIn("CLOUD_REGION", result["confirmed_config"])

    def test_direct_config_assignment_goes_to_named_field_not_current_pending(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc", "CLOUD_REGION": "asia-east1"}
        state["active_group"] = "provider_deployment"
        state["pending_question"] = {
            "group": "provider_deployment",
            "id": "CLOUD_ZONE",
            "field": "CLOUD_ZONE",
            "kind": "manual_value",
            "manual_input_allowed": True,
            "prompt": "Confirm CLOUD_ZONE.",
        }
        state["last_user_input"] = "MACHINE_TYPE=n2-standard-16"

        result = process_turn(state)

        self.assertEqual(result["confirmed_config"].get("MACHINE_TYPE"), "n2-standard-16")
        self.assertNotEqual(result["confirmed_config"].get("CLOUD_ZONE"), "MACHINE_TYPE=n2-standard-16")
        self.assertEqual(result["pending_question"]["id"], "CLOUD_ZONE")

    def test_comma_separated_direct_config_assignments_apply_each_field(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
        state["active_group"] = "provider_deployment"
        state["last_user_input"] = "CLOUD_REGION=us-1, CLOUD_ZONE=us-1-z, MACHINE_TYPE=n2"

        result = process_turn(state)

        self.assertEqual(result["confirmed_config"].get("CLOUD_REGION"), "us-1")
        self.assertEqual(result["confirmed_config"].get("CLOUD_ZONE"), "us-1-z")
        self.assertEqual(result["confirmed_config"].get("MACHINE_TYPE"), "n2")
        self.assertEqual(result["pending_question"]["id"], "LEDGER_DEVICE")

    def test_declining_preflight_smoke_pauses_instead_of_looping(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["active_group"] = "preflight_smoke_execution"
        state["pending_question"] = {
            "group": "preflight_smoke_execution",
            "id": "preflight_smoke_confirm",
            "field": "preflight_smoke_confirmed",
            "kind": "yes_no",
            "manual_input_allowed": False,
            "options": [{"label": "Y", "value": True}, {"label": "N", "value": False}],
            "prompt": "配置已收集。是否运行 preflight 和 smoke？",
        }
        state["last_user_input"] = "n"

        result = process_turn(state)

        self.assertEqual(result.get("pending_question"), {})
        self.assertFalse(result["preflight"]["approved"])
        self.assertIn("已暂停", "\n".join(result.get("visible_response") or []))

    def test_queue_rejects_target_mode_not_explicit_in_user_text(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["active_group"] = "opening"
        state["last_user_input"] = "我要用 fake-node 测试 BNB，用 mixed，QPS quick，并开启本地 Grafana"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {
                        "type": "choose_target_mode",
                        "target_mode": "real-node",
                        "target_mode_explicit": True,
                        "confidence": "high",
                        "reason": "bad model default",
                    },
                    {"type": "choose_chain", "chain_text": "BNB", "confidence": "high"},
                    {"type": "set_rpc_mode", "rpc_mode": "mixed", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        self.assertNotEqual(result.get("target_mode"), "real-node")
        self.assertEqual(result["chain_identity"]["canonical"], "bsc")
        self.assertEqual(result["rpc_mode"], "")
        self.assertEqual(result["pending_question"]["id"], "opening_next_action")
        self.assertTrue(result.get("action_queue"))

    def test_return_to_origin_group_does_not_preempt_new_blocking_group(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "ledger_disk"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {
            "BLOCKCHAIN_NODE": "bsc",
            "CLOUD_REGION": "asia-east1",
            "CLOUD_ZONE": "asia-east1-c",
            "MACHINE_TYPE": "n2-standard-16",
            "NETWORK_INTERFACE": "eth0",
            "NETWORK_MAX_BANDWIDTH_GBPS": "100",
        }
        state["pending_question"] = {
            "id": "LEDGER_DEVICE",
            "group": "ledger_disk",
            "kind": "device",
            "field": "LEDGER_DEVICE",
            "options": [{"label": "vda", "value": "vda"}, {"label": "vdb", "value": "vdb"}],
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "先配置 QPS quick，然后回到磁盘配置"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "change_group", "group": "qps_profile", "confidence": "high"},
                    {"type": "set_qps_mode", "qps_mode": "quick", "confidence": "high"},
                    {"type": "change_group", "group": "ledger_disk", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["qps_profile"]["mode"], "quick")
        self.assertEqual(result["pending_question"]["id"], "qps_profile_confirm")
        self.assertEqual(result["active_group"], "qps_profile")

    def test_pending_question_can_route_to_observability_without_being_swallowed(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "provider_deployment"
        state["chain_identity"] = {"raw": "solana", "canonical": "solana", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "solana"}
        state["pending_question"] = {
            "id": "CLOUD_REGION",
            "group": "provider_deployment",
            "kind": "manual_value",
            "field": "CLOUD_REGION",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "先确认一下，本地 Grafana 不开"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "change_group", "group": "observability", "confidence": "high"},
                    {"type": "set_observability", "observability_mode": "disabled", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["observability"]["mode"], "disabled")
        self.assertEqual(result["pending_question"]["id"], "CLOUD_REGION")
        self.assertIn("disabled", "\n".join(result["visible_response"]))

    def test_completed_observability_group_reports_status_when_revisited(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "provider_deployment"
        state["chain_identity"] = {"raw": "solana", "canonical": "solana", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "solana"}
        state["observability"] = {"mode": "disabled"}
        state["pending_question"] = {
            "id": "CLOUD_REGION",
            "group": "provider_deployment",
            "kind": "manual_value",
            "field": "CLOUD_REGION",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "Grafana 现在是不是不开"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "change_group", "group": "observability", "confidence": "high"}]}
            result = process_turn(state)

        self.assertEqual(result["pending_question"], {})
        self.assertIn("disabled", "\n".join(result["visible_response"]))

    def test_sync_observe_multi_action_records_demo_source(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["active_group"] = "opening"
        state["last_user_input"] = "我想观察 BNB 节点同步，不打 RPC 压测，并只做流程 demo"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "choose_target_mode", "target_mode": "sync-observe", "target_mode_explicit": True, "confidence": "high"},
                    {"type": "choose_chain", "chain_text": "BNB", "confidence": "high"},
                    {"type": "set_sync_observe_source", "sync_observe_source": "demo_only", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["target_mode"], "sync-observe")
        self.assertEqual(result["workflow_mode"], "sync_observe")
        self.assertEqual(result["chain_identity"]["canonical"], "bsc")
        self.assertEqual(result["sync_observe"]["source"], "demo_only")
        self.assertEqual(result["pending_question"]["id"], "CLOUD_REGION")

    def test_paused_action_queue_resumes_after_confirmation(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "provider_deployment"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["last_user_input"] = "换成 eth，不使用 fake-node，QPS quick"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "choose_target_mode", "target_mode": "real-node", "target_mode_explicit": True, "confidence": "high"},
                    {"type": "change_chain", "chain_text": "ethereum", "confidence": "high"},
                    {"type": "set_qps_mode", "qps_mode": "quick", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "target_mode_change_confirm")
        self.assertEqual(result["target_mode_change_candidate"], "real-node")
        self.assertEqual(len(result["action_queue"]), 2)

        result["last_user_input"] = "Y"
        result = process_turn(result)

        self.assertEqual(result["pending_question"]["id"], "chain_change_confirm")
        self.assertEqual(result["target_mode"], "real-node")
        self.assertEqual(len(result["action_queue"]), 1)

        result["last_user_input"] = "Y"
        result = process_turn(result)

        self.assertEqual(result["chain_identity"]["canonical"], "ethereum")
        self.assertEqual(result["qps_profile"]["mode"], "quick")
        self.assertEqual(result["pending_question"]["id"], "qps_profile_confirm")

    def test_declined_confirmation_discards_remaining_action_queue(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "provider_deployment"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["last_user_input"] = "use real-node and quick"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "choose_target_mode", "target_mode": "real-node", "target_mode_explicit": True, "confidence": "high"},
                    {"type": "set_qps_mode", "qps_mode": "quick", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "target_mode_change_confirm")
        result["last_user_input"] = "N"
        result = process_turn(result)

        self.assertEqual(result["target_mode"], "fake-node")
        self.assertEqual(result.get("action_queue"), [])
        self.assertFalse(result.get("qps_profile", {}).get("mode"))

    def test_new_chain_endpoint_probe_failure_remains_blocking(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "endpoint_process"
        state["chain_identity"] = {
            "raw": "flow",
            "canonical": "flow",
            "adapter_family": "jsonrpc",
            "status": "existing_family_needs_endpoint",
            "case": "existing_family",
        }
        state["pending_question"] = {
            "id": "new_chain_endpoint",
            "group": "endpoint_process",
            "kind": "url",
            "prompt": "Provide endpoint",
            "field": "new_chain_endpoint",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "https://example.invalid/rpc"

        with patch(
            "agent.harness.groups.validate_rpc_endpoint",
            return_value={"ready": False, "status": "failed", "error": "boom", "evidence_file": "probe.json"},
        ):
            result = process_turn(state)

        self.assertEqual(result["chain_identity"]["status"], "existing_family_needs_endpoint")
        self.assertFalse(result["endpoint_evidence"]["candidate_endpoint_ready"])
        self.assertEqual(result["pending_question"]["id"], "new_chain_endpoint")
        self.assertIn("endpoint validation failed", "\n".join(result.get("visible_response") or []))

    def test_change_group_activates_requested_group_instead_of_default_path(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "provider_deployment"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["last_user_input"] = "I want to adjust QPS first"

        with patch("agent.harness.groups.resolve_intent_action") as resolver:
            resolver.return_value = {"intent": "change_group", "group": "qps_profile", "confidence": "high"}
            result = process_turn(state)

        self.assertEqual(result["active_group"], "qps_profile")
        self.assertEqual(result["pending_question"]["id"], "benchmark_mode")
        self.assertEqual(result["group_history"][-1], "provider_deployment")

    def test_completed_group_jump_resumes_default_missing_group(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "provider_deployment"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["last_user_input"] = "I want to adjust QPS first"

        with patch("agent.harness.groups.resolve_intent_action", return_value={"intent": "change_group", "group": "qps_profile", "confidence": "high"}):
            state = process_turn(state)

        state["last_user_input"] = "1"
        state = process_turn(state)
        self.assertEqual(state["pending_question"]["id"], "qps_profile_confirm")

        state["last_user_input"] = "Y"
        state = process_turn(state)

        self.assertEqual(state["qps_profile"]["mode"], "quick")
        self.assertTrue(state["qps_profile"]["confirmed"])
        self.assertEqual(state["active_group"], "provider_deployment")
        self.assertEqual(state["pending_question"]["id"], "CLOUD_REGION")

    def test_go_back_uses_group_history_without_phrase_matching(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "qps_profile"
        state["group_history"] = ["opening", "provider_deployment", "workload_rpc"]
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["rpc_mode"] = "single"
        state["last_user_input"] = "previous"

        with patch("agent.harness.groups.resolve_action_queue", return_value={"actions": [{"type": "go_back", "confidence": "high"}]}):
            result = process_turn(state)

        self.assertEqual(result["active_group"], "workload_rpc")
        self.assertEqual(result["pending_question"]["id"], "workload_confirm")
        self.assertEqual(result["group_history"], ["opening", "provider_deployment"])

    def test_custom_rpc_choice_transfers_control_to_endpoint_group(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "workload_rpc"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["rpc_mode"] = "mixed"
        state["pending_question"] = {
            "id": "workload_confirm",
            "group": "workload_rpc",
            "kind": "numbered_choice",
            "field": "workload_choice",
            "options": [
                {"label": "Use defaults", "value": "default"},
                {"label": "Add custom RPC method", "value": "custom_rpc"},
                {"label": "Adjust mixed weights", "value": "weights"},
            ],
            "manual_input_allowed": False,
        }
        state["last_user_input"] = "2"

        result = process_turn(state)

        self.assertEqual(result["custom_rpc"]["status"], "needs_endpoint")
        self.assertEqual(result["active_group"], "endpoint_process")
        self.assertEqual(result["pending_question"]["id"], "custom_rpc_endpoint")

    def test_sync_observe_group_jump_requires_target_mode_confirmation(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "workload_rpc"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["last_user_input"] = "observe node sync instead"

        with patch("agent.harness.groups.resolve_intent_action") as resolver:
            resolver.return_value = {"intent": "change_group", "group": "sync_observe", "confidence": "high"}
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "target_mode_change_confirm")
        self.assertEqual(result["target_mode_change_candidate"], "sync-observe")
        self.assertEqual(result["target_mode"], "fake-node")

    def test_unknown_chain_uses_identity_gate_not_partial_coercion(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["pending_question"] = {
            "id": "chain",
            "group": "chain_identity",
            "kind": "chain",
            "field": "chain",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "sola"

        with patch("agent.harness.groups.resolve_unknown_chain_identity") as resolver:
            resolver.return_value = {
                "chain_exists": None,
                "canonical_chain_name": "sola",
                "adapter_family": "unknown",
                "confidence": "low",
            }
            result = process_turn(state)

        self.assertNotEqual(result.get("chain_identity", {}).get("canonical"), "solana")
        self.assertEqual(result["chain_identity"]["status"], "needs_identity_confirmation")
        self.assertEqual(result["pending_question"]["id"], "unknown_chain_identity_confirm")

    def test_unknown_chain_candidate_accepts_yes_and_advances(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "sola",
            "canonical": "solana",
            "status": "needs_known_chain_confirmation",
            "case": "known_candidate",
        }
        state["pending_question"] = {
            "id": "unknown_chain_identity_confirm",
            "group": "chain_identity",
            "kind": "numbered_choice",
            "field": "unknown_chain_decision",
            "options": [
                {"label": "Use `solana`", "value": "confirm_known_chain"},
                {"label": "No, re-enter chain name", "value": "reenter_chain"},
                {"label": "This is another real chain; choose protocol", "value": "choose_protocol"},
            ],
            "manual_input_allowed": False,
        }
        state["last_user_input"] = "Y"
        result = process_turn(state)

        self.assertEqual(result["chain_identity"]["status"], "confirmed")
        self.assertEqual(result["confirmed_config"]["BLOCKCHAIN_NODE"], "solana")
        self.assertNotEqual(result["pending_question"]["id"], "chain")

    def test_unknown_chain_pending_accepts_protocol_hint_as_case2_entry(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "abcd",
            "canonical": "abcd",
            "adapter_family": "unknown",
            "status": "needs_identity_confirmation",
            "case": "unknown",
        }
        state["pending_question"] = {
            "id": "unknown_chain_identity_confirm",
            "group": "chain_identity",
            "kind": "numbered_choice",
            "field": "unknown_chain_decision",
            "options": [
                {"label": "真实链，继续确认协议", "value": "choose_protocol"},
                {"label": "我要重新输入链名", "value": "reenter_chain"},
            ],
            "manual_input_allowed": False,
        }
        state["last_user_input"] = "它应该是 EVM/jsonrpc，先按这个协议验证"

        result = process_turn(state)

        self.assertEqual(result["chain_identity"]["adapter_family"], "jsonrpc")
        self.assertEqual(result["chain_identity"]["status"], "existing_family_needs_endpoint")
        self.assertEqual(result["pending_question"]["id"], "new_chain_endpoint")

    def test_unknown_chain_candidate_accepts_explicit_candidate_name(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "sola",
            "canonical": "solana",
            "status": "needs_known_chain_confirmation",
            "case": "known_candidate",
        }
        state["pending_question"] = {
            "id": "unknown_chain_identity_confirm",
            "group": "chain_identity",
            "kind": "numbered_choice",
            "field": "unknown_chain_decision",
            "options": [
                {"label": "Use `solana`", "value": "confirm_known_chain"},
                {"label": "No, re-enter chain name", "value": "reenter_chain"},
                {"label": "This is another real chain; choose protocol", "value": "choose_protocol"},
            ],
            "manual_input_allowed": False,
        }
        state["last_user_input"] = "solana"

        with patch("agent.harness.groups.resolve_intent_action") as resolver:
            resolver.return_value = {"intent": "choose_chain", "chain_text": "solana", "confidence": "high"}
            result = process_turn(state)

        self.assertEqual(result["chain_identity"]["status"], "confirmed")
        self.assertEqual(result["confirmed_config"]["BLOCKCHAIN_NODE"], "solana")
        self.assertNotEqual(result["pending_question"]["id"], "chain")

    def test_free_form_chain_change_does_not_fill_pending_region(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "solana", "canonical": "solana", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "solana"}
        state["pending_question"] = {
            "id": "CLOUD_REGION",
            "group": "provider_deployment",
            "kind": "manual_value",
            "field": "CLOUD_REGION",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "change to ethereum"

        with patch("agent.harness.groups.resolve_intent_action") as resolver:
            resolver.return_value = {"intent": "change_chain", "chain_text": "ethereum", "confidence": "high"}
            result = process_turn(state)

        self.assertNotIn("CLOUD_REGION", result.get("confirmed_config", {}))
        self.assertEqual(result["pending_question"]["id"], "chain_change_confirm")

        result["last_user_input"] = "y"
        result = process_turn(result)
        self.assertNotIn("CLOUD_REGION", result.get("confirmed_config", {}))
        self.assertEqual(result["chain_identity"]["canonical"], "ethereum")
        self.assertEqual(result["confirmed_config"]["BLOCKCHAIN_NODE"], "ethereum")

    def test_target_mode_change_interrupts_manual_value_and_requires_confirmation(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["pending_question"] = {
            "id": "CLOUD_REGION",
            "group": "provider_deployment",
            "kind": "manual_value",
            "field": "CLOUD_REGION",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "use real-node instead"

        with patch("agent.harness.groups.resolve_intent_action") as resolver:
            resolver.return_value = {"intent": "choose_target_mode", "target_mode": "real-node", "confidence": "high"}
            result = process_turn(state)

        self.assertNotIn("CLOUD_REGION", result.get("confirmed_config", {}))
        self.assertEqual(result["pending_question"]["id"], "target_mode_change_confirm")

        result["last_user_input"] = "Y"
        result = process_turn(result)

        self.assertEqual(result["target_mode"], "real-node")
        self.assertNotIn("CLOUD_REGION", result.get("confirmed_config", {}))

    def test_target_mode_change_candidate_survives_langgraph_checkpoint(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime
        from agent.harness.state import new_state

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoints.sqlite"
            runtime = AnyChainGraphRuntime(thread_id="unit-thread", checkpoint_path=checkpoint_path)
            initial = new_state("unit-thread")
            initial["target_mode"] = "fake-node"
            initial["workflow_mode"] = "rpc_benchmark"
            initial["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
            initial["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
            initial["pending_question"] = {
                "id": "CLOUD_REGION",
                "group": "provider_deployment",
                "kind": "manual_value",
                "field": "CLOUD_REGION",
                "manual_input_allowed": True,
            }
            runtime.graph.update_state({"configurable": {"thread_id": "unit-thread"}}, initial)

            with patch("agent.harness.groups.resolve_intent_action") as resolver:
                resolver.return_value = {"intent": "choose_target_mode", "target_mode": "real-node", "confidence": "high"}
                state = runtime.invoke("use real-node instead", language="en")

            self.assertEqual(state["pending_question"]["id"], "target_mode_change_confirm")
            self.assertEqual(state["target_mode_change_candidate"], "real-node")

            state = runtime.invoke("Y", language="en")

        self.assertEqual(state["target_mode"], "real-node")
        self.assertNotIn("CLOUD_REGION", state.get("confirmed_config", {}))

    def test_manual_value_rejects_bare_yes_without_default_value(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "solana", "canonical": "solana", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "solana"}
        state["pending_question"] = {
            "id": "CLOUD_REGION",
            "group": "provider_deployment",
            "kind": "manual_value",
            "field": "CLOUD_REGION",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "y"

        with patch("agent.harness.groups.resolve_intent_action") as resolver:
            resolver.return_value = {"intent": "unknown", "confidence": "low"}
            result = process_turn(state)

        self.assertNotIn("CLOUD_REGION", result.get("confirmed_config", {}))
        self.assertEqual(result["pending_question"]["id"], "CLOUD_REGION")
        self.assertIn("does not look like", result["visible_response"][0])

    def test_free_form_pending_choice_can_be_resolved_by_llm_mapper(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {
            "BLOCKCHAIN_NODE": "bsc",
            "CLOUD_REGION": "us-1",
            "CLOUD_ZONE": "us-1-z",
            "MACHINE_TYPE": "n2",
            "LEDGER_DEVICE": "vda",
            "DATA_VOL_TYPE": "hyperdisk-balanced",
            "DATA_VOL_SIZE": "926",
            "DATA_VOL_MAX_IOPS": "20000",
            "DATA_VOL_MAX_THROUGHPUT": "1000",
            "has_accounts_device": False,
            "NETWORK_INTERFACE": "eth0",
            "NETWORK_MAX_BANDWIDTH_GBPS": "100",
        }
        state["rpc_mode"] = "mixed"
        state["custom_rpc"] = {"status": "needs_scope", "method": "eth_blockNumber"}
        state["pending_question"] = {
            "id": "custom_rpc_scope",
            "group": "workload_rpc",
            "kind": "numbered_choice",
            "field": "custom_rpc_scope",
            "options": [
                {"label": "Use this method as single workload only", "value": "single_replace"},
                {"label": "Use only my custom methods in mixed", "value": "mixed_replace"},
                {"label": "Keep template defaults in mixed and add this method", "value": "mixed_add"},
            ],
            "manual_input_allowed": False,
        }
        state["last_user_input"] = "replace defaults"

        with (
            patch("agent.harness.groups.resolve_action_queue", return_value={"actions": [{"type": "unknown", "confidence": "low"}]}),
            patch("agent.harness.groups.resolve_pending_choice") as resolver,
        ):
            resolver.return_value = {
                "matched": True,
                "selected_value": "mixed_replace",
                "selected_label": "Use only my custom methods in mixed",
                "confidence": "high",
            }
            result = process_turn(state)

        self.assertEqual(result["custom_rpc"]["scope"], "mixed_replace")
        self.assertEqual(result["custom_rpc"]["status"], "needs_weights")
        self.assertEqual(result["pending_question"]["id"], "custom_rpc_weights")

    def test_assignment_text_does_not_satisfy_numbered_scope_choice(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {
            "BLOCKCHAIN_NODE": "bsc",
            "CLOUD_REGION": "us-1",
            "CLOUD_ZONE": "us-1-z",
            "MACHINE_TYPE": "n2",
            "LEDGER_DEVICE": "vda",
            "DATA_VOL_TYPE": "hyperdisk-balanced",
            "DATA_VOL_SIZE": "926",
            "DATA_VOL_MAX_IOPS": "20000",
            "DATA_VOL_MAX_THROUGHPUT": "1000",
            "has_accounts_device": False,
            "NETWORK_INTERFACE": "eth0",
            "NETWORK_MAX_BANDWIDTH_GBPS": "100",
        }
        state["rpc_mode"] = "mixed"
        state["custom_rpc"] = {"status": "needs_scope", "method": "eth_blockNumber"}
        state["pending_question"] = {
            "id": "custom_rpc_scope",
            "group": "workload_rpc",
            "kind": "numbered_choice",
            "field": "custom_rpc_scope",
            "options": [
                {"label": "Use this method as single workload only", "value": "single_replace"},
                {"label": "Use only my custom methods in mixed", "value": "mixed_replace"},
                {"label": "Keep template defaults in mixed and add this method", "value": "mixed_add"},
            ],
            "manual_input_allowed": False,
        }
        state["last_user_input"] = "eth_blockNumber=100"

        with patch("agent.harness.groups.resolve_pending_choice") as resolver, patch("agent.harness.groups.resolve_intent_action") as intent:
            intent.return_value = {"intent": "unknown", "confidence": "low"}
            result = process_turn(state)

        resolver.assert_not_called()
        self.assertEqual(result["custom_rpc"]["status"], "needs_scope")
        self.assertEqual(result["pending_question"]["id"], "custom_rpc_scope")

    def test_declined_chain_change_keeps_original_chain_and_region_pending(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["pending_question"] = {
            "id": "CLOUD_REGION",
            "group": "provider_deployment",
            "kind": "manual_value",
            "field": "CLOUD_REGION",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "change to ethereum"

        with patch("agent.harness.groups.resolve_intent_action") as resolver:
            resolver.return_value = {"intent": "change_chain", "chain_text": "ethereum", "confidence": "high"}
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "chain_change_confirm")
        result["last_user_input"] = "n"
        result = process_turn(result)

        self.assertEqual(result["chain_identity"]["canonical"], "bsc")
        self.assertEqual(result["confirmed_config"]["BLOCKCHAIN_NODE"], "bsc")
        self.assertNotIn("CLOUD_REGION", result.get("confirmed_config", {}))
        self.assertEqual(result["pending_question"]["id"], "CLOUD_REGION")

    def test_choose_chain_action_still_confirms_when_chain_already_exists(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["pending_question"] = {
            "id": "CLOUD_REGION",
            "group": "provider_deployment",
            "kind": "manual_value",
            "field": "CLOUD_REGION",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "ethereum"

        with patch("agent.harness.groups.resolve_intent_action") as resolver:
            resolver.return_value = {"intent": "choose_chain", "chain_text": "ethereum", "confidence": "high"}
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "chain_change_confirm")
        self.assertEqual(result["chain_identity"]["canonical"], "bsc")

    def test_qps_rejecting_defaults_enters_adjustment_subflow(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "solana", "canonical": "solana", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {
            "BLOCKCHAIN_NODE": "solana",
            "CLOUD_REGION": "us-1",
            "CLOUD_ZONE": "us-1-z",
            "MACHINE_TYPE": "n2",
            "LEDGER_DEVICE": "vda",
            "DATA_VOL_TYPE": "hyperdisk-balanced",
            "DATA_VOL_SIZE": "926",
            "DATA_VOL_MAX_IOPS": "20000",
            "DATA_VOL_MAX_THROUGHPUT": "1000",
            "has_accounts_device": False,
            "NETWORK_INTERFACE": "eth0",
            "NETWORK_MAX_BANDWIDTH_GBPS": "100",
        }
        state["rpc_mode"] = "single"
        state["workload"] = {"confirmed": True, "choice": "default"}
        state["pending_question"] = {
            "id": "benchmark_mode",
            "group": "qps_profile",
            "kind": "numbered_choice",
            "field": "benchmark_mode",
            "options": [{"label": "quick", "value": "quick"}, {"label": "standard", "value": "standard"}],
            "manual_input_allowed": False,
        }
        state["last_user_input"] = "1"
        state = process_turn(state)
        self.assertEqual(state["pending_question"]["id"], "qps_profile_confirm")

        state["last_user_input"] = "n"
        state = process_turn(state)
        self.assertEqual(state["pending_question"]["id"], "qps_adjust_field")

        state["last_user_input"] = "1"
        state = process_turn(state)
        self.assertEqual(state["pending_question"]["id"], "qps_adjust_value")

    def test_preflight_yes_calls_harness_execution_helper(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["pending_question"] = {
            "id": "preflight_smoke_confirm",
            "group": "preflight_smoke_execution",
            "kind": "yes_no",
            "field": "preflight_smoke_confirmed",
            "options": [{"label": "Y", "value": True}, {"label": "N", "value": False}],
            "manual_input_allowed": False,
        }
        state["last_user_input"] = "y"

        with patch("agent.harness.groups.run_approved_preflight_and_smoke") as execute:
            execute.return_value = {**state, "visible_response": ["executed"], "pending_question": {}, "_stop_after_response": True}
            result = process_turn(state)

        execute.assert_called_once()
        self.assertEqual(result["visible_response"], ["executed"])

    def test_existing_chain_custom_rpc_validates_endpoint_method_and_weights(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {
            "BLOCKCHAIN_NODE": "bsc",
            "CLOUD_REGION": "us-1",
            "CLOUD_ZONE": "us-1-z",
            "MACHINE_TYPE": "n2",
            "LEDGER_DEVICE": "vda",
            "DATA_VOL_TYPE": "hyperdisk-balanced",
            "DATA_VOL_SIZE": "926",
            "DATA_VOL_MAX_IOPS": "20000",
            "DATA_VOL_MAX_THROUGHPUT": "1000",
            "has_accounts_device": False,
            "NETWORK_INTERFACE": "eth0",
            "NETWORK_MAX_BANDWIDTH_GBPS": "100",
        }
        state["rpc_mode"] = "mixed"
        state["pending_question"] = {
            "id": "workload_confirm",
            "group": "workload_rpc",
            "kind": "numbered_choice",
            "field": "workload_choice",
            "options": [
                {"label": "Use defaults", "value": "default"},
                {"label": "Add custom RPC method", "value": "custom_rpc"},
            ],
            "manual_input_allowed": False,
        }
        state["last_user_input"] = "2"
        state = process_turn(state)
        self.assertEqual(state["custom_rpc"]["status"], "needs_endpoint")

        ok_probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}
        with patch("agent.harness.groups.validate_rpc_endpoint", return_value=ok_probe):
            state["last_user_input"] = "https://example.invalid/rpc"
            state = process_turn(state)
            self.assertEqual(state["custom_rpc"]["status"], "needs_method")

            state["last_user_input"] = "eth_blockNumber"
            state = process_turn(state)
            self.assertEqual(state["custom_rpc"]["status"], "needs_schema_evidence")

            state["last_user_input"] = "[]"
            state = process_turn(state)
            self.assertEqual(state["custom_rpc"]["status"], "method_validated_next")
            self.assertEqual(state["pending_question"]["id"], "custom_rpc_continue")

        state["last_user_input"] = "2"
        state = process_turn(state)
        self.assertEqual(state["custom_rpc"]["status"], "needs_scope")

        state["last_user_input"] = "2"
        state = process_turn(state)
        self.assertEqual(state["custom_rpc"]["status"], "needs_weights")

        state["last_user_input"] = "eth_blockNumber=70,eth_getBalance=20"
        state = process_turn(state)
        self.assertEqual(state["custom_rpc"]["status"], "needs_weights")
        self.assertEqual(state["pending_question"]["id"], "custom_rpc_weights")

        state["last_user_input"] = "eth_blockNumber=70,eth_getBalance=30"
        state = process_turn(state)
        self.assertEqual(state["custom_rpc"]["status"], "validated")
        self.assertTrue(state["workload"]["confirmed"])
        self.assertEqual(state["workload"]["mixed_weights"], {"eth_blockNumber": 70, "eth_getBalance": 30})

    def test_existing_chain_custom_rpc_extracts_schema_evidence_before_probe(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["custom_rpc"] = {
            "status": "needs_schema_evidence",
            "endpoint": "https://example.invalid/rpc",
            "endpoint_ready": True,
            "method": "eth_getBalance",
        }
        state["pending_question"] = {
            "id": "custom_rpc_schema_evidence",
            "group": "endpoint_process",
            "kind": "evidence",
            "field": "custom_rpc_schema_evidence",
            "manual_input_allowed": True,
        }
        draft = {
            "status": "draft",
            "method": "eth_getBalance",
            "params": [
                {"index": 0, "name": "address", "type": "string", "meaning": "account address", "example": "0x0000000000000000000000000000000000000000", "required": True},
                {"index": 1, "name": "block", "type": "string", "meaning": "block tag", "example": "latest", "required": True},
            ],
            "response_summary": "hex balance",
            "confidence": "high",
        }

        with patch("agent.harness.groups.extract_rpc_schema_from_evidence", return_value=draft):
            state["last_user_input"] = "curl --data '{\"method\":\"eth_getBalance\",\"params\":[\"0x0000000000000000000000000000000000000000\",\"latest\"]}'"
            state = process_turn(state)

        self.assertEqual(state["custom_rpc"]["status"], "schema_needs_confirmation")
        self.assertEqual(state["pending_question"]["id"], "custom_rpc_schema_confirm")

        ok_probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}
        with patch("agent.harness.groups.validate_rpc_endpoint", return_value=ok_probe) as probe:
            state["last_user_input"] = "y"
            state = process_turn(state)

        probe.assert_called_once()
        self.assertEqual(state["custom_rpc"]["params"], ["0x0000000000000000000000000000000000000000", "latest"])
        self.assertEqual(state["custom_rpc"]["status"], "method_validated_next")
        self.assertEqual(state["pending_question"]["id"], "custom_rpc_continue")

    def test_existing_chain_custom_rpc_accepts_json_rpc_request_as_direct_schema(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["custom_rpc"] = {
            "status": "needs_schema_evidence",
            "endpoint": "https://example.invalid/rpc",
            "endpoint_ready": True,
            "method": "eth_getBalance",
        }
        state["pending_question"] = {
            "id": "custom_rpc_schema_evidence",
            "group": "endpoint_process",
            "kind": "evidence",
            "field": "custom_rpc_schema_evidence",
            "manual_input_allowed": True,
        }
        ok_probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}

        with patch("agent.harness.groups.extract_rpc_schema_from_evidence") as extractor, patch("agent.harness.groups.validate_rpc_endpoint", return_value=ok_probe) as probe:
            state["last_user_input"] = '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}'
            state = process_turn(state)

        extractor.assert_not_called()
        probe.assert_called_once()
        self.assertEqual(state["custom_rpc"]["method"], "eth_blockNumber")
        self.assertEqual(state["custom_rpc"]["params"], [])
        self.assertEqual(state["custom_rpc"]["status"], "method_validated_next")
        self.assertEqual(state["pending_question"]["id"], "custom_rpc_continue")

    def test_custom_rpc_endpoint_turn_consumes_inline_jsonrpc_request(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["custom_rpc"] = {"status": "needs_endpoint"}
        state["pending_question"] = {
            "id": "custom_rpc_endpoint",
            "group": "endpoint_process",
            "kind": "url",
            "field": "custom_rpc_endpoint",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = 'endpoint 是 http://fake-node:19000，method 是 eth_chainId，请求 {"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}'

        ok_probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}
        with patch("agent.harness.groups.validate_rpc_endpoint", return_value=ok_probe):
            result = process_turn(state)

        self.assertEqual(result["custom_rpc"]["status"], "method_validated_next")
        self.assertEqual(result["custom_rpc"]["method"], "eth_chainId")
        self.assertEqual(result["pending_question"]["id"], "custom_rpc_continue")

    def test_real_node_requires_local_rpc_probe_before_workload(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "real-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "ethereum", "canonical": "ethereum", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {
            "BLOCKCHAIN_NODE": "ethereum",
            "CLOUD_REGION": "us-1",
            "CLOUD_ZONE": "us-1-z",
            "MACHINE_TYPE": "n2",
            "LEDGER_DEVICE": "vda",
            "DATA_VOL_TYPE": "hyperdisk-balanced",
            "DATA_VOL_SIZE": "926",
            "DATA_VOL_MAX_IOPS": "20000",
            "DATA_VOL_MAX_THROUGHPUT": "1000",
            "has_accounts_device": False,
            "NETWORK_INTERFACE": "eth0",
            "NETWORK_MAX_BANDWIDTH_GBPS": "100",
        }
        state["last_user_input"] = "continue"

        with (
            patch("agent.harness.groups.resolve_action_queue", return_value={"actions": [{"type": "unknown", "confidence": "low"}]}),
            patch("agent.harness.groups.resolve_intent_action") as resolver,
        ):
            resolver.return_value = {"intent": "unknown", "confidence": "low"}
            state = process_turn(state)

        self.assertEqual(state["pending_question"]["id"], "LOCAL_RPC_URL")

        ok_probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}
        with patch("agent.harness.groups.validate_rpc_endpoint", return_value=ok_probe):
            state["last_user_input"] = "https://example.invalid/rpc"
            state = process_turn(state)

        self.assertEqual(state["confirmed_config"]["LOCAL_RPC_URL"], "https://example.invalid/rpc")
        self.assertTrue(state["endpoint_evidence"]["local_rpc_url_ready"])
        self.assertEqual(state["pending_question"]["id"], "BLOCKCHAIN_PROCESS_NAMES")

    def test_new_chain_existing_family_validates_endpoint_method_workload_then_runtime_choice(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "flow-evm",
            "canonical": "flow-evm",
            "adapter_family": "jsonrpc",
            "status": "existing_family_needs_endpoint",
            "case": "case2",
        }
        state["pending_question"] = {
            "id": "new_chain_endpoint",
            "group": "endpoint_process",
            "kind": "url",
            "field": "new_chain_endpoint",
            "manual_input_allowed": True,
        }

        ok_probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}
        with patch("agent.harness.groups.validate_rpc_endpoint", return_value=ok_probe):
            state["last_user_input"] = "https://example.invalid/rpc"
            state = process_turn(state)
            self.assertEqual(state["chain_identity"]["status"], "existing_family_needs_method")
            self.assertEqual(state["pending_question"]["id"], "new_chain_method")

            state["last_user_input"] = "eth_blockNumber"
            state = process_turn(state)
            self.assertEqual(state["chain_identity"]["status"], "existing_family_needs_schema_evidence")
            self.assertEqual(state["pending_question"]["id"], "new_chain_schema_evidence")

            state["last_user_input"] = "[]"
            state = process_turn(state)

        self.assertEqual(state["chain_identity"]["status"], "existing_family_method_validated_next")
        self.assertEqual(state["pending_question"]["id"], "new_chain_method_continue")
        self.assertIn("new_chain_method_probe", state["endpoint_evidence"])
        self.assertNotIn("LOCAL_RPC_URL", state.get("confirmed_config", {}))
        self.assertEqual(state["active_group"], "endpoint_process")

        state["last_user_input"] = "2"
        state = process_turn(state)
        self.assertEqual(state["chain_identity"]["status"], "existing_family_needs_workload_scope")
        self.assertEqual(state["active_group"], "endpoint_process")
        self.assertEqual(state["pending_question"]["id"], "new_chain_workload_scope")

        state["last_user_input"] = "1"
        state = process_turn(state)
        self.assertEqual(state["chain_identity"]["status"], "existing_family_runtime_choice")
        self.assertEqual(state["active_group"], "target_samples_fixtures")
        self.assertEqual(state["pending_question"]["id"], "new_chain_runtime_choice")
        self.assertNotIn("LOCAL_RPC_URL", state.get("confirmed_config", {}))

        state["last_user_input"] = "1"
        state = process_turn(state)

        self.assertEqual(state["target_mode"], "real-node")
        self.assertEqual(state["chain_identity"]["status"], "confirmed")
        self.assertEqual(state["chain_identity"]["case"], "case2_runtime_override")
        self.assertEqual(state["confirmed_config"]["BLOCKCHAIN_NODE"], "flow-evm")
        self.assertEqual(state["confirmed_config"]["LOCAL_RPC_URL"], "https://example.invalid/rpc")
        self.assertEqual(state["workload"]["methods"], ["eth_blockNumber"])
        self.assertTrue(state["workload"]["job_local_override"])
        self.assertEqual(state["pending_question"]["id"], "CLOUD_REGION")

    def test_new_chain_existing_family_mixed_requires_validated_method_weights(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "flow-evm",
            "canonical": "flow-evm",
            "adapter_family": "jsonrpc",
            "status": "existing_family_method_validated_next",
            "case": "case2",
            "validated_methods": [
                {"method": "eth_blockNumber", "params": [], "evidence_file": ".agent/evidence/one.json"},
                {"method": "eth_chainId", "params": [], "evidence_file": ".agent/evidence/two.json"},
            ],
        }
        state["pending_question"] = {
            "id": "new_chain_method_continue",
            "group": "endpoint_process",
            "kind": "numbered_choice",
            "field": "new_chain_method_continue",
            "options": [
                {"label": "Add another RPC method", "value": "add_another"},
                {"label": "This is enough", "value": "finish"},
            ],
        }

        state["last_user_input"] = "2"
        state = process_turn(state)
        self.assertEqual(state["pending_question"]["id"], "new_chain_workload_scope")

        state["last_user_input"] = "2"
        state = process_turn(state)
        self.assertEqual(state["chain_identity"]["status"], "existing_family_needs_weights")
        self.assertEqual(state["pending_question"]["id"], "new_chain_custom_weights")

        state["last_user_input"] = "eth_blockNumber=100"
        state = process_turn(state)
        self.assertEqual(state["chain_identity"]["status"], "existing_family_needs_weights")
        self.assertIn("缺少已验证 method", "\n".join(state["visible_response"]))

        state["last_user_input"] = "那改成 eth_blockNumber=70,eth_chainId=30"
        state = process_turn(state)
        self.assertEqual(state["chain_identity"]["status"], "existing_family_runtime_choice")
        self.assertEqual(state["rpc_mode"], "mixed")
        self.assertEqual(state["workload"]["mixed_weights"], {"eth_blockNumber": 70, "eth_chainId": 30})
        self.assertTrue(state["workload"]["job_local_override"])
        self.assertEqual(state["pending_question"]["id"], "new_chain_runtime_choice")

    def test_unknown_chain_with_user_protocol_hint_enters_supported_family_confirmation(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["active_group"] = "opening"
        state["last_user_input"] = "我想测 Flow，它应该是 EVM/jsonrpc"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {
                        "type": "choose_chain",
                        "chain_text": "Flow",
                        "chain_exists": True,
                        "canonical_chain_name": "flow",
                        "adapter_family": "jsonrpc",
                        "confidence": "high",
                    }
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["chain_identity"]["canonical"], "flow")
        self.assertEqual(result["chain_identity"]["adapter_family"], "jsonrpc")
        self.assertEqual(result["pending_question"]["id"], "unknown_chain_identity_confirm")
        self.assertIn("协议族为 `jsonrpc`", result["visible_response"][0])

    def test_device_choice_does_not_accept_yes_as_custom_interface(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["pending_question"] = {
            "id": "network_interface",
            "group": "network",
            "kind": "device",
            "field": "NETWORK_INTERFACE",
            "options": [{"label": "eth0 (default)", "value": "eth0"}],
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "y"

        result = process_turn(state)

        self.assertNotIn("NETWORK_INTERFACE", result.get("confirmed_config", {}))
        self.assertEqual(result["pending_question"]["id"], "network_interface")

    def test_new_chain_existing_family_extracts_schema_evidence_before_runtime_choice(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "flow-evm",
            "canonical": "flow-evm",
            "adapter_family": "jsonrpc",
            "status": "existing_family_needs_schema_evidence",
            "case": "case2",
            "candidate_method": "eth_blockNumber",
        }
        state["endpoint_evidence"] = {"candidate_endpoint": "https://example.invalid/rpc"}
        state["pending_question"] = {
            "id": "new_chain_schema_evidence",
            "group": "endpoint_process",
            "kind": "evidence",
            "field": "new_chain_schema_evidence",
            "manual_input_allowed": True,
        }
        draft = {
            "status": "draft",
            "method": "eth_blockNumber",
            "params": [],
            "response_summary": "latest block number as hex quantity",
            "confidence": "high",
        }

        with patch("agent.harness.groups.extract_rpc_schema_from_evidence", return_value=draft):
            state["last_user_input"] = "curl --data '{\"method\":\"eth_blockNumber\",\"params\":[]}'"
            state = process_turn(state)

        self.assertEqual(state["chain_identity"]["status"], "existing_family_schema_needs_confirmation")
        self.assertEqual(state["pending_question"]["id"], "new_chain_schema_confirm")

        ok_probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}
        with patch("agent.harness.groups.validate_rpc_endpoint", return_value=ok_probe):
            state["last_user_input"] = "y"
            state = process_turn(state)

        self.assertEqual(state["chain_identity"]["candidate_params"], [])
        self.assertEqual(state["chain_identity"]["status"], "existing_family_method_validated_next")
        self.assertEqual(state["pending_question"]["id"], "new_chain_method_continue")
        self.assertIn("new_chain_method_probe", state["endpoint_evidence"])

    def test_pending_answer_clears_stale_action_queue_before_new_endpoint_question(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "flow",
            "canonical": "flow",
            "status": "needs_protocol_confirmation",
            "case": "unknown",
        }
        state["pending_question"] = {
            "id": "adapter_family_confirm",
            "group": "chain_identity",
            "kind": "numbered_choice",
            "field": "adapter_family",
            "options": [{"label": "jsonrpc / EVM", "value": "jsonrpc"}],
            "manual_input_allowed": False,
        }
        state["action_queue"] = [
            {
                "type": "propose_config_values",
                "config_values": {"LOCAL_RPC_URL": ""},
                "unmapped_values": {"endpoint": "provided by user"},
                "confidence": "high",
            }
        ]

        state["last_user_input"] = "1"
        state = process_turn(state)

        self.assertEqual(state["pending_question"]["id"], "new_chain_endpoint")
        self.assertEqual(state.get("action_queue"), [])

        ok_probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}
        with patch("agent.harness.groups.validate_rpc_endpoint", return_value=ok_probe):
            state["last_user_input"] = "https://example.invalid/rpc"
            state = process_turn(state)

        self.assertEqual(state["chain_identity"]["status"], "existing_family_needs_method")
        self.assertEqual(state["endpoint_evidence"]["candidate_endpoint"], "https://example.invalid/rpc")
        self.assertEqual(state["pending_question"]["id"], "new_chain_method")

    def test_unsupported_family_stops_with_development_handoff_message(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "new-protocol-chain",
            "canonical": "new-protocol-chain",
            "status": "needs_protocol_confirmation",
            "case": "unknown",
        }
        state["pending_question"] = {
            "id": "adapter_family_confirm",
            "group": "chain_identity",
            "kind": "numbered_choice",
            "field": "adapter_family",
            "options": [{"label": "unsupported", "value": "unsupported"}],
            "manual_input_allowed": False,
        }
        state["last_user_input"] = "1"

        state = process_turn(state)

        self.assertEqual(state["chain_identity"]["status"], "unsupported_family_handoff")
        self.assertEqual(state["chain_identity"]["case"], "case3")
        self.assertIn("outside the supported adapter families", state["visible_response"][0])

    def test_sync_observe_requires_real_data_source_before_process_or_stop_condition(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "sync-observe"
        state["workflow_mode"] = "sync_observe"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {
            "BLOCKCHAIN_NODE": "bsc",
            "CLOUD_REGION": "us-1",
            "CLOUD_ZONE": "us-1-z",
            "MACHINE_TYPE": "n2",
            "LEDGER_DEVICE": "vda",
            "DATA_VOL_TYPE": "hyperdisk-balanced",
            "DATA_VOL_SIZE": "926",
            "DATA_VOL_MAX_IOPS": "20000",
            "DATA_VOL_MAX_THROUGHPUT": "1000",
            "has_accounts_device": False,
            "NETWORK_INTERFACE": "eth0",
            "NETWORK_MAX_BANDWIDTH_GBPS": "100",
        }
        state["last_user_input"] = "continue"

        with (
            patch("agent.harness.groups.resolve_action_queue", return_value={"actions": [{"type": "unknown", "confidence": "low"}]}),
            patch("agent.harness.groups.resolve_intent_action", return_value={"intent": "unknown", "confidence": "low"}),
        ):
            state = process_turn(state)

        self.assertEqual(state["pending_question"]["id"], "sync_observe_source")
        self.assertIn("fake-node", state["visible_response"][0])

    def test_sync_observe_endpoint_only_probes_real_endpoint_and_skips_process_name(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "sync-observe"
        state["workflow_mode"] = "sync_observe"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {
            "BLOCKCHAIN_NODE": "bsc",
            "CLOUD_REGION": "us-1",
            "CLOUD_ZONE": "us-1-z",
            "MACHINE_TYPE": "n2",
            "LEDGER_DEVICE": "vda",
            "DATA_VOL_TYPE": "hyperdisk-balanced",
            "DATA_VOL_SIZE": "926",
            "DATA_VOL_MAX_IOPS": "20000",
            "DATA_VOL_MAX_THROUGHPUT": "1000",
            "has_accounts_device": False,
            "NETWORK_INTERFACE": "eth0",
            "NETWORK_MAX_BANDWIDTH_GBPS": "100",
        }
        state["pending_question"] = {
            "id": "sync_observe_source",
            "group": "sync_observe",
            "kind": "numbered_choice",
            "field": "sync_observe_source",
            "options": [{"label": "endpoint", "value": "endpoint_only"}],
            "manual_input_allowed": False,
        }
        state["last_user_input"] = "1"
        state = process_turn(state)
        self.assertEqual(state["pending_question"]["id"], "SYNC_OBSERVE_RPC_URL")

        ok_probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}
        with patch("agent.harness.groups.validate_rpc_endpoint", return_value=ok_probe):
            state["last_user_input"] = "https://example.invalid/rpc"
            state = process_turn(state)

        self.assertTrue(state["endpoint_evidence"]["sync_rpc_url_ready"])
        self.assertEqual(state["confirmed_config"]["LOCAL_RPC_URL"], "https://example.invalid/rpc")
        self.assertNotEqual(state.get("pending_question", {}).get("id"), "BLOCKCHAIN_PROCESS_NAMES")
        self.assertEqual(state["pending_question"]["id"], "MAINNET_RPC_URL_REVIEWED")

    def test_sync_observe_client_setup_without_google_search_stops_with_handoff(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "sync-observe"
        state["workflow_mode"] = "sync_observe"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["web_research"] = {"google_search_available": False}
        state["confirmed_config"] = {
            "BLOCKCHAIN_NODE": "bsc",
            "CLOUD_REGION": "us-1",
            "CLOUD_ZONE": "us-1-z",
            "MACHINE_TYPE": "n2",
            "LEDGER_DEVICE": "vda",
            "DATA_VOL_TYPE": "hyperdisk-balanced",
            "DATA_VOL_SIZE": "926",
            "DATA_VOL_MAX_IOPS": "20000",
            "DATA_VOL_MAX_THROUGHPUT": "1000",
            "has_accounts_device": False,
            "NETWORK_INTERFACE": "eth0",
            "NETWORK_MAX_BANDWIDTH_GBPS": "100",
        }
        state["pending_question"] = {
            "id": "sync_observe_source",
            "group": "sync_observe",
            "kind": "numbered_choice",
            "field": "sync_observe_source",
            "options": [{"label": "client setup", "value": "client_setup"}],
            "manual_input_allowed": False,
        }
        state["last_user_input"] = "1"
        state = process_turn(state)
        self.assertEqual(state["pending_question"]["id"], "sync_observe_client_setup_ack")

        state["last_user_input"] = "y"
        state = process_turn(state)

        self.assertIn("google_search is unavailable", state["visible_response"][0])
        self.assertEqual(state.get("pending_question"), {})
        self.assertNotEqual(state.get("active_group"), "preflight_smoke_execution")

    def test_retired_adk_runner_bridge_cannot_own_product_workflow(self) -> None:
        from agent.adk_app.runner_bridge import ADKRunnerBridge, run_text_once, runner_bridge_status, sanitize_adk_text

        bridge_file = Path(__file__).resolve().parents[1] / "agent" / "adk_app" / "runner_bridge.py"
        bridge_text = bridge_file.read_text(encoding="utf-8")

        self.assertNotIn("workflows.conversation_state", bridge_text)
        self.assertNotIn("build_terminal_turn_prompt", bridge_text)
        self.assertNotIn("build_root_agent", bridge_text)
        self.assertNotIn("google.genai", bridge_text)
        self.assertIsInstance(runner_bridge_status().as_dict(), dict)
        self.assertEqual(sanitize_adk_text("  ok  "), "ok")
        with self.assertRaisesRegex(RuntimeError, "retired"):
            ADKRunnerBridge()
        with self.assertRaisesRegex(RuntimeError, "retired"):
            run_text_once("Hi")

    def test_adk_root_and_registry_do_not_expose_retired_workflow_runtime(self) -> None:
        import sys

        agent_root = Path(__file__).resolve().parents[1] / "agent"
        if str(agent_root) not in sys.path:
            sys.path.insert(0, str(agent_root))

        from agent.adk_app.root_agent import ADK_MODEL_BRIDGE_INSTRUCTION
        from agent.adk_app.tools.registry import get_adk_tools

        repo = Path(__file__).resolve().parents[1]
        root_text = (repo / "agent" / "adk_app" / "root_agent.py").read_text(encoding="utf-8")
        registry_text = (repo / "agent" / "adk_app" / "tools" / "registry.py").read_text(encoding="utf-8")
        tool_names = {getattr(item, "__name__", str(item)) for item in get_adk_tools()}

        self.assertIn("LangGraph Harness", ADK_MODEL_BRIDGE_INSTRUCTION)
        self.assertNotIn("before_tool_callback", root_text)
        self.assertNotIn("after_model_callback", root_text)
        self.assertNotIn("build_domain_agents", root_text)
        self.assertNotIn("ROOT_INSTRUCTION", root_text)
        self.assertNotIn("workflow_state", registry_text)
        self.assertFalse({"load_workflow_state", "update_workflow_state", "answer_pending_question"} & tool_names)

    def test_custom_rpc_success_asks_continue_instead_of_looping_schema(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["custom_rpc"] = {
            "status": "schema_needs_confirmation",
            "endpoint": "http://fake-node:19000",
            "endpoint_ready": True,
            "method": "eth_blockNumber",
            "schema_draft": {
                "status": "draft",
                "evidence_kind": "jsonrpc_request",
                "transport": "jsonrpc",
                "method": "eth_blockNumber",
                "params": [],
                "params_json": [],
                "confidence": "high",
            },
        }
        state["active_group"] = "endpoint_process"
        state["pending_question"] = {
            "id": "custom_rpc_schema_confirm",
            "group": "endpoint_process",
            "kind": "yes_no",
            "field": "custom_rpc_schema_confirm",
            "options": [{"label": "Y", "value": True}, {"label": "N", "value": False}],
        }
        state["last_user_input"] = "Y"

        with patch("agent.harness.groups.validate_rpc_endpoint") as probe:
            probe.return_value = {"ready": True, "evidence_file": ".agent/evidence/endpoint-probes/demo.json"}
            result = process_turn(state)

        self.assertEqual(result["custom_rpc"]["status"], "method_validated_next")
        self.assertEqual(result["pending_question"]["id"], "custom_rpc_continue")
        self.assertIn("What should happen next", "\n".join(result["visible_response"]))

    def test_new_chain_rest_evidence_does_not_probe_as_jsonrpc_method(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "abcd",
            "canonical": "abcd",
            "adapter_family": "jsonrpc",
            "status": "existing_family_needs_schema_evidence",
            "case": "case2",
            "candidate_method": "GetBlockByID",
        }
        state["endpoint_evidence"] = {"candidate_endpoint": "https://example.invalid", "candidate_endpoint_ready": True}
        state["active_group"] = "endpoint_process"
        state["pending_question"] = {
            "id": "new_chain_schema_evidence",
            "group": "endpoint_process",
            "kind": "evidence",
            "field": "new_chain_schema_evidence",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "Get Blocks by ID\npath Parameters\nid required\nquery Parameters"

        with (
            patch("agent.harness.groups.extract_rpc_schema_from_evidence") as extract,
            patch("agent.harness.groups.validate_rpc_endpoint") as probe,
        ):
            extract.return_value = {
                "status": "draft",
                "evidence_kind": "docs_excerpt",
                "transport": "rest",
                "method": "GET /v1/blocks/{id}",
                "rest_path": "/v1/blocks/{id}",
                "params": [{"index": 0, "name": "id", "type": "string", "example": "0" * 64, "required": True}],
                "confidence": "medium",
            }
            result = process_turn(state)

        probe.assert_not_called()
        self.assertEqual(result["pending_question"]["id"], "adapter_family_confirm")
        self.assertIn("REST API", "\n".join(result["visible_response"]))

    def test_url_is_not_accepted_as_new_chain_rpc_method_name(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["chain_identity"] = {
            "raw": "abcd",
            "canonical": "abcd",
            "adapter_family": "jsonrpc",
            "status": "existing_family_needs_method",
            "case": "case2",
        }
        state["active_group"] = "endpoint_process"
        state["pending_question"] = {
            "id": "new_chain_method",
            "group": "endpoint_process",
            "kind": "manual_value",
            "field": "new_chain_method",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "https://rest-testnet.onflow.org/v1/blocks"

        result = process_turn(state)

        self.assertEqual(result["chain_identity"]["status"], "existing_family_needs_method")
        self.assertNotEqual(result["chain_identity"].get("candidate_method"), "https://rest-testnet.onflow.org/v1/blocks")
        self.assertIn("不像可直接验证的 RPC method 名称", "\n".join(result["visible_response"]))

    def test_new_chain_method_question_accepts_inline_jsonrpc_request(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["chain_identity"] = {
            "raw": "abcd",
            "canonical": "abcd",
            "adapter_family": "jsonrpc",
            "status": "existing_family_needs_method",
            "case": "case2",
        }
        state["endpoint_evidence"] = {"candidate_endpoint": "http://fake-node:19000", "candidate_endpoint_ready": True}
        state["active_group"] = "endpoint_process"
        state["pending_question"] = {
            "id": "new_chain_method",
            "group": "endpoint_process",
            "kind": "manual_value",
            "field": "new_chain_method",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = 'method 是 eth_blockNumber，请求是 {"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}'

        ok_probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}
        with patch("agent.harness.groups.validate_rpc_endpoint", return_value=ok_probe):
            result = process_turn(state)

        self.assertEqual(result["chain_identity"]["status"], "existing_family_method_validated_next")
        self.assertEqual(result["chain_identity"]["candidate_method"], "eth_blockNumber")
        self.assertEqual(result["pending_question"]["id"], "new_chain_method_continue")

    def test_docs_excerpt_at_new_chain_method_question_stays_in_current_group(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["chain_identity"] = {
            "raw": "abcd",
            "canonical": "abcd",
            "adapter_family": "jsonrpc",
            "status": "existing_family_needs_method",
            "case": "case2",
        }
        state["active_group"] = "endpoint_process"
        state["pending_question"] = {
            "id": "new_chain_method",
            "group": "endpoint_process",
            "kind": "manual_value",
            "field": "new_chain_method",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "Get Blocks by ID. path Parameters id required query Parameters expand select"

        with patch("agent.harness.groups.resolve_action_queue") as router:
            result = process_turn(state)

        router.assert_not_called()
        self.assertEqual(result["chain_identity"]["status"], "existing_family_needs_method")
        self.assertNotIn("candidate_method", result["chain_identity"])
        self.assertIn("不像可直接验证的 RPC method 名称", "\n".join(result["visible_response"]))

    def test_partial_chain_change_requires_identity_gate_not_silent_alias(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "solana", "canonical": "solana", "status": "confirmed", "case": "known"}
        state["active_group"] = "qps_profile"
        state["pending_question"] = {
            "id": "qps_profile_confirmed",
            "group": "qps_profile",
            "kind": "yes_no",
            "field": "qps_profile_confirmed",
            "options": [{"label": "Y", "value": True}, {"label": "N", "value": False}],
        }
        state["last_user_input"] = "我要切换链到 Sola"

        with (
            patch("agent.harness.groups.resolve_action_queue") as queue,
            patch("agent.harness.groups.resolve_unknown_chain_identity") as identify,
        ):
            queue.return_value = {"actions": [{"type": "change_chain", "chain_text": "Sola", "confidence": "high"}]}
            identify.return_value = {
                "chain_exists": None,
                "canonical_chain_name": "Sola",
                "adapter_family": "unknown",
                "possible_known_chain": "solana",
                "confidence": "medium",
            }
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "chain_change_confirm")
        self.assertEqual(result["pending_question"]["kind"], "numbered_choice")
        self.assertIn("Sola", result["pending_question"]["prompt"])
        self.assertIn("solana", result["pending_question"]["prompt"])
        self.assertEqual(result["chain_identity"]["canonical"], "solana")

    def test_ambiguous_chain_turn_asks_user_to_choose_candidate(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = "我想用 fake-node 测一下，链名可能是 sola 或 solana，QPS 用 quick"

        with patch("agent.harness.groups.resolve_action_queue") as queue:
            queue.return_value = {
                "actions": [
                    {"type": "choose_target_mode", "target_mode": "fake-node", "target_mode_explicit": True, "confidence": "high"},
                    {"type": "choose_chain", "chain_text": "sola", "confidence": "high"},
                    {"type": "set_qps_mode", "qps_mode": "quick", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "chain_ambiguity_confirm")
        rendered = "\n".join(result["visible_response"])
        self.assertIn("solana", rendered)
        self.assertIn("sola", rendered)

        result["last_user_input"] = "1"
        result = process_turn(result)

        self.assertEqual(result["chain_identity"]["canonical"], "solana")
        self.assertEqual(result["qps_profile"]["mode"], "quick")

    def test_ambiguous_chain_turn_still_prompts_when_llm_selects_known_candidate(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = "我想用 fake-node 测一下，链名可能是 sola 或 solana，QPS 用 quick"

        with patch("agent.harness.groups.resolve_action_queue") as queue:
            queue.return_value = {
                "actions": [
                    {"type": "choose_target_mode", "target_mode": "fake-node", "target_mode_explicit": True, "confidence": "high"},
                    {"type": "choose_chain", "chain_text": "solana", "confidence": "high"},
                    {"type": "set_qps_mode", "qps_mode": "quick", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "chain_ambiguity_confirm")
        rendered = "\n".join(result["visible_response"])
        self.assertIn("solana", rendered)
        self.assertIn("sola", rendered)

    def test_pending_question_chain_detour_uses_llm_chain_mention_gate(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "solana", "canonical": "solana", "status": "confirmed", "case": "known"}
        state["active_group"] = "qps_profile"
        state["pending_question"] = {
            "id": "qps_profile_confirmed",
            "group": "qps_profile",
            "kind": "yes_no",
            "field": "qps_profile_confirmed",
            "options": [{"label": "Y", "value": True}, {"label": "N", "value": False}],
        }
        state["last_user_input"] = "先等一下，如果我说的其实是 Sola，不是 Solana，你应该怎么处理？"

        with (
            patch("agent.harness.groups.resolve_action_queue", return_value={"actions": [{"type": "unknown", "confidence": "low"}]}),
            patch("agent.harness.groups.resolve_intent_action", return_value={"intent": "unknown", "confidence": "low"}),
            patch("agent.harness.groups.resolve_pending_choice", return_value={"matched": False}),
            patch("agent.harness.groups.extract_chain_mention") as mention,
            patch("agent.harness.groups.resolve_unknown_chain_identity") as identify,
        ):
            mention.return_value = {"found": True, "chain_text": "Sola", "confidence": "high", "reason": "user is correcting chain"}
            identify.return_value = {
                "chain_exists": None,
                "canonical_chain_name": "Sola",
                "adapter_family": "unknown",
                "possible_known_chain": "solana",
                "confidence": "medium",
            }
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "chain_change_confirm")
        self.assertIn("Sola", result["pending_question"]["prompt"])
        self.assertIn("保持当前链", result["pending_question"]["prompt"])
        self.assertNotIn("切换到 `solana`", result["pending_question"]["prompt"])

    def test_chain_change_preserves_user_protocol_hint_for_case2(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "provider_deployment"
        state["chain_identity"] = {"raw": "solana", "canonical": "solana", "status": "confirmed", "case": "known"}
        state["pending_question"] = {
            "id": "CLOUD_REGION",
            "group": "provider_deployment",
            "kind": "manual_value",
            "field": "CLOUD_REGION",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "我需要换成 abcd，它是 EVM/jsonrpc 链"

        with patch("agent.harness.groups.resolve_action_queue") as queue:
            queue.return_value = {
                "actions": [
                    {
                        "type": "change_chain",
                        "chain_text": "abcd",
                        "adapter_family": "jsonrpc",
                        "chain_exists": True,
                        "confidence": "high",
                    }
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "chain_change_confirm")
        self.assertIn("jsonrpc", result["pending_question"]["prompt"])
        self.assertIn("endpoint/RPC", result["pending_question"]["prompt"])

    def test_single_custom_rpc_method_accepts_inline_numeric_weight(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "endpoint_process"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["custom_rpc"] = {
            "status": "method_validated_next",
            "method": "eth_chainId",
            "params": [],
            "validated_methods": [{"method": "eth_chainId", "params": []}],
        }
        state["pending_question"] = {
            "id": "custom_rpc_continue",
            "group": "endpoint_process",
            "kind": "numbered_choice",
            "field": "custom_rpc_continue",
            "options": [
                {"label": "继续添加另一个自定义 RPC method", "value": "add_another"},
                {"label": "当前 method 已够，继续配置 workload", "value": "finish"},
            ],
            "manual_input_allowed": False,
        }
        state["last_user_input"] = "够了，只用这个 method 做 mixed，权重 100"

        result = process_turn(state)

        self.assertEqual(result["rpc_mode"], "mixed")
        self.assertEqual(result["workload"]["mixed_weights"], {"eth_chainId": 100})
        self.assertTrue(result["workload"]["replace_defaults"])

    def test_compound_custom_rpc_answer_resumes_remaining_action_queue(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "endpoint_process"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["custom_rpc"] = {
            "status": "method_validated_next",
            "method": "eth_chainId",
            "params": [],
            "validated_methods": [{"method": "eth_chainId", "params": []}],
        }
        state["action_queue"] = [
            {"type": "set_qps_mode", "qps_mode": "quick", "confidence": "high", "_origin_text": "quick Grafana 不开"},
            {"type": "set_observability", "observability_mode": "disabled", "confidence": "high", "_origin_text": "quick Grafana 不开"},
        ]
        state["pending_question"] = {
            "id": "custom_rpc_continue",
            "group": "endpoint_process",
            "kind": "numbered_choice",
            "field": "custom_rpc_continue",
            "options": [
                {"label": "继续添加另一个自定义 RPC method", "value": "add_another"},
                {"label": "当前 method 已够，继续配置 workload", "value": "finish"},
            ],
            "manual_input_allowed": False,
            "resume_action_queue": True,
        }
        state["last_user_input"] = "够了，只用这个 method 做 mixed，权重 100"

        result = process_turn(state)

        self.assertEqual(result["workload"]["mixed_weights"], {"eth_chainId": 100})
        self.assertEqual(result["qps_profile"]["mode"], "quick")
        self.assertEqual(result["observability"]["mode"], "disabled")
        self.assertEqual(result["pending_question"]["id"], "qps_profile_confirm")

    def test_generated_schema_question_preserves_remaining_action_queue(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "endpoint_process"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["custom_rpc"] = {
            "status": "needs_schema_evidence",
            "method": "eth_chainId",
            "endpoint_ready": True,
        }
        state["action_queue"] = [
            {"type": "set_qps_mode", "qps_mode": "quick", "confidence": "high", "_origin_text": "quick Grafana 不开"},
            {"type": "set_observability", "observability_mode": "disabled", "confidence": "high", "_origin_text": "quick Grafana 不开"},
        ]
        state["last_user_input"] = ""

        result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "custom_rpc_schema_evidence")
        self.assertTrue(result["pending_question"].get("resume_action_queue"))

    def test_schema_answer_propagates_resume_to_next_custom_rpc_question(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "endpoint_process"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["custom_rpc"] = {
            "status": "needs_schema_evidence",
            "method": "eth_chainId",
            "endpoint": "https://example.invalid/rpc",
            "endpoint_ready": True,
        }
        state["action_queue"] = [{"type": "set_qps_mode", "qps_mode": "quick", "confidence": "high", "_origin_text": "quick"}]
        state["pending_question"] = {
            "id": "custom_rpc_schema_evidence",
            "group": "endpoint_process",
            "kind": "evidence",
            "field": "custom_rpc_schema_evidence",
            "manual_input_allowed": True,
            "resume_action_queue": True,
        }
        state["last_user_input"] = "[]"
        probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}

        with patch("agent.harness.groups.validate_rpc_endpoint", return_value=probe):
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "custom_rpc_continue")
        self.assertTrue(result["pending_question"].get("resume_action_queue"))

    def test_custom_rpc_schema_accepts_no_parameters_natural_language(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "endpoint_process"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["custom_rpc"] = {
            "status": "needs_schema_evidence",
            "method": "eth_chainId",
            "endpoint": "https://example.invalid/rpc",
            "endpoint_ready": True,
        }
        state["pending_question"] = {
            "id": "custom_rpc_schema_evidence",
            "group": "endpoint_process",
            "kind": "evidence",
            "field": "custom_rpc_schema_evidence",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "没有参数"
        probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}

        with patch("agent.harness.groups.validate_rpc_endpoint", return_value=probe):
            result = process_turn(state)

        self.assertEqual(result["custom_rpc"]["schema_draft"]["params"], [])
        self.assertEqual(result["pending_question"]["id"], "custom_rpc_schema_confirm")

    def test_start_custom_rpc_uses_original_turn_as_schema_evidence(self) -> None:
        from agent.harness.groups import _apply_queue_action
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["last_user_input"] = (
            "我要用 fake-node 测 BSC，添加自定义 RPC method eth_chainId，"
            "endpoint 是 https://example.invalid/rpc，没有参数，mixed 只跑这个 method，权重 100"
        )
        state["pending_question"] = {}
        action = {
            "type": "start_custom_rpc",
            "rpc_method": "eth_chainId",
            "rpc_endpoint": "https://example.invalid/rpc",
            "confidence": "high",
        }
        probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}

        with patch("agent.harness.groups.validate_rpc_endpoint", return_value=probe):
            result = _apply_queue_action(state, action, state["last_user_input"])

        self.assertNotIn("LOCAL_RPC_URL", result.get("confirmed_config", {}))
        self.assertEqual(result["custom_rpc"]["params"], [])
        self.assertEqual(result["custom_rpc"]["status"], "validated")
        self.assertEqual(result["rpc_mode"], "mixed")
        self.assertEqual(result["workload"]["mixed_weights"], {"eth_chainId": 100})
        self.assertTrue(result["workload"]["replace_defaults"])
        self.assertIn("自定义 RPC mixed workload 已确认", "\n".join(result["visible_response"]))

    def test_custom_rpc_validation_recovers_inline_weight_hint_after_method_is_known(self) -> None:
        from agent.harness.groups import _validate_custom_rpc_schema
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["custom_rpc"] = {
            "method": "eth_chainId",
            "endpoint": "https://example.invalid/rpc",
            "endpoint_ready": True,
        }
        state["last_user_input"] = "没有参数，mixed 只跑这个 method，权重 100，quick，Grafana 不开"
        probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}

        with patch("agent.harness.groups.validate_rpc_endpoint", return_value=probe):
            result = _validate_custom_rpc_schema(state, [])

        self.assertEqual(result["custom_rpc"]["status"], "validated")
        self.assertEqual(result["workload"]["mixed_weights"], {"eth_chainId": 100})
        self.assertTrue(result["workload"]["replace_defaults"])

    def test_inline_weight_ignores_endpoint_version_numbers(self) -> None:
        from agent.harness.groups import _custom_rpc_inline_workload_hint

        text = "endpoint 是 https://example.invalid/v1/token，没有参数，mixed 只跑这个 method，权重 100"
        hint = _custom_rpc_inline_workload_hint(text, ["eth_chainId"])

        self.assertEqual(hint["weights"], {"eth_chainId": 100})

    def test_custom_rpc_schema_confirmation_uses_original_turn_for_weights(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["custom_rpc"] = {
            "method": "eth_chainId",
            "endpoint": "https://example.invalid/rpc",
            "endpoint_ready": True,
            "source_turn_text": "没有参数，mixed 只跑这个 method，权重 100，quick，Grafana 不开",
            "schema_draft": {
                "status": "draft",
                "method": "eth_chainId",
                "params": [],
                "params_json": [],
                "evidence_kind": "jsonrpc_request",
                "transport": "jsonrpc",
            },
            "status": "schema_needs_confirmation",
        }
        state["pending_question"] = {
            "id": "custom_rpc_schema_confirm",
            "group": "endpoint_process",
            "kind": "yes_no",
            "field": "custom_rpc_schema_confirm",
            "options": [{"label": "Y", "value": True}, {"label": "N", "value": False}],
        }
        state["last_user_input"] = "Y"
        probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}

        with patch("agent.harness.groups.validate_rpc_endpoint", return_value=probe):
            result = process_turn(state)

        self.assertEqual(result["custom_rpc"]["status"], "validated")
        self.assertEqual(result["workload"]["mixed_weights"], {"eth_chainId": 100})
        self.assertIn("自定义 RPC mixed workload 已确认", "\n".join(result["visible_response"]))

    def test_case2_protocol_confirm_consumes_queued_endpoint_and_method(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        source = (
            "我想改测 Flow，它是 EVM/jsonrpc 链，endpoint 是 https://example.invalid/rpc，"
            "method 用 eth_blockNumber，没有参数"
        )
        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "flow",
            "canonical": "flow",
            "adapter_family": "jsonrpc",
            "status": "needs_identity_confirmation",
            "case": "unknown",
            "llm_resolution": {"canonical_chain_name": "flow", "adapter_family": "jsonrpc", "chain_exists": True},
        }
        state["pending_question"] = {
            "id": "unknown_chain_identity_confirm",
            "group": "chain_identity",
            "kind": "numbered_choice",
            "field": "unknown_chain_decision",
            "options": [{"label": "确认", "value": "confirm_proposed_protocol"}],
        }
        state["action_queue"] = [
            {
                "type": "start_custom_rpc",
                "rpc_method": "eth_blockNumber",
                "rpc_endpoint": "https://example.invalid/rpc",
                "_origin_text": source,
                "confidence": "medium",
            }
        ]
        state["last_user_input"] = "1"
        probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}

        with patch("agent.harness.groups.resolve_pending_choice", return_value={"matched": True, "confidence": "high", "selected_value": "confirm_proposed_protocol"}), patch("agent.harness.groups.validate_rpc_endpoint", return_value=probe):
            result = process_turn(state)

        self.assertEqual(result["endpoint_evidence"]["candidate_endpoint"], "https://example.invalid/rpc")
        self.assertIn(result["chain_identity"]["status"], {"existing_family_schema_needs_confirmation", "existing_family_method_validated_next", "existing_family_runtime_choice"})
        self.assertNotEqual(result["pending_question"].get("id"), "new_chain_endpoint")

    def test_case3_handoff_stops_instead_of_falling_back_to_chain_prompt(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "sola",
            "canonical": "sola",
            "adapter_family": "unknown",
            "status": "needs_protocol_confirmation",
            "case": "unknown",
        }
        state["pending_question"] = {
            "id": "adapter_family_confirm",
            "group": "chain_identity",
            "kind": "numbered_choice",
            "field": "adapter_family",
            "options": [{"label": "不属于以上协议族", "value": "unsupported"}],
        }
        state["last_user_input"] = "unsupported"

        result = process_turn(state)

        self.assertEqual(result["chain_identity"]["status"], "unsupported_family_handoff")
        self.assertEqual(result.get("pending_question"), {})
        self.assertIn("二次开发", "\n".join(result["visible_response"]))
        self.assertNotIn("你想测试哪条链", "\n".join(result["visible_response"]))

    def test_case3_handoff_collects_followup_evidence_without_fallback(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "WeirdP2PChain",
            "canonical": "WeirdP2PChain",
            "adapter_family": "unsupported",
            "status": "unsupported_family_handoff",
            "case": "case3",
        }
        state["secondary_handoff"] = {"status": "collecting_evidence", "evidence": []}
        state["pending_question"] = {}
        state["last_user_input"] = "protocol: weird-p2p"

        result = process_turn(state)

        self.assertEqual(result["chain_identity"]["status"], "unsupported_family_handoff")
        self.assertEqual(result["secondary_handoff"]["evidence"], ["protocol: weird-p2p"])
        self.assertEqual(result.get("pending_question"), {})
        self.assertIn("二次开发证据", "\n".join(result.get("visible_response") or []))
        self.assertNotIn("你想测试哪条链", "\n".join(result.get("visible_response") or []))

    def test_case3_handoff_generation_request_uses_collected_evidence(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["chain_identity"] = {
            "raw": "WeirdP2PChain",
            "canonical": "WeirdP2PChain",
            "adapter_family": "unsupported",
            "status": "unsupported_family_handoff",
            "case": "case3",
        }
        state["secondary_handoff"] = {"status": "collecting_evidence", "evidence": ["protocol: weird-p2p"]}
        state["last_user_input"] = "我想生成给另一个 AI 的二次开发任务文档"

        result = process_turn(state)

        self.assertEqual(len(result["secondary_handoff"]["evidence"]), 2)
        self.assertEqual(result.get("pending_question"), {})
        self.assertIn("adapter", "\n".join(result.get("visible_response") or []))
        self.assertIn("二次开发交接草案", "\n".join(result.get("visible_response") or []))
        self.assertIn("protocol: weird-p2p", "\n".join(result.get("visible_response") or []))

    def test_case3_handoff_does_not_capture_report_analysis_request(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["latest_job_id"] = "job_demo"
        state["chain_identity"] = {
            "raw": "WeirdP2PChain",
            "canonical": "WeirdP2PChain",
            "adapter_family": "unsupported",
            "status": "unsupported_family_handoff",
            "case": "case3",
        }
        state["secondary_handoff"] = {"status": "collecting_evidence", "evidence": ["protocol: weird-p2p"]}
        state["last_user_input"] = "先别管之前配置，帮我分析最近一次 job 的报告和日志"

        with patch("agent.harness.groups.resolve_action_queue") as resolver, patch("agent.harness.groups.resume_job") as resume:
            resolver.return_value = {"actions": [{"type": "analyze_report", "confidence": "high"}]}
            resume.return_value = {
                "job_id": "job_demo",
                "status": "completed",
                "run_dir": ".agent/jobs/job_demo",
                "runtime_env_file": ".agent/jobs/job_demo/runtime.env",
                "artifact_index": ".agent/jobs/job_demo/artifact_index.json",
                "next_actions": ["status", "analyze"],
            }
            result = process_turn(state)

        self.assertEqual(result["secondary_handoff"]["evidence"], ["protocol: weird-p2p"])
        self.assertIn("job_demo", "\n".join(result.get("visible_response") or []))

    def test_case3_handoff_does_not_capture_generic_test_help_request(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["chain_identity"] = {
            "raw": "WeirdP2PChain",
            "canonical": "WeirdP2PChain",
            "adapter_family": "unsupported",
            "status": "unsupported_family_handoff",
            "case": "case3",
        }
        state["secondary_handoff"] = {"status": "collecting_evidence", "evidence": ["protocol: weird-p2p"]}
        state["last_user_input"] = "我需要测试，但是我不知道可以测试什么"

        with patch("agent.harness.groups.resolve_intent_action") as resolver:
            resolver.return_value = {"intent": "greeting", "confidence": "medium"}
            result = process_turn(state)

        self.assertEqual(result["secondary_handoff"]["evidence"], ["protocol: weird-p2p"])
        self.assertNotIn("二次开发证据", "\n".join(result.get("visible_response") or []))

    def test_case3_handoff_does_not_capture_confusion_question(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["chain_identity"] = {
            "raw": "WeirdP2PChain",
            "canonical": "WeirdP2PChain",
            "adapter_family": "unsupported",
            "status": "unsupported_family_handoff",
            "case": "case3",
        }
        state["secondary_handoff"] = {"status": "collecting_evidence", "evidence": ["protocol: weird-p2p"]}
        state["last_user_input"] = "什么意思？你在讲什么"

        result = process_turn(state)

        self.assertEqual(result["secondary_handoff"]["evidence"], ["protocol: weird-p2p"])
        self.assertNotIn("已记录第", "\n".join(result.get("visible_response") or []))

    def test_pending_config_review_merges_additional_yaml_lines(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["inferred_config"] = {
            "pending_review": {
                "config_values": {"CLOUD_REGION": "asia-east1"},
                "unmapped_values": {},
                "source_format": "mixed",
                "reason": "initial line",
            }
        }
        state["pending_question"] = {
            "id": "inferred_config_review",
            "group": "provider_deployment",
            "kind": "yes_no",
            "field": "inferred_config_review",
            "options": [{"label": "Y", "value": True}, {"label": "N", "value": False}],
        }
        state["last_user_input"] = "CLOUD_ZONE: asia-east1-c"

        result = process_turn(state)

        proposal = result["inferred_config"]["pending_review"]["config_values"]
        self.assertEqual(proposal["CLOUD_REGION"], "asia-east1")
        self.assertEqual(proposal["CLOUD_ZONE"], "asia-east1-c")
        self.assertEqual(result["pending_question"]["id"], "inferred_config_review")
        self.assertIn("CLOUD_ZONE", "\n".join(result["visible_response"]))

        result["last_user_input"] = "MACHINE_TYPE=n2-standard-16"
        merged = process_turn(result)

        proposal = merged["inferred_config"]["pending_review"]["config_values"]
        self.assertEqual(proposal["CLOUD_REGION"], "asia-east1")
        self.assertEqual(proposal["CLOUD_ZONE"], "asia-east1-c")
        self.assertEqual(proposal["MACHINE_TYPE"], "n2-standard-16")
        self.assertEqual(merged["pending_question"]["id"], "inferred_config_review")
        self.assertIn("MACHINE_TYPE", "\n".join(merged["visible_response"]))

    def test_pending_config_review_rerenders_in_current_language(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["inferred_config"] = {
            "pending_review": {
                "config_values": {"CLOUD_REGION": "asia-east1"},
                "unmapped_values": {},
                "source_format": "mixed",
                "reason": "additional config fragments",
            }
        }
        state["pending_question"] = {
            "id": "inferred_config_review",
            "group": "provider_deployment",
            "kind": "yes_no",
            "field": "inferred_config_review",
            "prompt": "I inferred these candidate config values from your pasted content:",
            "options": [{"label": "Y", "value": True}, {"label": "N", "value": False}],
        }
        state["last_user_input"] = "我还想用 fake-node 测 solana"

        result = process_turn(state)
        text = "\n".join(result["visible_response"])

        self.assertIn("我从你粘贴的内容中推断出这些配置候选值", text)
        self.assertNotIn("I inferred these candidate config values", text)

    def test_freeform_traceback_collects_multiple_lines_before_analysis(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = "Traceback (most recent call last):"
        first = process_turn(state)

        self.assertEqual(first["evidence_collection"]["question"]["id"], "freeform_evidence")
        self.assertIn("多行错误/日志证据", "\n".join(first["visible_response"]))

        first["last_user_input"] = '  File "agent/terminal/repl.py", line 123, in run'
        first["language"] = "en"
        second = process_turn(first)

        self.assertEqual(second["evidence_collection"]["question"]["id"], "freeform_evidence")
        self.assertNotIn("AnyChain Benchmark Agent", "\n".join(second["visible_response"]))
        self.assertIn("已记录第 2 行证据", "\n".join(second["visible_response"]))

        second["last_user_input"] = "这个错误是什么意思？"
        third = process_turn(second)

        self.assertEqual(third.get("evidence_collection"), {})
        self.assertTrue(third["evidence_buffer"])
        self.assertIn("Traceback", third["evidence_buffer"][-1]["text"])
        self.assertIn("错误/日志证据", "\n".join(third["visible_response"]))

    def test_multiline_rpc_evidence_is_collected_before_protocol_conflict(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "abcd",
            "canonical": "abcd",
            "adapter_family": "jsonrpc",
            "status": "existing_family_needs_schema_evidence",
            "case": "case2",
            "candidate_method": "eth_blockNumber",
        }
        state["endpoint_evidence"] = {"candidate_endpoint": "http://fake-node:19000", "candidate_endpoint_ready": True}
        state["active_group"] = "endpoint_process"
        state["pending_question"] = {
            "id": "new_chain_schema_evidence",
            "group": "endpoint_process",
            "kind": "evidence",
            "field": "new_chain_schema_evidence",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "curl --location 'http://fake-node:19000' \\"

        first = process_turn(state)
        self.assertEqual(first["evidence_collection"]["question"]["id"], "new_chain_schema_evidence")
        self.assertEqual(first.get("pending_question"), {})
        self.assertIn("多行 RPC 证据", "\n".join(first["visible_response"]))

        first["last_user_input"] = '--data \'{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}\''
        second = process_turn(first)
        self.assertTrue(second["evidence_collection"]["lines"])

        second["last_user_input"] = 'response: {"jsonrpc":"2.0","id":1,"result":"0x10"}'
        with patch("agent.harness.groups.extract_rpc_schema_from_evidence") as extract:
            extract.return_value = {
                "status": "draft",
                "evidence_kind": "jsonrpc_request",
                "transport": "jsonrpc",
                "method": "eth_blockNumber",
                "params": [],
                "params_json": [],
                "response_summary": "hex block number",
                "confidence": "high",
            }
            third = process_turn(second)

        self.assertEqual(third["chain_identity"]["status"], "existing_family_schema_needs_confirmation")
        self.assertEqual(third["pending_question"]["id"], "new_chain_schema_confirm")

    def test_opening_requirements_question_returns_preparation_checklist(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = "如果我需要测试，我都需要做什么，提供什么"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "answer_opening_question", "topic": "requirements", "confidence": "high"}]}
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("测试类型", text)
        self.assertIn("CLOUD_REGION", text)
        self.assertIn("Ledger/data", text)
        self.assertIn("preflight/smoke", text)
        self.assertNotIn("已知链：acala", text)

    def test_opening_correction_question_does_not_repeat_capabilities(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = "你是否理解我的问题？"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "answer_opening_question", "topic": "correction", "confidence": "high"}]}
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("我理解", text)
        self.assertIn("准备什么", text)
        self.assertNotIn("协议族分布", text)

    def test_opening_mode_comparison_has_sync_observe_without_vegeta(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = "fake-node real-node sync-observe 有什么区别"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "answer_opening_question", "topic": "mode_comparison", "confidence": "high"}]}
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("fake-node", text)
        self.assertIn("real-node", text)
        self.assertIn("sync-observe", text)
        self.assertIn("不走 vegeta", text)

    def test_pending_accounts_answer_with_extra_question_applies_answer_then_routes_question(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "real-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["pending_question"] = {
            "id": "has_accounts_device",
            "group": "accounts_disk",
            "kind": "yes_no",
            "field": "has_accounts_device",
            "options": [{"label": "Y", "value": True}, {"label": "N", "value": False}],
        }
        state["last_user_input"] = "没有 accounts 盘，顺便告诉我 real-node 需要提供什么 endpoint"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "answer_opening_question", "topic": "requirements", "confidence": "high"}]}
            result = process_turn(state)

        self.assertIs(result["confirmed_config"]["has_accounts_device"], False)
        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("LOCAL_RPC_URL", text)
        self.assertIn("endpoint", text)
        self.assertNotIn("这个节点是否有独立的 accounts/state 磁盘", text)

    def test_pending_accounts_plain_natural_answer_only_continues_fallback(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"canonical": "solana", "adapter_family": "jsonrpc", "status": "confirmed"}
        state["confirmed_config"] = {
            "CLOUD_REGION": "asia-east1",
            "CLOUD_ZONE": "asia-east1-c",
            "MACHINE_TYPE": "n2-standard-16",
            "LEDGER_DEVICE": "vda",
            "DATA_VOL_TYPE": "hyperdisk-balanced",
            "DATA_VOL_SIZE": "926",
            "DATA_VOL_MAX_IOPS": "20000",
            "DATA_VOL_MAX_THROUGHPUT": "1000",
        }
        state["pending_question"] = {
            "id": "has_accounts_device",
            "group": "accounts_disk",
            "kind": "yes_no",
            "field": "has_accounts_device",
            "options": [{"label": "Y", "value": True}, {"label": "N", "value": False}],
        }
        state["last_user_input"] = "没有 accounts 盘"

        result = process_turn(state)

        self.assertIs(result["confirmed_config"]["has_accounts_device"], False)
        self.assertNotEqual(result.get("pending_question", {}).get("id"), "has_accounts_device")

    def test_mixed_goal_and_config_turn_preserves_chain_and_target_mode_when_llm_returns_only_config(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = (
            "我要测试 BNB real-node，region asia-east1，zone asia-east1-c，机器 n2-standard-16，"
            "ledger vda，磁盘 hyperdisk-balanced，IOPS 20000，吞吐 1000"
        )

        with (
            patch("agent.harness.groups.resolve_action_queue") as resolver,
            patch("agent.harness.groups.extract_chain_mention") as mention,
        ):
            resolver.return_value = {
                "actions": [
                    {
                        "type": "propose_config_values",
                        "config_values": {
                            "CLOUD_REGION": "asia-east1",
                            "CLOUD_ZONE": "asia-east1-c",
                            "MACHINE_TYPE": "n2-standard-16",
                            "LEDGER_DEVICE": "vda",
                        },
                        "confidence": "high",
                    }
                ]
            }
            mention.return_value = {"found": True, "chain_text": "BNB", "confidence": "high"}
            result = process_turn(state)

        self.assertEqual(result["target_mode"], "real-node")
        self.assertEqual(result["chain_identity"]["canonical"], "bsc")
        self.assertEqual(result["pending_question"]["id"], "inferred_config_review")
        self.assertIn("CLOUD_REGION", "\n".join(result.get("visible_response") or []))

    def test_opening_current_config_question_reports_state(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "real-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed"}
        state["pending_question"] = {"id": "DATA_VOL_SIZE", "group": "ledger_disk", "kind": "confirm_or_value"}
        state["last_user_input"] = "当前链和模式是什么？"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "answer_opening_question", "topic": "current_config", "confidence": "high"}]}
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("chain: `bsc`", text)
        self.assertIn("target_mode: `real-node`", text)
        self.assertIn("DATA_VOL_SIZE", text)
        self.assertNotIn("已知链：", text)

    def test_opening_recommendation_for_unsure_quick_validation(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = "我想先随便跑一下，但不知道 fake-node 和 real-node 选哪个"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "answer_opening_question", "topic": "recommendation", "confidence": "high"}]}
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("建议先用 fake-node smoke", text)
        self.assertIn("solana", text)
        self.assertEqual(result.get("target_mode"), "")


if __name__ == "__main__":
    unittest.main()

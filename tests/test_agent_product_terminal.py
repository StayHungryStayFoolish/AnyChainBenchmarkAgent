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
from terminal.language import detect_language  # noqa: E402
from terminal.pending_answers import is_structural_pending_answer  # noqa: E402
from terminal.repl import AnyChainTerminal, TerminalSession, TerminalSessionStore, _classify_input_mode, _format_environment_inference, _format_pending_answer_blockers  # noqa: E402
from discovery.environment import discover_environment  # noqa: E402
from adk_app.callbacks import after_model_callback, before_tool_callback  # noqa: E402
from adk_app.runner_bridge import ADKRunnerBridge, sanitize_adk_text  # noqa: E402
from adk_app.tools.workflow_state import propose_benchmark_profile_choice, propose_benchmark_target_mode_choice, propose_chain_identity_resolution, propose_chain_protocol_resolution, request_unsupported_chain_handoff, request_custom_rpc_handoff, propose_chain_change_confirmation, propose_chain_selection_question, propose_custom_rpc_endpoint_gate, propose_disk_device_choice, propose_opening_help_choice, propose_quick_assumed_smoke_confirmation, propose_unsupported_chain_endpoint_gate, record_group_jump as adk_record_group_jump, update_workflow_state as adk_update_workflow_state, workflow_tool_session  # noqa: E402
from adk_app.instructions import build_terminal_turn_prompt, build_workflow_branch_context  # noqa: E402
from adk_app.workflow.product_context import build_product_decision_context, validate_product_intent  # noqa: E402
from validators.config_contract import build_missing_config_questions, validate_required_config  # noqa: E402
from validators.execution_gate import validate_execution_gate  # noqa: E402
from validators.rpc_workload import default_workload, validate_rpc_workload  # noqa: E402
from workflows.transition_executor import advance_after_pending_answer, ensure_next_benchmark_setup_question, render_pending_question  # noqa: E402
from workflows.conversation_state import DEFAULT_GROUP_ORDER, PENDING_QUESTION_REGISTRY, answer_pending_question, load_workflow_state, recompute_next_blocking_group, record_group_jump, reset_workflow_state, revert_workflow_state, update_workflow_state  # noqa: E402
from workflows.group_registry import GROUP_ORDER as REGISTRY_GROUP_ORDER, is_setup_question_id, product_node_for_group  # noqa: E402


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


class StaticBridge:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, dict]] = []

    def run_text(self, text: str, state_delta=None) -> str:
        self.calls.append((text, state_delta or {}))
        return self.response


class CancellingBridge:
    def run_text(self, text: str, state_delta=None) -> str:
        raise KeyboardInterrupt()


class FakeTool:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeToolContext:
    def __init__(self, state: dict) -> None:
        self.state = state


class FakeADKState:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def to_dict(self) -> dict:
        return dict(self.payload)


class AgentDependencyTests(unittest.TestCase):
    def test_environment_inference_prefers_deployment_type_over_unknown_cloud_platform(self):
        rendered = _format_environment_inference(
            {
                "cloud": {"provider": "other", "platform": "unknown"},
                "deployment": {"type": "container"},
                "host": {},
                "network": {},
                "disks": {},
            }
        )
        self.assertIn("- deployment: container", rendered)

    def test_environment_inference_lists_hardware_network_and_disk_candidates(self):
        rendered = _format_environment_inference(
            {
                "cloud": {"provider": "gcp", "platform": "gce", "region": "asia-east1", "zone": "asia-east1-c", "machine_type": "c3"},
                "deployment": {"type": "vm"},
                "host": {"cpu_count": 8, "memory_gib": 31.5},
                "network": {"default_interface": "eth0", "interfaces": ["eth0", "ens4"]},
                "disks": {
                    "proposed_ledger_device": "sdb",
                    "proposed_accounts_device": "sdc",
                    "candidates": [
                        {"name": "sdb", "type": "disk", "size": "2T", "mountpoint": "/ledger", "label": "ledger"},
                        {"name": "sdc", "type": "disk", "size": "1T", "mountpoint": "/accounts", "label": "accounts"},
                    ],
                },
            }
        )

        self.assertIn("- CPU: 8", rendered)
        self.assertIn("- Memory: 31.5 GiB", rendered)
        self.assertIn("[1] sdb type=disk size=2T mount=/ledger label=ledger", rendered)
        self.assertIn("[2] sdc type=disk size=1T mount=/accounts label=accounts", rendered)
        self.assertIn("- Network interface candidates:", rendered)
        self.assertIn("[1] eth0 (default)", rendered)
        self.assertIn("[2] ens4", rendered)

    def test_environment_discovery_filters_zero_size_block_devices(self):
        def fake_runner(command, timeout):
            if command[:3] == ["lsblk", "-J", "-o"]:
                return 0, (
                    '{"blockdevices": ['
                    '{"name": "nbd0", "type": "disk", "size": "0B", "mountpoint": null},'
                    '{"name": "vda", "type": "disk", "size": "200G", "mountpoint": null,'
                    '"children": ['
                    '{"name": "vda1", "type": "part", "size": "200G", "mountpoint": "/data"},'
                    '{"name": "vda2", "type": "part", "size": "200G", "mountpoint": "/etc/hosts"}'
                    ']}'
                    ']}'
                ), ""
            if command[:2] == ["bash", "-lc"]:
                script = command[2]
                if script.startswith("command -v "):
                    return 1, "", ""
                if "ip route" in script:
                    return 1, "", ""
            return 1, "", ""

        env = discover_environment(command_runner=fake_runner)
        names = [item["name"] for item in env["disks"]["candidates"]]
        self.assertNotIn("nbd0", names)
        self.assertNotIn("vda2", names)
        self.assertIn("vda", names)
        self.assertIn("vda1", names)

    def test_adk_requirements_include_prompt_toolkit(self):
        requirements = (REPO / "requirements-adk.txt").read_text(encoding="utf-8")
        self.assertIn("prompt-toolkit", requirements)
        self.assertIn("litellm", requirements)

    def test_adk_text_sanitizer_is_mechanical_not_semantic_filter(self):
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
            "\n".join([
                "我会确认你的目标，然后给出下一步。",
                "不要向用户展示 prepare_benchmark_run。",
                "不要向用户展示 knowledge_search。",
                "不要向用户展示 chain_rpc_onboarding_agent。",
                "不要告诉用户转给子代理。",
                "我能帮助你完成区块链节点性能基准测试。",
            ]),
        )

    def test_adk_text_sanitizer_does_not_phrase_filter_english_text(self):
        text = "Let me check the framework capabilities.\nPlease provide LOCAL_RPC_URL."
        sanitized = sanitize_adk_text(text)
        self.assertEqual(sanitized, text)

    def test_adk_text_sanitizer_does_not_phrase_filter_chinese_text(self):
        text = "我先确认环境。\n请提供 LOCAL_RPC_URL。"
        sanitized = sanitize_adk_text(text)
        self.assertEqual(sanitized, text)

    def test_adk_text_sanitizer_does_not_hide_normal_business_text_without_internal_markers(self):
        text = "我来处理 Solana 的自定义 RPC method 添加。\n请提供 method 名称。"
        sanitized = sanitize_adk_text(text)
        self.assertEqual(sanitized, text)

    def test_adk_text_sanitizer_preserves_compliant_visible_response_envelope(self):
        text = "internal preface\nVISIBLE_RESPONSE:\n请提供 LOCAL_RPC_URL。"
        self.assertEqual(sanitize_adk_text(text), "请提供 LOCAL_RPC_URL。")

    def test_adk_text_sanitizer_does_not_normalize_onboarding_status(self):
        text = "缺少 endpoint，会标记为 needs_real_recording。"
        sanitized = sanitize_adk_text(text)
        self.assertEqual(sanitized, text)

    def test_group_registry_is_source_for_state_and_context(self):
        self.assertEqual(DEFAULT_GROUP_ORDER, REGISTRY_GROUP_ORDER)
        self.assertEqual(product_node_for_group("ledger_disk"), "environment_config")
        self.assertEqual(product_node_for_group("workload_rpc"), "rpc_workload")
        self.assertTrue(is_setup_question_id("disk_ledger_choice"))
        self.assertTrue(is_setup_question_id("benchmark_profile_confirm"))

    def test_adk_text_sanitizer_does_not_phrase_filter_inline_text(self):
        text = "可以，当前支持 36 条链。不要向用户展示 prepare_benchmark_run。请提供 LOCAL_RPC_URL。"
        self.assertEqual(
            sanitize_adk_text(text),
            text,
        )

    def test_pending_answer_blocker_messages_do_not_leak_internal_fields(self):
        blockers = [
            "answer does not match pending_question options",
            "yes requires pending_question.default_option for this choice",
            "expected yes or no for pending_question",
            "ACCOUNTS_DEVICE must be different from LEDGER_DEVICE when a separate accounts/state disk is configured. Reply back and choose N if accounts/state uses the same disk, or choose a different accounts device.",
        ]
        zh = _format_pending_answer_blockers("zh", blockers)
        en = _format_pending_answer_blockers("en", blockers)

        self.assertNotIn("pending_question", zh)
        self.assertNotIn("default_option", zh)
        self.assertNotIn("ACCOUNTS_DEVICE must be different", zh)
        self.assertNotIn("pending_question", en)
        self.assertNotIn("default_option", en)
        self.assertNotIn("ACCOUNTS_DEVICE must be different", en)
        self.assertIn("独立 accounts/state 磁盘不能", zh)
        self.assertIn("accounts/state disk must be different", en)
        self.assertIn("当前", zh)
        self.assertIn("current", en)

    def test_terminal_turn_prompt_includes_pending_question_transitions(self):
        prompt = build_terminal_turn_prompt(
            "y",
            {
                "terminal_language": "en",
                "framework_summary": {
                    "chain_count": 36,
                    "family_count": 6,
                    "unique_rpc_method_count": 109,
                    "fake_node_fixture_file_count": 206,
                },
                "workflow_state": {
                    "active_group": "qps_profile",
                    "next_blocking_group": "qps_profile",
                    "group_progress": {"qps_profile": {"status": "in_progress"}},
                    "invalidated_fields": ["benchmark_profile"],
                    "pending_question": {
                        "id": "profile",
                        "prompt": "Use quick profile?",
                        "expected_answer": "yes_no",
                        "next_on_yes": {"workflow_step": "validate_profile"},
                        "next_on_no": {"workflow_step": "adjust_profile"},
                        "next_on_manual": {"workflow_step": "manual_profile"},
                        "validation_tool": "validate_required_config",
                    }
                },
            },
        )
        self.assertIn("next_on_yes", prompt)
        self.assertIn("next_on_no", prompt)
        self.assertIn("next_on_manual", prompt)
        self.assertIn("validate_required_config", prompt)
        self.assertIn("Product decision context", prompt)
        self.assertIn("workflow_group", prompt)
        self.assertIn("allowed_intents", prompt)
        self.assertIn("allowed_tools", prompt)
        self.assertIn("valid_transitions", prompt)
        self.assertIn("guardrails", prompt)
        self.assertIn("36", prompt)
        self.assertIn("Active workflow branch context", prompt)
        self.assertIn("group_progress", prompt)
        self.assertIn("qps_profile", prompt)
        self.assertIn("benchmark_profile", prompt)
        self.assertIn("group_action_contract", prompt)
        self.assertIn("record_group_jump", prompt)
        self.assertIn("required_tools", prompt)

    def test_product_decision_context_contains_bounded_llm_inputs(self):
        context = build_product_decision_context(
            workflow_state={
                "target_mode": "fake-node",
                "chain": "solana",
                "missing_fields": ["ledger_device"],
                "active_group": "ledger_disk",
                "next_blocking_group": "ledger_disk",
                "group_progress": {"ledger_disk": {"status": "in_progress"}},
                "invalidated_fields": ["smoke_result"],
                "pending_question": {
                    "id": "disk_ledger_choice",
                    "kind": "device",
                    "branch": "environment_config",
                    "next_on_manual": {"workflow_step": "next_missing_config_question"},
                },
            },
            framework_summary={
                "chain_count": 36,
                "family_count": 6,
                "unique_rpc_method_count": 109,
                "fake_node_fixture_file_count": 206,
            },
            input_mode="normal_user_turn",
            terminal_language="zh",
        )
        self.assertEqual(context["current_node"], "environment_config")
        self.assertIn("answer_current_question", context["allowed_intents"])
        self.assertIn("discover_environment", context["allowed_tools"])
        self.assertIn("record_group_jump", context["allowed_tools"])
        self.assertIn("recompute_next_blocking_group", context["allowed_tools"])
        self.assertIn("propose_chain_change_confirmation", context["allowed_tools"])
        self.assertIn("propose_benchmark_target_mode_choice", context["allowed_tools"])
        self.assertIn("propose_unsupported_chain_endpoint_gate", context["allowed_tools"])
        self.assertIn("propose_custom_rpc_endpoint_gate", context["allowed_tools"])
        self.assertIn("group_action_contract", context)
        self.assertIn("record_group_jump", context["group_action_contract"]["non_structural_turn"])
        self.assertIn("next_on_manual", context["valid_transitions"])
        self.assertIn("ledger_device", context["missing_required_fields"])
        self.assertEqual(context["framework_facts"]["chains"], 36)
        self.assertEqual(context["workflow_group"]["active_group"], "ledger_disk")
        self.assertEqual(context["workflow_group"]["next_blocking_group"], "ledger_disk")
        self.assertEqual(context["workflow_group"]["group_progress"]["ledger_disk"], "in_progress")
        self.assertIn("smoke_result", context["workflow_group"]["invalidated_fields"])
        self.assertTrue(context["guardrails"])

    def test_product_decision_context_maps_benchmark_setup_pending_questions(self):
        context = build_product_decision_context(
            workflow_state={
                "active_workflow": "benchmark_setup",
                "pending_question": {
                    "id": "data_vol_max_iops",
                    "kind": "manual_value",
                    "branch": "benchmark_setup",
                    "next_on_manual": {"workflow_step": "next_missing_config_question"},
                },
            },
            framework_summary={},
            input_mode="normal_user_turn",
            terminal_language="en",
        )
        self.assertEqual(context["current_node"], "environment_config")
        self.assertIn("build_missing_config_questions", context["allowed_tools"])
        self.assertIn("validate_required_config", context["allowed_tools"])

        prompt = build_terminal_turn_prompt(
            "我想手动输入这个值",
            {
                "terminal_language": "zh",
                "workflow_state": {
                    "active_workflow": "benchmark_setup",
                    "pending_question": {
                        "id": "data_vol_max_iops",
                        "kind": "manual_value",
                        "branch": "benchmark_setup",
                        "next_on_manual": {"workflow_step": "next_missing_config_question"},
                    },
                },
            },
        )
        self.assertIn("- branch: environment_config", prompt)
        self.assertIn("build_missing_config_questions", prompt)

    def test_product_decision_context_maps_registered_pending_questions_to_domain_tools(self):
        expected = {
            "dependency_install": ("dependency_setup", "audit_dependencies"),
            "target_mode": ("target_selection", "validate_required_config"),
            "quick_assumed_smoke_confirm": ("target_selection", "validate_required_config"),
            "chain_selection": ("chain_selection", "validate_chain_template"),
            "chain_identity_resolution": ("chain_selection", "validate_chain_template"),
            "chain_protocol_resolution": ("unsupported_chain", "validate_rpc_endpoint"),
            "real_node_local_rpc_url": ("real_node_endpoint", "validate_rpc_endpoint"),
            "cloud_region": ("environment_config", "build_missing_config_questions"),
            "volume_baseline_confirm": ("environment_config", "validate_required_config"),
            "network_interface_confirm": ("environment_config", "validate_required_config"),
            "process_names_confirm": ("environment_config", "validate_required_config"),
            "rpc_mode_choice": ("rpc_workload", "validate_rpc_workload"),
            "default_workload_confirm": ("rpc_workload", "validate_rpc_workload"),
            "workload_customization_choice": ("rpc_workload", "validate_rpc_workload"),
            "mixed_weights_confirm": ("rpc_workload", "validate_rpc_workload"),
            "benchmark_profile_adjust_item": ("rpc_workload", "validate_rpc_workload"),
            "observability_ports_confirm": ("observability", "validate_required_config"),
            "custom_rpc_endpoint_gate": ("custom_rpc", "validate_rpc_endpoint"),
            "unsupported_chain_endpoint_gate": ("unsupported_chain", "validate_rpc_endpoint"),
            "smoke_run_confirm": ("preflight_smoke", "prepare_benchmark_run"),
            "real_benchmark_submit_confirm": ("real_execution", "validate_execution_gate"),
            "job_follow_logs": ("job_resume", "tail_job_log"),
            "apply_pasted_evidence": ("evidence_repair", "load_workflow_state"),
        }
        benchmark_setup_ids = {
            "cloud_region",
            "volume_baseline_confirm",
            "network_interface_confirm",
            "process_names_confirm",
            "rpc_mode_choice",
            "default_workload_confirm",
            "workload_customization_choice",
            "mixed_weights_confirm",
            "benchmark_profile_adjust_item",
            "observability_ports_confirm",
        }
        for qid, (node, tool) in expected.items():
            with self.subTest(qid=qid):
                self.assertIn(qid, PENDING_QUESTION_REGISTRY)
                kind = PENDING_QUESTION_REGISTRY[qid]["kind"]
                branch = {
                    "dependency_install": "dependency_setup",
                    "target_mode": "target_selection",
                    "quick_assumed_smoke_confirm": "target_selection",
                    "chain_selection": "chain_selection",
                    "chain_identity_resolution": "chain_selection",
                    "chain_protocol_resolution": "unsupported_chain",
                    "real_node_local_rpc_url": "real_node_endpoint",
                    "custom_rpc_endpoint_gate": "custom_rpc",
                    "unsupported_chain_endpoint_gate": "unsupported_chain",
                    "smoke_run_confirm": "preflight_smoke",
                    "real_benchmark_submit_confirm": "real_execution",
                    "job_follow_logs": "job_resume",
                    "apply_pasted_evidence": "evidence_repair",
                }.get(qid, "benchmark_setup" if qid in benchmark_setup_ids else "")
                context = build_product_decision_context(
                    workflow_state={
                        "active_workflow": "benchmark_setup",
                        "pending_question": {
                            "id": qid,
                            "kind": kind,
                            "branch": branch,
                            "prompt": "Prompt",
                            "next_on_manual": {"workflow_step": "next_missing_config_question"},
                        },
                    },
                    framework_summary={},
                    input_mode="normal_user_turn",
                    terminal_language="en",
                )
                branch_context = build_workflow_branch_context({
                    "active_workflow": "benchmark_setup",
                    "pending_question": {
                        "id": qid,
                        "kind": kind,
                        "branch": branch,
                        "prompt": "Prompt",
                    },
                })
                self.assertEqual(context["current_node"], node)
                self.assertIn(tool, context["allowed_tools"])
                self.assertNotEqual(branch_context["branch"], "benchmark_setup")

    def test_workflow_tool_pending_question_ids_are_registered(self):
        emitted_ids = {
            "opening_help_choice",
            "target_mode",
            "chain_selection",
            "chain_identity_resolution",
            "chain_protocol_resolution",
            "confirm_chain_change",
            "quick_assumed_smoke_confirm",
            "real_node_local_rpc_url",
            "custom_rpc_endpoint_gate",
            "unsupported_chain_endpoint_gate",
            "benchmark_profile_choice",
            "proceed_discovery",
            "workload_customization_choice",
        }
        for qid in sorted(emitted_ids):
            with self.subTest(qid=qid):
                self.assertIn(qid, PENDING_QUESTION_REGISTRY)

    def test_product_intent_validator_blocks_invalid_tool_and_transition(self):
        context = {
            "pending_question": {"id": "disk_ledger_choice"},
            "allowed_tools": ["discover_environment"],
            "valid_transitions": {"default": "next_missing_config_question"},
        }
        errors = validate_product_intent(
            {
                "intent": "answer_current_question",
                "confidence": "high",
                "applies_to_pending_question": True,
                "state_patch": {},
                "requested_tool": "submit_benchmark_job",
                "next_node": "detached_job_submit",
                "requires_user_confirmation": False,
                "user_message": "ok",
            },
            context,
        )
        self.assertIn("requested_tool is not allowed", "; ".join(errors))
        self.assertIn("next_node is not valid", "; ".join(errors))

    def test_root_prompt_forbids_model_memory_for_workload_and_ports(self):
        from adk_app.instructions import ROOT_INSTRUCTION  # noqa: E402

        self.assertIn("load_default_workload", ROOT_INSTRUCTION)
        self.assertIn("EXPORTER_PORT=9108", ROOT_INSTRUCTION)
        self.assertIn("PROMETHEUS_PORT=9091", ROOT_INSTRUCTION)
        self.assertIn("GRAFANA_PORT=3001", ROOT_INSTRUCTION)
        self.assertIn("no endpoint but want a handoff", ROOT_INSTRUCTION)
        self.assertIn("Use the stated family", ROOT_INSTRUCTION)
        self.assertIn("as a hypothesis", ROOT_INSTRUCTION)

    def test_runner_bridge_does_not_expose_visible_prompt_shape_guards(self):
        self.assertFalse(hasattr(ADKRunnerBridge, "_needs_pending_question_repair"))
        self.assertFalse(hasattr(ADKRunnerBridge, "_ensure_visible_prompt_has_pending_question"))

    def test_runner_bridge_retries_transient_model_call_once_without_fallback(self):
        bridge = object.__new__(ADKRunnerBridge)
        bridge.turn_timeout_seconds = 10
        calls = {"count": 0}

        def flaky(_kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("transient")
            return "ok"

        bridge._run_once_with_timeout = flaky  # type: ignore[attr-defined]

        self.assertEqual(bridge._run_once_with_retry({"new_message": "x"}), "ok")
        self.assertEqual(calls["count"], 2)

    def test_branch_context_uses_active_workflow_state(self):
        context = build_workflow_branch_context(
            {
                "workflow_step": "custom_rpc_endpoint_required",
                "pending_question": {"id": "custom_rpc_endpoint_required", "kind": "url"},
                "blockers": ["endpoint missing"],
                "allowed_next_actions": ["tool:validate_rpc_endpoint"],
            }
        )
        self.assertEqual(context["branch"], "custom_rpc")
        self.assertIn("validate_rpc_endpoint", context["required_tools"])
        self.assertEqual(context["blockers"], ["endpoint missing"])

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

    def test_terminal_applies_pending_short_answer_before_adk(self):
        session_id = "unit-terminal-pending-answer"
        reset_workflow_state(session_id=session_id)
        update_workflow_state(
            {
                "pending_question": {
                    "id": "target_mode",
                    "kind": "numbered_choice",
                    "prompt": "Choose target mode",
                    "field": "target_mode",
                    "options": [
                        {"id": "1", "label": "fake-node", "value": "fake-node"},
                        {"id": "2", "label": "real-node", "value": "real-node"},
                    ],
                }
            },
            session_id=session_id,
        )
        fake_bridge = FakeBridge()
        io = CapturingIO()
        app = AnyChainTerminal(
            state=TerminalSession(language="en"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=io,
            bridge_factory=lambda: fake_bridge,
            session_id=session_id,
        )
        app._adk_available = True
        app.handle_user_text("1")

        state = load_workflow_state(session_id=session_id)
        self.assertEqual(state["target_mode"], "fake-node")
        self.assertEqual(state["pending_question"]["id"], "cloud_region")
        self.assertEqual(fake_bridge.calls, [])
        self.assertTrue(any("CLOUD_REGION" in item or "region" in item.lower() for item in io.messages))

    def test_workflow_pending_answer_wins_over_stale_terminal_install_confirmation(self):
        session_id = "unit-terminal-stale-install-state"
        reset_workflow_state(session_id=session_id)
        update_workflow_state(
            {
                "pending_question": {
                    "id": "disk_accounts_exists",
                    "kind": "yes_no",
                    "prompt": "Do you have an accounts disk?",
                    "field": "has_accounts_device",
                    "workflow_step": "confirm_accounts_exists",
                    "next_on_yes": {"workflow_step": "confirm_accounts_device"},
                    "next_on_no": {"workflow_step": "confirm_data_network"},
                }
            },
            session_id=session_id,
        )
        io = CapturingIO()
        app = AnyChainTerminal(
            state=TerminalSession(language="en", current_question_id="install_agent_runtime"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=io,
            bridge_factory=lambda: FakeBridge(),
            session_id=session_id,
        )
        app._adk_available = True

        with patch.object(app, "_install_agent_runtime") as install_runtime:
            app.handle_user_text("Y")

        state = load_workflow_state(session_id=session_id)
        self.assertEqual(state["workflow_step"], "confirm_accounts_device")
        self.assertEqual(state["confirmed_config"]["has_accounts_device"], True)
        install_runtime.assert_not_called()

    def test_terminal_runs_config_ready_smoke_confirmation_deterministically(self):
        session_id = "unit-terminal-smoke-confirm"
        reset_workflow_state(session_id=session_id)
        update_workflow_state(
            {
                "active_workflow": "benchmark_setup",
                "target_mode": "fake-node",
                "chain": "solana",
                "rpc_mode": "single",
                "confirmed_config": {
                    "chain": "solana",
                    "rpc_mode": "single",
                    "use_fake_node": True,
                    "cloud_region": "asia-east1",
                    "cloud_zone": "asia-east1-c",
                    "machine_type": "test-machine",
                    "ledger_device": "vda",
                    "data_vol_type": "ssd",
                    "data_vol_size": "926",
                    "data_vol_max_iops": "3000",
                    "data_vol_max_throughput": "125",
                    "has_accounts_device": False,
                    "network_interface": "eth0",
                    "network_max_bandwidth_gbps": "10",
                    "blockchain_process_names": "solana-validator",
                    "benchmark_mode_confirmed": "quick",
                    "qps_profile_confirmed": True,
                    "chain_template_reviewed": True,
                    "rpc_workload_confirmed": True,
                    "rpc_param_samples_confirmed": True,
                    "observability_choice_confirmed": "disabled",
                },
                "pending_question": {
                    "id": "observability_mode_choice",
                    "kind": "numbered_choice",
                    "field": "observability_choice_confirmed",
                    "prompt": "Choose observability",
                    "options": [
                        {
                            "id": "1",
                            "value": "disabled",
                            "label": "disabled",
                            "state_patch": {
                                "observability": {"mode": "disabled", "enabled": False},
                                "confirmed_config": {"OBSERVABILITY_STACK_MODE": "disabled"},
                            },
                        }
                    ],
                },
            },
            session_id=session_id,
        )
        fake_bridge = FakeBridge()
        io = CapturingIO()
        app = AnyChainTerminal(
            state=TerminalSession(language="zh"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=io,
            bridge_factory=lambda: fake_bridge,
            session_id=session_id,
        )
        app._adk_available = True

        app.handle_user_text("1")
        state = load_workflow_state(session_id=session_id)
        self.assertEqual(state["pending_question"]["id"], "smoke_run_confirm")
        self.assertIn("preflight", io.messages[-1])

        with (
            patch("adk_app.tools.planning.prepare_benchmark_run") as prepare,
            patch("adk_app.tools.actions.run_fake_node_smoke_benchmark") as run_smoke,
        ):
            prepare.return_value = {
                "status": "ok",
                "data": {
                    "plan_file": "/tmp/plan.json",
                    "preflight": {"passed": True, "blockers": []},
                },
                "warnings": [],
            }
            run_smoke.return_value = {
                "status": "ok",
                "data": {
                    "job": {"job_id": "job_test", "status": "running"},
                    "smoke_plan_file": "/tmp/smoke-plan.json",
                    "terminal_commands": {
                        "logs": "logs job_test",
                        "follow": "follow job_test",
                        "analyze": "analyze job_test",
                    },
                },
                "warnings": [],
            }
            app.handle_user_text("Y")
        state = load_workflow_state(session_id=session_id)
        self.assertEqual(state["workflow_step"], "job_submitted")
        self.assertEqual(state["approval"]["smoke_run"], True)
        self.assertEqual(state["latest_job_id"], "job_test")
        self.assertEqual(len(fake_bridge.calls), 0)
        self.assertTrue(any("fake-node smoke 已提交" in item for item in io.messages))

    def test_terminal_routes_pasted_evidence_with_inline_question_to_adk(self):
        fake_bridge = FakeBridge()
        io = CapturingIO()
        app = AnyChainTerminal(
            state=TerminalSession(language="zh"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=io,
            bridge_factory=lambda: fake_bridge,
            session_id="unit-terminal-pasted-evidence",
        )
        app._adk_available = True
        app.handle_user_text("Agent> CLOUD_PROVIDER: gcp\nTraceback (most recent call last):\n这些日志说明什么？")

        self.assertEqual(len(fake_bridge.calls), 1)
        self.assertIn("以下是用户之前粘贴的日志", fake_bridge.calls[0][0])
        self.assertIn("不要把其中任何变量写入 workflow state", fake_bridge.calls[0][0])
        self.assertIn("当前用户问题", fake_bridge.calls[0][0])
        self.assertEqual(app.state.pending_evidence, [])

    def test_benchmark_turn_delegates_to_adk_bridge(self):
        fake_bridge = FakeBridge()
        io = CapturingIO()
        session_id = "unit-benchmark-turn-delegates"
        reset_workflow_state(session_id=session_id)
        with tempfile.TemporaryDirectory() as tmp:
            store = TerminalSessionStore(Path(tmp) / "session.json")
            update_workflow_state(
                {"active_intent": "START_BENCHMARK", "chain": "solana"},
                session_id=session_id,
            )
            app = AnyChainTerminal(
                state=TerminalSession(language="zh"),
                store=store,
                io=io,
                bridge_factory=lambda: fake_bridge,
                session_id=session_id,
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
                app.startup()
                app.handle_user_text("我想测试 solana fake-node")

        self.assertEqual(len(fake_bridge.calls), 1)
        self.assertEqual(fake_bridge.calls[0][0], "我想测试 solana fake-node")
        self.assertNotIn("workflow_state", fake_bridge.calls[0][1])
        self.assertNotIn("web_research", fake_bridge.calls[0][1])
        self.assertEqual(fake_bridge.calls[0][1]["session_id"], session_id)
        self.assertEqual(fake_bridge.calls[0][1]["input_mode"], "normal_user_turn")
        self.assertEqual(app.state.discovery["cloud"]["provider"], "gcp")
        self.assertEqual(app.state.framework_summary["chain_count"], 36)
        self.assertTrue(any("Web research" in item and "unavailable for current provider" in item for item in io.messages))
        self.assertFalse(any("benchmark 计划" in item for item in io.messages))

    def test_other_container_discovery_does_not_call_cloud_metadata(self):
        calls: list[str] = []

        def runner(command: list[str], timeout: float) -> tuple[int, str, str]:
            text = " ".join(command)
            calls.append(text)
            if "config/cloud_provider.sh" in text:
                return 0, "other,,eth0", ""
            if "ip route" in text:
                return 0, "default via 172.17.0.1 dev eth0\n", ""
            if command[:1] == ["lsblk"]:
                return 0, '{"blockdevices":[{"name":"vda","type":"disk","size":"100G","mountpoint":"","fstype":"","label":""}]}', ""
            if "command -v" in text:
                tool = text.rsplit(" ", 1)[-1]
                return 0, f"/usr/bin/{tool}\n", ""
            return 1, "", ""

        with patch("discovery.environment._discover_container", return_value={"detected": True, "cgroup_hint": "docker"}):
            env = discover_environment(command_runner=runner)

        self.assertEqual(env["cloud"]["provider"], "other")
        self.assertEqual(env["deployment"]["type"], "container")
        self.assertFalse(any("metadata.google.internal" in item for item in calls))
        self.assertFalse(any("169.254.169.254" in item for item in calls))

    def test_runner_bridge_loads_workflow_state_by_session_id(self):
        bridge = object.__new__(ADKRunnerBridge)
        bridge.session_id = "terminal-session"
        with patch("adk_app.runner_bridge.load_workflow_state") as load_state:
            load_state.return_value = {"chain": "solana", "pending_question": {}}
            turn_state = bridge._turn_state({
                "terminal_language": "zh",
                "input_mode": "normal_user_turn",
                "session_id": "terminal-session",
            })
        self.assertEqual(turn_state["workflow_state"]["chain"], "solana")
        self.assertEqual(turn_state["terminal_language"], "zh")
        load_state.assert_called_once_with(session_id="terminal-session")

    def test_runner_bridge_does_not_generate_terminal_pending_questions(self):
        self.assertFalse(hasattr(ADKRunnerBridge, "_needs_pending_question_repair"))
        self.assertFalse(hasattr(ADKRunnerBridge, "_ensure_visible_prompt_has_pending_question"))
        self.assertNotIn("terminal_numbered_followup", PENDING_QUESTION_REGISTRY)
        self.assertNotIn("terminal_yes_no_followup", PENDING_QUESTION_REGISTRY)
        self.assertNotIn("terminal_free_text_followup", PENDING_QUESTION_REGISTRY)

    def test_target_mode_choice_locally_advances_to_chain_selection_without_llm(self):
        fake_bridge = FakeBridge()
        io = CapturingIO()
        session_id = "unit-target-mode-local-advance"
        reset_workflow_state(session_id=session_id)
        update_workflow_state(
            {
                "active_intent": "benchmark",
                "active_workflow": "benchmark_setup",
                "pending_question": {
                    "id": "target_mode",
                    "kind": "numbered_choice",
                    "prompt": "请选择目标模式",
                    "field": "target_mode",
                    "options": [
                        {
                            "id": "1",
                            "label": "fake-node",
                            "value": "fake-node",
                            "state_patch": {"target_mode": "fake-node", "workflow_step": "chain_selection"},
                            "transition": {"workflow_step": "chain_selection", "next_question_id": "chain_selection"},
                        },
                        {
                            "id": "2",
                            "label": "real-node",
                            "value": "real-node",
                            "state_patch": {"target_mode": "real-node", "workflow_step": "chain_selection"},
                            "transition": {"workflow_step": "chain_selection", "next_question_id": "chain_selection"},
                        },
                    ],
                },
            },
            session_id=session_id,
        )
        app = AnyChainTerminal(
            state=TerminalSession(language="zh"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=io,
            bridge_factory=lambda: fake_bridge,
            session_id=session_id,
        )
        app._adk_available = True
        app.handle_user_text("1")

        state = load_workflow_state(session_id=session_id)
        self.assertEqual(fake_bridge.calls, [])
        self.assertEqual(state["target_mode"], "fake-node")
        self.assertEqual(state["pending_question"]["id"], "chain_selection")
        self.assertTrue(any("你想测试哪条链" in item for item in io.messages))

    def test_opening_menu_fake_node_choice_locally_enters_group_setup_without_llm(self):
        fake_bridge = FakeBridge()
        io = CapturingIO()
        session_id = "unit-opening-menu-local-advance"
        reset_workflow_state(session_id=session_id)
        with workflow_tool_session(session_id):
            propose_opening_help_choice(language="zh", latest_job_id="")
        app = AnyChainTerminal(
            state=TerminalSession(language="zh"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=io,
            bridge_factory=lambda: fake_bridge,
            session_id=session_id,
        )
        app._adk_available = True
        app.handle_user_text("1")

        state = load_workflow_state(session_id=session_id)
        self.assertEqual(fake_bridge.calls, [])
        self.assertEqual(state["target_mode"], "fake-node")
        self.assertEqual(state["pending_question"]["id"], "chain_selection")
        self.assertTrue(any("你想测试哪条链" in item for item in io.messages))

    def test_opening_menu_english_fake_node_choice_must_ask_chain_before_environment(self):
        fake_bridge = FakeBridge()
        io = CapturingIO()
        session_id = "unit-opening-menu-en-chain-first"
        reset_workflow_state(session_id=session_id)
        with workflow_tool_session(session_id):
            propose_opening_help_choice(language="en", latest_job_id="")
        app = AnyChainTerminal(
            state=TerminalSession(language="en"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=io,
            bridge_factory=lambda: fake_bridge,
            session_id=session_id,
        )
        app._adk_available = True
        app.handle_user_text("1")

        state = load_workflow_state(session_id=session_id)
        self.assertEqual(fake_bridge.calls, [])
        self.assertEqual(state["target_mode"], "fake-node")
        self.assertEqual(state["chain"], "")
        self.assertEqual(state["pending_question"]["id"], "chain_selection")
        joined = "\n".join(io.messages)
        self.assertIn("Which chain do you want to benchmark?", joined)
        self.assertNotIn("Confirm CLOUD_REGION", joined)
        self.assertNotIn("solana single RPC workload", joined)

    def test_opening_menu_without_latest_job_does_not_offer_latest_job(self):
        session_id = "unit-opening-menu-no-latest-job"
        reset_workflow_state(session_id=session_id)
        with workflow_tool_session(session_id):
            payload = propose_opening_help_choice(language="zh", latest_job_id="")

        prompt = payload["data"]["prompt"]
        self.assertIn("当前没有历史任务可分析", prompt)
        self.assertIn("1. 启动 fake-node", prompt)
        self.assertNotIn("查看或分析最近一次任务", prompt)

    def test_ascii_greeting_switches_response_language_to_english(self):
        self.assertEqual(detect_language("Hi", "zh"), "en")
        self.assertEqual(detect_language("hello", "zh"), "en")
        self.assertEqual(detect_language("你好", "en"), "zh")
        self.assertEqual(detect_language("sdb", "zh"), "zh")
        self.assertEqual(detect_language("quick", "zh"), "zh")

    def test_chain_answer_locally_advances_to_disk_question_without_llm(self):
        fake_bridge = FakeBridge()
        io = CapturingIO()
        session_id = "unit-chain-local-advance"
        reset_workflow_state(session_id=session_id)
        update_workflow_state(
            {
                "active_intent": "benchmark",
                "active_workflow": "benchmark_setup",
                "target_mode": "fake-node",
                "pending_question": {
                    "id": "chain_selection",
                    "kind": "chain",
                    "prompt": "Which chain?",
                    "field": "chain",
                    "known_chains": ["solana", "ethereum", "bsc"],
                    "chain_aliases": {"bnb": "bsc", "eth": "ethereum"},
                    "next_on_manual": {"workflow_step": "validate_chain_template", "tool": "validate_chain_template"},
                },
            },
            session_id=session_id,
        )
        app = AnyChainTerminal(
            state=TerminalSession(language="en"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=io,
            bridge_factory=lambda: fake_bridge,
            session_id=session_id,
        )
        app._adk_available = True
        app.handle_user_text("solana")

        state = load_workflow_state(session_id=session_id)
        self.assertEqual(fake_bridge.calls, [])
        self.assertEqual(state["chain"], "solana")
        self.assertEqual(state["pending_question"]["id"], "cloud_region")
        self.assertTrue(any("CLOUD_REGION" in item or "region" in item.lower() for item in io.messages))

    def test_partial_chain_token_is_preserved_for_adk_identity_resolution(self):
        fake_bridge = StaticBridge("Which exact chain did you mean: solana, or a new unsupported chain?")
        io = CapturingIO()
        session_id = "unit-partial-chain-token-not-local"
        reset_workflow_state(session_id=session_id)
        update_workflow_state(
            {
                "active_intent": "benchmark",
                "active_workflow": "benchmark_setup",
                "target_mode": "fake-node",
                "pending_question": {
                    "id": "chain_selection",
                    "kind": "chain",
                    "prompt": "Which chain?",
                    "field": "chain",
                    "known_chains": ["solana", "ethereum", "bsc"],
                    "chain_aliases": {"bnb": "bsc", "eth": "ethereum"},
                    "next_on_manual": {"workflow_step": "validate_chain_template", "tool": "validate_chain_template"},
                },
            },
            session_id=session_id,
        )
        app = AnyChainTerminal(
            state=TerminalSession(language="en"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=io,
            bridge_factory=lambda: fake_bridge,
            session_id=session_id,
        )
        app._adk_available = True
        app.handle_user_text("sola")

        state = load_workflow_state(session_id=session_id)
        self.assertEqual(len(fake_bridge.calls), 1)
        self.assertEqual(state.get("chain"), "")
        self.assertEqual(state["last_user_change"]["type"], "unsupported_chain_candidate")
        self.assertEqual(state["last_user_change"]["raw"], "sola")
        self.assertEqual(state["pending_question"]["id"], "chain_selection")
        self.assertFalse(any("ledger/data disk" in item.lower() for item in io.messages))

    def test_full_unknown_chain_name_is_preserved_for_adk_identity_resolution(self):
        fake_bridge = StaticBridge("Confirm whether BNB Greenfield is the intended chain before any endpoint setup.")
        io = CapturingIO()
        session_id = "unit-full-unknown-chain-enters-identity-first"
        reset_workflow_state(session_id=session_id)
        update_workflow_state(
            {
                "active_intent": "benchmark",
                "active_workflow": "benchmark_setup",
                "target_mode": "fake-node",
                "pending_question": {
                    "id": "chain_selection",
                    "kind": "chain",
                    "prompt": "Which chain?",
                    "field": "chain",
                    "known_chains": ["solana", "ethereum", "bsc"],
                    "chain_aliases": {"bnb": "bsc", "eth": "ethereum"},
                    "next_on_manual": {"workflow_step": "validate_chain_template", "tool": "validate_chain_template"},
                },
            },
            session_id=session_id,
        )
        app = AnyChainTerminal(
            state=TerminalSession(language="en"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=io,
            bridge_factory=lambda: fake_bridge,
            session_id=session_id,
        )
        app._adk_available = True
        app.handle_user_text("bnb greenfield")

        state = load_workflow_state(session_id=session_id)
        self.assertEqual(len(fake_bridge.calls), 1)
        self.assertEqual(state.get("chain"), "")
        self.assertEqual(state["last_user_change"]["type"], "unsupported_chain_candidate")
        self.assertEqual(state["last_user_change"]["raw"], "bnb greenfield")
        self.assertEqual(state["pending_question"]["id"], "chain_selection")
        self.assertNotEqual(state["pending_question"].get("id"), "unsupported_chain_endpoint_gate")

    def test_exact_short_chain_aliases_are_supported_for_chain_question(self):
        for raw, expected in (("eth", "ethereum"), ("bnb", "bsc"), ("btc", "bitcoin")):
            with self.subTest(raw=raw):
                fake_bridge = FakeBridge()
                io = CapturingIO()
                session_id = f"unit-chain-alias-{raw}"
                reset_workflow_state(session_id=session_id)
                update_workflow_state(
                    {
                        "active_intent": "benchmark",
                        "active_workflow": "benchmark_setup",
                        "target_mode": "fake-node",
                        "pending_question": {
                            "id": "chain_selection",
                            "kind": "chain",
                            "prompt": "Which chain?",
                            "field": "chain",
                            "known_chains": ["ethereum", "bsc", "bitcoin"],
                            "chain_aliases": {"eth": "ethereum", "bnb": "bsc", "btc": "bitcoin"},
                            "next_on_manual": {"workflow_step": "validate_chain_template", "tool": "validate_chain_template"},
                        },
                    },
                    session_id=session_id,
                )
                app = AnyChainTerminal(
                    state=TerminalSession(language="en"),
                    store=TerminalSessionStore(Path("/tmp/unused-session.json")),
                    io=io,
                    bridge_factory=lambda: fake_bridge,
                    session_id=session_id,
                )
                app._adk_available = True
                app.handle_user_text(raw)

                state = load_workflow_state(session_id=session_id)
                self.assertEqual(fake_bridge.calls, [])
                self.assertEqual(state["chain"], expected)

    def test_free_text_chain_mentions_go_to_adk_not_terminal_alias_heuristics(self):
        for raw in ("eth", "bnb", "btc", "test eth fake-node", "test bnb greenfield fake-node", "sola"):
            with self.subTest(raw=raw):
                fake_bridge = FakeBridge()
                io = CapturingIO()
                session_id = f"unit-free-text-chain-{raw.replace(' ', '-')}"
                reset_workflow_state(session_id=session_id)
                app = AnyChainTerminal(
                    state=TerminalSession(language="en"),
                    store=TerminalSessionStore(Path("/tmp/unused-session.json")),
                    io=io,
                    bridge_factory=lambda: fake_bridge,
                    session_id=session_id,
                )
                app._adk_available = True
                app.handle_user_text(raw)

                state = load_workflow_state(session_id=session_id)
                self.assertEqual(len(fake_bridge.calls), 1)
                self.assertEqual(state.get("chain"), "")
                self.assertFalse(state.get("confirmed_config", {}).get("BLOCKCHAIN_NODE"))

    def test_named_profile_choice_is_local_structural_answer(self):
        fake_bridge = FakeBridge()
        io = CapturingIO()
        session_id = "unit-profile-name-local-answer"
        reset_workflow_state(session_id=session_id)
        update_workflow_state(
            {
                "active_intent": "benchmark",
                "active_workflow": "benchmark_setup",
                "target_mode": "fake-node",
                "chain": "solana",
                "pending_question": {
                    "id": "benchmark_profile_choice",
                    "kind": "numbered_choice",
                    "prompt": "Choose benchmark mode",
                    "field": "benchmark_mode_confirmed",
                    "options": [
                        {"id": "1", "value": "quick", "label": "quick", "state_patch": {"benchmark_profile": {"mode": "quick"}}},
                        {"id": "2", "value": "standard", "label": "standard", "state_patch": {"benchmark_profile": {"mode": "standard"}}},
                    ],
                },
            },
            session_id=session_id,
        )
        app = AnyChainTerminal(
            state=TerminalSession(language="en"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=io,
            bridge_factory=lambda: fake_bridge,
            session_id=session_id,
        )
        app._adk_available = True
        app.handle_user_text("quick")

        state = load_workflow_state(session_id=session_id)
        self.assertEqual(fake_bridge.calls, [])
        self.assertEqual(state["benchmark_profile"]["mode"], "quick")

    def test_manual_config_value_is_local_structural_answer(self):
        fake_bridge = FakeBridge()
        io = CapturingIO()
        session_id = "unit-manual-config-local-answer"
        reset_workflow_state(session_id=session_id)
        update_workflow_state(
            {
                "active_intent": "benchmark",
                "active_workflow": "benchmark_setup",
                "target_mode": "fake-node",
                "chain": "solana",
                "pending_question": {
                    "id": "cloud_region",
                    "kind": "manual_value",
                    "prompt": "Confirm CLOUD_REGION",
                    "field": "CLOUD_REGION",
                },
            },
            session_id=session_id,
        )
        app = AnyChainTerminal(
            state=TerminalSession(language="en"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=io,
            bridge_factory=lambda: fake_bridge,
            session_id=session_id,
        )
        app._adk_available = True
        app.handle_user_text("asia-east1")

        state = load_workflow_state(session_id=session_id)
        self.assertEqual(fake_bridge.calls, [])
        self.assertEqual(state["confirmed_config"]["CLOUD_REGION"], "asia-east1")

    def test_manual_config_yes_without_current_value_is_rejected_not_stored(self):
        fake_bridge = FakeBridge()
        io = CapturingIO()
        session_id = "unit-manual-config-yes-without-current-value"
        reset_workflow_state(session_id=session_id)
        update_workflow_state(
            {
                "active_intent": "benchmark",
                "active_workflow": "benchmark_setup",
                "target_mode": "fake-node",
                "chain": "solana",
                "pending_question": {
                    "id": "cloud_region",
                    "kind": "manual_value",
                    "prompt": "Confirm CLOUD_REGION",
                    "field": "CLOUD_REGION",
                    "current_value": "",
                },
            },
            session_id=session_id,
        )
        app = AnyChainTerminal(
            state=TerminalSession(language="en"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=io,
            bridge_factory=lambda: fake_bridge,
            session_id=session_id,
        )
        app._adk_available = True
        app.handle_user_text("Y")

        state = load_workflow_state(session_id=session_id)
        self.assertEqual(fake_bridge.calls, [])
        self.assertNotIn("CLOUD_REGION", state.get("confirmed_config", {}))
        self.assertEqual(state["pending_question"]["id"], "cloud_region")
        self.assertTrue(any("no detected/current value" in item for item in io.messages))

    def test_manual_config_yes_with_current_value_uses_current_value(self):
        fake_bridge = FakeBridge()
        io = CapturingIO()
        session_id = "unit-manual-config-yes-with-current-value"
        reset_workflow_state(session_id=session_id)
        update_workflow_state(
            {
                "active_intent": "benchmark",
                "active_workflow": "benchmark_setup",
                "target_mode": "fake-node",
                "chain": "solana",
                "pending_question": {
                    "id": "cloud_region",
                    "kind": "manual_value",
                    "prompt": "Confirm CLOUD_REGION",
                    "field": "CLOUD_REGION",
                    "current_value": "asia-east1",
                },
            },
            session_id=session_id,
        )
        app = AnyChainTerminal(
            state=TerminalSession(language="en"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=io,
            bridge_factory=lambda: fake_bridge,
            session_id=session_id,
        )
        app._adk_available = True
        app.handle_user_text("Y")

        state = load_workflow_state(session_id=session_id)
        self.assertEqual(fake_bridge.calls, [])
        self.assertEqual(state["confirmed_config"]["CLOUD_REGION"], "asia-east1")

    def test_config_transition_overwrites_stale_blockchain_node_when_chain_changes(self):
        session_id = "unit-config-transition-overwrites-stale-blockchain-node"
        reset_workflow_state(session_id=session_id)
        update_workflow_state(
            {
                "active_intent": "benchmark",
                "active_workflow": "benchmark_setup",
                "target_mode": "fake-node",
                "chain": "bsc",
                "rpc_mode": "mixed",
                "confirmed_config": {
                    "BLOCKCHAIN_NODE": "solana",
                    "chain": "solana",
                    "TARGET_MODE": "fake-node",
                    "use_fake_node": True,
                },
            },
            session_id=session_id,
        )
        advance_after_pending_answer(
            {"applied": True, "transition": {}},
            discovery={},
            language="en",
            session_id=session_id,
        )

        state = load_workflow_state(session_id=session_id)
        self.assertEqual(state["chain"], "bsc")
        self.assertEqual(state["confirmed_config"]["chain"], "bsc")
        self.assertEqual(state["confirmed_config"]["BLOCKCHAIN_NODE"], "bsc")

    def test_top_level_target_fields_sync_into_confirmed_config(self):
        session_id = "unit-top-level-fields-sync-to-confirmed-config"
        reset_workflow_state(session_id=session_id)
        update_workflow_state(
            {
                "active_intent": "benchmark",
                "active_workflow": "benchmark_setup",
                "target_mode": "fake-node",
                "chain": "solana",
                "rpc_mode": "single",
                "benchmark_profile": {"mode": "quick"},
            },
            session_id=session_id,
        )

        state = load_workflow_state(session_id=session_id)
        confirmed = state["confirmed_config"]
        self.assertEqual(confirmed["chain"], "solana")
        self.assertEqual(confirmed["BLOCKCHAIN_NODE"], "solana")
        self.assertEqual(confirmed["target_mode"], "fake-node")
        self.assertEqual(confirmed["TARGET_MODE"], "fake-node")
        self.assertEqual(confirmed["use_fake_node"], True)
        self.assertEqual(confirmed["rpc_mode"], "single")
        self.assertEqual(confirmed["RPC_MODE"], "single")
        self.assertEqual(confirmed["benchmark_mode_confirmed"], "quick")

    def test_benchmark_setup_turn_prints_typed_question_not_model_handwritten_flow(self):
        class StateUpdatingBridge:
            def __init__(self, session_id: str) -> None:
                self.session_id = session_id
                self.calls: list[tuple[str, dict]] = []

            def run_text(self, text: str, state_delta=None) -> str:
                self.calls.append((text, state_delta or {}))
                update_workflow_state(
                    {
                        "active_intent": "benchmark",
                        "active_workflow": "benchmark_setup",
                        "target_mode": "fake-node",
                        "chain": "solana",
                        "benchmark_profile": {"mode": "quick"},
                        "pending_question": {
                            "id": "disk_ledger_choice",
                            "kind": "device",
                            "field": "LEDGER_DEVICE",
                            "prompt": "请选择 Ledger/data 磁盘。",
                            "options": [
                                {"id": "1", "label": "sdb", "value": "sdb"},
                            ],
                        },
                    },
                    reason="unit_state_updating_bridge",
                    session_id=self.session_id,
                )
                return "好的，我会逐项确认。\n请选择 benchmark 模式：1 quick、2 standard、3 intensive。"

        session_id = "unit-setup-turn-typed-question-wins"
        reset_workflow_state(session_id=session_id)
        bridge = StateUpdatingBridge(session_id)
        io = CapturingIO()
        app = AnyChainTerminal(
            state=TerminalSession(language="zh"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=io,
            bridge_factory=lambda: bridge,
            session_id=session_id,
        )
        app._adk_available = True

        app.handle_user_text("我要测试 solana fake-node quick，请逐项确认环境变量")

        output = "\n".join(io.messages)
        self.assertIn("请选择 Ledger/data 磁盘", output)
        self.assertNotIn("请选择 benchmark 模式", output)
        self.assertNotIn("好的，我会逐项确认", output)
        state = load_workflow_state(session_id=session_id)
        self.assertEqual(state["pending_question"]["id"], "disk_ledger_choice")
        self.assertEqual(state["confirmed_config"]["BLOCKCHAIN_NODE"], "solana")
        self.assertEqual(state["confirmed_config"]["TARGET_MODE"], "fake-node")
        self.assertEqual(state["confirmed_config"]["benchmark_mode_confirmed"], "quick")

    def test_chain_change_confirmation_commits_new_chain_on_yes(self):
        fake_bridge = FakeBridge()
        io = CapturingIO()
        session_id = "unit-chain-change-confirmation-commits-new-chain"
        reset_workflow_state(session_id=session_id)
        with workflow_tool_session(session_id):
            result = propose_chain_change_confirmation(
                new_chain="eth",
                previous_chain="solana",
                target_mode="real-node",
                target_mode_explicit=True,
                language="zh",
            )
        self.assertEqual(result["data"]["pending_question_id"], "confirm_chain_change")
        self.assertIn("`ethereum`", result["data"]["prompt"])

        app = AnyChainTerminal(
            state=TerminalSession(language="zh"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=io,
            bridge_factory=lambda: fake_bridge,
            session_id=session_id,
        )
        app._adk_available = True
        app.handle_user_text("Y")

        state = load_workflow_state(session_id=session_id)
        self.assertEqual(fake_bridge.calls, [])
        self.assertEqual(state["chain"], "ethereum")
        self.assertEqual(state["confirmed_config"]["chain"], "ethereum")
        self.assertEqual(state["confirmed_config"]["BLOCKCHAIN_NODE"], "ethereum")
        self.assertEqual(state["target_mode"], "real-node")
        self.assertNotEqual(state["pending_question"].get("id"), "confirm_chain_change")

    def test_chain_change_without_explicit_mode_asks_target_mode_after_yes(self):
        fake_bridge = FakeBridge()
        io = CapturingIO()
        session_id = "unit-chain-change-asks-target-mode-after-yes"
        reset_workflow_state(session_id=session_id)
        update_workflow_state(
            {
                "active_intent": "benchmark",
                "active_workflow": "benchmark_setup",
                "target_mode": "fake-node",
                "chain": "solana",
            },
            session_id=session_id,
        )
        with workflow_tool_session(session_id):
            result = propose_chain_change_confirmation(
                new_chain="bsc",
                previous_chain="solana",
                language="zh",
            )
        self.assertIn("选择 fake-node/real-node", result["data"]["prompt"])

        app = AnyChainTerminal(
            state=TerminalSession(language="zh"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=io,
            bridge_factory=lambda: fake_bridge,
            session_id=session_id,
        )
        app._adk_available = True
        app.handle_user_text("Y")

        state = load_workflow_state(session_id=session_id)
        self.assertEqual(fake_bridge.calls, [])
        self.assertEqual(state["chain"], "bsc")
        self.assertEqual(state["target_mode"], "")
        self.assertEqual(state["pending_question"]["id"], "target_mode")
        self.assertTrue(any("fake-node 模式" in item and "real-node 模式" in item for item in io.messages))

    def test_chain_change_tool_extracts_exact_alias_from_natural_language(self):
        session_id = "unit-chain-change-extracts-exact-alias"
        reset_workflow_state(session_id=session_id)
        update_workflow_state(
            {
                "active_intent": "benchmark",
                "active_workflow": "benchmark_setup",
                "target_mode": "fake-node",
                "chain": "solana",
                "last_user_change": {
                    "type": "pending_question_interruption",
                    "raw": "我需要测试 BNB",
                    "pending_question_id": "cloud_region",
                },
            },
            session_id=session_id,
        )
        with workflow_tool_session(session_id):
            result = propose_chain_change_confirmation(
                new_chain="我需要测试 BNB",
                previous_chain="solana",
                language="zh",
            )

        state = load_workflow_state(session_id=session_id)
        self.assertEqual(result["data"]["pending_question_id"], "confirm_chain_change")
        self.assertEqual(result["data"]["new_chain"], "bsc")
        self.assertEqual(state["pending_question"]["id"], "confirm_chain_change")
        self.assertIn("`bsc`", state["pending_question"]["prompt"])

    def test_chain_change_tool_extracts_alias_with_rpc_workload_terms(self):
        session_id = "unit-chain-change-extracts-alias-with-rpc-workload-terms"
        reset_workflow_state(session_id=session_id)
        update_workflow_state(
            {
                "active_intent": "benchmark",
                "active_workflow": "benchmark_setup",
                "target_mode": "fake-node",
                "chain": "solana",
                "last_user_change": {
                    "type": "pending_question_interruption",
                    "raw": "我现在要换成 BNB，并重新确认 RPC method 和权重",
                    "pending_question_id": "cloud_region",
                },
            },
            session_id=session_id,
        )
        with workflow_tool_session(session_id):
            result = propose_chain_change_confirmation(
                new_chain="我现在要换成 BNB，并重新确认 RPC method 和权重",
                previous_chain="solana",
                language="zh",
            )

        state = load_workflow_state(session_id=session_id)
        self.assertEqual(result["data"]["pending_question_id"], "confirm_chain_change")
        self.assertEqual(result["data"]["new_chain"], "bsc")
        self.assertEqual(state["pending_question"]["id"], "confirm_chain_change")
        self.assertIn("`bsc`", state["pending_question"]["prompt"])

    def test_chain_change_tool_does_not_collapse_alias_prefix_product_name(self):
        session_id = "unit-chain-change-does-not-collapse-bnb-greenfield"
        reset_workflow_state(session_id=session_id)
        update_workflow_state(
            {
                "active_intent": "benchmark",
                "active_workflow": "benchmark_setup",
                "target_mode": "fake-node",
                "chain": "solana",
                "last_user_change": {
                    "type": "pending_question_interruption",
                    "raw": "test bnb greenfield",
                    "pending_question_id": "cloud_region",
                },
            },
            session_id=session_id,
        )
        with workflow_tool_session(session_id):
            result = propose_chain_change_confirmation(
                new_chain="bsc",
                previous_chain="solana",
                language="en",
            )

        state = load_workflow_state(session_id=session_id)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["data"]["pending_question_id"], "chain_selection")
        self.assertEqual(state["pending_question"]["id"], "chain_selection")
        self.assertEqual(state["chain_status"], "ambiguous_or_unknown")
        self.assertEqual(state["chain"], "")

    def test_chain_change_ignores_inherited_target_mode_without_explicit_flag(self):
        fake_bridge = FakeBridge()
        io = CapturingIO()
        session_id = "unit-chain-change-ignores-inherited-target-mode"
        reset_workflow_state(session_id=session_id)
        update_workflow_state(
            {
                "active_intent": "benchmark",
                "active_workflow": "benchmark_setup",
                "target_mode": "fake-node",
                "chain": "solana",
            },
            session_id=session_id,
        )
        with workflow_tool_session(session_id):
            result = propose_chain_change_confirmation(
                new_chain="bsc",
                previous_chain="solana",
                target_mode="fake-node",
                language="zh",
            )
        self.assertIn("选择 fake-node/real-node", result["data"]["prompt"])
        self.assertNotIn("继续使用 fake-node", result["data"]["prompt"])
        self.assertFalse(result["data"]["target_mode_explicit"])

        app = AnyChainTerminal(
            state=TerminalSession(language="zh"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=io,
            bridge_factory=lambda: fake_bridge,
            session_id=session_id,
        )
        app._adk_available = True
        app.handle_user_text("Y")

        state = load_workflow_state(session_id=session_id)
        self.assertEqual(fake_bridge.calls, [])
        self.assertEqual(state["chain"], "bsc")
        self.assertEqual(state["target_mode"], "")
        self.assertEqual(state["pending_question"]["id"], "target_mode")
        self.assertTrue(any("fake-node 模式" in item and "real-node 模式" in item for item in io.messages))

    def test_chain_change_explicit_real_node_commits_target_mode_on_yes(self):
        fake_bridge = FakeBridge()
        io = CapturingIO()
        session_id = "unit-chain-change-explicit-real-node"
        reset_workflow_state(session_id=session_id)
        update_workflow_state(
            {
                "active_intent": "benchmark",
                "active_workflow": "benchmark_setup",
                "target_mode": "fake-node",
                "chain": "bsc",
                "confirmed_config": {
                    "TARGET_MODE": "fake-node",
                    "use_fake_node": True,
                    "blockchain_process_names": ["fake-node"],
                    "BLOCKCHAIN_PROCESS_NAMES": ["fake-node"],
                    "BLOCKCHAIN_PROCESS_NAMES_STR": "fake-node",
                },
            },
            session_id=session_id,
        )
        with workflow_tool_session(session_id):
            result = propose_chain_change_confirmation(
                new_chain="ethereum",
                previous_chain="bsc",
                target_mode="real-node",
                target_mode_explicit=True,
                language="zh",
            )
        self.assertIn("real-node", result["data"]["prompt"])
        self.assertTrue(result["data"]["target_mode_explicit"])

        app = AnyChainTerminal(
            state=TerminalSession(language="zh"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=io,
            bridge_factory=lambda: fake_bridge,
            session_id=session_id,
        )
        app._adk_available = True
        app.handle_user_text("Y")

        state = load_workflow_state(session_id=session_id)
        self.assertEqual(state["chain"], "ethereum")
        self.assertEqual(state["target_mode"], "real-node")
        self.assertFalse(state["confirmed_config"]["use_fake_node"])
        self.assertNotIn("blockchain_process_names", state["confirmed_config"])
        self.assertNotIn("BLOCKCHAIN_PROCESS_NAMES", state["confirmed_config"])
        self.assertNotIn("BLOCKCHAIN_PROCESS_NAMES_STR", state["confirmed_config"])
        self.assertNotEqual(state["pending_question"].get("id"), "target_mode")

    def test_target_mode_interruption_enters_real_node_endpoint_gate(self):
        session_id = "unit-target-mode-interruption-enters-real-node-endpoint"
        reset_workflow_state(session_id=session_id)
        pending_question = {
            "id": "disk_ledger_choice",
            "kind": "device",
            "prompt": "请选择 Ledger/data 磁盘。",
            "field": "LEDGER_DEVICE",
        }
        update_workflow_state(
            {
                "active_intent": "benchmark",
                "active_workflow": "benchmark_setup",
                "workflow_step": "disk_ledger_choice",
                "target_mode": "fake-node",
                "chain": "solana",
                "pending_question": pending_question,
                "last_user_change": {
                    "type": "pending_question_interruption",
                    "raw": "我改主意了，不用 fake-node，切换到 real-node",
                    "pending_question_id": "disk_ledger_choice",
                },
            },
            session_id=session_id,
        )
        app = AnyChainTerminal(
            state=TerminalSession(language="zh"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=CapturingIO(),
            session_id=session_id,
        )

        result = app._setup_question_after_adk({"pending_question": pending_question})

        state = load_workflow_state(session_id=session_id)
        self.assertFalse(result["suppress_response"])
        self.assertEqual(state["target_mode"], "fake-node")
        self.assertEqual(state["pending_question"]["id"], "disk_ledger_choice")
        self.assertEqual(state["last_user_change"]["type"], "pending_question_interruption")

    def test_chain_change_no_returns_to_chain_selection_prompt(self):
        fake_bridge = FakeBridge()
        io = CapturingIO()
        session_id = "unit-chain-change-no-returns-chain-selection"
        reset_workflow_state(session_id=session_id)
        update_workflow_state(
            {
                "active_intent": "benchmark",
                "active_workflow": "benchmark_setup",
                "target_mode": "fake-node",
                "chain": "solana",
            },
            session_id=session_id,
        )
        with workflow_tool_session(session_id):
            propose_chain_change_confirmation(
                new_chain="ethereum",
                previous_chain="solana",
                language="zh",
            )

        app = AnyChainTerminal(
            state=TerminalSession(language="zh"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=io,
            bridge_factory=lambda: fake_bridge,
            session_id=session_id,
        )
        app._adk_available = True
        app.handle_user_text("N")

        state = load_workflow_state(session_id=session_id)
        self.assertEqual(fake_bridge.calls, [])
        self.assertEqual(state["chain"], "solana")
        self.assertEqual(state["pending_question"]["id"], "chain_selection")
        self.assertTrue(any("你想测试哪条链" in item for item in io.messages))

    def test_natural_language_during_manual_config_delegates_to_adk(self):
        fake_bridge = FakeBridge()
        io = CapturingIO()
        session_id = "unit-manual-config-natural-language-interruption"
        reset_workflow_state(session_id=session_id)
        update_workflow_state(
            {
                "active_intent": "benchmark",
                "active_workflow": "benchmark_setup",
                "workflow_step": "cloud_region",
                "target_mode": "fake-node",
                "chain": "solana",
                "pending_question": {
                    "id": "cloud_region",
                    "kind": "manual_value",
                    "prompt": "Confirm CLOUD_REGION",
                    "field": "CLOUD_REGION",
                    "branch": "benchmark_setup",
                },
            },
            session_id=session_id,
        )
        app = AnyChainTerminal(
            state=TerminalSession(language="zh"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=io,
            bridge_factory=lambda: fake_bridge,
            session_id=session_id,
        )
        app._adk_available = True
        app.handle_user_text("我需要测试 BNB")

        state = load_workflow_state(session_id=session_id)
        self.assertEqual(len(fake_bridge.calls), 1)
        self.assertEqual(fake_bridge.calls[0][0], "我需要测试 BNB")
        self.assertEqual(fake_bridge.calls[0][1]["input_mode"], "pending_question_interruption")
        self.assertNotIn("CLOUD_REGION", state.get("confirmed_config", {}))
        self.assertEqual(state["pending_question"]["id"], "cloud_region")

    def test_unknown_chain_interruption_during_manual_config_enters_identity_gate(self):
        class IdentityBridge:
            def __init__(self, session_id: str) -> None:
                self.session_id = session_id
                self.calls: list[tuple[str, dict]] = []

            def run_text(self, text: str, state_delta=None) -> str:
                self.calls.append((text, state_delta or {}))
                with workflow_tool_session(self.session_id):
                    propose_chain_identity_resolution(
                        candidate_chain="sola",
                        evidence_summary="user asked whether this chain exists during configuration",
                        language="en",
                    )
                return "Confirm the chain identity before continuing configuration."

        io = CapturingIO()
        session_id = "unit-unknown-chain-during-cloud-region"
        bridge = IdentityBridge(session_id)
        reset_workflow_state(session_id=session_id)
        update_workflow_state(
            {
                "active_intent": "benchmark",
                "active_workflow": "benchmark_setup",
                "workflow_step": "cloud_region",
                "target_mode": "fake-node",
                "chain": "solana",
                "pending_question": {
                    "id": "cloud_region",
                    "kind": "manual_value",
                    "prompt": "Confirm CLOUD_REGION",
                    "field": "CLOUD_REGION",
                    "branch": "benchmark_setup",
                },
            },
            session_id=session_id,
        )
        app = AnyChainTerminal(
            state=TerminalSession(language="en"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=io,
            bridge_factory=lambda: bridge,
            session_id=session_id,
        )
        app._adk_available = True
        app.handle_user_text("I want to switch to sola, check whether it is a real chain first")

        state = load_workflow_state(session_id=session_id)
        self.assertEqual(len(bridge.calls), 1)
        self.assertEqual(bridge.calls[0][1]["input_mode"], "pending_question_interruption")
        self.assertNotIn("CLOUD_REGION", state.get("confirmed_config", {}))
        self.assertEqual(state["chain_status"], "identity_needs_confirmation")
        self.assertEqual(state["chain_identity_candidate"]["normalized"], "sola")
        self.assertEqual(state["pending_question"]["id"], "chain_identity_resolution")

    def test_pending_question_clarification_does_not_append_stale_english_prompt(self):
        session_id = "unit-opening-clarification-language-sync"
        reset_workflow_state(session_id=session_id)
        with workflow_tool_session(session_id):
            propose_opening_help_choice(language="en", latest_job_id="")
        bridge = StaticBridge(
            "没关系，我简单解释一下：\n"
            "1. 模拟节点基准测试\n"
            "2. 真实节点基准测试\n"
            "3. 查看支持信息"
        )
        io = CapturingIO()
        app = AnyChainTerminal(
            state=TerminalSession(language="en"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=io,
            bridge_factory=lambda: bridge,
            session_id=session_id,
        )
        app._adk_available = True

        app.handle_user_text("我没看懂你在说什么")

        self.assertEqual(len(bridge.calls), 1)
        self.assertEqual(bridge.calls[0][1]["input_mode"], "normal_user_turn")
        output = "\n".join(io.messages)
        self.assertIn("没关系，我简单解释一下", output)
        self.assertNotIn("Start a fake-node benchmark", output)
        self.assertNotIn("No previous job is available", output)
        self.assertEqual(load_workflow_state(session_id=session_id).get("pending_question"), {})

    def test_terminal_does_not_rewrite_appended_stale_model_menu(self):
        session_id = "unit-opening-clarification-replaces-stale-english-menu"
        reset_workflow_state(session_id=session_id)
        with workflow_tool_session(session_id):
            propose_opening_help_choice(language="en", latest_job_id="")
        bridge = StaticBridge(
            "没关系，我简单解释一下：\n"
            "1. 模拟节点基准测试\n"
            "2. 真实节点基准测试\n"
            "3. 查看支持信息\n"
            "Hi, I am AnyChain Benchmark Agent. No previous job is available to analyze. What would you like to do?\n"
            "1. Start a fake-node benchmark (no real node required; validates the framework loop)\n"
            "2. Start a real-node benchmark (requires a real LOCAL_RPC_URL)\n"
            "3. Learn supported chains, RPC methods, and extension paths\n"
            "Reply `1`, `2`, or `3`, or type your goal directly."
        )
        io = CapturingIO()
        app = AnyChainTerminal(
            state=TerminalSession(language="en"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=io,
            bridge_factory=lambda: bridge,
            session_id=session_id,
        )
        app._adk_available = True

        app.handle_user_text("我没看懂你在说什么")

        output = "\n".join(io.messages)
        self.assertIn("Start a fake-node benchmark", output)
        self.assertIn("No previous job is available", output)
        self.assertEqual(load_workflow_state(session_id=session_id).get("pending_question"), {})

    def test_terminal_does_not_bind_unregistered_opening_menu(self):
        session_id = "unit-unregistered-opening-menu"
        reset_workflow_state(session_id=session_id)
        bridge = StaticBridge(
            "Hi, I am AnyChain Benchmark Agent. No previous job is available to analyze. What would you like to do?\n"
            "1. Start a fake-node benchmark (no real node required; validates the framework loop)\n"
            "2. Start a real-node benchmark (requires a real LOCAL_RPC_URL)\n"
            "3. Learn supported chains, RPC methods, and extension paths\n"
            "Reply `1`, `2`, or `3`, or type your goal directly."
        )
        io = CapturingIO()
        app = AnyChainTerminal(
            state=TerminalSession(language="en"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=io,
            bridge_factory=lambda: bridge,
            session_id=session_id,
        )
        app._adk_available = True

        app.handle_user_text("Hi")

        state = load_workflow_state(session_id=session_id)
        self.assertEqual(state.get("pending_question"), {})

    def test_opening_capability_choice_returns_framework_summary_without_benchmark_state(self):
        fake_bridge = FakeBridge()
        io = CapturingIO()
        session_id = "unit-opening-capability-choice-summary"
        reset_workflow_state(session_id=session_id)
        with workflow_tool_session(session_id):
            propose_opening_help_choice(language="en", latest_job_id="")
        app = AnyChainTerminal(
            state=TerminalSession(language="en"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=io,
            bridge_factory=lambda: fake_bridge,
            session_id=session_id,
        )
        app._adk_available = True

        app.handle_user_text("3")

        state = load_workflow_state(session_id=session_id)
        output = "\n".join(io.messages)
        self.assertEqual(fake_bridge.calls, [])
        self.assertIn("36 chains", output)
        self.assertIn("RPC method", output)
        self.assertNotEqual(state.get("active_workflow"), "benchmark_setup")
        self.assertFalse(state.get("pending_question"))

    def test_simple_chain_token_remains_local_structural_answer(self):
        fake_bridge = FakeBridge()
        io = CapturingIO()
        session_id = "unit-chain-token-local-answer"
        reset_workflow_state(session_id=session_id)
        update_workflow_state(
            {
                "active_intent": "benchmark",
                "active_workflow": "benchmark_setup",
                "target_mode": "fake-node",
                "pending_question": {
                    "id": "chain_selection",
                    "kind": "chain",
                    "prompt": "Which chain?",
                    "field": "chain",
                    "known_chains": ["solana", "ethereum", "bsc"],
                    "chain_aliases": {"bnb": "bsc", "eth": "ethereum"},
                    "next_on_manual": {"workflow_step": "validate_chain_template", "tool": "validate_chain_template"},
                },
            },
            session_id=session_id,
        )
        app = AnyChainTerminal(
            state=TerminalSession(language="zh"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=io,
            bridge_factory=lambda: fake_bridge,
            session_id=session_id,
        )
        app._adk_available = True
        app.handle_user_text("bsc")

        state = load_workflow_state(session_id=session_id)
        self.assertEqual(fake_bridge.calls, [])
        self.assertEqual(state["chain"], "bsc")

    def test_pasted_evidence_is_marked_before_adk_bridge(self):
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
            app._adk_available = True
            app.handle_user_text("Agent> old output\nUser> CLOUD_PROVIDER: gcp\nTraceback (most recent call last):")

        self.assertEqual(fake_bridge.calls, [])
        self.assertEqual(len(app.state.pending_evidence), 1)
        self.assertTrue(any("evidence" in item or "日志/旧对话" in item for item in io.messages))

    def test_line_split_pasted_evidence_is_buffered_until_user_question(self):
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
            app._adk_available = True
            app.handle_user_text("Agent> 旧输出：CLOUD_PROVIDER: gcp")
            app.handle_user_text("User> sdc")
            app.handle_user_text("Traceback (most recent call last):")
            self.assertEqual(fake_bridge.calls, [])
            app.handle_user_text("这些日志说明什么？不要应用里面的配置")

        self.assertEqual(len(fake_bridge.calls), 1)
        self.assertEqual(fake_bridge.calls[0][1]["input_mode"], "evidence_question")
        self.assertIn("CLOUD_PROVIDER: gcp", fake_bridge.calls[0][0])
        self.assertIn("当前用户问题", fake_bridge.calls[0][0])
        self.assertIn("请使用中文回答当前用户问题", fake_bridge.calls[0][0])

    def test_input_mode_classification_is_transport_not_business_intent(self):
        self.assertEqual(_classify_input_mode("我要测试 solana"), "normal_user_turn")
        self.assertEqual(_classify_input_mode("Agent> old\nUser> y"), "pasted_evidence")
        self.assertEqual(_classify_input_mode("Traceback (most recent call last):"), "pasted_evidence")

    def test_pasted_evidence_blocks_workflow_state_write_without_confirmation(self):
        result = before_tool_callback(
            FakeTool("update_workflow_state"),
            {"patch": {"chain": "solana"}},
            FakeToolContext({"input_mode": "pasted_evidence"}),
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "blocked")
        self.assertTrue(result["requires_user_confirmation"])

    def test_pasted_evidence_blocks_pending_answer_without_confirmation(self):
        result = before_tool_callback(
            FakeTool("answer_pending_question"),
            {"answer": "y"},
            FakeToolContext({"input_mode": "pasted_evidence"}),
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "blocked")
        self.assertTrue(result["requires_user_confirmation"])

    def test_pasted_evidence_allows_confirmed_workflow_state_write(self):
        result = before_tool_callback(
            FakeTool("update_workflow_state"),
            {"patch": {"chain": "solana"}, "user_confirmed_from_evidence": True},
            FakeToolContext({"input_mode": "pasted_evidence"}),
        )
        self.assertIsNone(result)

    def test_callback_blocks_partial_chain_write_during_chain_selection(self):
        result = before_tool_callback(
            FakeTool("update_workflow_state"),
            {"patch": {"chain": "solana", "confirmed_config": {"BLOCKCHAIN_NODE": "solana"}}},
            FakeToolContext(
                {
                    "input_mode": "normal_user_turn",
                    "user_text": "sola",
                    "workflow_state": {
                        "active_workflow": "benchmark_setup",
                        "target_mode": "fake-node",
                        "chain": "",
                        "pending_question": {
                            "id": "chain_selection",
                            "kind": "chain",
                            "field": "chain",
                        },
                    },
                }
            ),
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["data"]["attempted_chain"], "solana")

    def test_callback_reads_adk_state_to_dict_for_chain_guards(self):
        result = before_tool_callback(
            FakeTool("update_workflow_state"),
            {"patch": {"chain": "solana", "confirmed_config": {"BLOCKCHAIN_NODE": "solana"}}},
            type("Context", (), {
                "state": FakeADKState(
                    {
                        "input_mode": "normal_user_turn",
                        "user_text": "sola",
                        "workflow_state": {
                            "active_workflow": "benchmark_setup",
                            "target_mode": "fake-node",
                            "chain": "",
                            "pending_question": {
                                "id": "chain_selection",
                                "kind": "chain",
                                "field": "chain",
                            },
                        },
                    }
                )
            })(),
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["data"]["attempted_chain"], "solana")

    def test_after_model_callback_registers_unknown_chain_identity_gate_from_adk_state(self):
        session_id = "unit-after-model-unknown-chain-from-adk-state"
        reset_workflow_state(session_id=session_id)
        update_workflow_state(
            {
                "active_intent": "benchmark",
                "active_workflow": "benchmark_setup",
                "workflow_step": "chain_selection",
                "target_mode": "fake-node",
                "chain": "",
                "pending_question": {"id": "chain_selection", "kind": "chain", "field": "chain"},
            },
            session_id=session_id,
        )

        after_model_callback(
            type("Context", (), {
                "state": FakeADKState(
                    {
                        "session_id": session_id,
                        "terminal_language": "en",
                        "user_text": "sola",
                        "workflow_state": load_workflow_state(session_id=session_id),
                    }
                )
            })(),
            type("Response", (), {"text": "Did you mean solana? If not, provide official docs."})(),
        )

        state = load_workflow_state(session_id=session_id)
        self.assertEqual(state["chain"], "")
        self.assertEqual(state["chain_status"], "identity_needs_confirmation")
        self.assertEqual(state["pending_question"]["id"], "chain_identity_resolution")
        self.assertEqual(state["chain_identity_candidate"]["normalized"], "sola")

    def test_after_model_callback_routes_no_endpoint_unknown_chain_to_handoff(self):
        session_id = "unit-after-model-no-endpoint-unknown-chain-handoff"
        reset_workflow_state(session_id=session_id)
        with workflow_tool_session(session_id):
            propose_chain_identity_resolution(
                candidate_chain="foochain",
                suggested_adapter_family="jsonrpc",
                evidence_summary="FooChain is a possible EVM JSON-RPC chain.",
                language="zh",
            )

        after_model_callback(
            FakeToolContext(
                {
                    "session_id": session_id,
                    "terminal_language": "zh",
                    "user_text": "没有 endpoint，先给另一个 AI 一个可执行开发文档",
                    "workflow_state": load_workflow_state(session_id=session_id),
                }
            ),
            object(),
        )

        state = load_workflow_state(session_id=session_id)
        self.assertEqual(state["chain"], "foochain")
        self.assertEqual(state["chain_status"], "unsupported_needs_development_handoff")
        self.assertEqual(state["fixture_status"]["status"], "needs_review")
        self.assertFalse(state.get("pending_question"))

    def test_after_model_callback_records_compact_mixed_weights_and_blocks_invalid_total(self):
        session_id = "unit-after-model-compact-mixed-weight-invalid-total"
        reset_workflow_state(session_id=session_id)
        update_workflow_state(
            {
                "active_intent": "benchmark",
                "active_workflow": "benchmark_setup",
                "workflow_step": "benchmark_setup",
                "confirmed_config": {},
            },
            session_id=session_id,
        )

        after_model_callback(
            type("Context", (), {
                "state": FakeADKState(
                    {
                        "session_id": session_id,
                        "terminal_language": "en",
                        "user_text": "ethereum fake-node mixed eth_blockNumber 70 eth_getBalance 20",
                        "workflow_state": load_workflow_state(session_id=session_id),
                    }
                )
            })(),
            type("Response", (), {"text": "Benchmark Ethereum fake-node mixed workload."})(),
        )

        state = load_workflow_state(session_id=session_id)
        self.assertEqual(state["chain"], "ethereum")
        self.assertEqual(state["target_mode"], "fake-node")
        self.assertEqual(state["rpc_mode"], "mixed")
        self.assertEqual(state["mixed_weights"], {"eth_blockNumber": 70, "eth_getBalance": 20})
        self.assertEqual(state["active_group"], "workload_rpc")
        self.assertEqual(state["next_blocking_group"], "workload_rpc")
        self.assertEqual(state["allowed_next_actions"], ["ask:mixed_weights_confirm"])
        self.assertIn("mixed_weights total must be 100, got 90", state["blockers"])

    def test_transition_executor_honors_mixed_weight_allowed_action_without_pending(self):
        session_id = "unit-transition-honors-mixed-weight-allowed-action"
        reset_workflow_state(session_id=session_id)
        update_workflow_state(
            {
                "active_intent": "benchmark",
                "active_workflow": "benchmark_setup",
                "workflow_step": "mixed_weights_confirm",
                "active_group": "workload_rpc",
                "next_blocking_group": "workload_rpc",
                "chain": "ethereum",
                "target_mode": "fake-node",
                "rpc_mode": "mixed",
                "mixed_weights": {"eth_blockNumber": 70, "eth_getBalance": 20},
                "allowed_next_actions": ["ask:mixed_weights_confirm"],
                "confirmed_config": {
                    "chain": "ethereum",
                    "BLOCKCHAIN_NODE": "ethereum",
                    "target_mode": "fake-node",
                    "TARGET_MODE": "fake-node",
                    "rpc_mode": "mixed",
                    "RPC_MODE": "mixed",
                },
            },
            session_id=session_id,
        )

        advanced = ensure_next_benchmark_setup_question(
            load_workflow_state(session_id=session_id),
            language="en",
            session_id=session_id,
        )
        state = load_workflow_state(session_id=session_id)

        self.assertTrue(advanced["advanced"])
        self.assertEqual(advanced["pending_question"]["id"], "mixed_weights_confirm")
        self.assertEqual(state["pending_question"]["id"], "mixed_weights_confirm")
        self.assertIn("eth_blockNumber=70", advanced["message"])
        self.assertIn("eth_getBalance=20", advanced["message"])

    def test_partial_chain_candidate_enters_identity_resolution_gate(self):
        session_id = "unit-partial-chain-enters-identity-resolution"
        reset_workflow_state(session_id=session_id)
        update_workflow_state(
            {
                "active_intent": "benchmark",
                "active_workflow": "benchmark_setup",
                "workflow_step": "chain_selection",
                "target_mode": "fake-node",
                "chain": "",
                "pending_question": {"id": "chain_selection", "kind": "chain", "field": "chain"},
                "last_user_change": {
                    "type": "unsupported_chain_candidate",
                    "raw": "sola",
                    "pending_question_id": "chain_selection",
                },
            },
            session_id=session_id,
        )
        app = AnyChainTerminal(
            state=TerminalSession(language="en"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=CapturingIO(),
            session_id=session_id,
        )

        result = app._setup_question_after_adk({"pending_question": {"id": "chain_selection"}})

        state = load_workflow_state(session_id=session_id)
        self.assertFalse(result["suppress_response"])
        self.assertEqual(state["chain"], "")
        self.assertEqual(state["pending_question"]["id"], "chain_selection")
        self.assertEqual(state["last_user_change"]["raw"], "sola")

    def test_chain_identity_resolution_can_confirm_llm_supported_suggestion(self):
        session_id = "unit-chain-identity-confirms-supported-suggestion"
        reset_workflow_state(session_id=session_id)
        with workflow_tool_session(session_id):
            propose_chain_identity_resolution(
                candidate_chain="sola",
                suggested_supported_chain="solana",
                suggested_adapter_family="jsonrpc",
                evidence_summary="likely typo based on model knowledge",
                language="en",
            )
        answer = answer_pending_question("1", session_id=session_id)
        next_step = advance_after_pending_answer(answer, discovery={}, language="en", session_id=session_id)

        state = load_workflow_state(session_id=session_id)
        self.assertEqual(state["chain"], "solana")
        self.assertEqual(state["confirmed_config"]["BLOCKCHAIN_NODE"], "solana")
        self.assertNotEqual(state["pending_question"].get("id"), "chain_identity_resolution")
        self.assertTrue(next_step.get("advanced"))

    def test_chain_identity_supported_suggestion_accepts_yes_as_default(self):
        session_id = "unit-chain-identity-supported-suggestion-yes"
        reset_workflow_state(session_id=session_id)
        with workflow_tool_session(session_id):
            propose_chain_identity_resolution(
                candidate_chain="bnb",
                suggested_supported_chain="bsc",
                suggested_adapter_family="jsonrpc",
                evidence_summary="BNB commonly maps to BSC in this framework.",
                language="zh",
            )
        answer = answer_pending_question("Y", session_id=session_id)

        state = load_workflow_state(session_id=session_id)
        self.assertTrue(answer["applied"])
        self.assertEqual(state["chain"], "bsc")
        self.assertEqual(state["confirmed_config"]["BLOCKCHAIN_NODE"], "bsc")
        self.assertNotEqual(state["pending_question"].get("id"), "chain_identity_resolution")

    def test_chain_identity_resolution_routes_existing_family_new_chain_to_endpoint_gate(self):
        session_id = "unit-chain-identity-existing-family-endpoint-gate"
        reset_workflow_state(session_id=session_id)
        with workflow_tool_session(session_id):
            propose_chain_identity_resolution(
                candidate_chain="flow",
                suggested_adapter_family="jsonrpc",
                evidence_summary="Flow exposes an EVM-compatible JSON-RPC endpoint in user evidence",
                language="en",
            )
        answer = answer_pending_question("1", session_id=session_id)

        state = load_workflow_state(session_id=session_id)
        self.assertTrue(answer["applied"])
        self.assertEqual(state["chain"], "flow")
        self.assertEqual(state["chain_status"], "protocol_needs_confirmation")
        self.assertEqual(state["chain_protocol_candidate"]["chain"], "flow")
        self.assertEqual(state["confirmed_config"]["BLOCKCHAIN_NODE"], "flow")
        self.assertEqual(state["pending_question"]["id"], "chain_protocol_resolution")
        self.assertIn("protocol family", state["pending_question"]["prompt"].lower())

        answer = answer_pending_question("1", session_id=session_id)
        state = load_workflow_state(session_id=session_id)
        self.assertTrue(answer["applied"])
        self.assertEqual(state["chain"], "flow")
        self.assertEqual(state["chain_status"], "unsupported_needs_endpoint_validation")
        self.assertEqual(state["fixture_status"]["status"], "needs_endpoint")
        self.assertEqual(state["pending_question"]["id"], "unsupported_chain_endpoint_gate")
        self.assertIn("endpoint", state["pending_question"]["prompt"].lower())

    def test_chain_identity_evidence_is_sanitized_before_visible_prompts(self):
        session_id = "unit-chain-identity-visible-evidence-sanitized"
        reset_workflow_state(session_id=session_id)
        with workflow_tool_session(session_id):
            propose_chain_identity_resolution(
                candidate_chain="flow",
                suggested_adapter_family="jsonrpc",
                evidence_summary='Flow exposes JSON-RPC. The user says "EVM", suggesting jsonrpc. The user states it is EVM-compatible. The user described it as EVM.',
                language="zh",
            )

        state = load_workflow_state(session_id=session_id)
        first_prompt = state["pending_question"]["prompt"]
        self.assertNotIn("The user", first_prompt)
        self.assertIn("用户补充", first_prompt)

        answer = answer_pending_question("1", session_id=session_id)
        self.assertTrue(answer["applied"])
        state = load_workflow_state(session_id=session_id)
        second_prompt = state["pending_question"]["prompt"]
        self.assertNotIn("The user", second_prompt)
        self.assertIn("用户补充", second_prompt)
        self.assertEqual(
            state["chain_protocol_candidate"]["evidence_summary"],
            'Flow exposes JSON-RPC. 用户补充 "EVM", 这表明 jsonrpc. 用户补充 it is EVM-compatible. 用户补充 it as EVM.',
        )

    def test_unknown_chain_no_endpoint_handoff_can_skip_endpoint_gate(self):
        session_id = "unit-unknown-chain-no-endpoint-handoff"
        reset_workflow_state(session_id=session_id)
        with workflow_tool_session(session_id):
            propose_chain_identity_resolution(
                candidate_chain="foochain",
                suggested_adapter_family="jsonrpc",
                evidence_summary='FooChain is not supported. The user described it as EVM JSON-RPC.',
                language="zh",
            )
            result = request_unsupported_chain_handoff(
                reason="用户没有 endpoint，希望给另一个 AI 开发交接",
                language="zh",
            )

        state = load_workflow_state(session_id=session_id)
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(state["chain"], "foochain")
        self.assertEqual(state["chain_status"], "unsupported_needs_development_handoff")
        self.assertEqual(state["fixture_status"]["status"], "needs_review")
        self.assertFalse(state.get("pending_question"))
        self.assertIn("endpoint/request/response", state["blockers"][0])
        self.assertNotIn("The user", state["chain_identity_candidate"]["evidence_summary"])

    def test_chain_identity_evidence_removes_visible_response_and_tool_repr(self):
        session_id = "unit-chain-identity-visible-envelope-sanitized"
        reset_workflow_state(session_id=session_id)
        with workflow_tool_session(session_id):
            propose_chain_identity_resolution(
                candidate_chain="sola",
                suggested_adapter_family="unknown",
                evidence_summary="internal notes VISIBLE_RESPONSE: `sola` may be a typo for `solana`.",
                language="en",
            )
        state = load_workflow_state(session_id=session_id)
        self.assertNotIn("VISIBLE_RESPONSE", state["pending_question"]["prompt"])
        self.assertEqual(state["chain_identity_candidate"]["evidence_summary"], "`sola` may be a typo for `solana`.")

        session_id = "unit-chain-identity-tool-repr-hidden"
        reset_workflow_state(session_id=session_id)
        with workflow_tool_session(session_id):
            propose_chain_identity_resolution(
                candidate_chain="sola",
                suggested_adapter_family="unknown",
                evidence_summary="model_version='deepseek' content=Content(parts=[Part(function_call=FunctionCall(...))])",
                language="en",
            )
        state = load_workflow_state(session_id=session_id)
        self.assertEqual(state["chain_identity_candidate"].get("evidence_summary", ""), "")
        self.assertNotIn("FunctionCall", state["pending_question"]["prompt"])

    def test_chain_identity_resolution_routes_unsupported_family_to_handoff(self):
        session_id = "unit-chain-identity-unsupported-family-handoff"
        reset_workflow_state(session_id=session_id)
        with workflow_tool_session(session_id):
            propose_chain_identity_resolution(
                candidate_chain="unknown-new-chain",
                suggested_adapter_family="unknown",
                evidence_summary="No supported adapter family could be confirmed",
                language="en",
            )
        answer = answer_pending_question("1", session_id=session_id)

        state = load_workflow_state(session_id=session_id)
        self.assertTrue(answer["applied"])
        self.assertEqual(state["chain"], "unknown-new-chain")
        self.assertEqual(state["chain_status"], "protocol_needs_confirmation")
        self.assertEqual(state["pending_question"]["id"], "chain_protocol_resolution")

        answer = answer_pending_question("1", session_id=session_id)

        state = load_workflow_state(session_id=session_id)
        self.assertTrue(answer["applied"])
        self.assertEqual(state["chain"], "unknown-new-chain")
        self.assertEqual(state["chain_status"], "unsupported_needs_development_handoff")
        self.assertEqual(state["fixture_status"]["status"], "needs_review")
        self.assertFalse(state.get("pending_question"))

    def test_chain_change_guard_accepts_exact_supported_chain_names_in_natural_language(self):
        session_id = "unit-chain-change-natural-language-supported-chain"
        reset_workflow_state(session_id=session_id)
        update_workflow_state(
            {
                "target_mode": "real-node",
                "chain": "ethereum",
                "confirmed_config": {"TARGET_MODE": "real-node", "BLOCKCHAIN_NODE": "ethereum"},
                "last_user_change": {
                    "type": "pending_question_interruption",
                    "raw": "我改成 solana，并先用 fake-node 验证",
                    "pending_question_id": "cloud_region",
                },
            },
            session_id=session_id,
        )
        with workflow_tool_session(session_id):
            propose_chain_change_confirmation(
                new_chain="solana",
                previous_chain="ethereum",
                target_mode="fake-node",
                target_mode_explicit=True,
                language="zh",
            )
        state = load_workflow_state(session_id=session_id)
        self.assertEqual(state["pending_question"]["id"], "confirm_chain_change")
        self.assertIn("solana", state["pending_question"]["prompt"])
        self.assertNotEqual(state.get("chain_status"), "ambiguous_or_unknown")

    def test_chain_change_guard_does_not_collapse_long_new_chain_names_to_alias(self):
        session_id = "unit-chain-change-natural-language-new-chain-not-alias"
        reset_workflow_state(session_id=session_id)
        update_workflow_state(
            {
                "target_mode": "fake-node",
                "chain": "solana",
                "confirmed_config": {"TARGET_MODE": "fake-node", "BLOCKCHAIN_NODE": "solana"},
                "last_user_change": {
                    "type": "pending_question_interruption",
                    "raw": "我改成 BNB Greenfield，仍然用 fake-node",
                    "pending_question_id": "data_vol_type",
                },
            },
            session_id=session_id,
        )
        with workflow_tool_session(session_id):
            propose_chain_change_confirmation(
                new_chain="bsc",
                previous_chain="solana",
                target_mode="fake-node",
                target_mode_explicit=True,
                language="zh",
            )
        state = load_workflow_state(session_id=session_id)
        self.assertEqual(state["pending_question"]["id"], "chain_selection")
        self.assertEqual(state["chain_status"], "ambiguous_or_unknown")
        self.assertNotEqual(state.get("chain"), "bsc")

    def test_callback_blocks_downstream_setup_question_before_explicit_target_is_recorded(self):
        result = before_tool_callback(
            FakeTool("propose_disk_device_choice"),
            {"role": "ledger"},
            FakeToolContext(
                {
                    "user_text": "我要测试 solana fake-node quick，请逐项确认环境变量",
                    "workflow_state": {"active_workflow": "", "chain": "", "target_mode": ""},
                }
            ),
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "blocked")
        self.assertIn("chain=solana", result["data"]["required_state"])
        self.assertIn("target_mode=fake-node", result["data"]["required_state"])
        self.assertIn("benchmark_profile=quick", result["data"]["required_state"])

    def test_callback_blocks_downstream_setup_when_group_is_not_active(self):
        result = before_tool_callback(
            FakeTool("propose_disk_device_choice"),
            {"role": "ledger"},
            FakeToolContext(
                {
                    "user_text": "我要测试 solana fake-node quick，请逐项确认环境变量",
                    "workflow_state": {
                        "active_workflow": "benchmark_setup",
                        "chain": "solana",
                        "target_mode": "fake-node",
                        "benchmark_profile": {"mode": "quick"},
                    },
                }
            ),
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["data"]["reason"], "downstream setup question tool called outside its workflow group")
        self.assertIn("ledger_disk", result["data"]["expected_groups"])

    def test_callback_allows_downstream_setup_when_matching_group_is_active(self):
        result = before_tool_callback(
            FakeTool("propose_disk_device_choice"),
            {"role": "ledger"},
            FakeToolContext(
                {
                    "user_text": "我要测试 solana fake-node quick，请逐项确认环境变量",
                    "workflow_state": {
                        "active_workflow": "benchmark_setup",
                        "active_group": "ledger_disk",
                        "next_blocking_group": "ledger_disk",
                        "chain": "solana",
                        "target_mode": "fake-node",
                        "benchmark_profile": {"mode": "quick"},
                    },
                }
            ),
        )
        self.assertIsNone(result)

    def test_callback_allows_downstream_setup_when_matching_group_is_next_blocking(self):
        result = before_tool_callback(
            FakeTool("propose_benchmark_profile_choice"),
            {},
            FakeToolContext(
                {
                    "user_text": "我要测试 solana fake-node quick",
                    "workflow_state": {
                        "active_workflow": "benchmark_setup",
                        "active_group": "",
                        "next_blocking_group": "qps_profile",
                        "chain": "solana",
                        "target_mode": "fake-node",
                        "benchmark_profile": {"mode": "quick"},
                    },
                }
            ),
        )
        self.assertIsNone(result)

    def test_callback_blocks_workload_setup_when_qps_group_is_active(self):
        result = before_tool_callback(
            FakeTool("propose_workload_customization_choice"),
            {},
            FakeToolContext(
                {
                    "user_text": "我要调整 workload",
                    "workflow_state": {
                        "active_workflow": "benchmark_setup",
                        "active_group": "qps_profile",
                        "next_blocking_group": "qps_profile",
                        "chain": "solana",
                        "target_mode": "fake-node",
                    },
                }
            ),
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "blocked")
        self.assertIn("workload_rpc", result["data"]["expected_groups"])

    def test_callback_does_not_collapse_new_chain_name_to_supported_alias(self):
        result = before_tool_callback(
            FakeTool("propose_disk_device_choice"),
            {"role": "ledger"},
            FakeToolContext(
                {
                    "user_text": "test bnb greenfield fake-node",
                    "workflow_state": {"active_workflow": "benchmark_setup", "chain": "", "target_mode": ""},
                }
            ),
        )
        self.assertIsNotNone(result)
        self.assertNotIn("chain=bsc", result["data"]["required_state"])
        self.assertIn("target_mode=fake-node", result["data"]["required_state"])

    def test_callback_records_bnb_alias_with_rpc_workload_terms(self):
        session_id = "unit-callback-records-bnb-alias-with-rpc-workload-terms"
        reset_workflow_state(session_id=session_id)
        after_model_callback(
            FakeToolContext(
                {
                    "session_id": session_id,
                    "user_text": "我现在要换成 BNB，并重新确认 RPC method 和权重 fake-node",
                    "workflow_state": load_workflow_state(session_id=session_id),
                }
            ),
            object(),
        )
        state = load_workflow_state(session_id=session_id)
        self.assertEqual(state["chain"], "bsc")
        self.assertEqual(state["target_mode"], "fake-node")

    def test_after_model_callback_routes_explicit_custom_rpc_request_to_endpoint_gate(self):
        session_id = "unit-after-model-custom-rpc-route"
        reset_workflow_state(session_id=session_id)
        after_model_callback(
            FakeToolContext(
                {
                    "session_id": session_id,
                    "terminal_language": "zh",
                    "user_text": "我想给 solana 增加一个自定义 rpc method，需要 3 个参数",
                    "workflow_state": load_workflow_state(session_id=session_id),
                }
            ),
            object(),
        )
        state = load_workflow_state(session_id=session_id)
        self.assertEqual(state["chain"], "solana")
        self.assertEqual(state["active_group"], "target_samples_fixtures")
        self.assertEqual(state["workflow_step"], "custom_rpc_requested")
        self.assertIn("ask:custom_rpc_endpoint_gate", state["allowed_next_actions"])

    def test_after_model_callback_records_exact_benchmark_entities_only(self):
        session_id = "unit-after-model-records-exact-benchmark-entities"
        reset_workflow_state(session_id=session_id)
        result = after_model_callback(
            FakeToolContext(
                {
                    "session_id": session_id,
                    "user_text": "我要测试 solana fake-node quick，请逐项确认环境变量",
                    "workflow_state": load_workflow_state(session_id=session_id),
                }
            ),
            object(),
        )
        self.assertIsNone(result)
        state = load_workflow_state(session_id=session_id)
        self.assertEqual(state["chain"], "solana")
        self.assertEqual(state["target_mode"], "fake-node")
        self.assertEqual(state["benchmark_profile"]["mode"], "quick")
        self.assertEqual(state["confirmed_config"]["BLOCKCHAIN_NODE"], "solana")
        self.assertEqual(state["confirmed_config"]["TARGET_MODE"], "fake-node")
        self.assertEqual(state["confirmed_config"]["benchmark_mode_confirmed"], "quick")

        ambiguous_session = "unit-after-model-does-not-collapse-bnb-greenfield"
        reset_workflow_state(session_id=ambiguous_session)
        after_model_callback(
            FakeToolContext(
                {
                    "session_id": ambiguous_session,
                    "user_text": "test bnb greenfield fake-node",
                    "workflow_state": load_workflow_state(session_id=ambiguous_session),
                }
            ),
            object(),
        )
        ambiguous_state = load_workflow_state(session_id=ambiguous_session)
        self.assertEqual(ambiguous_state.get("chain"), "")
        self.assertNotEqual(ambiguous_state.get("confirmed_config", {}).get("BLOCKCHAIN_NODE"), "bsc")
        self.assertEqual(ambiguous_state.get("target_mode"), "fake-node")

    def test_ctrl_c_during_adk_turn_returns_to_session(self):
        io = CapturingIO()
        with tempfile.TemporaryDirectory() as tmp:
            app = AnyChainTerminal(
                state=TerminalSession(language="zh"),
                store=TerminalSessionStore(Path(tmp) / "session.json"),
                io=io,
                bridge_factory=lambda: CancellingBridge(),
            )
            app._adk_available = True
            app.handle_user_text("我要测试")

        self.assertTrue(any("已取消当前这一轮" in item for item in io.messages))

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

    def test_startup_completed_job_clears_stale_config_pending_question(self):
        fake_bridge = FakeBridge()
        io = CapturingIO()
        session_id = "unit-clear-stale-pending"
        reset_workflow_state(session_id=session_id)
        update_workflow_state(
            {
                "pending_question": {
                    "id": "observability_mode_choice",
                    "kind": "numbered_choice",
                    "prompt": "Choose observability mode",
                    "field": "observability_choice_confirmed",
                    "options": [
                        {"id": "1", "value": "disabled", "label": "disabled"},
                        {"id": "2", "value": "local", "label": "local"},
                    ],
                }
            },
            session_id=session_id,
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = TerminalSessionStore(Path(tmp) / "session.json")
            app = AnyChainTerminal(
                state=TerminalSession(language="zh"),
                store=store,
                io=io,
                bridge_factory=lambda: fake_bridge,
                session_id=session_id,
            )
            with patch("terminal.repl.load_startup_state") as startup_mock, \
                patch("terminal.repl.adk_status") as adk_status_mock, \
                patch("terminal.repl.runner_bridge_status") as bridge_status_mock, \
                patch("terminal.repl.web_research_status") as web_research_mock, \
                patch("terminal.repl.run_doctor") as doctor_mock, \
                patch("terminal.repl.load_framework_context") as context_mock, \
                patch("terminal.repl.load_framework_capabilities") as capabilities_mock:
                startup_mock.return_value = {
                    "latest_job": {"job_id": "job_done", "status": "completed"},
                    "next_actions": ["ask: analyze latest job", "start a new benchmark"],
                }
                adk_status_mock.return_value.as_dict.return_value = {"available": True, "reason": "ok"}
                bridge_status_mock.return_value.as_dict.return_value = {"available": True, "reason": "ok"}
                web_research_mock.return_value.as_dict.return_value = {
                    "enabled": False,
                    "mode": "disabled",
                    "reason": "unavailable for current provider",
                }
                doctor_mock.return_value = {
                    "status": "ready",
                    "capabilities": {"chain_count": 36, "unique_rpc_method_count": 109},
                    "environment": {
                        "cloud": {"provider": "other", "platform": "container"},
                        "deployment": {"type": "container"},
                        "host": {"cpu_count": 10, "memory_gib": 7.75},
                        "network": {"default_interface": "eth0"},
                        "disks": {"candidates": [], "proposed_ledger_device": "vda1"},
                        "dependencies": {"missing_required": []},
                    },
                }
                context_mock.return_value = {"capability_summary": {"chain_count": 36, "family_count": 6, "unique_rpc_method_count": 109}}
                capabilities_mock.return_value = {"chain_count": 36, "family_count": 6, "unique_rpc_method_count": 109}
                app.startup()
                app.handle_user_text("Hi")

        state = load_workflow_state(session_id=session_id)
        self.assertEqual(state["pending_question"], {})
        self.assertEqual(len(fake_bridge.calls), 1)
        self.assertEqual(fake_bridge.calls[0][0], "Hi")
        self.assertFalse(any("benchmark 计划" in item for item in io.messages))

    def test_startup_completed_job_clears_stale_benchmark_context(self):
        fake_bridge = FakeBridge()
        io = CapturingIO()
        session_id = "unit-clear-stale-context"
        reset_workflow_state(session_id=session_id)
        update_workflow_state(
            {
                "active_intent": "benchmark",
                "active_workflow": "benchmark_setup",
                "workflow_step": "workload_customization_choice",
                "target_mode": "fake-node",
                "chain": "solana",
                "rpc_mode": "mixed",
                "mixed_weights": {"getSlot": 70, "getBlockHeight": 30},
                "benchmark_profile": {"name": "quick"},
                "allowed_next_actions": ["ask:workload_customization_choice"],
            },
            session_id=session_id,
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = TerminalSessionStore(Path(tmp) / "session.json")
            app = AnyChainTerminal(
                state=TerminalSession(language="zh"),
                store=store,
                io=io,
                bridge_factory=lambda: fake_bridge,
                session_id=session_id,
            )
            with patch("terminal.repl.load_startup_state") as startup_mock, \
                patch("terminal.repl.adk_status") as adk_status_mock, \
                patch("terminal.repl.runner_bridge_status") as bridge_status_mock, \
                patch("terminal.repl.web_research_status") as web_research_mock, \
                patch("terminal.repl.run_doctor") as doctor_mock, \
                patch("terminal.repl.load_framework_context") as context_mock, \
                patch("terminal.repl.load_framework_capabilities") as capabilities_mock:
                startup_mock.return_value = {
                    "latest_job": {"job_id": "job_done", "status": "completed"},
                    "next_actions": ["ask: analyze latest job", "start a new benchmark"],
                }
                adk_status_mock.return_value.as_dict.return_value = {"available": True, "reason": "ok"}
                bridge_status_mock.return_value.as_dict.return_value = {"available": True, "reason": "ok"}
                web_research_mock.return_value.as_dict.return_value = {
                    "enabled": False,
                    "mode": "disabled",
                    "reason": "unavailable for current provider",
                }
                doctor_mock.return_value = {
                    "status": "ready",
                    "capabilities": {"chain_count": 36, "unique_rpc_method_count": 109},
                    "environment": {
                        "cloud": {"provider": "other", "platform": "container"},
                        "deployment": {"type": "container"},
                        "host": {"cpu_count": 10, "memory_gib": 7.75},
                        "network": {"default_interface": "eth0"},
                        "disks": {"candidates": [], "proposed_ledger_device": "vda1"},
                        "dependencies": {"missing_required": []},
                    },
                }
                context_mock.return_value = {"capability_summary": {"chain_count": 36, "family_count": 6, "unique_rpc_method_count": 109}}
                capabilities_mock.return_value = {"chain_count": 36, "family_count": 6, "unique_rpc_method_count": 109}
                app.startup()
                app.handle_user_text("Hi")

        state = load_workflow_state(session_id=session_id)
        self.assertEqual(state["chain"], "")
        self.assertEqual(state["rpc_mode"], "")
        self.assertEqual(state["mixed_weights"], {})
        self.assertEqual(state["benchmark_profile"], {})
        self.assertEqual(state["allowed_next_actions"], [])
        self.assertEqual(fake_bridge.calls[0][0], "Hi")

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

    def test_terminal_overrides_stale_model_text_when_handoff_state_is_needs_review(self):
        session_id = "unit-terminal-needs-review-state-override"
        reset_workflow_state(session_id=session_id)
        update_workflow_state(
            {
                "chain": "foochain",
                "workflow_step": "unsupported_chain_handoff_requested",
                "fixture_status": {"status": "needs_review"},
                "blockers": ["foochain requires endpoint/request/response evidence before fixtures or smoke can run"],
            },
            session_id=session_id,
        )
        app = AnyChainTerminal(
            state=TerminalSession(language="zh"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=CapturingIO(),
            session_id=session_id,
        )

        result = app._terminal_state_override_after_adk()

        self.assertTrue(result["suppress_response"])
        self.assertIn("needs_review", result["message"])
        self.assertIn("缺少已验证的 endpoint", result["message"])

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
        with patch("terminal.job_commands.tail_job_log") as tail_mock:
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

    def test_status_command_accepts_explicit_job_id_without_adk_routing(self):
        fake_bridge = FakeBridge()
        io = CapturingIO()
        app = AnyChainTerminal(
            state=TerminalSession(language="en", latest_job_id="job_latest"),
            store=TerminalSessionStore(Path("/tmp/unused-session.json")),
            io=io,
            bridge_factory=lambda: fake_bridge,
        )
        app._adk_available = True
        with patch("terminal.job_commands.get_job") as get_job_mock:
            get_job_mock.return_value = {"job_id": "job_abc", "status": "running"}
            app.handle_user_text("status job_abc")

        self.assertEqual(fake_bridge.calls, [])
        self.assertTrue(any("job_abc" in item and "running" in item for item in io.messages))

    def test_follow_ctrl_c_stops_log_follow_without_stopping_job_or_agent(self):
        fake_bridge = FakeBridge()
        io = CapturingIO()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "job_abc"
            run_dir.mkdir()
            (run_dir / "benchmark.log").write_text("benchmark still running\n", encoding="utf-8")
            app = AnyChainTerminal(
                state=TerminalSession(language="en", latest_job_id="job_abc"),
                store=TerminalSessionStore(Path(tmp) / "session.json"),
                io=io,
                bridge_factory=lambda: fake_bridge,
            )
            app._adk_available = True
            with patch("terminal.job_commands.get_job") as get_job_mock, \
                patch("terminal.job_commands.time.sleep", side_effect=KeyboardInterrupt):
                get_job_mock.return_value = {"job_id": "job_abc", "status": "running", "run_dir": str(run_dir)}
                app.handle_user_text("follow job_abc")
                app.handle_user_text("status job_abc")

        self.assertEqual(fake_bridge.calls, [])
        self.assertTrue(any("benchmark still running" in item for item in io.messages))
        self.assertTrue(any("Stopped log-follow mode" in item for item in io.messages))
        self.assertTrue(any("job_abc" in item and "running" in item for item in io.messages))

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
    def test_default_workflow_state_has_group_progress_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertEqual(list(loaded["group_progress"].keys()), list(DEFAULT_GROUP_ORDER))
        self.assertEqual(loaded["active_group"], "")
        self.assertEqual(loaded["last_completed_group"], "")
        self.assertEqual(loaded["next_blocking_group"], "")
        self.assertEqual(loaded["confirmed_fields"], [])
        self.assertEqual(loaded["invalidated_fields"], [])
        self.assertEqual(loaded["interruption_stack"], [])
        for group in DEFAULT_GROUP_ORDER:
            self.assertEqual(loaded["group_progress"][group]["status"], "pending")
            self.assertEqual(loaded["group_progress"][group]["confirmed_fields"], [])
            self.assertEqual(loaded["group_progress"][group]["invalidated_fields"], [])

    def test_workflow_state_normalizes_group_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            update_workflow_state(
                {
                    "active_group": "ledger-disk",
                    "last_completed_group": "hardware-discovery",
                    "next_blocking_group": "accounts-disk",
                    "confirmed_fields": ["LEDGER_DEVICE", "LEDGER_DEVICE", "DATA_VOL_TYPE"],
                    "invalidated_fields": ["smoke_result", "", "smoke_result"],
                    "group_progress": {
                        "ledger-disk": {
                            "status": "in_progress",
                            "confirmed_fields": ["LEDGER_DEVICE", "LEDGER_DEVICE"],
                            "invalidated_fields": ["smoke_result"],
                            "evidence": [{"source": "unit"}],
                        },
                        "not-a-real-group": {"status": "complete"},
                    },
                    "interruption_stack": [
                        {"group": "network", "reason": "user changed chain"},
                        {"group": "bad-group", "reason": "ignore"},
                    ],
                },
                reason="group state",
                session_id="test-session",
                state_root=root,
            )
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertEqual(loaded["active_group"], "ledger_disk")
        self.assertEqual(loaded["last_completed_group"], "hardware_discovery")
        self.assertEqual(loaded["next_blocking_group"], "accounts_disk")
        self.assertEqual(loaded["confirmed_fields"], ["LEDGER_DEVICE", "DATA_VOL_TYPE"])
        self.assertEqual(loaded["invalidated_fields"], ["smoke_result"])
        self.assertEqual(loaded["group_progress"]["ledger_disk"]["status"], "in_progress")
        self.assertEqual(loaded["group_progress"]["ledger_disk"]["confirmed_fields"], ["LEDGER_DEVICE"])
        self.assertEqual(loaded["group_progress"]["ledger_disk"]["invalidated_fields"], ["smoke_result"])
        self.assertEqual(loaded["group_progress"]["ledger_disk"]["evidence"], [{"source": "unit"}])
        self.assertEqual(loaded["interruption_stack"][0]["group"], "network")
        self.assertEqual(len(loaded["interruption_stack"]), 1)

    def test_workflow_state_rejects_unknown_group_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            update_workflow_state(
                {
                    "active_group": "keyword_router",
                    "last_completed_group": "prompt_patch",
                    "next_blocking_group": "fuzzy_match",
                },
                reason="bad groups",
                session_id="test-session",
                state_root=root,
            )
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertEqual(loaded["active_group"], "")
        self.assertEqual(loaded["last_completed_group"], "")
        self.assertEqual(loaded["next_blocking_group"], "")

    def test_record_group_jump_pauses_active_group_and_clears_stale_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            update_workflow_state(
                {
                    "active_workflow": "benchmark_setup",
                    "active_group": "ledger_disk",
                    "group_progress": {
                        "ledger_disk": {"status": "in_progress"},
                    },
                    "pending_question": {
                        "id": "disk_ledger_choice",
                        "kind": "device",
                        "prompt": "Choose ledger disk",
                        "field": "LEDGER_DEVICE",
                    },
                },
                reason="seed ledger group",
                session_id="test-session",
                state_root=root,
            )
            result = record_group_jump(
                "workload-rpc",
                reason="user changed rpc setup",
                session_id="test-session",
                state_root=root,
            )
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertTrue(result["applied"])
        self.assertEqual(result["paused_group"], "ledger_disk")
        self.assertEqual(result["target_group"], "workload_rpc")
        self.assertEqual(loaded["active_group"], "workload_rpc")
        self.assertEqual(loaded["next_blocking_group"], "workload_rpc")
        self.assertEqual(loaded["workflow_step"], "workload_rpc")
        self.assertEqual(loaded["pending_question"], {})
        self.assertEqual(loaded["group_progress"]["ledger_disk"]["status"], "in_progress")
        self.assertEqual(loaded["group_progress"]["workload_rpc"]["status"], "in_progress")
        self.assertEqual(len(loaded["interruption_stack"]), 1)
        self.assertEqual(loaded["interruption_stack"][0]["group"], "ledger_disk")
        self.assertEqual(loaded["interruption_stack"][0]["pending_question"]["id"], "disk_ledger_choice")

    def test_record_group_jump_rejects_unknown_group_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            update_workflow_state(
                {"active_group": "network"},
                reason="seed group",
                session_id="test-session",
                state_root=root,
            )
            result = record_group_jump(
                "keyword-router",
                reason="bad group",
                session_id="test-session",
                state_root=root,
            )
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertFalse(result["applied"])
        self.assertEqual(loaded["active_group"], "network")
        self.assertEqual(loaded["pending_question"], {})
        self.assertIn("unknown workflow group", result["blockers"][0])

    def test_recompute_next_blocking_group_uses_group_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            update_workflow_state(
                {
                    "active_group": "",
                    "next_blocking_group": "",
                    "group_progress": {
                        "provider_deployment": {"status": "complete"},
                        "hardware_discovery": {"status": "complete"},
                        "ledger_disk": {"status": "complete"},
                        "accounts_disk": {"status": "invalidated"},
                        "network": {"status": "pending"},
                    },
                    "pending_question": {
                        "id": "cloud_region",
                        "kind": "manual_value",
                        "prompt": "stale cloud prompt",
                        "field": "CLOUD_REGION",
                    },
                },
                reason="seed group progress",
                session_id="test-session",
                state_root=root,
            )
            result = recompute_next_blocking_group(
                reason="unit recompute",
                session_id="test-session",
                state_root=root,
            )
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertEqual(result["next_blocking_group"], "accounts_disk")
        self.assertEqual(loaded["next_blocking_group"], "accounts_disk")
        self.assertEqual(loaded["active_group"], "accounts_disk")
        self.assertEqual(loaded["pending_question"]["id"], "cloud_region")

    def test_recompute_next_blocking_group_treats_pending_as_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            update_workflow_state(
                {
                    "group_progress": {
                        "provider_deployment": {"status": "complete"},
                        "hardware_discovery": {"status": "complete"},
                        "ledger_disk": {"status": "complete"},
                        "accounts_disk": {"status": "complete"},
                        "network": {"status": "pending"},
                    },
                },
                reason="seed completed groups",
                session_id="test-session",
                state_root=root,
            )
            result = recompute_next_blocking_group(
                reason="unit recompute pending",
                session_id="test-session",
                state_root=root,
            )
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertEqual(result["next_blocking_group"], "network")
        self.assertEqual(loaded["next_blocking_group"], "network")
        self.assertEqual(loaded["active_group"], "network")

    def test_chain_change_invalidates_downstream_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            update_workflow_state(
                {
                    "chain": "solana",
                    "rpc_methods": ["getAccountInfo"],
                    "endpoint_validation": {"status": "ok"},
                    "fixture_status": {"status": "ok"},
                    "preflight_result": {"status": "pass"},
                    "smoke_result": {"status": "pass"},
                    "approval": {"approved": True},
                },
                reason="seed evidence",
                session_id="test-session",
                state_root=root,
            )
            update_workflow_state(
                {"chain": "ethereum"},
                reason="change chain",
                session_id="test-session",
                state_root=root,
            )
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertEqual(loaded["chain"], "ethereum")
        self.assertEqual(loaded["endpoint_validation"], {})
        self.assertEqual(loaded["fixture_status"], {})
        self.assertEqual(loaded["preflight_result"], {})
        self.assertEqual(loaded["smoke_result"], {})
        self.assertEqual(loaded["approval"], {})
        self.assertIn("smoke_result", loaded["invalidated_fields"])
        self.assertIn("fixture_status", loaded["invalidated_fields"])
        self.assertEqual(loaded["group_progress"]["workload_rpc"]["status"], "invalidated")
        self.assertEqual(loaded["next_blocking_group"], "chain_target")

    def test_disk_change_invalidates_execution_evidence_without_clearing_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            update_workflow_state(
                {
                    "chain": "solana",
                    "confirmed_config": {"LEDGER_DEVICE": "vda"},
                    "preflight_result": {"status": "pass"},
                    "smoke_result": {"status": "pass"},
                    "approval": {"approved": True},
                },
                reason="seed disk",
                session_id="test-session",
                state_root=root,
            )
            update_workflow_state(
                {"confirmed_config": {"LEDGER_DEVICE": "vdb"}},
                reason="change disk",
                session_id="test-session",
                state_root=root,
            )
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertEqual(loaded["chain"], "solana")
        self.assertEqual(loaded["confirmed_config"]["LEDGER_DEVICE"], "vdb")
        self.assertEqual(loaded["preflight_result"], {})
        self.assertEqual(loaded["smoke_result"], {})
        self.assertEqual(loaded["approval"], {})
        self.assertIn("disk_baseline", loaded["invalidated_fields"])
        self.assertEqual(loaded["next_blocking_group"], "ledger_disk")

    def test_qps_change_invalidates_execution_evidence_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            update_workflow_state(
                {
                    "chain": "solana",
                    "fixture_status": {"status": "ok"},
                    "benchmark_profile": {"mode": "quick", "initial_qps": 1},
                    "preflight_result": {"status": "pass"},
                    "smoke_result": {"status": "pass"},
                },
                reason="seed profile",
                session_id="test-session",
                state_root=root,
            )
            update_workflow_state(
                {"benchmark_profile": {"initial_qps": 10}},
                reason="change qps",
                session_id="test-session",
                state_root=root,
            )
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertEqual(loaded["fixture_status"], {"status": "ok"})
        self.assertEqual(loaded["benchmark_profile"]["mode"], "quick")
        self.assertEqual(loaded["benchmark_profile"]["initial_qps"], 10)
        self.assertEqual(loaded["preflight_result"], {})
        self.assertEqual(loaded["smoke_result"], {})
        self.assertIn("benchmark_profile", loaded["invalidated_fields"])
        self.assertEqual(loaded["next_blocking_group"], "qps_profile")

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

    def test_workflow_state_preserves_canonical_target_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            update_workflow_state(
                {"target_mode": "fake-node"},
                reason="normalize target mode",
                session_id="test-session",
                state_root=root,
            )
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertEqual(loaded["target_mode"], "fake-node")

    def test_adk_workflow_tools_inherit_current_terminal_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            with workflow_tool_session("current-cli-session", state_root=root):
                adk_update_workflow_state(
                    {"chain": "solana"},
                    reason="session inheritance test",
                )
            current = load_workflow_state(session_id="current-cli-session", state_root=root)
            default = load_workflow_state(session_id="terminal-session", state_root=root)

        self.assertEqual(current["chain"], "solana")
        self.assertEqual(default["chain"], "")

    def test_adk_group_jump_tool_inherits_current_terminal_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            update_workflow_state(
                {"active_group": "ledger_disk"},
                reason="seed active group",
                session_id="current-cli-session",
                state_root=root,
            )
            with workflow_tool_session("current-cli-session", state_root=root):
                result = adk_record_group_jump(
                    "workload_rpc",
                    reason="adk route user interruption",
                )
            current = load_workflow_state(session_id="current-cli-session", state_root=root)
            default = load_workflow_state(session_id="terminal-session", state_root=root)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(current["active_group"], "workload_rpc")
        self.assertEqual(current["interruption_stack"][0]["group"], "ledger_disk")
        self.assertEqual(default["active_group"], "")

    def test_workflow_state_normalizes_pending_question_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            result = update_workflow_state(
                {
                    "pending_question": {
                        "id": "ledger_device",
                        "prompt": "Which disk is LEDGER_DEVICE?",
                        "expected_answer": "device",
                        "options": [{"value": "sda"}, {"value": "sdb"}],
                        "field": "LEDGER_DEVICE",
                        "branch": "benchmark_setup",
                        "workflow_step": "confirm_ledger_device",
                        "validation_tool": "validate_required_config",
                        "source_tool": "build_missing_config_questions",
                        "next_on_yes": {
                            "workflow_step": "confirm_accounts_device",
                            "tool": "build_missing_config_questions",
                            "state_patch": {"confirmed_config": {"LEDGER_DEVICE": "sda"}},
                        },
                        "next_on_no": {
                            "workflow_step": "manual_ledger_device",
                            "message": "ask for a manual device path",
                        },
                    }
                },
                reason="ask disk",
                session_id="test-session",
                state_root=root,
            )
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertEqual(result["validation_warnings"], [])
        self.assertEqual(loaded["pending_question"]["expected_answer"], "device")
        self.assertTrue(loaded["pending_question"]["allow_manual_input"])
        self.assertEqual([item["value"] for item in loaded["pending_question"]["options"]], ["sda", "sdb"])
        self.assertEqual(loaded["pending_question"]["branch"], "benchmark_setup")
        self.assertEqual(loaded["pending_question"]["workflow_step"], "confirm_ledger_device")
        self.assertEqual(loaded["pending_question"]["validation_tool"], "validate_required_config")
        self.assertEqual(loaded["pending_question"]["next_on_yes"]["workflow_step"], "confirm_accounts_device")
        self.assertEqual(loaded["pending_question"]["next_on_yes"]["state_patch"]["confirmed_config"]["LEDGER_DEVICE"], "sda")
        self.assertEqual(loaded["pending_question"]["next_on_no"]["workflow_step"], "manual_ledger_device")
        self.assertTrue(loaded["pending_question"]["requires_revalidation"])

    def test_workflow_state_rejects_untyped_pending_question_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            result = update_workflow_state(
                {"pending_question": {"id": "bad", "expected_answer": "whatever"}},
                reason="bad shape",
                session_id="test-session",
                state_root=root,
            )
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertTrue(result["validation_warnings"])
        self.assertEqual(loaded["pending_question"]["expected_answer"], "free_text")

    def test_workflow_state_rejects_unknown_pending_question_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            result = update_workflow_state(
                {
                    "pending_question": {
                        "id": "target_mode",
                        "prompt": "Fake-node or real-node?",
                        "expected_answer": "numbered_choice",
                        "options": [{"value": "fake-node"}, {"value": "real-node"}],
                        "secret_router_hint": "force fake-node",
                    }
                },
                reason="ask target",
                session_id="test-session",
                state_root=root,
            )
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertTrue(any("ignored unsupported keys" in item for item in result["validation_warnings"]))
        self.assertNotIn("secret_router_hint", loaded["pending_question"])

    def test_config_validator_distinguishes_fake_node_and_real_node(self):
        fake = validate_required_config("fake-node", {"chain": "solana", "rpc_mode": "single"})
        real = validate_required_config("real-node", {"chain": "solana", "rpc_mode": "single"})
        self.assertIn("local_rpc_url", real["missing"])
        self.assertNotIn("local_rpc_url", fake["missing"])
        self.assertNotIn("blockchain_process_names", fake["missing"])

    def test_config_questions_can_prioritize_next_blocking_group(self):
        confirmed = {
            "chain": "solana",
            "use_fake_node": True,
            "rpc_mode": "single",
        }
        network = build_missing_config_questions("fake-node", confirmed, preferred_group="network")
        qps = build_missing_config_questions("fake-node", confirmed, preferred_group="qps-profile")

        self.assertEqual(network["next_question"]["id"], "network_interface")
        self.assertEqual(qps["next_question"]["id"], "benchmark_profile_choice")

    def test_transition_executor_uses_next_blocking_group_for_config_question(self):
        session_id = "unit-next-blocking-network-group"
        reset_workflow_state(session_id=session_id)
        update_workflow_state(
            {
                "active_workflow": "benchmark_setup",
                "target_mode": "fake-node",
                "chain": "solana",
                "rpc_mode": "single",
                "next_blocking_group": "network",
            },
            reason="seed next group",
            session_id=session_id,
        )
        state = load_workflow_state(session_id=session_id)
        result = ensure_next_benchmark_setup_question(
            state,
            discovery={},
            language="en",
            session_id=session_id,
        )

        self.assertEqual(result["pending_question"]["id"], "network_interface")

    def test_config_questions_include_disk_inventory(self):
        questions = build_missing_config_questions(
            "real-node",
            {"chain": "solana", "rpc_mode": "single"},
            {"disks": {"candidates": [{"name": "sdb", "type": "disk", "size": "2T", "mountpoint": "/ledger"}]}},
        )
        ledger = next(item for item in questions["questions"] if item["id"] == "ledger_device")
        self.assertEqual(ledger["candidates"][0]["name"], "sdb")
        self.assertTrue(ledger["manual_input_allowed"])
        self.assertTrue(ledger["allow_manual_input"])
        self.assertEqual(ledger["expected_answer"], "device")
        self.assertEqual(ledger["field"], "LEDGER_DEVICE")
        self.assertEqual(ledger["validation_tool"], "validate_required_config")
        self.assertEqual(ledger["options"][0]["value"], "sdb")
        self.assertIn("next_on_manual", ledger)
        self.assertIn("benchmark_mode_confirmed", questions["missing"])
        self.assertIn("qps_profile_confirmed", questions["missing"])
        self.assertIn("observability_choice_confirmed", questions["missing"])
        qps = next(item for item in questions["questions"] if item["id"] == "qps_profile_confirmed")
        self.assertEqual(qps["interaction_mode"], "accept_defaults_or_adjust_item")
        self.assertEqual(qps["expected_answer"], "yes_no")
        self.assertIn("initial_qps", qps["parameter_descriptions"])
        self.assertIn("duration_seconds", {item["id"] for item in qps["adjustable_items"]})
        self.assertIn("config/user_config.sh", qps["prompt"])

        fake_questions = build_missing_config_questions(
            "fake-node",
            {"chain": "solana", "rpc_mode": "single"},
            {"disks": {"candidates": [{"name": "sdb", "type": "disk", "size": "2T", "mountpoint": "/ledger"}]}},
        )
        self.assertNotIn("blockchain_process_names", fake_questions["missing"])
        fake_qps = next(item for item in fake_questions["questions"] if item["id"] == "qps_profile_confirmed")
        self.assertIn("fake-node smoke", fake_qps["prompt"])
        self.assertIn("Use the selected mode's default QPS profile", render_pending_question(fake_qps, language="en"))
        self.assertNotIn("是否使用所选模式", render_pending_question(fake_qps, language="en"))
        self.assertIn("是否使用所选模式", render_pending_question(fake_qps, language="zh"))

    def test_real_node_endpoint_prompt_is_localized_by_terminal_language(self):
        question = {
            "id": "real_node_local_rpc_url",
            "kind": "url",
            "field": "LOCAL_RPC_URL",
            "prompt": (
                "A solana real-node benchmark must validate the target endpoint first.\n"
                "Provide LOCAL_RPC_URL. If missing, I will list blockers."
            ),
        }

        rendered = render_pending_question(question, language="zh")

        self.assertIn("请提供 LOCAL_RPC_URL", rendered)
        self.assertNotIn("I will", rendered)

    def test_config_next_question_walks_ledger_accounts_sequence(self):
        discovery = {
            "disks": {
                "candidates": [
                    {"name": "sdb", "type": "disk", "size": "2T", "mountpoint": "/ledger"},
                    {"name": "sdc", "type": "disk", "size": "1T", "mountpoint": "/accounts"},
                ]
            }
        }
        base = {
            "chain": "solana",
            "rpc_mode": "single",
            "cloud_region": "asia-east1",
            "cloud_zone": "asia-east1-c",
            "machine_type": "c3-standard-22",
        }
        first = build_missing_config_questions("fake-node", base, discovery)
        self.assertEqual(first["next_question"]["id"], "disk_ledger_choice")
        self.assertEqual(first["next_question"]["kind"], "device")
        self.assertEqual(first["next_question"]["options"][0]["value"], "sdb")

        ledger_done = {
            **base,
            "ledger_device": "sdb",
            "data_vol_type": "hyperdisk-extreme",
            "data_vol_size": "2000",
            "data_vol_max_iops": "30000",
            "data_vol_max_throughput": "1000",
        }
        accounts_exists = build_missing_config_questions("fake-node", ledger_done, discovery)
        self.assertEqual(accounts_exists["next_question"]["id"], "disk_accounts_exists")
        self.assertEqual(accounts_exists["next_question"]["kind"], "yes_no")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            update_workflow_state(
                {"pending_question": accounts_exists["next_question"]},
                session_id="test-session",
                state_root=root,
            )
            result = answer_pending_question("y", session_id="test-session", state_root=root)
            self.assertTrue(result["applied"])
            self.assertIs(result["state"]["confirmed_config"]["has_accounts_device"], True)

        accounts_device = build_missing_config_questions(
            "fake-node",
            {**ledger_done, "has_accounts_device": True},
            discovery,
        )
        self.assertEqual(accounts_device["next_question"]["id"], "disk_accounts_choice")
        self.assertEqual(accounts_device["next_question"]["field"], "ACCOUNTS_DEVICE")

    def test_selected_disk_size_is_reused_as_detected_current_value(self):
        discovery = {
            "disks": {
                "candidates": [
                    {"name": "sdb", "type": "disk", "size": "2T", "mountpoint": "/ledger"},
                    {"name": "sdc", "type": "disk", "size": "1T", "mountpoint": "/accounts"},
                ]
            }
        }
        base = {
            "chain": "solana",
            "rpc_mode": "single",
            "cloud_region": "asia-east1",
            "cloud_zone": "asia-east1-c",
            "machine_type": "c3-standard-22",
            "ledger_device": "sdb",
            "data_vol_type": "hyperdisk-extreme",
        }
        data_size = build_missing_config_questions("fake-node", base, discovery)["next_question"]
        self.assertEqual(data_size["id"], "data_vol_size")
        self.assertEqual(data_size["current_value"], "2048")
        self.assertIn("检测到 DATA_VOL_SIZE 为 `2048`", render_pending_question(data_size, language="zh"))
        self.assertIn("Reply `Y` to use it", render_pending_question(data_size, language="en"))

        accounts_size = build_missing_config_questions(
            "fake-node",
            {
                **base,
                "data_vol_size": "2048",
                "data_vol_max_iops": "30000",
                "data_vol_max_throughput": "1000",
                "has_accounts_device": True,
                "accounts_device": "sdc",
                "accounts_vol_type": "hyperdisk-extreme",
            },
            discovery,
        )["next_question"]
        self.assertEqual(accounts_size["id"], "accounts_vol_size")
        self.assertEqual(accounts_size["current_value"], "1024")
        self.assertIn("检测到 ACCOUNTS_VOL_SIZE 为 `1024`", render_pending_question(accounts_size, language="zh"))

    def test_direct_confirmed_config_disk_patch_is_normalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            result = update_workflow_state(
                {"confirmed_config": {"LEDGER_DEVICE": "/dev/sdb", "chain": "eth", "rpc_mode": "mixed"}},
                session_id="test-session",
                state_root=root,
            )
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertEqual(result["validation_warnings"], [])
        self.assertEqual(loaded["confirmed_config"]["LEDGER_DEVICE"], "sdb")
        self.assertEqual(loaded["confirmed_config"]["chain"], "ethereum")
        self.assertEqual(loaded["confirmed_config"]["rpc_mode"], "mixed")

    def test_direct_accounts_patch_cannot_bypass_separate_disk_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            result = update_workflow_state(
                {
                    "confirmed_config": {
                        "LEDGER_DEVICE": "sdb",
                        "has_accounts_device": True,
                        "ACCOUNTS_DEVICE": "/dev/sdb",
                    }
                },
                session_id="test-session",
                state_root=root,
            )
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertIn("ignored ACCOUNTS_DEVICE", "; ".join(result["validation_warnings"]))
        self.assertEqual(loaded["confirmed_config"]["LEDGER_DEVICE"], "sdb")
        self.assertNotIn("ACCOUNTS_DEVICE", loaded["confirmed_config"])

    def test_satisfied_pending_question_advances_from_structured_state(self):
        session_id = "unit-structured-advance-satisfied-ledger-pending"
        reset_workflow_state(session_id=session_id)
        update_workflow_state(
            {
                "active_workflow": "benchmark_setup",
                "target_mode": "fake-node",
                "chain": "solana",
                "rpc_mode": "single",
                "confirmed_config": {
                    "chain": "solana",
                    "rpc_mode": "single",
                    "cloud_region": "asia-east1",
                    "cloud_zone": "asia-east1-c",
                    "machine_type": "c3-standard-22",
                    "LEDGER_DEVICE": "/dev/sdb",
                },
                "pending_question": {
                    "id": "disk_ledger_choice",
                    "kind": "device",
                    "field": "LEDGER_DEVICE",
                    "prompt": "Choose ledger disk",
                    "workflow_step": "confirm_ledger_device",
                    "options": [{"id": "1", "label": "sdb", "value": "sdb"}],
                    "manual_input_allowed": True,
                },
            },
            session_id=session_id,
        )
        result = ensure_next_benchmark_setup_question(
            load_workflow_state(session_id=session_id),
            discovery={
                "disks": {
                    "candidates": [
                        {"name": "sdb", "type": "disk", "size": "2T", "mountpoint": "/ledger"},
                    ]
                }
            },
            language="zh",
            session_id=session_id,
        )
        loaded = load_workflow_state(session_id=session_id)

        self.assertTrue(result["advanced"])
        self.assertEqual(loaded["confirmed_config"]["LEDGER_DEVICE"], "sdb")
        self.assertEqual(loaded["pending_question"]["id"], "data_vol_type")
        self.assertIn("Ledger/data 磁盘类型", result["message"])

    def test_accounts_iops_and_throughput_prompts_are_specific(self):
        discovery = {
            "disks": {
                "candidates": [
                    {"name": "sdb", "type": "disk", "size": "2T", "mountpoint": "/ledger"},
                    {"name": "sdc", "type": "disk", "size": "1T", "mountpoint": "/accounts"},
                ]
            }
        }
        base = {
            "chain": "solana",
            "rpc_mode": "single",
            "cloud_region": "asia-east1",
            "cloud_zone": "asia-east1-c",
            "machine_type": "c3-standard-22",
            "ledger_device": "sdb",
            "data_vol_type": "hyperdisk-extreme",
            "data_vol_size": "2048",
            "data_vol_max_iops": "30000",
            "data_vol_max_throughput": "1000",
            "has_accounts_device": True,
            "accounts_device": "sdc",
            "accounts_vol_type": "hyperdisk-extreme",
            "accounts_vol_size": "1024",
        }
        iops = build_missing_config_questions("fake-node", base, discovery)["next_question"]
        self.assertEqual(iops["id"], "accounts_vol_max_iops")
        self.assertIn("ACCOUNTS_VOL_MAX_IOPS", iops["prompt"])

        throughput = build_missing_config_questions(
            "fake-node",
            {**base, "accounts_vol_max_iops": "30000"},
            discovery,
        )["next_question"]
        self.assertEqual(throughput["id"], "accounts_vol_max_throughput")
        self.assertIn("ACCOUNTS_VOL_MAX_THROUGHPUT", throughput["prompt"])

    def test_detected_network_interface_accepts_yes_confirmation(self):
        confirmed = {
            "chain": "solana",
            "rpc_mode": "single",
            "benchmark_mode_confirmed": "quick",
            "qps_profile_confirmed": True,
            "chain_template_reviewed": True,
            "rpc_workload_confirmed": True,
            "rpc_param_samples_confirmed": True,
            "cloud_region": "asia-east1",
            "cloud_zone": "asia-east1-c",
            "machine_type": "c3-standard-22",
            "blockchain_process_names": ["fake-node"],
            "ledger_device": "sdb",
            "data_vol_type": "hyperdisk-extreme",
            "data_vol_size": "2048",
            "data_vol_max_iops": "30000",
            "data_vol_max_throughput": "1000",
            "has_accounts_device": False,
        }
        question = build_missing_config_questions(
            "fake-node",
            confirmed,
            {"network": {"default_interface": "eth0"}},
        )["next_question"]
        self.assertEqual(question["id"], "network_interface")
        self.assertEqual(question["kind"], "manual_value")
        self.assertEqual(question["current_value"], "eth0")
        self.assertIn("检测到 NETWORK_INTERFACE 为 `eth0`", render_pending_question(question, language="zh"))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            update_workflow_state(
                {"pending_question": question},
                session_id="test-session",
                state_root=root,
            )
            result = answer_pending_question("Y", session_id="test-session", state_root=root)
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertTrue(result["applied"])
        self.assertEqual(loaded["confirmed_config"]["NETWORK_INTERFACE"], "eth0")

    def test_network_interface_manual_override_then_bandwidth_prompt(self):
        question = {
            "id": "network_interface",
            "kind": "manual_value",
            "field": "NETWORK_INTERFACE",
            "prompt": "Confirm network interface.",
            "current_value": "eth0",
            "manual_input_allowed": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            update_workflow_state(
                {"pending_question": question},
                session_id="test-session",
                state_root=root,
            )
            result = answer_pending_question("ens4", session_id="test-session", state_root=root)
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertTrue(result["applied"])
        self.assertEqual(loaded["confirmed_config"]["NETWORK_INTERFACE"], "ens4")

        confirmed = {
            "chain": "solana",
            "rpc_mode": "single",
            "benchmark_mode_confirmed": "quick",
            "qps_profile_confirmed": True,
            "chain_template_reviewed": True,
            "rpc_workload_confirmed": True,
            "rpc_param_samples_confirmed": True,
            "cloud_region": "asia-east1",
            "cloud_zone": "asia-east1-c",
            "machine_type": "c3-standard-22",
            "blockchain_process_names": ["fake-node"],
            "ledger_device": "sdb",
            "data_vol_type": "hyperdisk-extreme",
            "data_vol_size": "2048",
            "data_vol_max_iops": "30000",
            "data_vol_max_throughput": "1000",
            "has_accounts_device": False,
            "network_interface": "ens4",
        }
        next_question = build_missing_config_questions("fake-node", confirmed)["next_question"]
        self.assertEqual(next_question["id"], "network_max_bandwidth_gbps")
        self.assertEqual(next_question["field"], "NETWORK_MAX_BANDWIDTH_GBPS")

    def test_custom_cloud_region_is_accepted_as_structural_pending_answer(self):
        from terminal.pending_answers import is_structural_pending_answer

        question = {
            "id": "cloud_region",
            "kind": "manual_value",
            "field": "CLOUD_REGION",
            "prompt": "Confirm CLOUD_REGION; use the detected value or enter a custom region.",
            "manual_input_allowed": True,
        }
        self.assertTrue(is_structural_pending_answer("us-1", question))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            update_workflow_state(
                {"pending_question": question},
                session_id="test-session",
                state_root=root,
            )
            result = answer_pending_question("us-1", session_id="test-session", state_root=root)
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertTrue(result["applied"])
        self.assertEqual(loaded["confirmed_config"]["CLOUD_REGION"], "us-1")

    def test_scalar_pending_answer_trims_wrapping_punctuation(self):
        question = {
            "id": "data_vol_type",
            "kind": "manual_value",
            "field": "DATA_VOL_TYPE",
            "prompt": "Confirm DATA_VOL_TYPE",
            "manual_input_allowed": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            update_workflow_state(
                {"pending_question": question},
                session_id="test-session",
                state_root=root,
            )
            result = answer_pending_question("  hyperdisk-balanced,  ", session_id="test-session", state_root=root)
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertTrue(result["applied"])
        self.assertEqual(loaded["confirmed_config"]["DATA_VOL_TYPE"], "hyperdisk-balanced")

    def test_terminal_structural_answer_trims_wrapping_punctuation(self):
        question = {
            "id": "data_vol_type",
            "kind": "manual_value",
            "field": "DATA_VOL_TYPE",
            "prompt": "Confirm DATA_VOL_TYPE",
            "manual_input_allowed": True,
        }

        self.assertTrue(is_structural_pending_answer("  hyperdisk-balanced,  ", question))
        self.assertTrue(is_structural_pending_answer("`pd-ssd`", question))

    def test_numbered_choice_pending_answer_trims_trailing_punctuation(self):
        question = {
            "id": "target_mode",
            "kind": "numbered_choice",
            "field": "target_mode",
            "prompt": "Choose target mode",
            "options": [
                {"id": "1", "value": "fake-node", "label": "fake-node"},
                {"id": "2", "value": "real-node", "label": "real-node"},
            ],
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            update_workflow_state(
                {"pending_question": question},
                session_id="test-session",
                state_root=root,
            )
            result = answer_pending_question("1,", session_id="test-session", state_root=root)
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertTrue(is_structural_pending_answer("1,", question))
        self.assertTrue(result["applied"])
        self.assertEqual(loaded["target_mode"], "fake-node")

    def test_qps_profile_no_enters_adjustment_flow(self):
        session_id = "unit-qps-profile-adjustment-flow"
        reset_workflow_state(session_id=session_id)
        question = {
            "id": "benchmark_profile_confirm",
            "kind": "yes_no",
            "field": "qps_profile_confirmed",
            "prompt": "Use selected QPS defaults?",
            "next_on_no": {
                "workflow_step": "benchmark_profile_adjust_item",
                "next_question_id": "benchmark_profile_adjust_item",
            },
        }
        update_workflow_state(
            {
                "active_workflow": "benchmark_setup",
                "target_mode": "fake-node",
                "chain": "solana",
                "rpc_mode": "single",
                "benchmark_profile": {"mode": "quick"},
                "pending_question": question,
            },
            session_id=session_id,
        )
        first = answer_pending_question("n", session_id=session_id)
        advanced = advance_after_pending_answer(first, language="en", session_id=session_id)
        loaded = load_workflow_state(session_id=session_id)

        self.assertTrue(first["applied"])
        self.assertEqual(advanced["pending_question"]["id"], "benchmark_profile_adjust_item")
        self.assertEqual(loaded["pending_question"]["id"], "benchmark_profile_adjust_item")

        second = answer_pending_question("2", session_id=session_id)
        advanced = advance_after_pending_answer(second, language="en", session_id=session_id)
        loaded = load_workflow_state(session_id=session_id)

        self.assertTrue(second["applied"])
        self.assertEqual(loaded["benchmark_profile"]["adjust_item"], "max_qps")
        self.assertEqual(advanced["pending_question"]["id"], "benchmark_profile_adjust_value")
        self.assertEqual(advanced["pending_question"]["field"], "QUICK_MAX_QPS")

        third = answer_pending_question("25", session_id=session_id)
        loaded = load_workflow_state(session_id=session_id)

        self.assertTrue(third["applied"])
        self.assertEqual(loaded["confirmed_config"]["QUICK_MAX_QPS"], "25")
        self.assertTrue(loaded["confirmed_config"]["qps_profile_confirmed"])

    def test_ledger_group_progress_completes_before_accounts_group(self):
        session_id = "unit-ledger-group-progress-completes"
        reset_workflow_state(session_id=session_id)
        discovery = {
            "disks": {
                "candidates": [
                    {"name": "vda", "type": "disk", "size": "926.3G", "mountpoint": "", "label": ""},
                    {"name": "vdb", "type": "disk", "size": "624.9M", "mountpoint": "", "label": ""},
                ]
            }
        }
        update_workflow_state(
            {
                "active_workflow": "benchmark_setup",
                "target_mode": "fake-node",
                "chain": "solana",
                "rpc_mode": "single",
                "confirmed_config": {
                    "chain": "solana",
                    "rpc_mode": "single",
                    "CLOUD_REGION": "us-1",
                    "CLOUD_ZONE": "us-1-z",
                    "MACHINE_TYPE": "n2",
                },
                "pending_question": build_missing_config_questions(
                    "fake-node",
                    {
                        "chain": "solana",
                        "rpc_mode": "single",
                        "CLOUD_REGION": "us-1",
                        "CLOUD_ZONE": "us-1-z",
                        "MACHINE_TYPE": "n2",
                    },
                    discovery,
                )["next_question"],
            },
            session_id=session_id,
        )

        for answer in ("1", "hyperdisk-balanced", "Y", "3000", "1000"):
            result = answer_pending_question(answer, session_id=session_id)
            self.assertTrue(result["applied"], result)
            advance_after_pending_answer(result, discovery=discovery, language="en", session_id=session_id)

        loaded = load_workflow_state(session_id=session_id)
        self.assertEqual(loaded["confirmed_config"]["LEDGER_DEVICE"], "vda")
        self.assertEqual(loaded["confirmed_config"]["DATA_VOL_SIZE"], "926")
        self.assertEqual(loaded["group_progress"]["ledger_disk"]["status"], "complete")
        self.assertEqual(loaded["active_group"], "accounts_disk")
        self.assertEqual(loaded["pending_question"]["id"], "disk_accounts_exists")

    def test_benchmark_setup_preempts_workload_until_environment_config_is_ready(self):
        session_id = "unit-config-preempts-workload"
        reset_workflow_state(session_id=session_id)
        update_workflow_state(
            {
                "active_workflow": "benchmark_setup",
                "target_mode": "fake-node",
                "chain": "solana",
                "rpc_mode": "single",
                "benchmark_profile": {"mode": "quick"},
                "pending_question": {
                    "id": "workload_customization_choice",
                    "kind": "numbered_choice",
                    "branch": "rpc_workload",
                    "prompt": "Choose workload",
                    "options": [
                        {"id": "1", "value": "default", "label": "default"},
                        {"id": "2", "value": "custom", "label": "custom"},
                    ],
                },
            },
            session_id=session_id,
        )

        result = ensure_next_benchmark_setup_question(
            load_workflow_state(session_id=session_id),
            discovery={"disks": {"items": [{"name": "sdb", "type": "disk", "size": "2T"}]}},
            language="en",
            session_id=session_id,
        )
        loaded = load_workflow_state(session_id=session_id)

        self.assertEqual(result["pending_question"]["id"], "cloud_region")
        self.assertEqual(loaded["pending_question"]["id"], "cloud_region")

    def test_chain_aliases_are_canonicalized_in_workflow_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            update_workflow_state(
                {"chain": "eth", "confirmed_config": {"chain": "eth"}},
                session_id="test-session",
                state_root=root,
            )
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertEqual(loaded["chain"], "ethereum")

    def test_separate_accounts_disk_cannot_reuse_ledger_device(self):
        question = {
            "id": "disk_accounts_choice",
            "kind": "device",
            "field": "ACCOUNTS_DEVICE",
            "options": [{"id": "1", "value": "sdb", "label": "sdb"}],
            "manual_input_allowed": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            update_workflow_state(
                {
                    "confirmed_config": {"LEDGER_DEVICE": "sdb", "has_accounts_device": True},
                    "pending_question": question,
                },
                session_id="test-session",
                state_root=root,
            )
            by_number = answer_pending_question("1", session_id="test-session", state_root=root)
            by_name = answer_pending_question("sdb", session_id="test-session", state_root=root)

        self.assertFalse(by_number["applied"])
        self.assertFalse(by_name["applied"])
        self.assertIn("must be different", by_number["blockers"][0])
        self.assertIn("must be different", by_name["blockers"][0])

    def test_config_next_question_accepts_observability_number_choice(self):
        confirmed = {
            "chain": "solana",
            "rpc_mode": "single",
            "benchmark_mode_confirmed": True,
            "qps_profile_confirmed": True,
            "chain_template_reviewed": True,
            "rpc_workload_confirmed": True,
            "rpc_param_samples_confirmed": True,
            "cloud_region": "asia-east1",
            "cloud_zone": "asia-east1-c",
            "machine_type": "c3-standard-22",
            "blockchain_process_names": ["fake-node"],
            "ledger_device": "sdb",
            "data_vol_type": "hyperdisk-extreme",
            "data_vol_size": "2000",
            "data_vol_max_iops": "30000",
            "data_vol_max_throughput": "1000",
            "has_accounts_device": False,
            "network_interface": "eth0",
            "network_max_bandwidth_gbps": "20",
        }
        questions = build_missing_config_questions("fake-node", confirmed, {})
        self.assertEqual(questions["next_question"]["id"], "observability_mode_choice")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            update_workflow_state(
                {"pending_question": questions["next_question"]},
                session_id="test-session",
                state_root=root,
            )
            result = answer_pending_question("1", session_id="test-session", state_root=root)
            self.assertTrue(result["applied"])
            self.assertEqual(result["state"]["observability"]["mode"], "disabled")
            self.assertEqual(result["state"]["confirmed_config"]["OBSERVABILITY_STACK_MODE"], "disabled")
            self.assertEqual(result["state"]["confirmed_config"]["observability_choice_confirmed"], "disabled")

    def test_workload_review_uses_typed_numbered_menu(self):
        confirmed = {
            "chain": "ethereum",
            "rpc_mode": "mixed",
            "benchmark_mode_confirmed": "quick",
            "qps_profile_confirmed": True,
            "chain_template_reviewed": True,
            "rpc_workload_confirmed": True,
            "rpc_param_samples_confirmed": True,
            "observability_choice_confirmed": "disabled",
            "cloud_region": "asia-east1",
            "cloud_zone": "asia-east1-c",
            "machine_type": "c3-standard-22",
            "blockchain_process_names": ["geth"],
            "ledger_device": "sdb",
            "data_vol_type": "hyperdisk-extreme",
            "data_vol_size": "2000",
            "data_vol_max_iops": "30000",
            "data_vol_max_throughput": "1000",
            "has_accounts_device": False,
            "network_interface": "eth0",
            "network_max_bandwidth_gbps": "20",
        }
        questions = build_missing_config_questions("fake-node", confirmed, {})
        self.assertEqual(questions["next_question"]["id"], "workload_customization_choice")
        self.assertEqual(questions["next_question"]["kind"], "numbered_choice")
        self.assertEqual(
            [option["value"] for option in questions["next_question"]["options"]],
            ["use_defaults", "add_custom_rpc", "adjust_weights", "change_chain_or_mode"],
        )

    def test_workload_customization_default_derives_template_methods(self):
        confirmed = {
            "chain": "solana",
            "rpc_mode": "single",
            "benchmark_mode_confirmed": "quick",
            "qps_profile_confirmed": True,
            "observability_choice_confirmed": "disabled",
            "cloud_region": "asia-east1",
            "cloud_zone": "asia-east1-c",
            "machine_type": "c3-standard-22",
            "blockchain_process_names": ["fake-node"],
            "ledger_device": "sdb",
            "data_vol_type": "hyperdisk-extreme",
            "data_vol_size": "2000",
            "data_vol_max_iops": "30000",
            "data_vol_max_throughput": "1000",
            "has_accounts_device": False,
            "network_interface": "eth0",
            "network_max_bandwidth_gbps": "20",
        }
        question = build_missing_config_questions("fake-node", confirmed, {})["next_question"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            update_workflow_state(
                {
                    "chain": "solana",
                    "rpc_mode": "single",
                    "pending_question": question,
                    "confirmed_config": confirmed,
                },
                session_id="test-session",
                state_root=root,
            )
            result = answer_pending_question("1", session_id="test-session", state_root=root)
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertTrue(result["applied"])
        self.assertEqual(loaded["rpc_methods"], ["getAccountInfo"])
        self.assertTrue(loaded["confirmed_config"]["chain_template_reviewed"])
        self.assertTrue(loaded["confirmed_config"]["rpc_workload_confirmed"])
        self.assertTrue(loaded["confirmed_config"]["rpc_param_samples_confirmed"])

    def test_workload_customization_routes_custom_rpc_and_weight_adjustment(self):
        base = {
            "chain": "ethereum",
            "rpc_mode": "mixed",
            "benchmark_mode_confirmed": "quick",
            "qps_profile_confirmed": True,
            "observability_choice_confirmed": "disabled",
            "cloud_region": "asia-east1",
            "cloud_zone": "asia-east1-c",
            "machine_type": "c3-standard-22",
            "blockchain_process_names": ["fake-node"],
            "ledger_device": "sdb",
            "data_vol_type": "hyperdisk-extreme",
            "data_vol_size": "2000",
            "data_vol_max_iops": "30000",
            "data_vol_max_throughput": "1000",
            "has_accounts_device": False,
            "network_interface": "eth0",
            "network_max_bandwidth_gbps": "20",
        }
        question = build_missing_config_questions("fake-node", base, {})["next_question"]
        custom_session = "unit-workload-custom-rpc"
        reset_workflow_state(session_id=custom_session)
        update_workflow_state(
            {"chain": "ethereum", "rpc_mode": "mixed", "pending_question": question, "confirmed_config": base},
            session_id=custom_session,
        )
        custom = answer_pending_question("2", session_id=custom_session)
        custom_advanced = advance_after_pending_answer(custom, session_id=custom_session)
        custom_state = load_workflow_state(session_id=custom_session)

        self.assertTrue(custom["applied"])
        self.assertEqual(custom_advanced["pending_question"]["id"], "custom_rpc_endpoint_gate")
        self.assertEqual(custom_state["pending_question"]["id"], "custom_rpc_endpoint_gate")

        weight_session = "unit-workload-adjust-weights"
        reset_workflow_state(session_id=weight_session)
        update_workflow_state(
            {"chain": "ethereum", "rpc_mode": "mixed", "pending_question": question, "confirmed_config": base},
            session_id=weight_session,
        )
        weights = answer_pending_question("3", session_id=weight_session)
        weights_advanced = advance_after_pending_answer(weights, session_id=weight_session)
        weights_state = load_workflow_state(session_id=weight_session)

        self.assertTrue(weights["applied"])
        self.assertEqual(weights_advanced["pending_question"]["id"], "mixed_weights_confirm")
        self.assertEqual(weights_state["pending_question"]["kind"], "rpc_weight")

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

    def test_pending_answer_rejects_bare_yes_without_active_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            result = answer_pending_question("y", session_id="test-session", state_root=root)

        self.assertFalse(result["applied"])
        self.assertEqual(result["answer_kind"], "unbound")
        self.assertIn("no active pending_question", result["blockers"][0])

    def test_pending_answer_numbered_choice_updates_root_target_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            update_workflow_state(
                {
                    "pending_question": {
                        "id": "target_mode",
                        "kind": "numbered_choice",
                        "prompt": "Choose target mode",
                        "field": "target_mode",
                        "workflow_step": "select_target_mode",
                        "options": [{"label": "fake-node", "value": "fake-node"}, {"label": "real-node", "value": "real-node"}],
                        "next_on_choice": {"workflow_step": "select_chain", "tool_order": ["load_framework_capabilities"]},
                    }
                },
                session_id="test-session",
                state_root=root,
            )
            result = answer_pending_question("1", session_id="test-session", state_root=root)
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertTrue(result["applied"])
        self.assertEqual(result["answer_kind"], "choice")
        self.assertEqual(loaded["target_mode"], "fake-node")
        self.assertEqual(loaded["workflow_step"], "select_chain")
        self.assertEqual(loaded["pending_question"], {})
        self.assertIn("tool:load_framework_capabilities", loaded["allowed_next_actions"])

    def test_pending_answer_numbered_choice_accepts_yes_only_with_default_option(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            update_workflow_state(
                {
                    "pending_question": {
                        "id": "benchmark_profile_choice",
                        "kind": "numbered_choice",
                        "prompt": "Choose benchmark profile",
                        "field": "benchmark_profile",
                        "options": [
                            {"id": "1", "label": "quick", "value": "quick"},
                            {"id": "2", "label": "standard", "value": "standard"},
                        ],
                        "default_option": "quick",
                        "next_on_choice": {"workflow_step": "confirm_profile"},
                    }
                },
                session_id="test-session",
                state_root=root,
            )
            result = answer_pending_question("y", session_id="test-session", state_root=root)
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertTrue(result["applied"])
        self.assertEqual(result["answer_kind"], "choice")
        self.assertEqual(result["selected"]["value"], "quick")
        self.assertEqual(loaded["benchmark_profile"]["value"], "quick")
        self.assertEqual(loaded["workflow_step"], "confirm_profile")

    def test_pending_answer_numbered_choice_rejects_yes_without_default_option(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            update_workflow_state(
                {
                    "pending_question": {
                        "id": "target_mode",
                        "kind": "numbered_choice",
                        "prompt": "Choose target mode",
                        "field": "target_mode",
                        "options": [
                            {"label": "fake-node", "value": "fake-node"},
                            {"label": "real-node", "value": "real-node"},
                        ],
                    }
                },
                session_id="test-session",
                state_root=root,
            )
            result = answer_pending_question("Y", session_id="test-session", state_root=root)
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertFalse(result["applied"])
        self.assertIn("default_option", result["blockers"][0])
        self.assertEqual(loaded["pending_question"]["id"], "target_mode")
        self.assertEqual(loaded["target_mode"], "")

    def test_pending_answer_confirmation_accepts_yes_no_transitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            update_workflow_state(
                {
                    "pending_question": {
                        "id": "unsupported_chain_handoff_confirm",
                        "kind": "confirmation",
                        "prompt": "Generate a handoff?",
                        "field": "approval",
                        "next_on_yes": {
                            "workflow_step": "generate_onboarding_handoff",
                            "tool_order": ["draft_secondary_development_handoff"],
                        },
                    }
                },
                session_id="test-session",
                state_root=root,
            )
            result = answer_pending_question("y", session_id="test-session", state_root=root)
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertTrue(result["applied"])
        self.assertEqual(result["answer_kind"], "yes")
        self.assertEqual(loaded["approval"]["value"], True)
        self.assertEqual(loaded["workflow_step"], "generate_onboarding_handoff")
        self.assertIn("tool:draft_secondary_development_handoff", loaded["allowed_next_actions"])

    def test_pending_answer_quick_assumed_smoke_sets_assumed_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            update_workflow_state(
                {
                    "pending_question": {
                        "id": "confirm_assumed_smoke",
                        "kind": "yes_no",
                        "prompt": "Use quick assumed smoke?",
                        "field": "approval.smoke_approved",
                    }
                },
                session_id="test-session",
                state_root=root,
            )
            result = answer_pending_question("Y", session_id="test-session", state_root=root)
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertTrue(result["applied"])
        self.assertTrue(loaded["assumed_for_smoke"])
        self.assertEqual(loaded["target_mode"], "fake-node")
        self.assertEqual(loaded["allowed_next_actions"], ["tool:run_quick_assumed_fake_node_smoke"])

    def test_quick_assumed_smoke_confirmation_delegates_to_adk_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            update_workflow_state(
                {
                    "active_workflow": "quick_assumed_smoke",
                    "target_mode": "fake-node",
                    "pending_question": {
                        "id": "quick_assumed_smoke_confirm",
                        "kind": "yes_no",
                        "prompt": "Use quick assumed smoke?",
                        "field": "approval.quick_assumed_smoke",
                        "next_on_yes": {
                            "workflow_step": "quick_assumed_smoke_approved",
                            "tool": "run_quick_assumed_fake_node_smoke",
                        },
                    },
                },
                session_id="test-session",
                state_root=root,
            )
            result = answer_pending_question("Y", session_id="test-session", state_root=root)
            next_step = advance_after_pending_answer(
                result,
                discovery={},
                language="en",
                session_id="test-session",
            )

        self.assertTrue(result["applied"])
        self.assertTrue(next_step.get("delegate_to_adk"))
        self.assertFalse(next_step.get("message"))

    def test_quick_assumed_smoke_tool_registers_typed_pending_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            with workflow_tool_session("test-session", state_root=root):
                payload = propose_quick_assumed_smoke_confirmation(
                    source_prompt="just verify it can run",
                    chain="solana",
                    rpc_mode="single",
                    language="zh",
                )
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(loaded["pending_question"]["id"], "quick_assumed_smoke_confirm")
        self.assertEqual(loaded["pending_question"]["source"], "workflow_tool")
        self.assertEqual(loaded["pending_question"]["source_tool"], "propose_quick_assumed_smoke_confirmation")
        self.assertIn("assumed_for_smoke=true", loaded["pending_question"]["prompt"])
        self.assertTrue(loaded["assumed_for_smoke"])
        self.assertEqual(loaded["target_mode"], "fake-node")
        self.assertEqual(loaded["chain"], "solana")

    def test_target_mode_tool_registers_numbered_pending_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            with workflow_tool_session("test-session", state_root=root):
                payload = propose_benchmark_target_mode_choice(
                    source_prompt="我要测试",
                    language="zh",
                    explicit_benchmark_intent=True,
                )
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(loaded["pending_question"]["id"], "target_mode")
        self.assertEqual(loaded["pending_question"]["source"], "workflow_tool")
        self.assertEqual(loaded["pending_question"]["source_tool"], "propose_benchmark_target_mode_choice")
        self.assertEqual([item["value"] for item in loaded["pending_question"]["options"]], ["fake-node", "real-node", "explain"])

    def test_opening_help_choice_registers_fake_node_entry_without_benchmark_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            with workflow_tool_session("test-session", state_root=root):
                payload = propose_opening_help_choice(language="zh", latest_job_id="job_123")
            loaded = load_workflow_state(session_id="test-session", state_root=root)
            result = answer_pending_question("2", session_id="test-session", state_root=root)
            answered = load_workflow_state(session_id="test-session", state_root=root)

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(loaded["active_intent"], "opening")
        self.assertEqual(loaded["pending_question"]["id"], "opening_help_choice")
        self.assertEqual(loaded["pending_question"]["source_tool"], "propose_opening_help_choice")
        self.assertTrue(result["applied"])
        self.assertEqual(answered["active_intent"], "benchmark")
        self.assertEqual(answered["target_mode"], "fake-node")
        self.assertEqual(answered["workflow_step"], "benchmark_setup")
        self.assertEqual(answered["pending_question"], {})

    def test_target_mode_numbered_answer_sets_fake_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            with workflow_tool_session("test-session", state_root=root):
                propose_benchmark_target_mode_choice(
                    source_prompt="test",
                    language="en",
                    explicit_benchmark_intent=True,
                )
            result = answer_pending_question("1", session_id="test-session", state_root=root)
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertTrue(result["applied"])
        self.assertEqual(loaded["target_mode"], "fake-node")
        self.assertEqual(loaded["workflow_step"], "benchmark_setup")
        self.assertEqual(loaded["pending_question"], {})

    def test_yes_no_pending_question_rejects_number_without_losing_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            with workflow_tool_session("test-session", state_root=root):
                propose_quick_assumed_smoke_confirmation(
                    source_prompt="fake-node 测试",
                    language="zh",
                )
            result = answer_pending_question("2", session_id="test-session", state_root=root)
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertFalse(result["applied"])
        self.assertIn("expected yes or no", "; ".join(result["blockers"]))
        self.assertEqual(loaded["pending_question"]["id"], "quick_assumed_smoke_confirm")

    def test_target_mode_tool_blocks_without_explicit_benchmark_intent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            with workflow_tool_session("test-session", state_root=root):
                payload = propose_benchmark_target_mode_choice(
                    source_prompt="Hi",
                    language="zh",
                )
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(loaded["pending_question"], {})
        self.assertFalse(loaded.get("target_mode"))

    def test_target_mode_tool_blocks_when_target_mode_already_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            with workflow_tool_session("test-session", state_root=root):
                payload = propose_benchmark_target_mode_choice(
                    source_prompt="fake-node 测试",
                    language="zh",
                    explicit_benchmark_intent=True,
                )
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(loaded["pending_question"], {})
        self.assertFalse(loaded.get("target_mode"))

    def test_chain_selection_tool_registers_manual_chain_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            with workflow_tool_session("test-session", state_root=root):
                propose_chain_selection_question(target_mode="fake-node", language="zh")
            loaded = load_workflow_state(session_id="test-session", state_root=root)
            result = answer_pending_question("ethereum", session_id="test-session", state_root=root)
            answered = load_workflow_state(session_id="test-session", state_root=root)

        self.assertEqual(loaded["pending_question"]["id"], "chain_selection")
        self.assertEqual(loaded["pending_question"]["source_tool"], "propose_chain_selection_question")
        self.assertEqual(loaded["target_mode"], "fake-node")
        self.assertTrue(result["applied"])
        self.assertEqual(answered["chain"], "ethereum")
        self.assertEqual(answered["workflow_step"], "validate_chain_template")
        self.assertEqual(answered["pending_question"], {})

    def test_chain_selection_rejects_bare_number_without_losing_pending_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            with workflow_tool_session("test-session", state_root=root):
                propose_chain_selection_question(target_mode="fake-node", language="zh")
            result = answer_pending_question("2", session_id="test-session", state_root=root)
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertFalse(result["applied"])
        self.assertIn("expected a chain name", "; ".join(result["blockers"]))
        self.assertEqual(loaded["pending_question"]["id"], "chain_selection")
        self.assertEqual(loaded["target_mode"], "fake-node")

    def test_unsupported_chain_endpoint_gate_registers_typed_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            with workflow_tool_session("test-session", state_root=root):
                propose_unsupported_chain_endpoint_gate(
                    chain="FooChain",
                    adapter_family="jsonrpc",
                    language="zh",
                )
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertEqual(loaded["pending_question"]["id"], "unsupported_chain_endpoint_gate")
        self.assertEqual(loaded["pending_question"]["source"], "workflow_tool")
        self.assertEqual(loaded["pending_question"]["source_tool"], "propose_unsupported_chain_endpoint_gate")
        self.assertEqual(loaded["fixture_status"]["status"], "needs_endpoint")
        self.assertIn("endpoint", loaded["pending_question"]["prompt"])

    def test_unsupported_chain_gate_preserves_full_raw_chain_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            update_workflow_state(
                {
                    "pending_question": {"id": "chain_selection", "kind": "chain"},
                    "last_user_change": {
                        "type": "unsupported_chain_candidate",
                        "raw": "bnb greenfield",
                    },
                },
                session_id="test-session",
                state_root=root,
            )
            with workflow_tool_session("test-session", state_root=root):
                propose_unsupported_chain_endpoint_gate(
                    chain="greenfield",
                    adapter_family="jsonrpc",
                    language="en",
                )
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertEqual(loaded["chain"], "bnb-greenfield")
        self.assertIn("bnb-greenfield", loaded["pending_question"]["prompt"])

    def test_unsupported_chain_gate_does_not_keep_model_shortened_supported_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            update_workflow_state(
                {
                    "last_user_change": {
                        "type": "unsupported_chain_candidate",
                        "raw": "我现在换成 BNB Greenfield，仍然用 fake-node",
                    },
                },
                session_id="test-session",
                state_root=root,
            )
            with workflow_tool_session("test-session", state_root=root):
                propose_unsupported_chain_endpoint_gate(
                    chain="bsc",
                    adapter_family="rest",
                    language="zh",
                )
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertEqual(loaded["chain"], "bnb-greenfield")
        self.assertEqual(loaded["confirmed_config"]["BLOCKCHAIN_NODE"], "bnb-greenfield")
        self.assertIn("bnb-greenfield", loaded["pending_question"]["prompt"])
        self.assertIn("我有 endpoint", loaded["pending_question"]["options"][0]["label"])

    def test_custom_rpc_endpoint_gate_registers_needs_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            with workflow_tool_session("test-session", state_root=root):
                propose_custom_rpc_endpoint_gate(
                    chain="solana",
                    language="zh",
                )
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertEqual(loaded["pending_question"]["id"], "custom_rpc_endpoint_gate")
        self.assertEqual(loaded["pending_question"]["source"], "workflow_tool")
        self.assertEqual(loaded["pending_question"]["source_tool"], "propose_custom_rpc_endpoint_gate")
        self.assertEqual(loaded["fixture_status"]["status"], "needs_endpoint")
        self.assertIn("needs_review", loaded["pending_question"]["prompt"])

    def test_custom_rpc_no_endpoint_handoff_marks_needs_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            update_workflow_state(
                {"chain": "solana", "active_group": "target_samples_fixtures"},
                session_id="test-session",
                state_root=root,
            )
            with workflow_tool_session("test-session", state_root=root):
                result = request_custom_rpc_handoff(
                    chain="solana",
                    methods=["getHealth"],
                    reason="没有 endpoint",
                    language="zh",
                )
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(loaded["active_group"], "target_samples_fixtures")
        self.assertEqual(loaded["workflow_step"], "custom_rpc_handoff_requested")
        self.assertEqual(loaded["fixture_status"]["status"], "needs_review")
        self.assertFalse(loaded.get("pending_question"))

    def test_benchmark_profile_tool_registers_numbered_choice(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            with workflow_tool_session("test-session", state_root=root):
                propose_benchmark_profile_choice(language="zh")
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertEqual(loaded["pending_question"]["id"], "benchmark_profile_choice")
        self.assertEqual(loaded["pending_question"]["source"], "workflow_tool")
        self.assertEqual(loaded["pending_question"]["source_tool"], "propose_benchmark_profile_choice")
        self.assertEqual([item["value"] for item in loaded["pending_question"]["options"]], ["quick", "standard", "intensive"])

    def test_benchmark_profile_answer_sets_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            with workflow_tool_session("test-session", state_root=root):
                propose_benchmark_profile_choice(language="en")
            result = answer_pending_question("2", session_id="test-session", state_root=root)
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertTrue(result["applied"])
        self.assertEqual(loaded["benchmark_profile"]["name"], "standard")
        self.assertEqual(loaded["benchmark_profile"]["max"], 5000)
        self.assertEqual(loaded["workflow_step"], "benchmark_profile_confirmed")

    def test_pending_answer_default_workload_confirm_derives_rpc_methods(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            update_workflow_state(
                {
                    "chain": "solana",
                    "rpc_mode": "single",
                    "pending_question": {
                        "id": "workload_confirm",
                        "kind": "yes_no",
                        "prompt": "Use default single RPC method?",
                        "field": "rpc_methods",
                    },
                },
                session_id="test-session",
                state_root=root,
            )
            result = answer_pending_question("y", session_id="test-session", state_root=root)
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertTrue(result["applied"])
        self.assertEqual(loaded["rpc_methods"], ["getAccountInfo"])
        self.assertNotEqual(loaded["confirmed_config"].get("rpc_methods"), True)

    def test_pending_answer_confirm_default_field_derives_rpc_methods(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            update_workflow_state(
                {
                    "chain": "solana",
                    "rpc_mode": "single",
                    "pending_question": {
                        "id": "confirm_default",
                        "kind": "confirmation",
                        "prompt": "Accept the default workload?",
                        "field": "confirm_default",
                    },
                },
                session_id="test-session",
                state_root=root,
            )
            result = answer_pending_question("y", session_id="test-session", state_root=root)
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertTrue(result["applied"])
        self.assertEqual(loaded["rpc_methods"], ["getAccountInfo"])
        self.assertEqual(loaded["confirmed_config"]["confirm_default"], True)

    def test_pending_answer_generic_confirmation_field_derives_rpc_methods(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            update_workflow_state(
                {
                    "chain": "solana",
                    "rpc_mode": "single",
                    "pending_question": {
                        "id": "confirm_env_metadata",
                        "kind": "yes_no",
                        "prompt": "Use inferred metadata and default workload?",
                        "field": "confirmations",
                    },
                },
                session_id="test-session",
                state_root=root,
            )
            result = answer_pending_question("Y", session_id="test-session", state_root=root)
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertTrue(result["applied"])
        self.assertEqual(loaded["rpc_methods"], ["getAccountInfo"])
        self.assertEqual(loaded["confirmed_config"]["confirmations"], True)

    def test_pending_answer_manual_disk_updates_confirmed_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            update_workflow_state(
                {
                    "pending_question": {
                        "id": "disk_ledger_choice",
                        "kind": "device",
                        "prompt": "Which disk is LEDGER_DEVICE?",
                        "field": "LEDGER_DEVICE",
                        "workflow_step": "confirm_ledger_device",
                        "options": [{"value": "sda"}, {"value": "sdb"}],
                        "manual_input_allowed": True,
                        "next_on_manual": {"workflow_step": "confirm_accounts_disk", "validator": "validate_required_config"},
                    }
                },
                session_id="test-session",
                state_root=root,
            )
            result = answer_pending_question("nvme1n1", session_id="test-session", state_root=root)
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertTrue(result["applied"])
        self.assertEqual(result["answer_kind"], "manual")
        self.assertEqual(loaded["confirmed_config"]["LEDGER_DEVICE"], "nvme1n1")
        self.assertEqual(loaded["workflow_step"], "confirm_accounts_disk")
        self.assertIn("validator:validate_required_config", loaded["allowed_next_actions"])

    def test_pending_answer_normalizes_dev_prefix_for_disk_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            update_workflow_state(
                {
                    "pending_question": {
                        "id": "disk_ledger_choice",
                        "kind": "device",
                        "prompt": "Which disk is LEDGER_DEVICE?",
                        "field": "LEDGER_DEVICE",
                        "workflow_step": "confirm_ledger_device",
                        "options": [{"id": "1", "label": "sdb", "value": "/dev/sdb"}],
                        "manual_input_allowed": True,
                    }
                },
                session_id="test-session",
                state_root=root,
            )
            first = answer_pending_question("1", session_id="test-session", state_root=root)
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertTrue(first["applied"])
        self.assertEqual(loaded["confirmed_config"]["LEDGER_DEVICE"], "sdb")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            update_workflow_state(
                {
                    "pending_question": {
                        "id": "disk_ledger_choice",
                        "kind": "device",
                        "prompt": "Which disk is LEDGER_DEVICE?",
                        "field": "LEDGER_DEVICE",
                        "workflow_step": "confirm_ledger_device",
                        "options": [{"id": "1", "label": "sdb", "value": "/dev/sdb"}],
                        "manual_input_allowed": True,
                    }
                },
                session_id="test-session",
                state_root=root,
            )
            second = answer_pending_question("/dev/sdb", session_id="test-session", state_root=root)
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertTrue(second["applied"])
        self.assertEqual(loaded["confirmed_config"]["LEDGER_DEVICE"], "sdb")

    def test_pending_answer_rejects_non_device_token_for_disk_question(self):
        question = {
            "id": "disk_ledger_choice",
            "kind": "device",
            "prompt": "Which disk is LEDGER_DEVICE?",
            "field": "LEDGER_DEVICE",
            "options": [{"value": "sda"}, {"value": "sdb"}],
            "manual_input_allowed": True,
        }
        self.assertTrue(is_structural_pending_answer("banana", question))
        self.assertTrue(is_structural_pending_answer("sdb", question))
        self.assertTrue(is_structural_pending_answer("/dev/sdd", question))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            update_workflow_state({"pending_question": question}, session_id="test-session", state_root=root)
            result = answer_pending_question("banana", session_id="test-session", state_root=root)
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertFalse(result["applied"])
        self.assertIn("expected a Linux block device", "; ".join(result["blockers"]))
        self.assertNotIn("LEDGER_DEVICE", loaded.get("confirmed_config", {}))
        self.assertEqual(loaded["pending_question"]["id"], "disk_ledger_choice")

    def test_disk_device_tool_replaces_multi_question_prompt_with_single_role_prompt(self):
        session_id = "unit-disk-device-tool-one-question"
        reset_workflow_state(session_id=session_id)
        with workflow_tool_session(session_id):
            payload = propose_disk_device_choice(
                device_role="ledger",
                prompt="请确认 LEDGER_DEVICE，另外是否需要单独的 ACCOUNTS_DEVICE？",
                device_options=["/dev/sdb (2T ledger)", "/dev/sdc (1T accounts)", "manual"],
            )
        prompt = payload["data"]["prompt"]
        loaded = load_workflow_state(session_id=session_id)

        self.assertIn("LEDGER_DEVICE", prompt)
        self.assertNotIn("ACCOUNTS_DEVICE", prompt)
        self.assertNotIn("manual", prompt)
        self.assertEqual(loaded["pending_question"]["id"], "disk_ledger_choice")
        self.assertEqual(loaded["pending_question"]["prompt"], prompt)

    def test_pending_answer_validates_url_shape_before_state_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            update_workflow_state(
                {
                    "pending_question": {
                        "id": "real_node_local_rpc_url",
                        "kind": "url",
                        "prompt": "Provide LOCAL_RPC_URL",
                        "field": "LOCAL_RPC_URL",
                        "workflow_step": "confirm_local_rpc_url",
                    }
                },
                session_id="test-session",
                state_root=root,
            )
            result = answer_pending_question("not-a-url", session_id="test-session", state_root=root)
            loaded = load_workflow_state(session_id="test-session", state_root=root)

        self.assertFalse(result["applied"])
        self.assertIn("expected a URL", result["blockers"][0])
        self.assertNotIn("LOCAL_RPC_URL", loaded["confirmed_config"])
        self.assertEqual(loaded["pending_question"]["id"], "real_node_local_rpc_url")

    def test_pending_answer_back_reverts_previous_state_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            update_workflow_state(
                {"confirmed_config": {"LEDGER_DEVICE": "sda"}},
                session_id="test-session",
                state_root=root,
            )
            update_workflow_state(
                {
                    "confirmed_config": {"LEDGER_DEVICE": "sdb"},
                    "pending_question": {
                        "id": "disk_accounts_exists",
                        "kind": "yes_no",
                        "prompt": "Do you have ACCOUNTS_DEVICE?",
                        "workflow_step": "confirm_accounts_exists",
                    },
                },
                session_id="test-session",
                state_root=root,
            )
            result = answer_pending_question("back", session_id="test-session", state_root=root)

        self.assertTrue(result["applied"])
        self.assertEqual(result["answer_kind"], "back")
        self.assertEqual(result["state"]["confirmed_config"]["LEDGER_DEVICE"], "sda")

    def test_pending_answer_chinese_back_reverts_previous_state_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sessions"
            update_workflow_state(
                {"confirmed_config": {"benchmark_mode_confirmed": "quick"}},
                session_id="test-session",
                state_root=root,
            )
            update_workflow_state(
                {
                    "confirmed_config": {"benchmark_mode_confirmed": "standard"},
                    "pending_question": {
                        "id": "benchmark_profile_choice",
                        "kind": "numbered_choice",
                        "prompt": "Choose benchmark mode",
                        "workflow_step": "benchmark_profile_choice",
                        "options": [
                            {"id": "1", "value": "quick", "label": "quick"},
                            {"id": "2", "value": "standard", "label": "standard"},
                        ],
                    },
                },
                session_id="test-session",
                state_root=root,
            )
            result = answer_pending_question("回到上一步", session_id="test-session", state_root=root)

        self.assertTrue(result["applied"])
        self.assertEqual(result["answer_kind"], "back")
        self.assertEqual(result["state"]["confirmed_config"]["benchmark_mode_confirmed"], "quick")

    def test_rpc_workload_validator_blocks_custom_methods_until_review_gates_pass(self):
        custom = validate_rpc_workload("solana", "mixed", mixed_weights={"getSlot": 70, "customMethod": 30})
        self.assertFalse(custom["ready"])
        self.assertIn("customMethod", custom["custom_methods"])
        self.assertTrue(custom["requires_fixture_review"])
        self.assertTrue(any("validated endpoint" in item for item in custom["errors"]))

    def test_rpc_workload_validator_blocks_unknown_chain_template(self):
        result = validate_rpc_workload("not-a-real-chain", "single", methods=["foo"])
        self.assertFalse(result["ready"])
        self.assertIn("chain template not found: not-a-real-chain", result["errors"])

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

"""Product decision context for one AnyChain Agent LLM turn.

This module builds the bounded context package passed to the ADK/model layer.
It is not an intent classifier and does not mutate workflow state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from workflows.group_registry import product_node_for_group


ALLOWED_INTENTS = {
    "greeting_or_help",
    "answer_current_question",
    "ask_concept",
    "change_target",
    "change_previous_value",
    "go_back",
    "switch_target_mode",
    "start_new_benchmark",
    "rerun_or_inspect_previous_job",
    "request_assumed_smoke",
    "custom_rpc_onboarding",
    "unsupported_chain_onboarding",
    "endpoint_or_sample_evidence",
    "analyze_job_or_report",
    "paste_evidence",
    "unknown_requires_clarification",
}


@dataclass(frozen=True)
class ProductIntent:
    intent: str
    confidence: str = "medium"
    applies_to_pending_question: bool = False
    state_patch: dict[str, Any] = field(default_factory=dict)
    requested_tool: str | None = None
    next_node: str | None = None
    requires_user_confirmation: bool = True
    user_message: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_product_intent(payload: dict[str, Any], context: dict[str, Any] | None = None) -> list[str]:
    """Return validation errors for a model-produced ProductIntent payload."""
    errors: list[str] = []
    context = context or {}
    intent = payload.get("intent")
    if intent not in ALLOWED_INTENTS:
        errors.append("intent must be one of the allowed product intents")
    if payload.get("confidence") not in {"low", "medium", "high"}:
        errors.append("confidence must be low, medium, or high")
    if not isinstance(payload.get("applies_to_pending_question", False), bool):
        errors.append("applies_to_pending_question must be boolean")
    if not isinstance(payload.get("state_patch", {}), dict):
        errors.append("state_patch must be an object")
    if not isinstance(payload.get("requires_user_confirmation", True), bool):
        errors.append("requires_user_confirmation must be boolean")
    user_message = payload.get("user_message", "")
    if user_message is not None and not isinstance(user_message, str):
        errors.append("user_message must be a string")

    requested_tool = payload.get("requested_tool")
    allowed_tools = set(context.get("allowed_tools") or [])
    if requested_tool and allowed_tools and requested_tool not in allowed_tools:
        errors.append(f"requested_tool is not allowed in the current decision node: {requested_tool}")

    next_node = payload.get("next_node")
    valid_transitions = set((context.get("valid_transitions") or {}).values())
    valid_transitions.update((context.get("valid_transitions") or {}).keys())
    if next_node and valid_transitions and next_node not in valid_transitions:
        errors.append(f"next_node is not valid from the current decision node: {next_node}")

    if payload.get("applies_to_pending_question") and not (context.get("pending_question") or {}):
        errors.append("applies_to_pending_question cannot be true without an active pending_question")

    return errors


def build_product_decision_context(
    *,
    workflow_state: dict[str, Any] | None,
    framework_summary: dict[str, Any] | None = None,
    input_mode: str = "normal_user_turn",
    terminal_language: str = "en",
) -> dict[str, Any]:
    """Build the bounded context needed for one LLM intent-resolution turn."""
    state = dict(workflow_state or {})
    pending = state.get("pending_question") or {}
    node = _current_node(state, pending, input_mode)
    allowed_intents = _allowed_intents(node, bool(pending), input_mode)
    allowed_tools = _allowed_tools(node)
    transitions = _valid_transitions(node, pending)
    group_context = _workflow_group_context(state)
    return {
        "identity": {
            "agent": "AnyChain Benchmark Agent",
            "purpose": "configure, validate, run, resume, and analyze blockchain node benchmark jobs",
            "language": terminal_language,
        },
        "framework_facts": _framework_facts(framework_summary or {}),
        "current_node": node,
        "workflow_group": group_context,
        "input_mode": input_mode,
        "pending_question": pending,
        "state_summary": _state_summary(state),
        "group_action_contract": _group_action_contract(bool(pending), input_mode),
        "missing_required_fields": list(state.get("missing_fields") or []),
        "blockers": list(state.get("blockers") or []),
        "allowed_intents": allowed_intents,
        "allowed_tools": allowed_tools,
        "valid_transitions": transitions,
        "guardrails": _guardrails(node, bool(pending)),
        "few_shot_patterns": _few_shot_patterns(node),
        "output_contract": {
            "format": "visible_response",
            "internal_intent_schema": ProductIntent(
                intent="unknown_requires_clarification",
                confidence="medium",
                requires_user_confirmation=True,
                user_message="Ask one concise clarification question.",
            ).as_dict(),
        },
    }


def render_product_decision_context(context: dict[str, Any]) -> str:
    """Render the context package compactly for the terminal-turn prompt."""
    return (
        "Product decision context:\n"
        f"- current_node: {context.get('current_node')}\n"
        f"- workflow_group: {context.get('workflow_group')}\n"
        f"- input_mode: {context.get('input_mode')}\n"
        f"- framework_facts: {context.get('framework_facts')}\n"
        f"- state_summary: {context.get('state_summary')}\n"
        f"- group_action_contract: {context.get('group_action_contract')}\n"
        f"- pending_question: {context.get('pending_question')}\n"
        f"- missing_required_fields: {context.get('missing_required_fields')}\n"
        f"- blockers: {context.get('blockers')}\n"
        f"- allowed_intents: {context.get('allowed_intents')}\n"
        f"- allowed_tools: {context.get('allowed_tools')}\n"
        f"- valid_transitions: {context.get('valid_transitions')}\n"
        f"- guardrails: {context.get('guardrails')}\n"
        f"- few_shot_patterns: {context.get('few_shot_patterns')}\n"
        "When user input is free text rather than a structural answer, infer a "
        "ProductIntent internally, validate it against this context, and route "
        "back into the decision tree through tools/state. Do not invent a new "
        "workflow outside valid_transitions.\n"
    )


def _current_node(state: dict[str, Any], pending: dict[str, Any], input_mode: str) -> str:
    if input_mode in {"pasted_evidence", "evidence_question"}:
        return "evidence_review"
    if pending:
        question_id = str(pending.get("id") or "")
        branch = str(pending.get("branch") or "")
        if branch == "benchmark_setup":
            question_id = str(pending.get("id") or "")
            if question_id.startswith("disk_") or question_id in {
                "cloud_region",
                "cloud_zone",
                "machine_type",
                "volume_baseline_confirm",
                "data_vol_type",
                "data_vol_size",
                "data_vol_max_iops",
                "data_vol_max_throughput",
                "accounts_vol_type",
                "accounts_vol_size",
                "accounts_vol_max_iops",
                "accounts_vol_max_throughput",
                "network_interface",
                "network_interface_confirm",
                "network_max_bandwidth_gbps",
                "network_bandwidth_confirm",
                "blockchain_process_names",
                "process_names_confirm",
            }:
                return "environment_config"
            if question_id in {
                "benchmark_profile_choice",
                "benchmark_profile_confirm",
                "benchmark_profile_adjust_item",
                "workload_customization_choice",
                "chain_template_reviewed",
                "rpc_mode",
                "rpc_mode_choice",
                "default_workload_confirm",
                "workload_confirm",
                "rpc_workload_confirmed",
                "mixed_weights_confirmed",
                "mixed_weights_confirm",
                "custom_rpc_add",
                "rpc_param_samples_confirmed",
            }:
                return "rpc_workload"
            if question_id in {"observability_mode_choice", "observability_ports_confirm"}:
                return "observability"
        if branch:
            return branch
        if question_id:
            return question_id
    step = str(state.get("workflow_step") or "")
    active = str(state.get("active_workflow") or state.get("active_intent") or "")
    if step:
        return step
    active_group = str(state.get("active_group") or state.get("next_blocking_group") or "").strip()
    group_node = _node_for_group(active_group)
    if group_node:
        return group_node
    if active:
        return active
    if state.get("latest_job_id"):
        return "previous_job_available"
    return "startup"


def _framework_facts(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "chains": summary.get("chain_count", "unknown"),
        "adapter_families": summary.get("family_count", "unknown"),
        "rpc_methods": summary.get("unique_rpc_method_count", "unknown"),
        "fake_node_fixtures": summary.get("fake_node_fixture_file_count", "unknown"),
        "fake_node_semantics": "closed-loop framework validation with fixture-backed responses; not real node performance",
        "real_node_semantics": "benchmark a reachable user endpoint after endpoint validation",
        "smoke_semantics": "complete isolated benchmark lifecycle, not a mock and not only a quick profile",
    }


def _state_summary(state: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "active_intent",
        "active_workflow",
        "workflow_step",
        "active_group",
        "next_blocking_group",
        "last_completed_group",
        "target_mode",
        "chain",
        "chain_status",
        "rpc_mode",
        "benchmark_profile",
        "assumed_for_smoke",
        "latest_job_id",
    )
    return {key: state.get(key) for key in keys if state.get(key) not in (None, "", [], {})}


def _workflow_group_context(state: dict[str, Any]) -> dict[str, Any]:
    progress = state.get("group_progress") if isinstance(state.get("group_progress"), dict) else {}
    compact_progress: dict[str, str] = {}
    for group, item in progress.items():
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "").strip()
        if status and status != "pending":
            compact_progress[str(group)] = status
    stack = []
    for item in list(state.get("interruption_stack") or [])[-3:]:
        if isinstance(item, dict) and item.get("group"):
            stack.append({"group": item.get("group"), "reason": item.get("reason", "")})
    return {
        "active_group": state.get("active_group") or "",
        "next_blocking_group": state.get("next_blocking_group") or "",
        "last_completed_group": state.get("last_completed_group") or "",
        "group_progress": compact_progress,
        "invalidated_fields": list(state.get("invalidated_fields") or [])[:20],
        "interruption_stack": stack,
    }


def _group_action_contract(has_pending: bool, input_mode: str) -> dict[str, Any]:
    structural = (
        "If the reply exactly fits the current typed pending_question, use answer_pending_question and then recompute_next_blocking_group."
        if has_pending
        else "There is no pending_question; do not treat bare Y/N or a bare number as approval."
    )
    if input_mode in {"pasted_evidence", "evidence_question"}:
        non_structural = (
            "Treat this turn as evidence for error/report analysis. Do not apply extracted config values until the user confirms them in a later turn."
        )
    else:
        non_structural = (
            "If the user changes chain, target mode, RPC workload, QPS, observability, disk, endpoint, or analysis topic, classify the target group, call record_group_jump, then continue through deterministic group tools and validators."
        )
    return {
        "structural_answer": structural,
        "non_structural_turn": non_structural,
        "after_group_change": "After a group completes or changes dependencies, call recompute_next_blocking_group.",
        "forbidden": "Do not patch visible transcripts, guess from prefixes, or write user prose into a config field.",
    }


def _node_for_group(group: str) -> str:
    return product_node_for_group(group)


def _allowed_intents(node: str, has_pending: bool, input_mode: str) -> list[str]:
    base = ["ask_concept", "change_previous_value", "go_back", "unknown_requires_clarification"]
    if has_pending:
        base.insert(0, "answer_current_question")
    if input_mode == "pending_question_interruption":
        return [
            "answer_current_question",
            "change_target",
            "switch_target_mode",
            "change_previous_value",
            "go_back",
            "ask_concept",
            "request_assumed_smoke",
            "custom_rpc_onboarding",
            "unsupported_chain_onboarding",
            "unknown_requires_clarification",
        ]
    if input_mode in {"pasted_evidence", "evidence_question"}:
        return ["paste_evidence", "ask_concept", "change_previous_value", "unknown_requires_clarification"]
    if node in {"startup", "previous_job_available"}:
        return ["greeting_or_help", "rerun_or_inspect_previous_job", "start_new_benchmark", "ask_concept", "unknown_requires_clarification"]
    if "custom_rpc" in node:
        return base + ["custom_rpc_onboarding", "endpoint_or_sample_evidence"]
    if "unsupported" in node or "onboarding" in node:
        return base + ["unsupported_chain_onboarding", "endpoint_or_sample_evidence"]
    if "job" in node or "analysis" in node:
        return base + ["analyze_job_or_report", "rerun_or_inspect_previous_job", "start_new_benchmark"]
    return base + ["switch_target_mode", "request_assumed_smoke", "custom_rpc_onboarding", "unsupported_chain_onboarding"]


def _allowed_tools(node: str) -> list[str]:
    mapping = {
        "startup": ["load_framework_context", "latest_job", "run_doctor"],
        "previous_job_available": ["latest_job", "job_status", "tail_job_log", "analyze_artifacts"],
        "benchmark_setup": ["discover_environment", "build_missing_config_questions", "validate_required_config", "validate_chain_template", "validate_rpc_workload"],
        "dependency_setup": ["audit_dependencies", "run_doctor"],
        "target_selection": ["load_workflow_state", "update_workflow_state", "build_missing_config_questions", "validate_required_config"],
        "environment_config": ["discover_environment", "build_missing_config_questions", "validate_required_config"],
        "chain_selection": ["validate_chain_template", "request_unsupported_chain_handoff", "build_onboarding_handoff", "load_framework_context"],
        "real_node_endpoint": ["validate_rpc_endpoint", "validate_required_config", "build_missing_config_questions"],
        "rpc_workload": ["load_default_workload", "validate_rpc_workload"],
        "custom_rpc": ["validate_rpc_endpoint", "validate_rpc_workload", "request_custom_rpc_handoff", "build_onboarding_handoff"],
        "unsupported_chain": ["load_framework_context", "request_unsupported_chain_handoff", "build_onboarding_handoff", "validate_rpc_endpoint"],
        "observability": ["validate_required_config"],
        "preflight_smoke": ["prepare_benchmark_run", "validate_execution_gate", "run_fake_node_smoke_benchmark"],
        "real_execution": ["validate_execution_gate", "submit_benchmark_job"],
        "job_resume": ["latest_job", "job_status", "tail_job_log", "analyze_artifacts"],
        "analysis": ["latest_job", "analyze_artifacts"],
        "evidence_review": ["load_workflow_state", "update_workflow_state"],
        "evidence_repair": ["load_workflow_state", "update_workflow_state"],
        "advanced_threshold_review": ["build_missing_config_questions", "validate_required_config"],
    }
    if node in mapping:
        return _with_group_workflow_tools(mapping[node])
    if node.startswith("disk_") or node in {"cloud_region", "cloud_zone", "machine_type"}:
        return _with_group_workflow_tools(mapping["environment_config"])
    if "custom_rpc" in node:
        return _with_group_workflow_tools(mapping["custom_rpc"])
    if "unsupported" in node:
        return _with_group_workflow_tools(mapping["unsupported_chain"])
    return _with_group_workflow_tools([])


def _with_group_workflow_tools(tools: list[str]) -> list[str]:
    required = [
        "load_workflow_state",
        "update_workflow_state",
        "record_group_jump",
        "recompute_next_blocking_group",
        "propose_opening_help_choice",
        "propose_benchmark_target_mode_choice",
        "propose_chain_selection_question",
        "propose_chain_identity_resolution",
        "propose_chain_protocol_resolution",
        "request_unsupported_chain_handoff",
        "request_custom_rpc_handoff",
        "propose_chain_change_confirmation",
        "propose_unsupported_chain_endpoint_gate",
        "propose_custom_rpc_endpoint_gate",
        "propose_real_node_endpoint_gate",
        "propose_disk_device_choice",
        "propose_benchmark_profile_choice",
        "propose_workload_customization_choice",
    ]
    result: list[str] = []
    for tool in [*required, *tools]:
        if tool not in result:
            result.append(tool)
    return result


def _valid_transitions(node: str, pending: dict[str, Any]) -> dict[str, str]:
    transitions: dict[str, str] = {}
    for key in ("next_on_yes", "next_on_no", "next_on_manual"):
        target = pending.get(key) or {}
        if isinstance(target, dict):
            step = str(target.get("workflow_step") or target.get("tool") or "")
            if step:
                transitions[key] = step
    if transitions:
        return transitions
    defaults = {
        "startup": "opening_help",
        "previous_job_available": "job_or_new_benchmark",
        "chain_selection": "validate_chain_template",
        "environment_config": "next_missing_config_question",
        "rpc_workload": "validate_rpc_workload",
        "observability": "validate_required_config",
        "preflight_smoke": "smoke_or_repair",
        "real_execution": "detached_job_submit",
        "evidence_review": "confirm_apply_evidence",
    }
    if node in defaults:
        transitions["default"] = defaults[node]
    elif node.startswith("disk_"):
        transitions["default"] = "next_missing_config_question"
    else:
        transitions["default"] = "clarify_or_update_workflow_state"
    return transitions


def _guardrails(node: str, has_pending: bool) -> list[str]:
    rules = [
        "Do not bypass validators, endpoint probes, preflight, smoke, or approval gates.",
        "Do not ask multiple unrelated blocking questions in one response.",
        "Do not treat quick profile selection as smoke validation.",
        "Do not claim unsupported chains or custom RPC methods are supported without validation evidence.",
    ]
    if not has_pending:
        rules.append("Do not accept bare Y/N as approval without an active pending_question.")
    if has_pending:
        rules.append("If free text does not structurally answer the active pending_question, classify whether it changes target, asks a question, goes back, or is invalid; do not write it into the pending field.")
        rules.append("If the user jumps to another configuration group, call record_group_jump before asking that group's next question.")
    rules.append("After completing or invalidating a group, call recompute_next_blocking_group instead of returning to a stale literal prompt.")
    if "custom_rpc" in node or "unsupported" in node:
        rules.append("Endpoint and request/response samples must be probed before they are trusted.")
    return rules


def _few_shot_patterns(node: str) -> list[dict[str, str]]:
    common = [
        {
            "user": "我刚才说错了，换成 real-node",
            "intent": "switch_target_mode",
            "next": "ask for LOCAL_RPC_URL and invalidate fake-node-only assumptions",
        },
        {
            "user": "先解释一下 fake-node 和 real-node 区别",
            "intent": "ask_concept",
            "next": "answer concept question without mutating benchmark config",
        },
        {
            "user": "随便帮我 smoke 一下",
            "intent": "request_assumed_smoke",
            "next": "use discovered values plus marked assumptions; fake-node only; run complete isolated smoke",
        },
    ]
    if "custom_rpc" in node:
        common.append({
            "user": "这个 method 用这个样本就行",
            "intent": "endpoint_or_sample_evidence",
            "next": "probe endpoint and compare live response before trusting sample",
        })
    if node in {"environment_config", "chain_selection", "rpc_workload"}:
        common.append({
            "user": "我需要测试 BNB",
            "intent": "change_target",
            "next": "call propose_chain_change_confirmation for BNB Smart Chain (bsc) with target_mode_explicit=false unless this same user turn explicitly says fake-node or real-node; do not store the text as the active pending field",
        })
    return common

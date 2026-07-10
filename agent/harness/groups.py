"""Deterministic group workflow engine for the LangGraph Harness."""

from __future__ import annotations

import json
import re
from typing import Any

try:
    from agent.knowledge.chain_identity import canonicalize_chain_scalar, repo_chain_names
    from agent.knowledge.framework_capabilities import load_framework_capabilities
    from agent.runners.job_manager import list_jobs, resume_job
    from agent.validators.rpc_workload import default_workload
    from agent.validators.endpoint_probe import validate_rpc_endpoint
except ModuleNotFoundError:  # product script adds agent/ to sys.path
    from knowledge.chain_identity import canonicalize_chain_scalar, repo_chain_names
    from knowledge.framework_capabilities import load_framework_capabilities
    from runners.job_manager import list_jobs, resume_job
    from validators.rpc_workload import default_workload
    from validators.endpoint_probe import validate_rpc_endpoint

from .state import AgentGraphState, PendingQuestion, new_state
from .intent import (
    ALLOWED_GROUPS,
    extract_chain_mention,
    extract_rpc_schema_from_evidence,
    resolve_action_queue,
    resolve_intent_action,
    resolve_pending_choice,
    resolve_unknown_chain_identity,
)
from .nodes.execution import run_approved_preflight_and_smoke
from .oracle import compute_next_action, format_current_context, format_current_state, format_recommended_next_action
from .turns import adjudicate_turn


SUPPORTED_ADAPTER_FAMILIES = {"jsonrpc", "substrate", "rest", "tendermint", "bitcoin_jsonrpc", "hedera_dual"}
CONFIRMABLE_CONFIG_FIELDS = {
    "CLOUD_REGION",
    "CLOUD_ZONE",
    "MACHINE_TYPE",
    "LEDGER_DEVICE",
    "DATA_VOL_TYPE",
    "DATA_VOL_SIZE",
    "DATA_VOL_MAX_IOPS",
    "DATA_VOL_MAX_THROUGHPUT",
    "ACCOUNTS_DEVICE",
    "ACCOUNTS_VOL_TYPE",
    "ACCOUNTS_VOL_SIZE",
    "ACCOUNTS_VOL_MAX_IOPS",
    "ACCOUNTS_VOL_MAX_THROUGHPUT",
    "NETWORK_INTERFACE",
    "NETWORK_MAX_BANDWIDTH_GBPS",
    "BLOCKCHAIN_PROCESS_NAMES",
}
PROPOSED_ENDPOINT_FIELDS = {"LOCAL_RPC_URL", "MAINNET_RPC_URL"}
SPECIAL_CONFIG_FIELDS = {"RPC_MODE", "HAS_ACCOUNTS_DEVICE"}
QUEUE_RESUME_PENDING_IDS = {
    "inferred_config_review",
    "target_mode_change_confirm",
    "chain_change_confirm",
    "chain_ambiguity_confirm",
    "unknown_chain_identity_confirm",
}


def process_turn(state: AgentGraphState) -> AgentGraphState:
    state = _copy_state(state)
    text = str(state.get("last_user_input") or "").strip()
    language = state.get("language") or "en"
    state["visible_response"] = []
    turn = adjudicate_turn(state, text)

    collecting = state.get("evidence_collection") or {}
    if turn.kind == "evidence_continuation":
        if not text:
            return _prompt_evidence_collection_waiting(state, collecting)
        evidence_question = collecting.get("question") if isinstance(collecting.get("question"), dict) else {}
        if str(evidence_question.get("id") or "") == "freeform_evidence" and _looks_like_evidence_analysis_request(text):
            collected = _continue_evidence_collection(state, text, collecting)
            if collected.pop("_stop_after_response", False) or collected.get("pending_question"):
                return collected
            return _ask_next_blocking_question(collected)
        if (
            not _looks_like_evidence_fragment(text, evidence_question)
            and not _evidence_collection_complete(_non_empty_evidence_lines(text))
        ):
            saved_collection = dict(collecting)
            state["evidence_collection"] = {}
            routed = _route_free_text(state, text)
            if routed and (routed.get("_stop_after_response") or routed.get("pending_question") or routed.get("visible_response")):
                return routed
            state["evidence_collection"] = saved_collection
        collected = _continue_evidence_collection(state, text, collecting)
        if collected.pop("_stop_after_response", False) or collected.get("pending_question"):
            return collected
        return _ask_next_blocking_question(collected)

    if turn.kind == "empty":
        return _ask_next_blocking_question(state)

    pending = state.get("pending_question") or {}
    if str(pending.get("id") or "") == "inferred_config_review":
        merged = _merge_pending_config_proposal_from_text(state, text)
        if merged:
            return merged

    direct_assignments = {} if "\n" in text else _parse_known_config_assignments(text)
    if direct_assignments:
        state = _apply_direct_config_assignments(state, direct_assignments)
        return _ask_next_blocking_question(state)

    if state.get("evidence_buffer") and _looks_like_evidence_analysis_request(text):
        return _analyze_saved_evidence(state, text)

    handoff = _route_secondary_handoff_text(state, text)
    if handoff:
        return handoff

    if pending and str(pending.get("kind") or "") == "evidence" and _should_start_evidence_collection(text):
        return _start_evidence_collection(state, text, pending)

    if pending and _answer_fits_pending(text, pending):
        pending_question_id = str(pending.get("id") or "")
        state = _apply_pending_answer(state, text, pending)
        if _post_pending_answer_context_requested(state, text, pending):
            state["visible_response"] = [_pending_context_response(state, state.get("pending_question") or {}, language)]
            state["_stop_after_response"] = True
            return state
        if _pending_answer_contains_extra_intent(text, pending) and not state.get("pending_question") and not state.get("_stop_after_response"):
            routed_after_answer = _route_free_text(state, text)
            if routed_after_answer:
                state = routed_after_answer
        if (pending.get("resume_action_queue") or pending_question_id in QUEUE_RESUME_PENDING_IDS) and state.get("pending_question"):
            state["pending_question"]["resume_action_queue"] = True
        if pending_question_id == "opening_next_action":
            state["action_queue"] = []
        elif not pending.get("resume_action_queue") and pending_question_id not in QUEUE_RESUME_PENDING_IDS:
            state["action_queue"] = []
        if not state.get("pending_question") and not state.get("_stop_after_response") and state.get("action_queue"):
            state = _process_action_queue(state, list(state.get("action_queue") or []), text) or state
        if state.pop("_stop_after_response", False) or state.get("pending_question"):
            return state
        return _ask_next_blocking_question(state)

    if pending:
        compound = _apply_compound_pending_answer(state, text, pending)
        if compound:
            if pending.get("resume_action_queue") and compound.get("pending_question"):
                compound["pending_question"]["resume_action_queue"] = True
            if pending.get("resume_action_queue") and not compound.get("pending_question") and not compound.get("_stop_after_response") and compound.get("action_queue"):
                compound = _process_action_queue(compound, list(compound.get("action_queue") or []), text) or compound
            if compound.pop("_stop_after_response", False) or compound.get("pending_question"):
                return compound
            return _ask_next_blocking_question(compound)
        if str(pending.get("id") or "") == "inferred_config_review":
            merged = _merge_pending_config_proposal_from_text(state, text)
            if merged:
                return merged
            proposal = state.setdefault("inferred_config", {}).get("pending_review")
            if isinstance(proposal, dict):
                pending["prompt"] = _format_config_proposal_prompt(state, proposal)
                state["pending_question"] = pending
            _defer_actions_until_pending_confirmed(state, text)
            state["visible_response"] = [_localized(
                language,
                "请先确认是否应用刚才推断出的配置。回复 `Y` 应用，或 `N` 放弃；确认后我会继续处理后续配置。",
                "Confirm whether to apply the inferred configuration first. Reply `Y` to apply it or `N` to discard it; after that I can continue with the next configuration.",
            ), _render_question(pending, language)]
            return state

    if pending and str(pending.get("kind") or "") == "device" and text.lower() in {"y", "yes", "n", "no"}:
        state["visible_response"] = [_localized(
            language,
            "这个问题需要选择编号或直接输入设备/接口名；`Y/N` 不能唯一确定候选。",
            "This question needs an option number or a device/interface name; `Y/N` does not uniquely identify a candidate.",
        ), _render_question(pending, language)]
        return state

    if pending and _pending_endpoint_context_question(pending, text):
        state["visible_response"] = [_pending_context_response(state, pending, language)]
        state["_stop_after_response"] = True
        return state

    routed = None
    if not (pending and str(pending.get("kind") or "") == "numbered_choice" and _looks_like_assignment_answer(text)):
        routed = _route_free_text(state, text)
    if routed:
        if routed.pop("_stop_after_response", False) or routed.get("pending_question"):
            return routed
        return _ask_next_blocking_question(routed)

    if pending and str(pending.get("kind") or "") in {"numbered_choice", "yes_no"} and not _looks_like_assignment_answer(text):
        choice = resolve_pending_choice(state, text, pending)
        if bool(choice.get("matched")) and str(choice.get("confidence") or "").lower() in {"medium", "high"}:
            selected = choice.get("selected_value")
            if _pending_option_value_exists(selected, pending):
                state = _apply_pending_answer(state, str(selected), pending)
                if (pending.get("resume_action_queue") or str(pending.get("id") or "") in QUEUE_RESUME_PENDING_IDS) and state.get("pending_question"):
                    state["pending_question"]["resume_action_queue"] = True
                if not pending.get("resume_action_queue") and str(pending.get("id") or "") not in QUEUE_RESUME_PENDING_IDS:
                    state["action_queue"] = []
                if not state.get("pending_question") and not state.get("_stop_after_response") and state.get("action_queue"):
                    state = _process_action_queue(state, list(state.get("action_queue") or []), text) or state
                if state.pop("_stop_after_response", False) or state.get("pending_question"):
                    return state
                return _ask_next_blocking_question(state)

    if pending:
        chain_detour = _route_pending_chain_detour(state, text, pending)
        if chain_detour:
            return chain_detour
        if _looks_like_user_question(text):
            state["visible_response"] = [_pending_context_response(state, pending, language)]
            state["_stop_after_response"] = True
            return state
        state["visible_response"] = [_localized(
            language,
            "这条回复不像当前问题的答案。我会先保持当前问题不变；你也可以直接说明要切换到哪个配置项。",
            "That reply does not look like an answer to the current question. I will keep the current question active; you can also describe which configuration area to change.",
        ), _render_question(pending, language)]
        return state
    return _ask_next_blocking_question(state)


def _route_pending_chain_detour(state: AgentGraphState, text: str, pending: PendingQuestion) -> AgentGraphState | None:
    if str(pending.get("group") or "") == "chain_identity":
        return None
    current_chain = str((state.get("chain_identity") or {}).get("canonical") or "").strip()
    if not current_chain:
        return None
    mention = extract_chain_mention(state, text)
    if not bool(mention.get("found")) or str(mention.get("confidence") or "").lower() not in {"medium", "high"}:
        return None
    chain_text = _strip_scalar(str(mention.get("chain_text") or ""))
    if not chain_text:
        return None
    return _request_chain_change_confirmation(state, chain_text, {"chain_text": chain_text, "reason": mention.get("reason") or ""})


def _copy_state(state: AgentGraphState) -> AgentGraphState:
    output: AgentGraphState = dict(state)
    output.setdefault("action_queue", [])
    output.setdefault("current_action", {})
    output.setdefault("completed_actions", [])
    output.setdefault("action_errors", [])
    output.setdefault("group_states", {})
    output.setdefault("group_history", [])
    output.setdefault("confirmed_config", {})
    output.setdefault("inferred_config", {})
    output.setdefault("invalidated_groups", [])
    output.setdefault("interruption_stack", [])
    output.setdefault("chain_identity", {})
    output.setdefault("workload", {})
    output.setdefault("custom_rpc", {})
    output.setdefault("endpoint_evidence", {})
    output.setdefault("fixture_evidence", {})
    output.setdefault("sync_observe", {})
    output.setdefault("observability", {})
    output.setdefault("evidence_buffer", [])
    output.setdefault("evidence_collection", {})
    output.setdefault("audit_events", [])
    return output


def _reset_workflow_state(state: AgentGraphState) -> AgentGraphState:
    """Clear Agent workflow configuration while preserving startup facts."""

    reset = new_state(str(state.get("thread_id") or "default"), language=str(state.get("language") or "en"))
    for key in ("discovery", "framework_summary", "web_research", "latest_job_id", "job", "report_context"):
        if key in state:
            reset[key] = state.get(key)  # type: ignore[literal-required]
    reset["audit_events"] = list(state.get("audit_events") or []) + [{"event": "workflow_reset"}]
    return reset


def _route_free_text(state: AgentGraphState, text: str) -> AgentGraphState | None:
    if _looks_like_pasted_evidence(text):
        return _start_freeform_evidence_collection(state, text)

    queue = resolve_action_queue(state, text)
    actions = _normalized_action_queue(queue)
    config_proposal = _extract_structured_config_proposal(text)
    if config_proposal and not any(str(item.get("type") or "") == "propose_config_values" for item in actions):
        actions.append(config_proposal)
    actions = _augment_actions_with_missing_goal_context(state, actions, text)
    if _has_meaningful_queue(actions):
        routed = _process_action_queue(state, actions, text)
        if routed:
            return routed

    action = resolve_intent_action(state, text)
    return _route_single_action(state, action, text)


def _route_single_action(state: AgentGraphState, action: dict[str, Any], text: str) -> AgentGraphState | None:
    intent = str(action.get("intent") or "unknown").strip()
    target_mode = _normalized_target_mode(action.get("target_mode"))
    if intent in {"greeting", "unknown"} and not target_mode and not action.get("chain_text") and not action.get("group"):
        if intent == "greeting":
            state["active_group"] = "opening"
            state["pending_question"] = {}
            return state
        return None
    chain_candidate = _strip_scalar(str(action.get("chain_text") or ""))
    if intent in {"choose_chain", "change_chain"} and chain_candidate:
        if target_mode:
            action["target_mode"] = target_mode
            current_mode = str(state.get("target_mode") or "").strip()
            if not current_mode:
                state["target_mode"] = target_mode
                state["workflow_mode"] = "sync_observe" if target_mode == "sync-observe" else "rpc_benchmark"
                _invalidate_for_target_mode(state)
        ambiguity = _chain_ambiguity_question(state, text, chain_candidate)
        if ambiguity:
            state["pending_question"] = ambiguity
            state["active_group"] = "chain_identity"
            state["visible_response"] = [_render_question(ambiguity, state.get("language", "en"))]
            return state
        if intent == "change_chain" or (state.get("chain_identity") or {}).get("canonical"):
            return _request_chain_change_confirmation(state, chain_candidate, action)
        return _handle_chain_candidate(state, chain_candidate, action)
    if target_mode and not (state.get("chain_identity") or {}).get("canonical"):
        mention = extract_chain_mention(state, text)
        if bool(mention.get("found")) and str(mention.get("confidence") or "").lower() in {"medium", "high"}:
            mentioned_chain = _strip_scalar(str(mention.get("chain_text") or ""))
            if mentioned_chain:
                state["target_mode"] = target_mode
                state["workflow_mode"] = "sync_observe" if target_mode == "sync-observe" else "rpc_benchmark"
                _invalidate_for_target_mode(state)
                return _handle_chain_candidate(state, mentioned_chain, mention)
    if intent == "choose_target_mode" and target_mode:
        current_mode = str(state.get("target_mode") or "").strip()
        if current_mode and current_mode != target_mode:
            return _request_target_mode_change_confirmation(state, target_mode)
        state["target_mode"] = target_mode
        state["workflow_mode"] = "sync_observe" if target_mode == "sync-observe" else "rpc_benchmark"
        _invalidate_for_target_mode(state)
        state["active_group"] = "chain_identity"
        state["pending_question"] = {}
        return state
    if target_mode:
        current_mode = str(state.get("target_mode") or "").strip()
        if current_mode and current_mode != target_mode:
            return _request_target_mode_change_confirmation(state, target_mode)
        state["target_mode"] = target_mode
        state["workflow_mode"] = "sync_observe" if target_mode == "sync-observe" else "rpc_benchmark"
        _invalidate_for_target_mode(state)
        state["active_group"] = "chain_identity"
        state["pending_question"] = {}
        return state
    group = str(action.get("group") or "").strip()
    if intent == "change_group" and group in ALLOWED_GROUPS:
        return _activate_group_question(state, group)
    if intent == "go_back":
        previous_group = _pop_previous_group(state)
        if previous_group:
            return _activate_group_question(state, previous_group, record_history=False)
        state["visible_response"] = [_localized(
            state.get("language", "en"),
            "当前没有可回退的配置组。你可以直接说明要回到哪个配置项，例如 RPC、QPS、磁盘或可观测性。",
            "There is no previous configuration group to return to. You can directly name the area, such as RPC, QPS, disk, or observability.",
        )]
        state["_stop_after_response"] = True
        return state
    if intent == "ask_capabilities":
        state["active_group"] = "report_artifact_analysis"
        state["pending_question"] = {}
        state["visible_response"] = [_framework_capability_summary(state)]
        state["_stop_after_response"] = True
        return state
    if intent == "answer_opening_question":
        if not state.get("pending_question"):
            state["active_group"] = "opening"
        state["visible_response"] = [_opening_consultation_response(state, action, text)]
        state["_stop_after_response"] = True
        return state
    if intent == "analyze_evidence":
        if not _looks_like_pasted_evidence(text):
            state["active_group"] = "error_evidence_analysis"
            state["pending_question"] = {}
            state["visible_response"] = [_evidence_help_response(state)]
            state["_stop_after_response"] = True
            return state
        state["active_group"] = "error_evidence_analysis"
        state.setdefault("evidence_buffer", []).append({"text": text})
        state["pending_question"] = {}
        state["visible_response"] = [_localized(
            state.get("language", "en"),
            "我已把这段内容作为证据保存。下一步会结合框架上下文分析原因和建议动作。",
            "I saved this as evidence. Next I can analyze it with the framework context and suggest actions.",
        )]
        state["_stop_after_response"] = True
        return state
    if intent == "analyze_report":
        state["active_group"] = "report_artifact_analysis"
        state["pending_question"] = {}
        state["visible_response"] = [_report_artifact_entry_response(state)]
        state["_stop_after_response"] = True
        return state
    return None


def _normalized_action_queue(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_actions = payload.get("actions") if isinstance(payload, dict) else None
    if isinstance(raw_actions, dict):
        raw_actions = [raw_actions]
    if not isinstance(raw_actions, list):
        return []
    output: list[dict[str, Any]] = []
    for raw in raw_actions[:12]:
        if not isinstance(raw, dict):
            continue
        action = dict(raw)
        action_type = str(action.get("type") or action.get("intent") or "unknown").strip()
        action["type"] = action_type
        action["confidence"] = str(action.get("confidence") or "medium").strip().lower()
        output.append(action)
    return output


def _augment_actions_with_missing_goal_context(state: AgentGraphState, actions: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
    """Ensure explicit benchmark goals survive mixed natural-language config turns.

    LLM extraction sometimes focuses on pasted config values and drops the
    surrounding "test BNB real-node" goal. The Harness owns workflow state, so
    it must restore explicit target-mode and chain actions before applying the
    config proposal.
    """

    if not actions:
        return actions
    prepared = [dict(item) for item in actions]
    existing_types = {str(item.get("type") or "") for item in prepared}
    prefix: list[dict[str, Any]] = []
    target_mode = _explicit_target_mode_from_text(text)
    if target_mode and "choose_target_mode" not in existing_types and not state.get("target_mode"):
        prefix.append({"type": "choose_target_mode", "target_mode": target_mode, "target_mode_explicit": True, "confidence": "high"})
        existing_types.add("choose_target_mode")
    if not (state.get("chain_identity") or {}).get("canonical") and "choose_chain" not in existing_types and "change_chain" not in existing_types:
        mention = extract_chain_mention(state, text)
        if bool(mention.get("found")) and str(mention.get("confidence") or "").lower() in {"medium", "high"}:
            chain_text = _strip_scalar(str(mention.get("chain_text") or ""))
            if chain_text:
                prefix.append({"type": "choose_chain", "chain_text": chain_text, "confidence": "high"})
    return _drop_ambiguous_custom_rpc_start_actions(_drop_conflicting_answer_actions(prefix + prepared), text)


def _drop_conflicting_answer_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    topics = {
        str(item.get("topic") or "").strip().lower()
        for item in actions
        if str(item.get("type") or "") == "answer_opening_question"
    }
    if not (topics & {"performance_benchmark_guidance", "mode_comparison"}):
        return actions
    pruned: list[dict[str, Any]] = []
    for item in actions:
        action_type = str(item.get("type") or "").strip()
        topic = str(item.get("topic") or "").strip().lower()
        target_mode = _normalized_target_mode(item.get("target_mode"))
        if action_type == "answer_opening_question" and topic in {"recommendation", "recommend_start"}:
            continue
        if "performance_benchmark_guidance" in topics and action_type == "choose_target_mode" and target_mode == "fake-node":
            continue
        pruned.append(item)
    return pruned


def _drop_ambiguous_custom_rpc_start_actions(actions: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
    """Prevent capability questions from starting the custom-RPC workflow.

    A user asking "can I add a custom RPC method?" needs an explanation, not an
    endpoint prompt. Starting the workflow is only safe when the turn includes a
    concrete method, endpoint, schema evidence, or imperative wording.
    """

    if not actions:
        return actions
    raw = str(text or "").strip()
    lowered = raw.lower()
    is_question = _looks_like_user_question(raw)
    capability_question = is_question and any(
        token in lowered
        for token in (
            "can i",
            "can we",
            "could i",
            "could we",
            "is it possible",
            "how to",
            "how do",
            "可以",
            "能不能",
            "能否",
            "怎么",
            "如何",
        )
    )
    if not capability_question:
        return actions

    filtered: list[dict[str, Any]] = []
    replaced = False
    for item in actions:
        if str(item.get("type") or "") != "start_custom_rpc":
            filtered.append(item)
            continue
        has_concrete_payload = any(
            _strip_scalar(str(item.get(key) or ""))
            for key in ("rpc_method", "rpc_endpoint", "rpc_schema_evidence")
        )
        if has_concrete_payload:
            filtered.append(item)
            continue
        replaced = True
    if replaced and not any(str(item.get("type") or "") == "answer_opening_question" and str(item.get("topic") or "") == "extension" for item in filtered):
        filtered.append({"type": "answer_opening_question", "topic": "extension", "confidence": "high"})
    return filtered


def _explicit_target_mode_from_text(text: str) -> str:
    raw = str(text or "").strip().lower()
    if _text_is_mode_consultation(raw):
        return ""
    if any(item in raw for item in ("not sure", "unsure", "don't know", "do not know", "不确定", "不知道", "没想好")):
        return ""
    if any(item in raw for item in ("sync-observe", "sync observe", "observe sync", "observe block", "block sync", "block import", "import observation", "观察同步", "节点同步", "同步观察", "观察追块", "追块")):
        return "sync-observe"
    if any(item in raw for item in ("real-node", "real node", "realnode", "真实节点", "实际节点")):
        return "real-node"
    if any(item in raw for item in ("fake-node", "fake node", "fakenode", "mock node", "mock-node", "模拟节点", "假节点")):
        return "fake-node"
    return ""


def _has_meaningful_queue(actions: list[dict[str, Any]]) -> bool:
    meaningful = [item for item in actions if str(item.get("type") or "") != "unknown"]
    if len(meaningful) > 1:
        return True
    queue_only_types = {
        "choose_target_mode",
        "choose_chain",
        "set_rpc_mode",
        "set_qps_mode",
        "set_qps_override",
        "set_observability",
        "set_sync_observe_source",
        "set_accounts_presence",
        "start_custom_rpc",
        "propose_config_values",
        "reset_session",
        "change_group",
        "go_back",
        "change_chain",
        "ask_capabilities",
        "answer_opening_question",
        "analyze_evidence",
        "analyze_report",
    }
    return any(str(item.get("type") or "") in queue_only_types for item in meaningful)


def _process_action_queue(state: AgentGraphState, actions: list[dict[str, Any]], text: str) -> AgentGraphState | None:
    prepared_actions = []
    origin_group = str(state.get("active_group") or "").strip()
    for item in actions:
        action = dict(item)
        action.setdefault("_origin_text", text)
        action.setdefault("_queue_origin_group", origin_group)
        prepared_actions.append(action)
    prepared_actions = _order_action_queue(prepared_actions)
    state["action_queue"] = prepared_actions
    state["completed_actions"] = []
    state["action_errors"] = []
    changed = False
    while state.get("action_queue"):
        action = dict(state["action_queue"].pop(0))
        state["current_action"] = action
        before_pending = bool(state.get("pending_question"))
        routed = _apply_queue_action(state, action, text)
        if routed is None:
            state.setdefault("action_errors", []).append({"action": action, "error": "unsupported_or_low_confidence"})
            continue
        state = routed
        state.setdefault("completed_actions", []).append(action)
        state["completed_actions"] = state["completed_actions"][-20:]
        changed = True
        if state.get("pending_question") and not before_pending:
            state["pending_question"]["resume_action_queue"] = True
        if state.pop("_queue_pause", False):
            if state.get("pending_question") and state.get("action_queue"):
                state["pending_question"]["resume_action_queue"] = True
            break
        if state.get("_stop_after_response"):
            break
        if state.get("pending_question") and not before_pending:
            break
    state["current_action"] = {}
    return state if changed else None


def _order_action_queue(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply Harness execution ordering without changing action semantics.

    LLMs may list a custom RPC workflow before the config proposal extracted
    from the same turn. The proposal is a user-confirmation gate; it must happen
    before custom RPC starts asking endpoint/schema questions, otherwise the
    queue can be stranded behind a pending workflow question.
    """

    if not any(str(item.get("type") or "") == "propose_config_values" for item in actions):
        return actions
    first_custom = next((idx for idx, item in enumerate(actions) if str(item.get("type") or "") == "start_custom_rpc"), None)
    if first_custom is None:
        return actions
    proposals = [item for item in actions if str(item.get("type") or "") == "propose_config_values"]
    others = [item for item in actions if str(item.get("type") or "") != "propose_config_values"]
    insert_at = next((idx for idx, item in enumerate(others) if str(item.get("type") or "") == "start_custom_rpc"), len(others))
    return others[:insert_at] + proposals + others[insert_at:]


def _apply_queue_action(state: AgentGraphState, action: dict[str, Any], text: str) -> AgentGraphState | None:
    action_type = str(action.get("type") or "unknown").strip()
    confidence = str(action.get("confidence") or "medium").strip().lower()
    origin_text = str(action.get("_origin_text") or text)
    if confidence == "low" and action_type not in {"greeting", "unknown"}:
        state.setdefault("action_errors", []).append({"action": action, "error": "low_confidence"})
        return None
    if action_type == "greeting":
        state["active_group"] = "opening"
        state["pending_question"] = {}
        return state
    if action_type == "unknown":
        return None
    if action_type == "reset_session":
        reset = _reset_workflow_state(state)
        reset["visible_response"] = [_localized(
            reset.get("language", "en"),
            "已清空之前的 Agent 配置。请告诉我这次要测试什么，或先选择 fake-node、real-node、sync-observe。",
            "Cleared the previous Agent configuration. Tell me what to test, or choose fake-node, real-node, or sync-observe.",
        )]
        reset["_stop_after_response"] = True
        return reset
    if action_type == "choose_target_mode":
        target_mode = _normalized_target_mode(action.get("target_mode"))
        if not target_mode:
            return None
        if not state.get("target_mode") and not _target_mode_is_explicit_in_text(origin_text, target_mode, action):
            state.setdefault("action_errors", []).append({"action": action, "error": "target_mode_not_explicit"})
            return None
        current_mode = str(state.get("target_mode") or "").strip()
        if current_mode and current_mode != target_mode:
            state = _request_target_mode_change_confirmation(state, target_mode)
            state["_queue_pause"] = True
            return state
        state["target_mode"] = target_mode
        state["workflow_mode"] = "sync_observe" if target_mode == "sync-observe" else "rpc_benchmark"
        _invalidate_for_target_mode(state)
        state["active_group"] = "chain_identity" if not _chain_confirmed(state) else _next_group(state)
        state["pending_question"] = {}
        return state
    if action_type in {"choose_chain", "change_chain"}:
        chain_candidate = _strip_scalar(str(action.get("chain_text") or ""))
        candidate_mode = _normalized_target_mode(action.get("target_mode"))
        target_mode = candidate_mode if _target_mode_is_explicit_in_text(origin_text, candidate_mode, action) else ""
        if target_mode and not state.get("target_mode"):
            state["target_mode"] = target_mode
            state["workflow_mode"] = "sync_observe" if target_mode == "sync-observe" else "rpc_benchmark"
            _invalidate_for_target_mode(state)
        if not chain_candidate and target_mode:
            mention = extract_chain_mention(state, origin_text)
            if bool(mention.get("found")) and str(mention.get("confidence") or "").lower() in {"medium", "high"}:
                chain_candidate = _strip_scalar(str(mention.get("chain_text") or ""))
        if not chain_candidate:
            state["active_group"] = "chain_identity"
            state["pending_question"] = _chain_question(state)
            state["visible_response"] = [_render_question(state["pending_question"], state.get("language", "en"))]
            state["_queue_pause"] = True
            return state
        current_chain = canonicalize_chain_scalar(
            str((state.get("chain_identity") or {}).get("canonical") or ""),
            known_chains=set(repo_chain_names()),
        )
        candidate_chain = canonicalize_chain_scalar(chain_candidate, known_chains=set(repo_chain_names()))
        if current_chain and candidate_chain and current_chain == candidate_chain:
            state["pending_question"] = {}
            return state
        ambiguity = _chain_ambiguity_question(state, origin_text, chain_candidate)
        if ambiguity:
            state["pending_question"] = ambiguity
            state["active_group"] = "chain_identity"
            state["visible_response"] = [_render_question(ambiguity, state.get("language", "en"))]
            state["_queue_pause"] = True
            return state
        if action_type == "change_chain" or (state.get("chain_identity") or {}).get("canonical"):
            state = _request_chain_change_confirmation(state, chain_candidate, action)
            state["_queue_pause"] = True
            return state
        state = _handle_chain_candidate(state, chain_candidate, action)
        if not state.get("target_mode") and state.get("action_queue"):
            state["_queue_pause"] = True
        return state
    if action_type == "change_group":
        group = str(action.get("group") or "").strip()
        if group not in ALLOWED_GROUPS:
            return None
        if group == str(action.get("_queue_origin_group") or "").strip() and _active_group_has_blocking_question(state):
            return state
        if _queue_has_followup_for_group(state, group):
            _record_group_transition(state, group)
            state["pending_question"] = {}
            return state
        state = _activate_group_question(state, group)
        state["_queue_pause"] = True
        return state
    if action_type == "go_back":
        previous_group = _pop_previous_group(state)
        if previous_group:
            state = _activate_group_question(state, previous_group, record_history=False)
        else:
            state["visible_response"] = [_localized(
                state.get("language", "en"),
                "当前没有可回退的配置组。你可以直接说明要回到哪个配置项，例如 RPC、QPS、磁盘或可观测性。",
                "There is no previous configuration group to return to. You can directly name the area, such as RPC, QPS, disk, or observability.",
            )]
            state["_stop_after_response"] = True
        state["_queue_pause"] = True
        return state
    if action_type == "ask_capabilities":
        state["active_group"] = "report_artifact_analysis"
        state["pending_question"] = {}
        response = list(state.get("visible_response") or [])
        response.append(_framework_capability_summary(state))
        state["visible_response"] = response
        if not state.get("action_queue"):
            state["_stop_after_response"] = True
        return state
    if action_type == "answer_opening_question":
        if not state.get("pending_question"):
            state["active_group"] = "opening"
        response = list(state.get("visible_response") or [])
        response.append(_opening_consultation_response(state, action, origin_text))
        state["visible_response"] = response
        if not state.get("action_queue"):
            state["_stop_after_response"] = True
        return state
    if action_type == "analyze_evidence":
        if not _looks_like_pasted_evidence(origin_text):
            state["active_group"] = "error_evidence_analysis"
            state["pending_question"] = {}
            state["visible_response"] = [_evidence_help_response(state)]
            state["_stop_after_response"] = True
            return state
        state["active_group"] = "error_evidence_analysis"
        state.setdefault("evidence_buffer", []).append({"text": text})
        state["pending_question"] = {}
        state["visible_response"] = [_localized(
            state.get("language", "en"),
            "我已把这段内容作为证据保存。下一步会结合框架上下文分析原因和建议动作。",
            "I saved this as evidence. Next I can analyze it with the framework context and suggest actions.",
        )]
        state["_stop_after_response"] = True
        return state
    if action_type == "analyze_report":
        state["active_group"] = "report_artifact_analysis"
        state["pending_question"] = {}
        state["visible_response"] = [_report_artifact_entry_response(state)]
        state["_stop_after_response"] = True
        return state
    if action_type == "set_rpc_mode":
        rpc_mode = str(action.get("rpc_mode") or "").strip().lower()
        if rpc_mode not in {"single", "mixed"}:
            return None
        previous = str(state.get("rpc_mode") or "").strip()
        if previous != rpc_mode:
            state["workload"] = {}
            state["custom_rpc"] = {}
            state["fixture_evidence"] = {}
            state["preflight"] = {}
        state["smoke"] = {}
        state.setdefault("invalidated_groups", []).extend(["workload_rpc", "target_samples_fixtures", "preflight_smoke_execution"])
        state["rpc_mode"] = rpc_mode
        state["active_group"] = ""
        state["pending_question"] = {}
        if not state.get("workload", {}).get("confirmed"):
            state = _enter_workload_confirmation_gate(state)
            state["_queue_pause"] = True
        return state
    if action_type == "set_qps_mode":
        qps_mode = str(action.get("qps_mode") or "").strip().lower()
        if qps_mode not in {"quick", "standard", "intensive"}:
            return None
        if not _qps_mode_is_explicit_in_text(origin_text, qps_mode):
            state.setdefault("action_errors", []).append({"action": action, "error": "qps_mode_not_explicit"})
            return None
        qps = state.setdefault("qps_profile", {})
        if qps.get("mode") != qps_mode:
            qps.clear()
        qps["mode"] = qps_mode
        qps["confirmed"] = False
        qps["default_decision_made"] = False
        state["preflight"] = {}
        state["smoke"] = {}
        state.setdefault("invalidated_groups", []).extend(["qps_profile", "preflight_smoke_execution"])
        state["active_group"] = "qps_profile"
        state["pending_question"] = {}
        return state
    if action_type == "set_qps_override":
        overrides = action.get("qps_overrides")
        if not isinstance(overrides, dict) or not overrides:
            return None
        qps = state.setdefault("qps_profile", {})
        qps.setdefault("overrides", {}).update({str(k): _strip_scalar(str(v)) for k, v in overrides.items()})
        qps["default_decision_made"] = True
        qps["confirmed"] = True
        state["preflight"] = {}
        state["smoke"] = {}
        state.setdefault("invalidated_groups", []).extend(["qps_profile", "preflight_smoke_execution"])
        state["active_group"] = ""
        state["pending_question"] = {}
        return state
    if action_type == "set_observability":
        mode = str(action.get("observability_mode") or "").strip().lower()
        aliases = {"prometheus": "local", "grafana": "local", "local": "local", "disabled": "disabled", "disable": "disabled", "exporter": "exporter"}
        mode = aliases.get(mode, mode)
        if mode not in {"disabled", "local", "exporter"}:
            return None
        state.setdefault("observability", {})["mode"] = mode
        response = list(state.get("visible_response") or [])
        response.append(_localized(
            state.get("language", "en"),
            f"可观测性模式已设置为 `{mode}`。",
            f"Observability mode is set to `{mode}`.",
        ))
        state["visible_response"] = response
        state["preflight"] = {}
        state["smoke"] = {}
        state.setdefault("invalidated_groups", []).extend(["observability", "preflight_smoke_execution"])
        state["pending_question"] = {}
        return state
    if action_type == "set_sync_observe_source":
        source = str(action.get("sync_observe_source") or "").strip().lower()
        if source not in {"existing_local_node", "endpoint_only", "client_setup", "demo_only"}:
            return None
        if state.get("workflow_mode") != "sync_observe":
            state.setdefault("action_errors", []).append({"action": action, "error": "sync_source_without_sync_observe"})
            return None
        sync = state.setdefault("sync_observe", {})
        sync["source"] = source
        if source == "endpoint_only":
            sync["local_attribution_available"] = False
        if source == "demo_only":
            sync["demo_acknowledged"] = False
        if source == "client_setup":
            sync["client_setup_acknowledged"] = False
        state["preflight"] = {}
        state["smoke"] = {}
        state.setdefault("invalidated_groups", []).extend(["sync_observe", "preflight_smoke_execution"])
        state["active_group"] = ""
        state["pending_question"] = {}
        return state
    if action_type == "set_accounts_presence":
        if "has_accounts_device" not in action:
            return None
        has_accounts = bool(action.get("has_accounts_device"))
        confirmed = state.setdefault("confirmed_config", {})
        confirmed["has_accounts_device"] = has_accounts
        if not has_accounts:
            for key in (
                "ACCOUNTS_DEVICE",
                "ACCOUNTS_VOL_TYPE",
                "ACCOUNTS_VOL_SIZE",
                "ACCOUNTS_VOL_MAX_IOPS",
                "ACCOUNTS_VOL_MAX_THROUGHPUT",
            ):
                confirmed.pop(key, None)
        state.setdefault("invalidated_groups", []).extend(["accounts_disk", "preflight_smoke_execution"])
        state["pending_question"] = {}
        return state
    if action_type == "start_custom_rpc":
        custom = state.setdefault("custom_rpc", {})
        method = _strip_scalar(str(action.get("rpc_method") or ""))
        endpoint = _extract_url_candidate(str(action.get("rpc_endpoint") or ""))
        evidence = str(action.get("rpc_schema_evidence") or "").strip()
        custom["source_turn_text"] = text
        if not evidence:
            evidence = _schema_evidence_from_turn_text(text, method_hint=method)
        if method:
            custom["method"] = method
            hint = _custom_rpc_inline_workload_hint(text, [method])
            if hint:
                custom["inline_workload_hint"] = hint
        custom["status"] = "needs_endpoint"
        state.setdefault("workload", {})["choice"] = "custom_rpc"
        state["active_group"] = "endpoint_process"
        state["pending_question"] = {}
        state["preflight"] = {}
        state["smoke"] = {}
        state.setdefault("invalidated_groups", []).extend(["endpoint_process", "workload_rpc", "target_samples_fixtures", "preflight_smoke_execution"])
        if endpoint:
            state = _apply_endpoint_answer(
                state,
                endpoint,
                {
                    "group": "endpoint_process",
                    "id": "custom_rpc_endpoint",
                    "field": "custom_rpc_endpoint",
                    "kind": "url",
                    "manual_input_allowed": True,
                },
            )
            if custom.get("status") == "probe_failed":
                state["_queue_pause"] = True
                return state
        if evidence and custom.get("endpoint_ready"):
            state = _apply_endpoint_answer(
                state,
                evidence,
                {
                    "group": "endpoint_process",
                    "id": "custom_rpc_schema_evidence",
                    "field": "custom_rpc_schema_evidence",
                    "kind": "evidence",
                    "manual_input_allowed": True,
                },
            )
        state["_queue_pause"] = True
        return state
    if action_type == "propose_config_values":
        config_values = action.get("config_values")
        if not isinstance(config_values, dict) or not config_values:
            return None
        proposal = _build_config_proposal(action)
        if not proposal["config_values"] and not proposal["unmapped_values"]:
            return None
        state.setdefault("inferred_config", {})["pending_review"] = proposal
        state["pending_question"] = _option_question(
            "config_inference",
            "inferred_config_review",
            _format_config_proposal_prompt(state, proposal),
            [{"label": "Y", "value": True}, {"label": "N", "value": False}],
            field="inferred_config_review",
            kind="yes_no",
            manual_input_allowed=False,
        )
        state["visible_response"] = [_render_question(state["pending_question"], state.get("language", "en"))]
        state["_queue_pause"] = True
        return state
    if action_type == "answer_pending":
        state.setdefault("action_errors", []).append({"action": action, "error": "answer_pending_not_allowed_in_queue"})
        return None
    return None


def _queue_has_followup_for_group(state: AgentGraphState, group: str) -> bool:
    action_types = {str(item.get("type") or "").strip() for item in state.get("action_queue") or [] if isinstance(item, dict)}
    if group == "qps_profile":
        return bool(action_types & {"set_qps_mode", "set_qps_override"})
    if group == "observability":
        return "set_observability" in action_types
    if group == "workload_rpc":
        return "set_rpc_mode" in action_types
    if group == "sync_observe":
        return "set_sync_observe_source" in action_types
    return False


def _active_group_has_blocking_question(state: AgentGraphState) -> bool:
    active_group = str(state.get("active_group") or "").strip()
    if not active_group:
        return False
    return bool(_question_for_group(state, active_group))


def _ask_next_blocking_question(state: AgentGraphState) -> AgentGraphState:
    active_group = str(state.get("active_group") or "").strip()
    if active_group and active_group not in {"opening", "job_monitoring", "error_evidence_analysis", "report_artifact_analysis"}:
        active_question = _question_for_group(state, active_group)
        if active_question:
            state["pending_question"] = active_question
            if state.get("action_queue"):
                state["pending_question"]["resume_action_queue"] = True
            prefix = list(state.get("visible_response") or [])
            state["visible_response"] = prefix + [_render_question(active_question, state.get("language", "en"))]
            return state
    group = _next_group(state)
    _record_group_transition(state, group)
    question = _question_for_group(state, group)
    prefix = list(state.get("visible_response") or [])
    if question:
        state["pending_question"] = question
        if state.get("action_queue"):
            state["pending_question"]["resume_action_queue"] = True
        state["visible_response"] = prefix + [_render_question(question, state.get("language", "en"))]
    else:
        next_action = compute_next_action(state)
        state["pending_question"] = {}
        state["visible_response"] = prefix + [_localized(
            state.get("language", "en"),
            f"当前配置没有新的阻塞项。建议下一步：{format_recommended_next_action(next_action, state.get('language', 'zh'))}",
            f"No blocking configuration item remains. Recommended next action: {format_recommended_next_action(next_action, state.get('language', 'en'))}",
        )]
    return state


def _activate_group_question(state: AgentGraphState, group: str, *, record_history: bool = True) -> AgentGraphState:
    if group == "sync_observe":
        current_mode = str(state.get("target_mode") or "").strip()
        if current_mode and current_mode != "sync-observe":
            return _request_target_mode_change_confirmation(state, "sync-observe")
        state["target_mode"] = "sync-observe"
        state["workflow_mode"] = "sync_observe"
        _invalidate_for_target_mode(state)
    _record_group_transition(state, group, record_history=record_history)
    question = _question_for_group(state, group)
    if question:
        state["pending_question"] = question
        state["visible_response"] = [_render_question(question, state.get("language", "en"))]
    else:
        status = _completed_group_status(state, group)
        if status:
            state["pending_question"] = {}
            state["visible_response"] = [status]
            state["_stop_after_response"] = True
            return state
        state["pending_question"] = {}
        state["visible_response"] = [_localized(
            state.get("language", "en"),
            "这个配置组当前没有阻塞项。我会继续寻找下一项必须确认的配置。",
            "This configuration group has no blocking item right now. I will continue to the next required configuration item.",
        )]
        state = _ask_next_blocking_question(state)
        return state
    state["_stop_after_response"] = True
    return state


def _completed_group_status(state: AgentGraphState, group: str) -> str:
    language = state.get("language", "en")
    if group == "observability":
        mode = str((state.get("observability") or {}).get("mode") or "").strip()
        if mode:
            return _localized(
                language,
                f"当前可观测性模式已设置为 `{mode}`。如果要修改，可以直接说“切换到本地 Prometheus/Grafana”、“只启用 exporter”或“禁用可观测性”。",
                f"Observability is currently set to `{mode}`. To change it, say local Prometheus/Grafana, exporter only, or disable observability.",
            )
    if group == "qps_profile":
        qps = state.get("qps_profile") or {}
        if qps.get("mode") and qps.get("confirmed"):
            overrides = qps.get("overrides") or {}
            detail = ", ".join(f"{key}={value}" for key, value in sorted(overrides.items())) or "default profile"
            return _localized(
                language,
                f"当前 QPS profile：`{qps.get('mode')}`，配置：{detail}。如需调整，可以直接说要改 INITIAL_QPS、MAX_QPS、QPS_STEP 或 DURATION。",
                f"Current QPS profile: `{qps.get('mode')}`, settings: {detail}. To adjust it, name INITIAL_QPS, MAX_QPS, QPS_STEP, or DURATION.",
            )
    if group == "workload_rpc":
        workload = state.get("workload") or {}
        if state.get("rpc_mode") and workload.get("confirmed"):
            return _localized(
                language,
                f"当前 RPC workload 已确认：mode=`{state.get('rpc_mode')}`。如需修改，可以说“切换 single/mixed”、“添加自定义 RPC method”或“调整 mixed 权重”。",
                f"Current RPC workload is confirmed: mode=`{state.get('rpc_mode')}`. To change it, say switch single/mixed, add custom RPC method, or adjust mixed weights.",
            )
    return ""


def _record_group_transition(state: AgentGraphState, next_group: str, *, record_history: bool = True) -> None:
    current = str(state.get("active_group") or "").strip()
    if record_history and current and current != next_group:
        history = list(state.get("group_history") or [])
        if not history or history[-1] != current:
            history.append(current)
        state["group_history"] = history[-20:]
    state["active_group"] = next_group


def _pop_previous_group(state: AgentGraphState) -> str:
    history = list(state.get("group_history") or [])
    current = str(state.get("active_group") or "").strip()
    while history:
        candidate = str(history.pop() or "").strip()
        if candidate and candidate != current and candidate in ALLOWED_GROUPS:
            state["group_history"] = history
            return candidate
    state["group_history"] = history
    return ""


def _next_group(state: AgentGraphState) -> str:
    if not state.get("target_mode"):
        return "opening"
    identity = state.get("chain_identity") or {}
    if identity.get("status") == "existing_family_needs_endpoint":
        return "endpoint_process"
    if identity.get("status") in {
        "existing_family_needs_method",
        "existing_family_needs_schema_evidence",
        "existing_family_schema_needs_confirmation",
        "existing_family_needs_workload_scope",
        "existing_family_needs_single_method",
        "existing_family_needs_weights",
    }:
        return "endpoint_process"
    if identity.get("status") == "existing_family_runtime_choice":
        return "target_samples_fixtures"
    if not _chain_confirmed(state):
        return "chain_identity"
    confirmed = state.get("confirmed_config") or {}
    for key in ("CLOUD_REGION", "CLOUD_ZONE", "MACHINE_TYPE"):
        if not confirmed.get(key):
            return "provider_deployment"
    for key in ("LEDGER_DEVICE", "DATA_VOL_TYPE", "DATA_VOL_SIZE", "DATA_VOL_MAX_IOPS", "DATA_VOL_MAX_THROUGHPUT"):
        if not confirmed.get(key):
            return "ledger_disk"
    if "has_accounts_device" not in confirmed:
        return "accounts_disk"
    if confirmed.get("has_accounts_device"):
        for key in ("ACCOUNTS_DEVICE", "ACCOUNTS_VOL_TYPE", "ACCOUNTS_VOL_SIZE", "ACCOUNTS_VOL_MAX_IOPS", "ACCOUNTS_VOL_MAX_THROUGHPUT"):
            if not confirmed.get(key):
                return "accounts_disk"
    for key in ("NETWORK_INTERFACE", "NETWORK_MAX_BANDWIDTH_GBPS"):
        if not confirmed.get(key):
            return "network"
    if state.get("target_mode") == "real-node" and not state.get("endpoint_evidence", {}).get("local_rpc_url_ready"):
        return "endpoint_process"
    if state.get("target_mode") == "real-node" and not confirmed.get("BLOCKCHAIN_PROCESS_NAMES"):
        return "endpoint_process"
    if state.get("target_mode") == "real-node" and not confirmed.get("MAINNET_RPC_URL_REVIEWED"):
        return "endpoint_process"
    if state.get("workflow_mode") == "sync_observe" and not state.get("sync_observe", {}).get("source"):
        return "sync_observe"
    if state.get("workflow_mode") == "sync_observe" and state.get("sync_observe", {}).get("source") in {"existing_local_node", "endpoint_only"} and not state.get("endpoint_evidence", {}).get("sync_rpc_url_ready"):
        return "endpoint_process"
    if state.get("workflow_mode") == "sync_observe" and state.get("sync_observe", {}).get("source") == "existing_local_node" and not confirmed.get("BLOCKCHAIN_PROCESS_NAMES"):
        return "endpoint_process"
    if state.get("workflow_mode") == "sync_observe" and state.get("sync_observe", {}).get("source") == "client_setup" and not state.get("sync_observe", {}).get("client_setup_acknowledged"):
        return "sync_observe"
    if state.get("workflow_mode") == "sync_observe" and state.get("sync_observe", {}).get("source") in {"existing_local_node", "endpoint_only"} and not confirmed.get("MAINNET_RPC_URL_REVIEWED"):
        return "endpoint_process"
    custom_rpc = state.get("custom_rpc") or {}
    if custom_rpc.get("status") in {
        "needs_endpoint",
        "needs_method",
        "needs_schema_evidence",
        "schema_needs_confirmation",
        "needs_scope",
        "needs_weights",
        "probe_failed",
    }:
        return "endpoint_process"
    if state.get("workflow_mode") == "sync_observe":
        if not state.get("sync_observe", {}).get("stop_condition"):
            return "sync_observe"
        if state.get("sync_observe", {}).get("stop_condition") == "duration" and not state.get("sync_observe", {}).get("duration_seconds"):
            return "sync_observe"
        if not state.get("observability", {}).get("mode"):
            return "observability"
        if not state.get("preflight", {}).get("approved"):
            return "preflight_smoke_execution"
        return "job_monitoring"
    if not state.get("rpc_mode"):
        return "workload_rpc"
    if not state.get("workload", {}).get("confirmed"):
        return "workload_rpc"
    qps_profile = state.get("qps_profile") or {}
    if not qps_profile.get("mode"):
        return "qps_profile"
    if not qps_profile.get("confirmed"):
        return "qps_profile"
    if not state.get("observability", {}).get("mode"):
        return "observability"
    if not state.get("preflight", {}).get("approved"):
        return "preflight_smoke_execution"
    return "job_monitoring"


def _question_for_group(state: AgentGraphState, group: str) -> PendingQuestion | None:
    language = state.get("language", "en")
    confirmed = state.get("confirmed_config") or {}
    discovery = state.get("discovery") or {}
    if group == "opening":
        return _opening_question(state, language)
    if group == "chain_identity":
        identity = state.get("chain_identity") or {}
        if identity.get("status") == "needs_protocol_confirmation":
            return _protocol_family_question(state)
        return _chain_question(state)
    if group == "provider_deployment":
        for key, prompt_en, prompt_zh in (
            ("CLOUD_REGION", "Confirm CLOUD_REGION; use the detected value or enter a custom region.", "请输入 CLOUD_REGION（云区域），可以使用检测值或输入自定义值。"),
            ("CLOUD_ZONE", "Confirm CLOUD_ZONE; use the detected value or enter a custom zone.", "请输入 CLOUD_ZONE（可用区），可以使用检测值或输入自定义值。"),
            ("MACHINE_TYPE", "Confirm MACHINE_TYPE or instance type for report metadata.", "请输入 MACHINE_TYPE（机器或实例规格），用于报告元数据。"),
        ):
            if not confirmed.get(key):
                return _manual_question(group, key, _localized(language, prompt_zh, prompt_en), kind="manual_value")
    if group == "ledger_disk":
        return _disk_group_question(state, prefix="DATA", device_key="LEDGER_DEVICE", group="ledger_disk")
    if group == "accounts_disk":
        if "has_accounts_device" not in confirmed:
            return _option_question(
                group,
                "has_accounts_device",
                _localized(language, "这个节点是否有独立的 accounts/state 磁盘？", "Does this node have a separate accounts/state disk?"),
                [{"label": "Y", "value": True}, {"label": "N", "value": False}],
                field="has_accounts_device",
                kind="yes_no",
                manual_input_allowed=False,
            )
        if confirmed.get("has_accounts_device"):
            return _disk_group_question(state, prefix="ACCOUNTS", device_key="ACCOUNTS_DEVICE", group="accounts_disk")
    if group == "network":
        if not confirmed.get("NETWORK_INTERFACE"):
            interfaces = _usable_interfaces(
                list((discovery.get("network") or {}).get("interfaces") or []),
                str((discovery.get("network") or {}).get("default_interface") or ""),
            )
            default = (discovery.get("network") or {}).get("default_interface")
            return _choice_question(
                group,
                "network_interface",
                _localized(language, "请选择网络接口，或直接输入接口名。", "Choose the network interface, or type the interface name."),
                [{"label": f"{name}{' (default)' if name == default else ''}", "value": name} for name in interfaces],
                "NETWORK_INTERFACE",
                kind="device",
            )
        if not confirmed.get("NETWORK_MAX_BANDWIDTH_GBPS"):
            return _manual_question(group, "NETWORK_MAX_BANDWIDTH_GBPS", _localized(language, "请输入 NETWORK_MAX_BANDWIDTH_GBPS。", "Confirm NETWORK_MAX_BANDWIDTH_GBPS for saturation analysis."), kind="manual_value")
    if group == "endpoint_process":
        identity = state.get("chain_identity") or {}
        custom_rpc = state.get("custom_rpc") or {}
        sync = state.get("sync_observe") or {}
        active_validation_question = _endpoint_active_validation_question(state, group, language, custom_rpc, identity)
        if active_validation_question:
            return active_validation_question
        if state.get("target_mode") == "real-node" and not state.get("endpoint_evidence", {}).get("local_rpc_url_ready"):
            return _manual_question(
                group,
                "LOCAL_RPC_URL",
                _localized(
                    language,
                    "请提供被测真实节点的 LOCAL_RPC_URL。该 endpoint 会先被探测验证。",
                    "Provide the LOCAL_RPC_URL for the real node under test. The endpoint will be probed before it is trusted.",
                ),
                kind="url",
            )
        if state.get("target_mode") == "real-node" and not confirmed.get("BLOCKCHAIN_PROCESS_NAMES"):
            return _manual_question(
                group,
                "BLOCKCHAIN_PROCESS_NAMES",
                _localized(
                    language,
                    "请输入真实节点进程名或命令行片段，用于资源归因。",
                    "Enter the real node process name or command-line fragment for resource attribution.",
                ),
                kind="url",
            )
        if state.get("target_mode") == "real-node" and not confirmed.get("MAINNET_RPC_URL_REVIEWED"):
            return _option_question(
                group,
                "MAINNET_RPC_URL_REVIEWED",
                _localized(
                    language,
                    "是否使用当前链模板的 MAINNET_RPC_URL / sync-health 逻辑进行对比？如果你有自定义 mainnet endpoint，可以直接输入 URL。",
                    "Use the current chain template MAINNET_RPC_URL / sync-health behavior for comparison? If you have a custom mainnet endpoint, type the URL directly.",
                ),
                [{"label": "Y", "value": True}, {"label": "N", "value": False}],
                field="MAINNET_RPC_URL_REVIEWED",
                kind="confirm_or_value",
                manual_input_allowed=True,
            )
        if state.get("workflow_mode") == "sync_observe" and sync.get("source") in {"existing_local_node", "endpoint_only"} and not state.get("endpoint_evidence", {}).get("sync_rpc_url_ready"):
            return _manual_question(
                group,
                "SYNC_OBSERVE_RPC_URL",
                _localized(
                    language,
                    "请提供用于 sync-observe 的真实节点 RPC endpoint。fake-node 不提供真实同步/import metrics，因此这里必须验证真实 endpoint。",
                    "Provide a real node RPC endpoint for sync-observe. fake-node cannot provide real sync/import metrics, so this endpoint must be probed.",
                ),
                kind="url",
            )
        if state.get("workflow_mode") == "sync_observe" and sync.get("source") == "existing_local_node" and not confirmed.get("BLOCKCHAIN_PROCESS_NAMES"):
            return _manual_question(
                group,
                "BLOCKCHAIN_PROCESS_NAMES",
                _localized(
                    language,
                    "请输入节点进程名或命令行片段，用于 sync-observe 的 CPU/线程归因。",
                    "Enter the node process name or command-line fragment for sync-observe CPU/thread attribution.",
                ),
                kind="url",
            )
        if state.get("workflow_mode") == "sync_observe" and sync.get("source") in {"existing_local_node", "endpoint_only"} and not confirmed.get("MAINNET_RPC_URL_REVIEWED"):
            return _option_question(
                group,
                "MAINNET_RPC_URL_REVIEWED",
                _localized(
                    language,
                    "是否使用当前链模板的 sync-health / MAINNET_RPC_URL 逻辑进行高度对比？如果你有自定义 mainnet endpoint，可以直接输入 URL。",
                    "Use the current chain template sync-health / MAINNET_RPC_URL behavior for height comparison? If you have a custom mainnet endpoint, type the URL directly.",
                ),
                [{"label": "Y", "value": True}, {"label": "N", "value": False}],
                field="MAINNET_RPC_URL_REVIEWED",
                kind="confirm_or_value",
                manual_input_allowed=True,
            )
        if identity.get("status") == "existing_family_needs_workload_scope":
            zh = str(language or "").startswith("zh")
            return _option_question(
                group,
                "new_chain_workload_scope",
                _localized(language, "请选择新链已验证 RPC methods 如何作为本次 workload 使用。", "Choose how to use the validated new-chain RPC methods for this workload."),
                [
                    {
                        "label": "single 中使用一个已验证 method" if zh else "Use one validated method as single",
                        "value": "single_replace",
                    },
                    {
                        "label": "mixed 中只使用这些已验证 methods，并配置权重" if zh else "Use only these validated methods in mixed and configure weights",
                        "value": "mixed_replace",
                    },
                ],
                field="new_chain_workload_scope",
                kind="numbered_choice",
            )
        if identity.get("status") == "existing_family_needs_single_method":
            methods = _validated_new_chain_methods(identity)
            return _option_question(
                group,
                "new_chain_single_method",
                _localized(language, "请选择 single workload 使用哪个已验证 method。", "Choose which validated method to use as the single workload."),
                [{"label": method, "value": method} for method in methods],
                field="new_chain_single_method",
                kind="numbered_choice",
            )
        if identity.get("status") == "existing_family_needs_weights":
            methods = _validated_new_chain_methods(identity)
            example = _single_method_weight_example(methods)
            return _manual_question(
                group,
                "new_chain_custom_weights",
                _localized(
                    language,
                    f"请输入新链 mixed 权重，总和必须为 100。已验证 methods：{', '.join(methods) or '<none>'}。格式示例：`{example}`。",
                    f"Enter mixed weights for the new chain. The total must be 100. Validated methods: {', '.join(methods) or '<none>'}. Example: `{example}`.",
                ),
                kind="manual_value",
            )
    if group == "workload_rpc":
        if not state.get("rpc_mode"):
            return _option_question(
                group,
                "rpc_mode",
                _localized(language, "请选择 RPC 模式。", "Choose RPC mode."),
                [{"label": "single", "value": "single"}, {"label": "mixed", "value": "mixed"}],
                field="rpc_mode",
                kind="numbered_choice",
            )
        if not state.get("workload", {}).get("confirmed"):
            chain = state.get("chain_identity", {}).get("canonical") or "selected chain"
            zh = str(language or "").startswith("zh")
            return _option_question(
                group,
                "workload_confirm",
                _workload_default_prompt(state, language),
                [
                    {"label": "使用默认值" if zh else "Use defaults", "value": "default"},
                    {"label": "添加自定义 RPC method" if zh else "Add custom RPC method", "value": "custom_rpc"},
                    {"label": "调整 mixed 权重" if zh else "Adjust mixed weights", "value": "weights"},
                    {"label": "更换链或目标模式" if zh else "Change chain or target mode", "value": "change_target"},
                ],
                field="workload_choice",
                kind="numbered_choice",
            )
    if group == "target_samples_fixtures":
        identity = state.get("chain_identity") or {}
        if identity.get("status") == "existing_family_runtime_choice":
            zh = str(language or "").startswith("zh")
            return _option_question(
                group,
                "new_chain_runtime_choice",
                _localized(
                    language,
                    "新链 endpoint 和 RPC schema 已验证。当前 fake-node 还没有该链的 fixture，不能直接声称 fake-node 可运行。请选择下一步。",
                    "The new-chain endpoint and RPC schema are validated. fake-node has no fixture for this chain yet, so it cannot be claimed runnable as fake-node. Choose the next step.",
                ),
                [
                    {
                        "label": "使用已验证 endpoint 切到 real-node 路径继续" if zh else "Use the verified endpoint and continue as real-node",
                        "value": "use_verified_endpoint_real_node",
                    },
                    {
                        "label": "生成 chain template / fixture / smoke 二次开发交接" if zh else "Generate chain-template / fixture / smoke development handoff",
                        "value": "generate_handoff",
                    },
                ],
                field="new_chain_runtime_choice",
                kind="numbered_choice",
            )
    if group == "qps_profile":
        qps = state.setdefault("qps_profile", {})
        if not qps.get("mode"):
            return _option_question(
                group,
                "benchmark_mode",
                _localized(language, "请选择 benchmark 模式。", "Choose benchmark mode."),
                [{"label": "quick", "value": "quick"}, {"label": "standard", "value": "standard"}, {"label": "intensive", "value": "intensive"}],
                field="benchmark_mode",
                kind="numbered_choice",
            )
        if not qps.get("default_decision_made"):
            return _option_question(
                group,
                "qps_profile_confirm",
                _qps_default_prompt(str(qps.get("mode") or ""), language),
                [{"label": "Y", "value": True}, {"label": "N", "value": False}],
                field="qps_profile_confirmed",
                kind="yes_no",
                manual_input_allowed=False,
            )
        if not qps.get("confirmed"):
            if not qps.get("adjust_field"):
                zh = str(language or "").startswith("zh")
                return _option_question(
                    group,
                    "qps_adjust_field",
                    _localized(language, f"请选择要调整的 {qps.get('mode')} QPS 参数。", f"Choose the {qps.get('mode')} QPS parameter to adjust."),
                    [
                        {"label": "INITIAL_QPS / 起始 QPS" if zh else "INITIAL_QPS / initial QPS", "value": "INITIAL_QPS"},
                        {"label": "MAX_QPS / 最高 QPS" if zh else "MAX_QPS / max QPS", "value": "MAX_QPS"},
                        {"label": "QPS_STEP / 每级递增" if zh else "QPS_STEP / increment", "value": "QPS_STEP"},
                        {"label": "DURATION / 每档持续秒数" if zh else "DURATION / seconds per step", "value": "DURATION"},
                        {"label": "完成调整" if zh else "Finish QPS adjustments", "value": "done"},
                    ],
                    field="qps_adjust_field",
                    kind="numbered_choice",
                )
            return _manual_question(
                group,
                "qps_adjust_value",
                _localized(language, f"请输入 {qps.get('adjust_field')} 的值。", f"Enter the value for {qps.get('adjust_field')}."),
                kind="manual_value",
            )
    if group == "sync_observe":
        sync = state.setdefault("sync_observe", {})
        if not sync.get("source"):
            zh = str(language or "").startswith("zh")
            return _option_question(
                group,
                "sync_observe_source",
                _localized(
                    language,
                    "请选择 sync-observe 的真实数据来源。fake-node 只能验证流程，不能提供真实同步/import、CPU/磁盘或 MGas/s 指标。",
                    "Choose the real data source for sync-observe. fake-node can only validate plumbing; it cannot provide real sync/import, CPU/disk, or MGas/s metrics.",
                ),
                [
                    {"label": "已有本机真实节点进程" if zh else "Existing local real node process", "value": "existing_local_node"},
                    {"label": "只有真实 RPC/metrics endpoint" if zh else "Real RPC/metrics endpoint only", "value": "endpoint_only"},
                    {"label": "让 Agent 准备真实节点客户端" if zh else "Ask Agent to prepare a real node client", "value": "client_setup"},
                    {"label": "只做流程 demo，不声明真实性能" if zh else "Demo plumbing only; no real performance claim", "value": "demo_only"},
                ],
                field="sync_observe_source",
                kind="numbered_choice",
            )
        if sync.get("source") == "client_setup" and not sync.get("client_setup_acknowledged"):
            return _option_question(
                group,
                "sync_observe_client_setup_ack",
                _sync_client_setup_prompt(state),
                [{"label": "Y", "value": True}, {"label": "N", "value": False}],
                field="sync_observe_client_setup_ack",
                kind="yes_no",
                manual_input_allowed=False,
            )
        if sync.get("source") == "client_setup" and sync.get("client_setup_acknowledged"):
            return _option_question(
                group,
                "sync_observe_after_client_setup",
                _localized(
                    language,
                    "真实节点客户端准备完成后，请选择下一步数据来源。未提供真实 endpoint 或本机进程前，不会启动 sync-observe。",
                    "After real node client setup, choose the next data source. sync-observe will not start until a real endpoint or local process is provided.",
                ),
                [
                    {"label": "已有本机真实节点进程" if str(language or "").startswith("zh") else "Existing local real node process", "value": "existing_local_node"},
                    {"label": "只有真实 RPC/metrics endpoint" if str(language or "").startswith("zh") else "Real RPC/metrics endpoint only", "value": "endpoint_only"},
                ],
                field="sync_observe_after_client_setup",
                kind="numbered_choice",
            )
        if sync.get("source") == "demo_only" and not sync.get("demo_acknowledged"):
            return _option_question(
                group,
                "sync_observe_demo_ack",
                _localized(
                    language,
                    "确认只做 sync-observe 流程 demo：它不会使用 fake-node 声称真实同步性能，也不会产生真实 MGas/s/CPU/磁盘归因结论。是否继续？",
                    "Confirm sync-observe demo only: it will not use fake-node to claim real sync performance and will not produce real MGas/s/CPU/disk attribution conclusions. Continue?",
                ),
                [{"label": "Y", "value": True}, {"label": "N", "value": False}],
                field="sync_observe_demo_ack",
                kind="yes_no",
                manual_input_allowed=False,
            )
        if sync.get("source") == "demo_only" and sync.get("demo_acknowledged"):
            return _option_question(
                group,
                "sync_observe_after_demo",
                _localized(
                    language,
                    "流程 demo 已结束。要生成真实 sync-observe 报告，需要切换到真实数据来源。",
                    "The demo is complete. To generate a real sync-observe report, switch to a real data source.",
                ),
                [
                    {"label": "已有本机真实节点进程" if str(language or "").startswith("zh") else "Existing local real node process", "value": "existing_local_node"},
                    {"label": "只有真实 RPC/metrics endpoint" if str(language or "").startswith("zh") else "Real RPC/metrics endpoint only", "value": "endpoint_only"},
                    {"label": "让 Agent 准备真实节点客户端" if str(language or "").startswith("zh") else "Ask Agent to prepare a real node client", "value": "client_setup"},
                ],
                field="sync_observe_after_demo",
                kind="numbered_choice",
            )
        if sync.get("stop_condition") == "duration" and not sync.get("duration_seconds"):
            return _manual_question(
                group,
                "sync_observe_duration_seconds",
                _localized(language, "请输入 sync-observe 观察时长，单位秒。", "Enter the sync-observe duration in seconds."),
                kind="manual_value",
            )
        zh = str(language or "").startswith("zh")
        return _option_question(
            group,
            "sync_observe_stop_condition",
            _localized(language, "请选择 sync-observe 停止条件。", "Choose sync-observe stop condition."),
            [
                {"label": "一直运行直到用户停止" if zh else "Run until stopped", "value": "until_stopped"},
                {"label": "固定时长" if zh else "Fixed duration", "value": "duration"},
                {"label": "同步完成后停止" if zh else "Stop after synced", "value": "until_synced"},
            ],
            field="sync_observe_stop_condition",
            kind="numbered_choice",
        )
    if group == "observability":
        if state.get("observability", {}).get("mode"):
            return None
        zh = str(language or "").startswith("zh")
        return _option_question(
            group,
            "observability_mode",
            _localized(language, "请选择可观测性模式。", "Choose observability mode."),
            [
                {"label": "禁用" if zh else "Disabled", "value": "disabled"},
                {"label": "本地 Prometheus/Grafana" if zh else "Local Prometheus/Grafana", "value": "local"},
                {"label": "仅 exporter，对接已有 Prometheus" if zh else "Exporter only for existing Prometheus", "value": "exporter"},
            ],
            field="observability_mode",
            kind="numbered_choice",
        )
    if group == "preflight_smoke_execution":
        return _option_question(
            group,
            "preflight_smoke_confirm",
            _localized(language, "配置已收集。是否运行 preflight 和 smoke？", "Configuration is collected. Run preflight and smoke?"),
            [{"label": "Y", "value": True}, {"label": "N", "value": False}],
            field="preflight_smoke_confirmed",
            kind="yes_no",
            manual_input_allowed=False,
        )
    return None


def _apply_pending_answer(state: AgentGraphState, text: str, question: PendingQuestion) -> AgentGraphState:
    group = question.get("group", "")
    field = question.get("field", "")
    value = _coerce_answer(text, question)
    confirmed = state.setdefault("confirmed_config", {})
    question_id = str(question.get("id") or "")
    if group:
        state["active_group"] = str(group)
    if question_id == "inferred_config_review":
        return _apply_inferred_config_review(state, bool(value))
    if question.get("id") == "target_mode_change_confirm":
        requested_mode = str(state.get("target_mode_change_candidate") or "").strip()
        state["target_mode_change_candidate"] = ""
        if value is False:
            state["action_queue"] = []
            interrupted_group = str(question.get("interrupted_group") or "").strip()
            if interrupted_group:
                state["active_group"] = interrupted_group
            state["pending_question"] = {}
            return state
        if requested_mode:
            state["target_mode"] = requested_mode
            state["workflow_mode"] = "sync_observe" if requested_mode == "sync-observe" else "rpc_benchmark"
            _invalidate_for_target_mode(state)
            state["pending_question"] = {}
            state["active_group"] = "chain_identity" if not _chain_confirmed(state) else _next_group(state)
            state["visible_response"] = [_localized(
                state.get("language", "en"),
                f"已切换到 `{requested_mode}` 模式。",
                f"Switched to `{requested_mode}` mode.",
            )]
            return state
        state["pending_question"] = {}
        return state
    if question_id.startswith("custom_rpc_"):
        state["active_group"] = "endpoint_process"
        return _apply_endpoint_answer(state, value, question)
    if group == "opening":
        if value in {"fake-node", "real-node", "sync-observe"}:
            state["target_mode"] = value
            state["workflow_mode"] = "sync_observe" if value == "sync-observe" else "rpc_benchmark"
            state["pending_question"] = {}
            _invalidate_for_target_mode(state)
            return state
        state["active_group"] = "report_artifact_analysis"
        state["pending_question"] = {}
        return state
    if group == "chain_identity":
        if question.get("id") == "chain_ambiguity_confirm":
            if value == "reenter_chain":
                state["chain_identity"] = {}
                state["pending_question"] = {}
                return state
            if isinstance(value, dict) and value.get("chain_choice"):
                chain = canonicalize_chain_scalar(str(value.get("chain_choice") or ""), known_chains=set(repo_chain_names()))
                if chain:
                    previous = state.get("chain_identity", {}).get("canonical")
                    if previous and previous != chain:
                        _invalidate_for_chain_change(state)
                    state["chain_identity"] = {"raw": str(value.get("chain_choice") or ""), "canonical": chain, "status": "confirmed", "case": "known"}
                    state.setdefault("confirmed_config", {})["BLOCKCHAIN_NODE"] = chain
                    _drop_resolved_chain_actions(state)
                    state["active_group"] = ""
                    state["pending_question"] = {}
                    state["visible_response"] = [_localized(
                        state.get("language", "en"),
                        f"已确认链为 `{chain}`。",
                        f"Confirmed chain: `{chain}`.",
                    )]
                    return state
            if isinstance(value, dict) and value.get("unknown_chain_choice"):
                return _handle_chain_candidate(state, str(value.get("unknown_chain_choice") or ""))
        if question.get("id") == "chain_change_confirm":
            candidate = state.get("chain_identity", {}).get("change_candidate") or {}
            if value is False:
                state["action_queue"] = []
                state.setdefault("chain_identity", {}).pop("change_candidate", None)
                interrupted_group = str(candidate.get("interrupted_group") or "").strip()
                if interrupted_group:
                    state["active_group"] = interrupted_group
                state["pending_question"] = {}
                return state
            chain_text = str(candidate.get("raw") or "").strip()
            requested_mode = _normalized_target_mode(candidate.get("target_mode"))
            if requested_mode:
                state["target_mode"] = requested_mode
                state["workflow_mode"] = "sync_observe" if requested_mode == "sync-observe" else "rpc_benchmark"
                _invalidate_for_target_mode(state)
            _invalidate_for_chain_change(state)
            state.setdefault("chain_identity", {}).pop("change_candidate", None)
            if value == "confirm_known_chain":
                resolution = candidate.get("resolution") if isinstance(candidate.get("resolution"), dict) else {}
                possible_known = canonicalize_chain_scalar(
                    str(resolution.get("possible_known_chain") or ""),
                    known_chains=set(repo_chain_names()),
                )
                if possible_known:
                    state["chain_identity"] = {"raw": chain_text, "canonical": possible_known, "status": "confirmed", "case": "known"}
                    state.setdefault("confirmed_config", {})["BLOCKCHAIN_NODE"] = possible_known
                    state["active_group"] = ""
                    state["pending_question"] = {}
                    state["visible_response"] = [_localized(
                        state.get("language", "en"),
                        f"已确认链为 `{possible_known}`。",
                        f"Confirmed chain: `{possible_known}`.",
                    )]
                    return state
            if value == "confirm_proposed_protocol":
                resolution = candidate.get("resolution") if isinstance(candidate.get("resolution"), dict) else {}
                state["chain_identity"] = {
                    "raw": chain_text,
                    "canonical": str(resolution.get("canonical_chain_name") or chain_text).strip(),
                    "adapter_family": str(resolution.get("adapter_family") or "unknown").strip(),
                    "status": "needs_identity_confirmation",
                    "case": "unknown",
                    "identity_confirmed": True,
                    "llm_resolution": resolution,
                }
                return _enter_case_for_adapter_family(state, str(resolution.get("adapter_family") or "unknown").strip())
            if value == "choose_protocol":
                resolution = candidate.get("resolution") if isinstance(candidate.get("resolution"), dict) else {}
                state["chain_identity"] = {
                    "raw": chain_text,
                    "canonical": str(resolution.get("canonical_chain_name") or chain_text).strip(),
                    "adapter_family": str(resolution.get("adapter_family") or "unknown").strip(),
                    "status": "needs_protocol_confirmation",
                    "case": "unknown",
                    "identity_confirmed": True,
                    "llm_resolution": resolution,
                }
                state["pending_question"] = _protocol_family_question(state)
                state["visible_response"] = [_render_question(state["pending_question"], state.get("language", "en"))]
                return state
            return _handle_chain_candidate(state, chain_text, candidate.get("resolution") if isinstance(candidate.get("resolution"), dict) else None)
        if question.get("id") == "unknown_chain_identity_confirm":
            if isinstance(value, dict) and value.get("choose_protocol_family"):
                family = str(value.get("choose_protocol_family") or "").strip()
                _convert_known_candidate_to_unknown_chain(state)
                state.setdefault("chain_identity", {})["status"] = "needs_protocol_confirmation"
                state.setdefault("chain_identity", {})["identity_confirmed"] = True
                return _enter_case_for_adapter_family(state, family)
            if value == "reenter_chain":
                state["chain_identity"] = {}
                state["pending_question"] = {}
                return state
            if value == "confirm_known_chain":
                canonical = str(state.get("chain_identity", {}).get("canonical") or "").strip()
                if canonical:
                    state["chain_identity"] = {"raw": text, "canonical": canonical, "status": "confirmed", "case": "known"}
                    state.setdefault("confirmed_config", {})["BLOCKCHAIN_NODE"] = canonical
                    state["pending_question"] = {}
                    state["active_group"] = "provider_deployment"
                    return state
            if value == "choose_protocol":
                _convert_known_candidate_to_unknown_chain(state)
                state.setdefault("chain_identity", {})["status"] = "needs_protocol_confirmation"
                state.setdefault("chain_identity", {})["identity_confirmed"] = True
                state["pending_question"] = _protocol_family_question(state)
                state["visible_response"] = [_render_question(state["pending_question"], state.get("language", "en"))]
                return state
            if value == "confirm_proposed_protocol":
                identity = state.setdefault("chain_identity", {})
                family = str(identity.get("adapter_family") or "").strip()
                return _enter_case_for_adapter_family(state, family)
        if question.get("id") == "adapter_family_confirm":
            family = str(value or "").strip()
            return _enter_case_for_adapter_family(state, family)
        return _handle_chain_candidate(state, str(value or text).strip())
    if group == "provider_deployment" and field:
        confirmed[field] = _strip_scalar(str(value))
    elif group in {"ledger_disk", "accounts_disk"}:
        _apply_disk_answer(state, value, question)
    elif group == "network":
        if field:
            confirmed[field] = _strip_scalar(str(value))
    elif group == "endpoint_process":
        return _apply_endpoint_answer(state, value, question)
    elif group == "workload_rpc":
        if field == "rpc_mode":
            state["rpc_mode"] = str(value)
            if not state.get("workload", {}).get("confirmed"):
                return _enter_workload_confirmation_gate(state)
        elif field == "workload_choice":
            if value == "default":
                state.setdefault("workload", {})["confirmed"] = True
                state["workload"]["choice"] = "default"
            elif value == "change_target":
                state["target_mode"] = ""
                state["chain_identity"] = {}
            else:
                state.setdefault("custom_rpc", {})["status"] = "needs_endpoint"
                state.setdefault("workload", {})["choice"] = value
                state["active_group"] = "endpoint_process"
                state["visible_response"] = [_localized(
                    state.get("language", "en"),
                    "自定义 RPC/权重需要先验证 endpoint，再验证 method/params，最后校验权重；所有变更只生成 job-local workload override，不会修改 config/chains 原始模板。",
                    "Custom RPC/weights require endpoint validation, then method/params validation, then weight validation. Changes are job-local workload overrides only; config/chains templates are not modified.",
                )]
                state["pending_question"] = {}
                return state
    elif group == "target_samples_fixtures":
        if field == "new_chain_runtime_choice":
            if value == "use_verified_endpoint_real_node":
                return _promote_new_chain_verified_endpoint_to_real_node(state)
            if value == "generate_handoff":
                identity = state.setdefault("chain_identity", {})
                identity["status"] = "needs_review_handoff"
                state["pending_question"] = {}
                state["visible_response"] = [_new_chain_handoff_message(state)]
                state["_stop_after_response"] = True
                return state
    elif group == "qps_profile":
        qps = state.setdefault("qps_profile", {})
        if field == "benchmark_mode":
            qps["mode"] = str(value)
            qps["confirmed"] = False
            qps["default_decision_made"] = False
            qps.pop("adjust_field", None)
        elif field == "qps_profile_confirmed":
            qps["default_decision_made"] = True
            qps["confirmed"] = bool(value)
            if not value:
                qps["adjusting"] = True
        elif field == "qps_adjust_field":
            if value == "done":
                qps["confirmed"] = True
                qps.pop("adjust_field", None)
            else:
                qps["adjust_field"] = str(value)
        elif field == "qps_adjust_value":
            adjust_field = str(qps.get("adjust_field") or "")
            if adjust_field:
                qps.setdefault("overrides", {})[adjust_field] = _strip_scalar(str(value))
            qps.pop("adjust_field", None)
    elif group == "sync_observe":
        sync = state.setdefault("sync_observe", {})
        if field == "sync_observe_source":
            sync["source"] = str(value)
            if value == "endpoint_only":
                sync["local_attribution_available"] = False
            if value in {"endpoint_only", "existing_local_node"}:
                state["active_group"] = "endpoint_process"
        elif field == "sync_observe_client_setup_ack":
            sync["client_setup_acknowledged"] = bool(value)
            if not value:
                sync["source"] = ""
            else:
                state["pending_question"] = {}
                state["visible_response"] = [_sync_client_setup_handoff_message(state)]
                state["_stop_after_response"] = True
                return state
        elif field == "sync_observe_after_client_setup":
            sync["source"] = str(value)
            sync.pop("client_setup_acknowledged", None)
            if value == "endpoint_only":
                sync["local_attribution_available"] = False
            if value in {"endpoint_only", "existing_local_node"}:
                state["active_group"] = "endpoint_process"
        elif field == "sync_observe_demo_ack":
            sync["demo_acknowledged"] = bool(value)
            if value:
                state["pending_question"] = {}
                state["visible_response"] = [_localized(
                    state.get("language", "en"),
                    "已确认只做 sync-observe 流程 demo。当前 Harness 不会用 fake-node 伪造同步性能数据；如需生成真实报告，请提供真实节点 RPC/metrics endpoint 或本机节点进程。",
                    "Confirmed sync-observe demo only. The Harness will not fake sync performance with fake-node; provide a real node RPC/metrics endpoint or local node process to generate a real report.",
                )]
                state["_stop_after_response"] = True
                return state
            else:
                sync["source"] = ""
        elif field == "sync_observe_after_demo":
            sync["source"] = str(value)
            sync.pop("demo_acknowledged", None)
            if value == "endpoint_only":
                sync["local_attribution_available"] = False
            if value in {"endpoint_only", "existing_local_node"}:
                state["active_group"] = "endpoint_process"
        elif field == "sync_observe_stop_condition":
            sync["stop_condition"] = str(value)
        elif field == "sync_observe_duration_seconds":
            sync["duration_seconds"] = _strip_scalar(str(value))
    elif group == "observability":
        state.setdefault("observability", {})["mode"] = str(value)
    elif group == "preflight_smoke_execution":
        state.setdefault("preflight", {})["approved"] = bool(value)
        if value:
            state["pending_question"] = {}
            return run_approved_preflight_and_smoke(state)
        state["pending_question"] = {}
        state["visible_response"] = [_localized(
            state.get("language", "en"),
            "已暂停 preflight/smoke。你可以继续修改配置，例如链、RPC、QPS、磁盘或可观测性；准备好后再告诉我运行 preflight 和 smoke。",
            "Paused preflight/smoke. You can change configuration such as chain, RPC, QPS, disk, or observability; tell me to run preflight and smoke when ready.",
        )]
        state["_stop_after_response"] = True
        return state
    state["pending_question"] = {}
    return state


def _enter_workload_confirmation_gate(state: AgentGraphState) -> AgentGraphState:
    _record_group_transition(state, "workload_rpc")
    question = _question_for_group(state, "workload_rpc")
    if question:
        if state.get("action_queue"):
            question["resume_action_queue"] = True
        state["pending_question"] = question
        state["visible_response"] = [_render_question(question, state.get("language", "en"))]
    return state


def _defer_actions_until_pending_confirmed(state: AgentGraphState, text: str) -> None:
    queue = resolve_action_queue(state, text)
    actions = _normalized_action_queue(queue)
    if not actions:
        return
    existing = list(state.get("action_queue") or [])
    seen = {json.dumps(item, sort_keys=True, ensure_ascii=False) for item in existing if isinstance(item, dict)}
    for action in actions:
        action = dict(action)
        action.setdefault("_origin_text", text)
        marker = json.dumps(action, sort_keys=True, ensure_ascii=False)
        if marker not in seen:
            existing.append(action)
            seen.add(marker)
    state["action_queue"] = _order_action_queue(existing)
    if state.get("pending_question"):
        state["pending_question"]["resume_action_queue"] = True


def _build_config_proposal(action: dict[str, Any]) -> dict[str, Any]:
    raw_values = action.get("config_values") if isinstance(action.get("config_values"), dict) else {}
    config_values: dict[str, Any] = {}
    unmapped: dict[str, Any] = {}
    allowed = CONFIRMABLE_CONFIG_FIELDS | PROPOSED_ENDPOINT_FIELDS | SPECIAL_CONFIG_FIELDS
    for key, value in raw_values.items():
        normalized_key = str(key or "").strip().upper()
        mapped_key, mapped_value = _normalize_proposed_config_value(normalized_key, value)
        if mapped_key in allowed and mapped_value not in {"", None}:
            config_values[mapped_key] = mapped_value
        elif normalized_key and normalized_key not in allowed:
            unmapped[normalized_key] = value
    if "HAS_ACCOUNTS_DEVICE" not in config_values and _text_mentions_no_accounts_disk(action.get("_origin_text") or ""):
        config_values["HAS_ACCOUNTS_DEVICE"] = False
    raw_unmapped = action.get("unmapped_values")
    if isinstance(raw_unmapped, dict):
        for key, value in raw_unmapped.items():
            text_key = str(key or "").strip()
            if text_key:
                unmapped[text_key] = value
    elif isinstance(raw_unmapped, list):
        for index, value in enumerate(raw_unmapped, start=1):
            unmapped[f"unmapped_{index}"] = value
    return {
        "config_values": config_values,
        "unmapped_values": unmapped,
        "source_format": _strip_scalar(str(action.get("source_format") or "mixed")),
        "reason": _strip_scalar(str(action.get("reason") or "")),
    }


def _extract_structured_config_proposal(text: str) -> dict[str, Any] | None:
    """Extract explicit config facts from pasted JSON/YAML/env snippets.

    This is not an intent router. It only maps known AnyChain configuration
    fields from structured technical paste, then the normal review question
    still requires user confirmation before anything is applied.
    """

    raw = str(text or "")
    if "\n" not in raw and "{" not in raw and ":" not in raw:
        return None
    config_values: dict[str, Any] = {}
    unmapped: dict[str, Any] = {}
    config_values.update(_parse_known_config_assignments(raw))

    json_values, json_unmapped = _extract_json_config_values(raw)
    config_values.update(json_values)
    unmapped.update(json_unmapped)

    yaml_values, yaml_unmapped = _extract_yaml_like_config_values(raw)
    config_values.update(yaml_values)
    unmapped.update(yaml_unmapped)

    if not config_values and not unmapped:
        return None
    return {
        "type": "propose_config_values",
        "source_format": "mixed",
        "config_values": config_values,
        "unmapped_values": unmapped,
        "confidence": "high",
        "reason": "structured config paste",
    }


def _merge_pending_config_proposal_from_text(state: AgentGraphState, text: str) -> AgentGraphState | None:
    proposal_action = _extract_structured_config_proposal(text)
    if proposal_action is None:
        direct = _parse_known_config_assignments(text)
        if direct:
            proposal_action = {
                "type": "propose_config_values",
                "source_format": "mixed",
                "config_values": direct,
                "unmapped_values": {},
                "confidence": "high",
                "reason": "additional config line",
            }
    if proposal_action is None:
        return None
    new_proposal = _build_config_proposal(proposal_action)
    if not new_proposal["config_values"] and not new_proposal["unmapped_values"]:
        return None
    inferred = state.setdefault("inferred_config", {})
    current = inferred.get("pending_review") if isinstance(inferred.get("pending_review"), dict) else {}
    current_values = dict(current.get("config_values") or {})
    current_unmapped = dict(current.get("unmapped_values") or {})
    current_values.update(new_proposal.get("config_values") or {})
    current_unmapped.update(new_proposal.get("unmapped_values") or {})
    merged = {
        "config_values": current_values,
        "unmapped_values": current_unmapped,
        "source_format": "mixed",
        "reason": _localized(state.get("language", "en"), "持续追加的配置片段", "additional config fragments"),
    }
    inferred["pending_review"] = merged
    state["pending_question"] = _option_question(
        "provider_deployment",
        "inferred_config_review",
        _format_config_proposal_prompt(state, merged),
        [{"label": "Y", "value": True}, {"label": "N", "value": False}],
        field="inferred_config_review",
        kind="yes_no",
    )
    state["visible_response"] = [_render_question(state["pending_question"], state.get("language", "en"))]
    return state


def _extract_json_config_values(text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}, {}
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}, {}
    flattened = _flatten_mapping(parsed)
    return _map_flat_config_values(flattened)


def _extract_yaml_like_config_values(text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    flattened: dict[str, Any] = {}
    stack: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith(("#", "- ")):
            continue
        if ":" not in raw_line:
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, value = raw_line.strip().split(":", 1)
        key = key.strip().strip("\"'")
        value = value.strip().strip(",").strip()
        if not key:
            continue
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not value:
            stack.append((indent, key))
            continue
        path = ".".join([item[1] for item in stack] + [key])
        flattened[path] = value.strip("\"'")
    return _map_flat_config_values(flattened)


def _flatten_mapping(value: Any, prefix: str = "") -> dict[str, Any]:
    output: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            next_key = f"{prefix}.{key}" if prefix else str(key)
            output.update(_flatten_mapping(item, next_key))
    else:
        output[prefix] = value
    return output


def _map_flat_config_values(flattened: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    aliases = {
        "cloud.region": "CLOUD_REGION",
        "region": "CLOUD_REGION",
        "cloud_region": "CLOUD_REGION",
        "cloud.zone": "CLOUD_ZONE",
        "zone": "CLOUD_ZONE",
        "cloud_zone": "CLOUD_ZONE",
        "cloud.machine_type": "MACHINE_TYPE",
        "machine": "MACHINE_TYPE",
        "machine_type": "MACHINE_TYPE",
        "instance_type": "MACHINE_TYPE",
        "disk.ledger": "LEDGER_DEVICE",
        "disk.ledger_device": "LEDGER_DEVICE",
        "storage.ledger": "LEDGER_DEVICE",
        "storage.ledger_device": "LEDGER_DEVICE",
        "ledger_device": "LEDGER_DEVICE",
        "disk.data_vol_type": "DATA_VOL_TYPE",
        "storage.data_vol_type": "DATA_VOL_TYPE",
        "data_vol_type": "DATA_VOL_TYPE",
        "disk.data_vol_size": "DATA_VOL_SIZE",
        "storage.data_vol_size": "DATA_VOL_SIZE",
        "data_vol_size": "DATA_VOL_SIZE",
        "disk.size_gib": "DATA_VOL_SIZE",
        "storage.size_gib": "DATA_VOL_SIZE",
        "disk.data_vol_max_iops": "DATA_VOL_MAX_IOPS",
        "storage.data_vol_max_iops": "DATA_VOL_MAX_IOPS",
        "data_vol_max_iops": "DATA_VOL_MAX_IOPS",
        "disk.iops": "DATA_VOL_MAX_IOPS",
        "storage.iops": "DATA_VOL_MAX_IOPS",
        "iops": "DATA_VOL_MAX_IOPS",
        "disk.data_vol_max_throughput": "DATA_VOL_MAX_THROUGHPUT",
        "storage.data_vol_max_throughput": "DATA_VOL_MAX_THROUGHPUT",
        "data_vol_max_throughput": "DATA_VOL_MAX_THROUGHPUT",
        "disk.throughput": "DATA_VOL_MAX_THROUGHPUT",
        "storage.throughput": "DATA_VOL_MAX_THROUGHPUT",
        "throughput": "DATA_VOL_MAX_THROUGHPUT",
        "network.interface": "NETWORK_INTERFACE",
        "network_interface": "NETWORK_INTERFACE",
        "interface": "NETWORK_INTERFACE",
        "network.bandwidth_gbps": "NETWORK_MAX_BANDWIDTH_GBPS",
        "network_max_bandwidth_gbps": "NETWORK_MAX_BANDWIDTH_GBPS",
        "network.bandwidth": "NETWORK_MAX_BANDWIDTH_GBPS",
        "bandwidth": "NETWORK_MAX_BANDWIDTH_GBPS",
        "accounts.has_accounts_device": "HAS_ACCOUNTS_DEVICE",
        "has_accounts_device": "HAS_ACCOUNTS_DEVICE",
        "local_rpc_url": "LOCAL_RPC_URL",
        "mainnet_rpc_url": "MAINNET_RPC_URL",
        "blockchain_process_names": "BLOCKCHAIN_PROCESS_NAMES",
        "rpc_mode": "RPC_MODE",
    }
    config_values: dict[str, Any] = {}
    unmapped: dict[str, Any] = {}
    allowed = CONFIRMABLE_CONFIG_FIELDS | PROPOSED_ENDPOINT_FIELDS | SPECIAL_CONFIG_FIELDS
    for raw_key, value in flattened.items():
        key = str(raw_key or "").strip()
        normalized = key.upper()
        alias = aliases.get(key.lower().replace("-", "_"))
        mapped_key = normalized if normalized in allowed else alias
        scalar = _strip_scalar(str(value)).strip().strip("\"'")
        if not scalar:
            continue
        if mapped_key:
            final_key, final_value = _normalize_proposed_config_value(mapped_key, scalar)
            if final_key and final_value not in {"", None}:
                config_values[final_key] = final_value
        elif "." in key or "_" in key:
            unmapped[key] = scalar
    return config_values, unmapped


def _parse_known_config_assignments(text: str) -> dict[str, Any]:
    """Parse explicit AnyChain config assignments from terminal input.

    This handles real terminal paste behavior where each ``KEY=value`` line may
    arrive as a separate turn. It intentionally accepts only known AnyChain
    fields so generic method-weight input such as ``eth_blockNumber=100`` stays
    with the active RPC workflow.
    """

    values: dict[str, Any] = {}
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        for key, raw_value in _iter_config_assignments(line):
            if key not in CONFIRMABLE_CONFIG_FIELDS and key not in PROPOSED_ENDPOINT_FIELDS and key not in SPECIAL_CONFIG_FIELDS:
                continue
            final_key, value = _normalize_proposed_config_value(key, raw_value)
            if final_key and value not in {"", None}:
                values[final_key] = value
    return values


def _iter_config_assignments(line: str) -> list[tuple[str, str]]:
    """Return KEY=value pairs from one shell/env-style line.

    Users often paste comma-separated facts into one terminal turn. Values are
    read until the next `, KEY=` boundary instead of treating the whole suffix
    as one value.
    """

    pattern = re.compile(
        r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
        r"(?P<value>.*?)(?=\s*[,，;；]\s*[A-Za-z_][A-Za-z0-9_]*\s*=|$)"
    )
    pairs: list[tuple[str, str]] = []
    for match in pattern.finditer(line):
        key = match.group("key").strip().upper()
        value = _strip_scalar(match.group("value"))
        if key and value:
            pairs.append((key, value))
    return pairs


def _apply_direct_config_assignments(state: AgentGraphState, config_values: dict[str, Any]) -> AgentGraphState:
    confirmed = state.setdefault("confirmed_config", {})
    endpoint_proposals = state.setdefault("endpoint_evidence", {}).setdefault("proposed_values", {})
    applied: dict[str, str] = {}
    endpoint_saved: dict[str, str] = {}
    for key, value in config_values.items():
        key = str(key or "").strip().upper()
        key, scalar = _normalize_proposed_config_value(key, value)
        if scalar in {"", None}:
            continue
        if key in CONFIRMABLE_CONFIG_FIELDS:
            confirmed[key] = scalar
            applied[key] = scalar
        elif key in PROPOSED_ENDPOINT_FIELDS:
            endpoint_proposals[key] = scalar
            endpoint_saved[key] = scalar
        elif key == "RPC_MODE":
            mode = scalar.lower()
            if mode in {"single", "mixed"}:
                state["rpc_mode"] = mode
                state["workload"] = {}
                state["custom_rpc"] = {}
                state["fixture_evidence"] = {}
                state.setdefault("invalidated_groups", []).extend(["workload_rpc", "target_samples_fixtures", "preflight_smoke_execution"])
                applied[key] = mode
        elif key == "HAS_ACCOUNTS_DEVICE":
            has_accounts = bool(scalar)
            confirmed["has_accounts_device"] = has_accounts
            if not has_accounts:
                for accounts_key in ("ACCOUNTS_DEVICE", "ACCOUNTS_VOL_TYPE", "ACCOUNTS_VOL_SIZE", "ACCOUNTS_VOL_MAX_IOPS", "ACCOUNTS_VOL_MAX_THROUGHPUT"):
                    confirmed.pop(accounts_key, None)
            applied["has_accounts_device"] = has_accounts
    messages = []
    if applied:
        messages.append(_localized(
            state.get("language", "en"),
            "已记录你直接提供的配置：" + ", ".join(f"{key}={value}" for key, value in sorted(applied.items())),
            "Recorded the config values you provided directly: " + ", ".join(f"{key}={value}" for key, value in sorted(applied.items())),
        ))
    if endpoint_saved:
        messages.append(_localized(
            state.get("language", "en"),
            "已保存 endpoint 候选值，后续仍会验证：" + ", ".join(f"{key}={value}" for key, value in sorted(endpoint_saved.items())),
            "Saved endpoint candidates for later validation: " + ", ".join(f"{key}={value}" for key, value in sorted(endpoint_saved.items())),
        ))
    state["visible_response"] = messages
    state["pending_question"] = {}
    state["active_group"] = ""
    return state


def _should_start_evidence_collection(text: str) -> bool:
    raw = str(text or "").strip()
    if "\n" in raw or "\r" in raw:
        return False
    lowered = raw.lower()
    if _parse_rpc_params_or_request(raw)[1] is not None:
        return False
    return (
        raw.endswith("\\")
        or (lowered.startswith("curl ") and "--data" not in lowered)
        or lowered.startswith(("--header ", "--data ", "-h ", "-d "))
        or lowered in {"request:", "response:"}
        or lowered.startswith(("request:", "response:"))
    )


def _start_evidence_collection(state: AgentGraphState, text: str, pending: PendingQuestion) -> AgentGraphState:
    lines = _non_empty_evidence_lines(text)
    state["evidence_collection"] = {
        "question": dict(pending),
        "lines": lines,
        "language": state.get("language", "en"),
    }
    state["pending_question"] = {}
    if _evidence_collection_complete(lines):
        return _finish_evidence_collection(state)
    state["visible_response"] = [_localized(
        state.get("language", "en"),
        "已开始接收多行 RPC 证据。请继续粘贴 request/response/docs；完成后输入 `END`。我会在完整证据后再解析，不会用半截内容改变协议。",
        "Started collecting multi-line RPC evidence. Continue pasting request/response/docs; type `END` when done. I will parse only complete evidence and will not change protocol from a partial line.",
    )]
    state["_stop_after_response"] = True
    return state


def _start_freeform_evidence_collection(state: AgentGraphState, text: str) -> AgentGraphState:
    lines = _non_empty_evidence_lines(text)
    state["evidence_collection"] = {
        "question": {"id": "freeform_evidence", "kind": "log_evidence"},
        "lines": lines,
        "language": state.get("language", "en"),
    }
    state["pending_question"] = {}
    state["visible_response"] = [_localized(
        state.get("language", "en"),
        f"我已开始接收多行错误/日志证据，当前收到 {len(lines)} 行。请继续粘贴；完成后输入 `END`，或直接问“这是什么意思”。",
        f"Started collecting multi-line error/log evidence; received {len(lines)} lines so far. Continue pasting; type `END`, or ask what it means when done.",
    )]
    state["_stop_after_response"] = True
    return state


def _continue_evidence_collection(state: AgentGraphState, text: str, collecting: dict[str, Any]) -> AgentGraphState:
    lines = [str(item) for item in list(collecting.get("lines") or [])]
    raw = str(text or "").rstrip("\n")
    question = collecting.get("question") if isinstance(collecting.get("question"), dict) else {}
    collection_language = str(collecting.get("language") or state.get("language") or "en")
    state["language"] = collection_language
    if str(question.get("id") or "") == "freeform_evidence" and _looks_like_evidence_analysis_request(raw):
        state["evidence_collection"] = {"question": question, "lines": lines, "language": collection_language}
        state = _finish_evidence_collection(state)
        state["visible_response"] = list(state.get("visible_response") or []) + [_localized(
            state.get("language", "en"),
            "这类内容会作为错误/日志证据分析；如果你希望继续配置 benchmark，也可以直接说明要回到哪个配置项。",
            "I will analyze this as error/log evidence. If you want to continue benchmark configuration instead, name the configuration area.",
        )]
        state["_stop_after_response"] = True
        return state
    if raw.strip().upper() in {"END", "DONE", "结束"}:
        state["evidence_collection"] = {"question": collecting.get("question") or {}, "lines": lines, "language": collection_language}
        return _finish_evidence_collection(state)
    lines.append(raw)
    state["evidence_collection"] = {"question": collecting.get("question") or {}, "lines": lines, "language": collection_language}
    if _evidence_collection_complete(lines):
        return _finish_evidence_collection(state)
    state["visible_response"] = [_localized(
        collection_language,
        f"已记录第 {len(lines)} 行证据。请继续粘贴，完成后输入 `END`。",
        f"Recorded evidence line {len(lines)}. Continue pasting, or type `END` when done.",
    )]
    state["_stop_after_response"] = True
    return state


def _prompt_evidence_collection_waiting(state: AgentGraphState, collecting: dict[str, Any]) -> AgentGraphState:
    question = collecting.get("question") if isinstance(collecting.get("question"), dict) else {}
    lines = [str(item) for item in list(collecting.get("lines") or []) if str(item).strip()]
    collection_language = str(collecting.get("language") or state.get("language") or "en")
    state["language"] = collection_language
    state["evidence_collection"] = {"question": question, "lines": lines, "language": collection_language}
    if str(question.get("id") or "") == "freeform_evidence":
        state["visible_response"] = [_localized(
            collection_language,
            "还没有收到新的日志内容。请继续粘贴真实日志/错误栈，完成后输入 `END`；如果要退出日志分析，可以直接说明要回到哪个配置项。",
            "No new log content was received. Paste the actual log/stack trace and type `END` when done; to leave log analysis, name the configuration area to return to.",
        )]
    else:
        state["visible_response"] = [_localized(
            collection_language,
            "还没有收到新的 RPC 证据。请继续粘贴 request/response/docs，完成后输入 `END`；如果要退出，请直接说明要回到哪个配置项。",
            "No new RPC evidence was received. Continue pasting request/response/docs and type `END` when done; to leave this flow, name the configuration area to return to.",
        )]
    state["_stop_after_response"] = True
    return state


def _finish_evidence_collection(state: AgentGraphState) -> AgentGraphState:
    collecting = state.get("evidence_collection") or {}
    question = collecting.get("question") if isinstance(collecting.get("question"), dict) else {}
    lines = [str(item) for item in list(collecting.get("lines") or []) if str(item).strip()]
    state["evidence_collection"] = {}
    if not question or not lines:
        state["visible_response"] = [_localized(
            state.get("language", "en"),
            "没有可解析的多行证据。请重新提供 request/response/docs，或直接输入 params JSON。",
            "No parseable multi-line evidence was collected. Provide request/response/docs again, or enter params JSON directly.",
        )]
        state["_stop_after_response"] = True
        return state
    if str(question.get("id") or "") == "freeform_evidence":
        text = "\n".join(lines)
        state.setdefault("evidence_buffer", []).append({"text": text})
        state["pending_question"] = {}
        state["visible_response"] = [_localized(
            state.get("language", "en"),
            f"已保存 {len(lines)} 行错误/日志证据。你可以继续问我分析原因，或让我基于这些证据生成修复/重试建议。",
            f"Saved {len(lines)} lines of error/log evidence. You can ask me to analyze the cause or generate a fix/retry plan from it.",
        )]
        state["_stop_after_response"] = True
        return state
    return _apply_pending_answer(state, "\n".join(lines), question)  # type: ignore[arg-type]


def _non_empty_evidence_lines(text: str) -> list[str]:
    lines = [line.rstrip("\n") for line in str(text or "").splitlines() if line.strip()]
    return lines or [str(text or "").strip()]


def _evidence_collection_complete(lines: list[str]) -> bool:
    text = "\n".join(lines)
    lowered = text.lower()
    if "response" in lowered and ("--data" in lowered or "\"method\"" in lowered or "'method'" in lowered):
        return True
    if "jsonrpc" in lowered and "\"result\"" in lowered and ("\"method\"" in lowered or "'method'" in lowered):
        return True
    return False


def _looks_like_evidence_analysis_request(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return False
    return any(token in lowered for token in ("什么意思", "什么原因", "分析", "怎么修", "修复", "下一步", "why", "what does", "what means", "explain", "fix", "next step"))


def _format_config_proposal_prompt(state: AgentGraphState, proposal: dict[str, Any]) -> str:
    language = state.get("language", "en")
    config_values = proposal.get("config_values") if isinstance(proposal.get("config_values"), dict) else {}
    unmapped = proposal.get("unmapped_values") if isinstance(proposal.get("unmapped_values"), dict) else {}
    lines = []
    if config_values:
        lines.append(_localized(language, "我从你粘贴的内容中推断出这些配置候选值：", "I inferred these candidate config values from your pasted content:"))
        for key in sorted(config_values):
            lines.append(f"- {key}: `{config_values[key]}`")
    if unmapped:
        lines.append(_localized(language, "以下内容没有自动映射到已知配置项，不会自动写入：", "These extracted values did not map to known config fields and will not be applied automatically:"))
        for key in sorted(unmapped):
            lines.append(f"- {key}: `{unmapped[key]}`")
    if proposal.get("reason"):
        lines.append(_localized(language, f"推断依据：{proposal.get('reason')}", f"Reason: {proposal.get('reason')}"))
    lines.append(_localized(language, "是否确认应用这些已映射的候选值？", "Apply the mapped candidate values?"))
    lines.append(_localized(language, "endpoint 类值只会保存为待验证候选，后续仍必须通过 endpoint probe。", "Endpoint values are saved only as candidates; they must still pass endpoint probing later."))
    return "\n".join(lines)


def _apply_inferred_config_review(state: AgentGraphState, accepted: bool) -> AgentGraphState:
    inferred = state.setdefault("inferred_config", {})
    proposal = inferred.pop("pending_review", {}) if isinstance(inferred.get("pending_review"), dict) else {}
    _drop_resolved_config_proposal_actions(state)
    if not accepted:
        state["pending_question"] = {}
        state["visible_response"] = [_localized(
            state.get("language", "en"),
            "已丢弃这次推断的候选配置。你可以重新粘贴更完整的信息，或直接告诉我要修改哪个配置组。",
            "Discarded these inferred candidate values. Paste corrected information, or tell me which configuration group to change.",
        )]
        state["_stop_after_response"] = True
        return state

    config_values = proposal.get("config_values") if isinstance(proposal.get("config_values"), dict) else {}
    confirmed = state.setdefault("confirmed_config", {})
    endpoint_proposals = state.setdefault("endpoint_evidence", {}).setdefault("proposed_values", {})
    applied: dict[str, Any] = {}
    endpoint_saved: dict[str, Any] = {}
    for key, value in config_values.items():
        key = str(key or "").strip().upper()
        key, scalar = _normalize_proposed_config_value(key, value)
        if scalar in {"", None}:
            continue
        if key in CONFIRMABLE_CONFIG_FIELDS:
            confirmed[key] = scalar
            applied[key] = scalar
        elif key in PROPOSED_ENDPOINT_FIELDS:
            endpoint_proposals[key] = scalar
            endpoint_saved[key] = scalar
        elif key == "RPC_MODE":
            mode = scalar.lower()
            if mode in {"single", "mixed"}:
                state["rpc_mode"] = mode
                state["workload"] = {}
                state["custom_rpc"] = {}
                state["fixture_evidence"] = {}
                state.setdefault("invalidated_groups", []).extend(["workload_rpc", "target_samples_fixtures", "preflight_smoke_execution"])
                applied[key] = mode
        elif key == "HAS_ACCOUNTS_DEVICE":
            has_accounts = bool(scalar)
            confirmed["has_accounts_device"] = has_accounts
            if not has_accounts:
                for accounts_key in ("ACCOUNTS_DEVICE", "ACCOUNTS_VOL_TYPE", "ACCOUNTS_VOL_SIZE", "ACCOUNTS_VOL_MAX_IOPS", "ACCOUNTS_VOL_MAX_THROUGHPUT"):
                    confirmed.pop(accounts_key, None)
            applied["has_accounts_device"] = has_accounts
    inferred.setdefault("accepted_reviews", []).append({
        "applied": applied,
        "endpoint_proposals": endpoint_saved,
        "unmapped_values": proposal.get("unmapped_values") or {},
        "source_format": proposal.get("source_format") or "",
    })
    state["pending_question"] = {}
    messages = []
    if applied:
        messages.append(_localized(
            state.get("language", "en"),
            "已应用已确认的配置候选值：" + ", ".join(f"{key}={value}" for key, value in sorted(applied.items())),
            "Applied confirmed candidate values: " + ", ".join(f"{key}={value}" for key, value in sorted(applied.items())),
        ))
    if endpoint_saved:
        messages.append(_localized(
            state.get("language", "en"),
            "已保存 endpoint 候选值，后续会要求验证：" + ", ".join(f"{key}={value}" for key, value in sorted(endpoint_saved.items())),
            "Saved endpoint candidate values for later validation: " + ", ".join(f"{key}={value}" for key, value in sorted(endpoint_saved.items())),
        ))
    if proposal.get("unmapped_values"):
        messages.append(_localized(
            state.get("language", "en"),
            "未映射字段已作为证据保留，不会自动写入配置。",
            "Unmapped values were kept as evidence and were not applied automatically.",
        ))
    state["visible_response"] = messages
    state["active_group"] = ""
    return state


def _drop_resolved_config_proposal_actions(state: AgentGraphState) -> None:
    """Remove stale config proposals after the user resolves the review gate.

    The remaining queue may still contain real follow-up intents from the same
    user turn, such as "QPS 用 quick" or "RPC 模式 single". Those should resume
    after confirmation, but older propose_config_values entries must not regain
    control and ask the user to review a partial snapshot again.
    """

    state["action_queue"] = [
        item
        for item in state.get("action_queue") or []
        if not isinstance(item, dict) or str(item.get("type") or "") != "propose_config_values"
    ]


def _endpoint_active_validation_question(
    state: AgentGraphState,
    group: str,
    language: str,
    custom_rpc: dict[str, Any],
    identity: dict[str, Any],
) -> PendingQuestion | None:
    """Return the active evidence-only endpoint validation question.

    Custom RPC and new-chain validation endpoints are not final benchmark
    endpoints. They must take precedence over LOCAL_RPC_URL/SYNC_OBSERVE_RPC_URL
    so examples pasted for schema validation cannot be accidentally bound as
    the node-under-test endpoint.
    """

    if custom_rpc.get("status") in {"needs_endpoint", "probe_failed"} and not custom_rpc.get("endpoint_ready"):
        return _manual_question(
            group,
            "custom_rpc_endpoint",
            _localized(
                language,
                "请提供可访问的 RPC endpoint，用于验证自定义 RPC method。不会修改 config/chains 原始模板，也不会自动作为最终压测的 LOCAL_RPC_URL。",
                "Provide a reachable RPC endpoint to validate the custom RPC method. The original config/chains template will not be modified, and this will not automatically become the final benchmark LOCAL_RPC_URL.",
            ),
            kind="url",
        )
    if custom_rpc.get("status") == "needs_method" and not custom_rpc.get("method"):
        return _manual_question(group, "custom_rpc_method", _localized(language, "请输入要验证的自定义 RPC method 名称。", "Enter the custom RPC method name to validate."), kind="manual_value")
    if custom_rpc.get("status") == "needs_schema_evidence" and "params" not in custom_rpc:
        return _manual_question(
            group,
            "custom_rpc_schema_evidence",
            _localized(
                language,
                "请提供该 method 的 schema 证据：可以粘贴 params JSON、curl/request、response 示例或官方文档片段。没有参数时可直接输入 `[]`。",
                "Provide schema evidence for this method: params JSON, curl/request, response sample, or docs excerpt. Use `[]` when there are no params.",
            ),
            kind="evidence",
        )
    if custom_rpc.get("status") == "schema_needs_confirmation":
        return _option_question(group, "custom_rpc_schema_confirm", _format_schema_confirmation_prompt(language, custom_rpc.get("schema_draft") or {}), [{"label": "Y", "value": True}, {"label": "N", "value": False}], field="custom_rpc_schema_confirm", kind="yes_no")
    if custom_rpc.get("status") == "method_validated_next":
        return _option_question(
            group,
            "custom_rpc_continue",
            _localized(language, "这个自定义 RPC method 已验证通过。下一步怎么处理？", "This custom RPC method has been validated. What should happen next?"),
            [
                {"label": "继续添加另一个自定义 RPC method" if str(language or "").startswith("zh") else "Add another custom RPC method", "value": "add_another"},
                {"label": "当前 method 已够，继续配置 workload" if str(language or "").startswith("zh") else "This is enough; continue workload setup", "value": "finish"},
            ],
            field="custom_rpc_continue",
            kind="numbered_choice",
        )
    if custom_rpc.get("status") == "needs_scope":
        zh = str(language or "").startswith("zh")
        return _option_question(
            group,
            "custom_rpc_scope",
            _localized(language, "请选择自定义 RPC workload 如何应用。", "Choose how to apply this custom RPC workload."),
            [
                {"label": "仅使用这个 method 作为 single workload" if zh else "Use this method as single workload only", "value": "single_replace", "description": "Replace the selected workload with one custom RPC method in single mode."},
                {"label": "mixed 中只使用我提供的自定义 methods" if zh else "Use only my custom methods in mixed", "value": "mixed_replace", "description": "Remove/replace template default mixed methods and use only user-provided custom RPC methods."},
                {"label": "mixed 中保留模板默认 methods，并追加自定义 method" if zh else "Keep template defaults in mixed and add this method", "value": "mixed_add", "description": "Keep template default mixed methods and add the user-provided custom RPC method."},
            ],
            field="custom_rpc_scope",
            kind="numbered_choice",
        )
    if custom_rpc.get("status") == "needs_weights":
        methods = _validated_custom_methods(custom_rpc)
        example = _single_method_weight_example(methods)
        return _manual_question(
            group,
            "custom_rpc_weights",
            _localized(language, f"请输入 mixed 权重，总和必须为 100。已验证 methods：{', '.join(methods) or '<none>'}。格式示例：`{example}`。", f"Enter mixed weights. The total must be 100. Validated methods: {', '.join(methods) or '<none>'}. Example: `{example}`."),
            kind="url",
        )
    if identity.get("status") == "existing_family_needs_endpoint" and not state.get("endpoint_evidence", {}).get("candidate_endpoint_ready"):
        return _manual_question(group, "new_chain_endpoint", _localized(language, "请提供可访问的 RPC endpoint，用于验证该新链和 RPC method。", "Provide a reachable RPC endpoint to validate this new chain and RPC methods."), kind="url")
    if identity.get("status") == "existing_family_needs_method":
        return _manual_question(group, "new_chain_method", _localized(language, "请输入要验证的 RPC method 名称。", "Enter the RPC method name to validate."), kind="manual_value")
    if identity.get("status") == "existing_family_needs_schema_evidence":
        return _manual_question(group, "new_chain_schema_evidence", _localized(language, "请提供该 method 的 schema 证据：params JSON、curl/request、response 示例或官方文档片段。没有参数时可直接输入 `[]`。", "Provide schema evidence for this method: params JSON, curl/request, response sample, or docs excerpt. Use `[]` when there are no params."), kind="evidence")
    if identity.get("status") == "existing_family_schema_needs_confirmation":
        return _option_question(group, "new_chain_schema_confirm", _format_schema_confirmation_prompt(language, identity.get("schema_draft") or {}), [{"label": "Y", "value": True}, {"label": "N", "value": False}], field="new_chain_schema_confirm", kind="yes_no")
    if identity.get("status") == "existing_family_method_validated_next":
        return _option_question(
            group,
            "new_chain_method_continue",
            _localized(language, "新链 RPC method 已验证通过。下一步怎么处理？", "The new-chain RPC method has been validated. What should happen next?"),
            [
                {"label": "继续添加另一个 RPC method" if str(language or "").startswith("zh") else "Add another RPC method", "value": "add_another"},
                {"label": "当前 method 已够，继续后续流程" if str(language or "").startswith("zh") else "This is enough; continue the next workflow", "value": "finish"},
            ],
            field="new_chain_method_continue",
            kind="numbered_choice",
        )
    return None


def _apply_endpoint_answer(state: AgentGraphState, value: Any, question: PendingQuestion) -> AgentGraphState:
    question_id = str(question.get("id") or "")
    language = state.get("language", "en")
    custom = state.setdefault("custom_rpc", {})
    if question_id == "BLOCKCHAIN_PROCESS_NAMES":
        state.setdefault("confirmed_config", {})["BLOCKCHAIN_PROCESS_NAMES"] = _strip_scalar(str(value))
        state["pending_question"] = {}
        return state
    if question_id == "LOCAL_RPC_URL":
        endpoint = _extract_url_candidate(str(value)) or _strip_scalar(str(value))
        chain = (state.get("chain_identity") or {}).get("canonical") or ""
        result = validate_rpc_endpoint(
            chain=chain,
            endpoint=endpoint,
            methods=["eth_chainId"] if _chain_adapter_family(state) == "jsonrpc" else None,
            adapter_family=_chain_adapter_family(state),
            method_params={"eth_chainId": []},
            timeout=3.0,
        )
        state.setdefault("endpoint_evidence", {})["local_rpc_url_probe"] = result
        if not result.get("ready"):
            state["pending_question"] = {}
            state["visible_response"] = [_localized(
                language,
                f"LOCAL_RPC_URL 验证失败：{result.get('error') or result.get('status')}。证据：{result.get('evidence_file') or '<none>'}。请提供可访问的 endpoint。",
                f"LOCAL_RPC_URL validation failed: {result.get('error') or result.get('status')}. Evidence: {result.get('evidence_file') or '<none>'}. Provide a reachable endpoint.",
            )]
            return state
        state.setdefault("confirmed_config", {})["LOCAL_RPC_URL"] = endpoint
        state.setdefault("endpoint_evidence", {})["local_rpc_url_ready"] = True
        state["pending_question"] = {}
        state["visible_response"] = [_localized(
            language,
            f"LOCAL_RPC_URL 验证通过。证据：{result.get('evidence_file') or '<none>'}。",
            f"LOCAL_RPC_URL validation passed. Evidence: {result.get('evidence_file') or '<none>'}.",
        )]
        return state
    if question_id == "SYNC_OBSERVE_RPC_URL":
        endpoint = _extract_url_candidate(str(value)) or _strip_scalar(str(value))
        chain = (state.get("chain_identity") or {}).get("canonical") or ""
        result = validate_rpc_endpoint(
            chain=chain,
            endpoint=endpoint,
            methods=None,
            adapter_family=_chain_adapter_family(state),
            timeout=3.0,
        )
        state.setdefault("endpoint_evidence", {})["sync_rpc_url_probe"] = result
        if not result.get("ready"):
            state.setdefault("endpoint_evidence", {})["sync_rpc_url_ready"] = False
            state["pending_question"] = {}
            state["visible_response"] = [_localized(
                language,
                f"sync-observe endpoint 验证失败：{result.get('error') or result.get('status')}。证据：{result.get('evidence_file') or '<none>'}。请提供真实可访问的节点 RPC endpoint。",
                f"Sync-observe endpoint validation failed: {result.get('error') or result.get('status')}. Evidence: {result.get('evidence_file') or '<none>'}. Provide a real reachable node RPC endpoint.",
            )]
            return state
        state.setdefault("endpoint_evidence", {})["sync_rpc_url_ready"] = True
        state.setdefault("confirmed_config", {})["LOCAL_RPC_URL"] = endpoint
        state["pending_question"] = {}
        state["visible_response"] = [_localized(
            language,
            f"sync-observe endpoint 验证通过。证据：{result.get('evidence_file') or '<none>'}。",
            f"Sync-observe endpoint validation passed. Evidence: {result.get('evidence_file') or '<none>'}.",
        )]
        return state
    if question_id == "MAINNET_RPC_URL_REVIEWED":
        confirmed = state.setdefault("confirmed_config", {})
        if value is True:
            confirmed["MAINNET_RPC_URL_REVIEWED"] = True
        elif value is False:
            state.setdefault("endpoint_evidence", {})["mainnet_review_declined"] = True
            confirmed["MAINNET_RPC_URL_REVIEWED"] = True
        else:
            confirmed["MAINNET_RPC_URL"] = _strip_scalar(str(value))
            confirmed["MAINNET_RPC_URL_REVIEWED"] = True
        state["pending_question"] = {}
        return state
    if question_id == "custom_rpc_endpoint":
        endpoint = _extract_url_candidate(str(value)) or _strip_scalar(str(value))
        custom["endpoint"] = endpoint
        chain = (state.get("chain_identity") or {}).get("canonical") or ""
        result = validate_rpc_endpoint(
            chain=chain,
            endpoint=endpoint,
            methods=["eth_chainId"] if _chain_adapter_family(state) == "jsonrpc" else None,
            adapter_family=_chain_adapter_family(state),
            method_params={"eth_chainId": []},
            timeout=3.0,
        )
        custom["endpoint_probe"] = result
        state.setdefault("endpoint_evidence", {})["custom_rpc_endpoint_probe"] = result
        if not result.get("ready"):
            custom["status"] = "probe_failed"
            state["pending_question"] = {}
            state["visible_response"] = [_localized(
                language,
                f"endpoint 验证失败：{result.get('error') or result.get('status')}。证据：{result.get('evidence_file') or '<none>'}。请提供可访问的 HTTP RPC endpoint。",
                f"Endpoint validation failed: {result.get('error') or result.get('status')}. Evidence: {result.get('evidence_file') or '<none>'}. Provide a reachable HTTP RPC endpoint.",
            )]
            return state
        custom["endpoint_ready"] = True
        custom["status"] = "needs_schema_evidence" if custom.get("method") else "needs_method"
        state["pending_question"] = {}
        state["visible_response"] = [_localized(
            language,
            f"endpoint 验证通过。证据：{result.get('evidence_file') or '<none>'}。这个 endpoint 只作为自定义 RPC method/schema 验证证据，不会自动作为最终压测的 LOCAL_RPC_URL。",
            f"Endpoint validation passed. Evidence: {result.get('evidence_file') or '<none>'}. This endpoint is stored only as custom RPC method/schema validation evidence and will not automatically become the final benchmark LOCAL_RPC_URL.",
        )]
        if _extract_json_object_or_array(str(value)):
            return _apply_endpoint_answer(
                state,
                value,
                _manual_question("endpoint_process", "custom_rpc_method", "", kind="manual_value"),
            )
        return state
    if question_id == "new_chain_endpoint":
        endpoint = _extract_url_candidate(str(value)) or _strip_scalar(str(value))
        identity = state.setdefault("chain_identity", {})
        chain = str(identity.get("canonical") or identity.get("raw") or "").strip()
        adapter_family = _chain_adapter_family(state)
        result = validate_rpc_endpoint(
            chain=chain,
            endpoint=endpoint,
            methods=["eth_chainId"] if adapter_family == "jsonrpc" else None,
            adapter_family=adapter_family,
            method_params={"eth_chainId": []},
            timeout=3.0,
        )
        state.setdefault("endpoint_evidence", {})["candidate_endpoint"] = endpoint
        state.setdefault("endpoint_evidence", {})["new_chain_endpoint_probe"] = result
        if not result.get("ready"):
            identity["status"] = "existing_family_needs_endpoint"
            state.setdefault("endpoint_evidence", {})["candidate_endpoint_ready"] = False
            state["pending_question"] = {}
            state["visible_response"] = [_localized(
                language,
                f"新链 endpoint 验证失败：{result.get('error') or result.get('status')}。证据：{result.get('evidence_file') or '<none>'}。请提供可访问的 endpoint。",
                f"New-chain endpoint validation failed: {result.get('error') or result.get('status')}. Evidence: {result.get('evidence_file') or '<none>'}. Provide a reachable endpoint.",
            )]
            return state
        identity["status"] = "existing_family_needs_method"
        state.setdefault("endpoint_evidence", {})["candidate_endpoint_ready"] = True
        state["pending_question"] = {}
        state["visible_response"] = [_localized(
            language,
            f"新链 endpoint 验证通过。证据：{result.get('evidence_file') or '<none>'}。",
            f"New-chain endpoint validation passed. Evidence: {result.get('evidence_file') or '<none>'}.",
        )]
        if _extract_json_object_or_array(str(value)):
            return _apply_endpoint_answer(
                state,
                value,
                _manual_question("endpoint_process", "new_chain_method", "", kind="manual_value"),
            )
        return state
    if question_id == "new_chain_method":
        identity = state.setdefault("chain_identity", {})
        method_value = _strip_scalar(str(value))
        if _looks_like_url_value(method_value) or _looks_like_rest_path_or_doc_method(method_value):
            identity["status"] = "existing_family_needs_method"
            state["pending_question"] = {}
            state["visible_response"] = [_localized(
                language,
                "这看起来像 endpoint、REST path 或文档标题，不像可直接验证的 RPC method 名称。请提供 method 名称；如果这是 REST API，请先切换/确认协议族为 `rest`，再提供 REST path 和 request/response 证据。",
                "This looks like an endpoint, REST path, or documentation title rather than a directly verifiable RPC method name. Provide the method name; if this is a REST API, switch/confirm the adapter family as `rest` first, then provide the REST path and request/response evidence.",
            )]
            return state
        parsed_method, parsed = _parse_rpc_params_or_request(_extract_json_object_or_array(str(value)) or str(value))
        if parsed_method and parsed is not None:
            identity["candidate_method"] = parsed_method
            identity["schema_draft"] = {
                "status": "draft",
                "method": parsed_method,
                "params_json": parsed,
                "params": parsed if isinstance(parsed, list) else [],
                "evidence_kind": "jsonrpc_request",
                "transport": "jsonrpc",
            }
            return _validate_new_chain_rpc_schema(state, parsed)
        identity["candidate_method"] = method_value
        identity["status"] = "existing_family_needs_schema_evidence"
        state["pending_question"] = {}
        return state
    if question_id == "new_chain_schema_evidence":
        identity = state.setdefault("chain_identity", {})
        parsed_method, parsed = _parse_rpc_params_or_request(str(value))
        if parsed is not None:
            if parsed_method:
                identity["candidate_method"] = parsed_method
            if _is_direct_json_rpc_input(str(value)):
                return _validate_new_chain_rpc_schema(state, parsed)
            identity["schema_evidence"] = str(value)
            identity["schema_draft"] = _jsonrpc_schema_draft(parsed_method or str(identity.get("candidate_method") or ""), parsed)
            identity["status"] = "existing_family_schema_needs_confirmation"
            state["pending_question"] = _option_question(
                "endpoint_process",
                "new_chain_schema_confirm",
                _format_schema_confirmation_prompt(language, identity.get("schema_draft") or {}),
                [{"label": "Y", "value": True}, {"label": "N", "value": False}],
                field="new_chain_schema_confirm",
                kind="yes_no",
            )
            state["visible_response"] = [_render_question(state["pending_question"], language)]
            return state
        draft = extract_rpc_schema_from_evidence(state, str(value), method_hint=str(identity.get("candidate_method") or ""))
        identity["schema_evidence"] = str(value)
        identity["schema_draft"] = draft
        conflict = _new_chain_schema_conflict(state, draft)
        if conflict:
            return conflict
        if not _schema_draft_has_params(draft):
            identity["status"] = "existing_family_needs_schema_evidence"
            state["pending_question"] = {}
            state["visible_response"] = [_localized(
                language,
                "没有从证据中提取到可验证的 params。请提供更完整的 request/response/docs，或直接输入 params JSON；没有参数时输入 `[]`。",
                "I could not extract verifiable params from the evidence. Provide clearer request/response/docs, or enter params JSON directly; use `[]` when there are no params.",
            )]
            return state
        identity["status"] = "existing_family_schema_needs_confirmation"
        state["pending_question"] = _option_question(
            "endpoint_process",
            "new_chain_schema_confirm",
            _format_schema_confirmation_prompt(language, identity.get("schema_draft") or {}),
            [{"label": "Y", "value": True}, {"label": "N", "value": False}],
            field="new_chain_schema_confirm",
            kind="yes_no",
        )
        state["visible_response"] = [_render_question(state["pending_question"], language)]
        return state
    if question_id == "new_chain_schema_confirm":
        identity = state.setdefault("chain_identity", {})
        if value is False:
            identity["status"] = "existing_family_needs_schema_evidence"
            state["pending_question"] = {}
            state["visible_response"] = [_localized(
                language,
                "请提供修正后的 schema 证据或直接输入 params JSON。",
                "Provide corrected schema evidence or direct params JSON.",
            )]
            return state
        parsed = _schema_params_from_draft(identity.get("schema_draft") or {})
        return _validate_new_chain_rpc_schema(state, parsed)
    if question_id == "new_chain_method_continue":
        identity = state.setdefault("chain_identity", {})
        if value == "add_another":
            identity["status"] = "existing_family_needs_method"
            for key in ("candidate_method", "candidate_params", "schema_draft", "schema_evidence"):
                identity.pop(key, None)
            state["active_group"] = "endpoint_process"
            state["pending_question"] = {}
            return state
        identity["status"] = "existing_family_needs_workload_scope"
        state["active_group"] = "endpoint_process"
        state["pending_question"] = {}
        return state
    if question_id == "new_chain_workload_scope":
        identity = state.setdefault("chain_identity", {})
        identity["workload_scope"] = str(value)
        methods = _validated_new_chain_methods(identity)
        if value == "single_replace":
            if len(methods) > 1:
                identity["status"] = "existing_family_needs_single_method"
                state["active_group"] = "endpoint_process"
                state["pending_question"] = {}
                return state
            method = methods[0] if methods else str(identity.get("candidate_method") or "")
            state["rpc_mode"] = "single"
            state["workload"] = {
                "confirmed": True,
                "choice": "new_chain_verified_method",
                "methods": [method],
                "replace_defaults": True,
                "job_local_override": True,
            }
            identity["status"] = "existing_family_runtime_choice"
            state["active_group"] = "target_samples_fixtures"
            state["pending_question"] = {}
            return state
        identity["status"] = "existing_family_needs_weights"
        state["active_group"] = "endpoint_process"
        state["pending_question"] = {}
        return state
    if question_id == "new_chain_single_method":
        method = str(value)
        identity = state.setdefault("chain_identity", {})
        state["rpc_mode"] = "single"
        state["workload"] = {
            "confirmed": True,
            "choice": "new_chain_verified_method",
            "methods": [method],
            "replace_defaults": True,
            "job_local_override": True,
        }
        identity["status"] = "existing_family_runtime_choice"
        state["active_group"] = "target_samples_fixtures"
        state["pending_question"] = {}
        return state
    if question_id == "new_chain_custom_weights":
        identity = state.setdefault("chain_identity", {})
        methods = _validated_new_chain_methods(identity)
        weights = _parse_weight_spec_for_methods(str(value), methods)
        validated = set(methods)
        if not weights:
            identity["status"] = "existing_family_needs_weights"
            state["pending_question"] = {}
            state["visible_response"] = [_localized(
                language,
                "没有识别到有效权重。请使用 `method=weight,method2=weight2` 格式。",
                "No valid weights were found. Use `method=weight,method2=weight2` format.",
            )]
            return state
        missing = [method for method in validated if method not in weights]
        unknown = [method for method in weights if validated and method not in validated]
        total = sum(weights.values())
        if missing or unknown or total != 100:
            identity["status"] = "existing_family_needs_weights"
            identity["weights"] = weights
            state["pending_question"] = {}
            details = []
            if missing:
                details.append(f"缺少已验证 method：{', '.join(missing)}")
            if unknown:
                details.append(f"包含未验证 method：{', '.join(unknown)}")
            if total != 100:
                details.append(f"权重总和为 {total}，必须等于 100")
            state["visible_response"] = [_localized(
                language,
                f"新链 mixed 权重需要调整：{'; '.join(details)}。当前配置：{_format_weights(weights)}。请重新输入。",
                f"New-chain mixed weights need adjustment: {'; '.join(details)}. Current weights: {_format_weights(weights)}. Enter the weights again.",
            )]
            return state
        state["rpc_mode"] = "mixed"
        state["workload"] = {
            "confirmed": True,
            "choice": "new_chain_verified_methods",
            "methods": list(weights),
            "mixed_weights": weights,
            "replace_defaults": True,
            "job_local_override": True,
        }
        identity["status"] = "existing_family_runtime_choice"
        state["active_group"] = "target_samples_fixtures"
        state["pending_question"] = {}
        return state
    if question_id == "custom_rpc_method":
        method = _strip_scalar(str(value))
        if _looks_like_url_value(method) or _looks_like_rest_path_or_doc_method(method):
            custom["status"] = "needs_method"
            state["pending_question"] = {}
            state["visible_response"] = [_localized(
                language,
                "这看起来像 endpoint、REST path 或文档标题，不像当前链可直接验证的 RPC method 名称。请提供 method 名称；如果你要改协议/链，请直接说明要切换到哪个链或协议族。",
                "This looks like an endpoint, REST path, or documentation title rather than a verifiable RPC method name for the current chain. Provide the method name; if you need to change protocol or chain, say which chain or adapter family to switch to.",
            )]
            return state
        parsed_method, parsed = _parse_rpc_params_or_request(_extract_json_object_or_array(str(value)) or str(value))
        if parsed_method and parsed is not None:
            custom["method"] = parsed_method
            custom["schema_draft"] = {
                "status": "draft",
                "method": parsed_method,
                "params_json": parsed,
                "params": parsed if isinstance(parsed, list) else [],
                "evidence_kind": "jsonrpc_request",
                "transport": "jsonrpc",
            }
            return _validate_custom_rpc_schema(state, parsed)
        custom["method"] = method
        custom["status"] = "needs_schema_evidence"
        state["pending_question"] = {}
        return state
    if question_id == "custom_rpc_schema_evidence":
        parsed_method, parsed = _parse_rpc_params_or_request(str(value))
        if parsed is not None:
            if parsed_method:
                custom["method"] = parsed_method
            if _is_direct_json_rpc_input(str(value)):
                return _validate_custom_rpc_schema(state, parsed)
            custom["schema_evidence"] = str(value)
            custom["schema_draft"] = _jsonrpc_schema_draft(parsed_method or str(custom.get("method") or ""), parsed)
            custom["status"] = "schema_needs_confirmation"
            state["pending_question"] = _option_question(
                "endpoint_process",
                "custom_rpc_schema_confirm",
                _format_schema_confirmation_prompt(language, custom.get("schema_draft") or {}),
                [{"label": "Y", "value": True}, {"label": "N", "value": False}],
                field="custom_rpc_schema_confirm",
                kind="yes_no",
            )
            state["visible_response"] = [_render_question(state["pending_question"], language)]
            return state
        draft = extract_rpc_schema_from_evidence(state, str(value), method_hint=str(custom.get("method") or ""))
        custom["schema_evidence"] = str(value)
        custom["schema_draft"] = draft
        conflict = _custom_rpc_schema_conflict(state, draft)
        if conflict:
            return conflict
        if not _schema_draft_has_params(draft):
            state["pending_question"] = {}
            state["visible_response"] = [_localized(
                language,
                "没有从证据中提取到可验证的 params。请提供更完整的 request/response/docs，或直接输入 params JSON；没有参数时输入 `[]`。",
                "I could not extract verifiable params from the evidence. Provide clearer request/response/docs, or enter params JSON directly; use `[]` when there are no params.",
            )]
            custom["status"] = "needs_schema_evidence"
            return state
        custom["status"] = "schema_needs_confirmation"
        state["pending_question"] = _option_question(
            "endpoint_process",
            "custom_rpc_schema_confirm",
            _format_schema_confirmation_prompt(language, custom.get("schema_draft") or {}),
            [{"label": "Y", "value": True}, {"label": "N", "value": False}],
            field="custom_rpc_schema_confirm",
            kind="yes_no",
        )
        state["visible_response"] = [_render_question(state["pending_question"], language)]
        return state
    if question_id == "custom_rpc_schema_confirm":
        if value is False:
            custom["status"] = "needs_schema_evidence"
            state["pending_question"] = {}
            state["visible_response"] = [_localized(
                language,
                "请提供修正后的 schema 证据或直接输入 params JSON。",
                "Provide corrected schema evidence or direct params JSON.",
            )]
            return state
        parsed = _schema_params_from_draft(custom.get("schema_draft") or {})
        return _validate_custom_rpc_schema(state, parsed)
    if question_id == "custom_rpc_continue":
        if value == "add_another":
            custom["status"] = "needs_method"
            for key in ("method", "params", "schema_draft", "schema_evidence"):
                custom.pop(key, None)
            state["active_group"] = "endpoint_process"
            state["pending_question"] = {}
            return state
        custom["status"] = "needs_scope"
        state["active_group"] = "endpoint_process"
        state["pending_question"] = {}
        return state
    if question_id == "custom_rpc_scope":
        custom["scope"] = str(value)
        validated_methods = _validated_custom_methods(custom)
        method = validated_methods[0] if validated_methods else str(custom.get("method") or "")
        if value == "single_replace":
            state["rpc_mode"] = "single"
            state["workload"] = {"confirmed": True, "choice": "custom_rpc", "methods": [method], "replace_defaults": True}
            custom["status"] = "validated"
            custom["methods"] = [method]
            state["pending_question"] = {}
            return state
        custom["status"] = "needs_weights"
        state["pending_question"] = {}
        return state
    if question_id == "custom_rpc_weights":
        weights = _parse_weight_spec_for_methods(str(value), _validated_custom_methods(custom))
        if not weights:
            custom["status"] = "needs_weights"
            state["pending_question"] = {}
            state["visible_response"] = [_localized(
                language,
                "没有识别到有效权重。请使用 `method=weight,method2=weight2` 格式。",
                "No valid weights were found. Use `method=weight,method2=weight2` format.",
            )]
            return state
        total = sum(weights.values())
        if total != 100:
            custom["status"] = "needs_weights"
            custom["weights"] = weights
            state["pending_question"] = {}
            state["visible_response"] = [_localized(
                language,
                f"当前权重总和为 {total}，必须等于 100。当前配置：{_format_weights(weights)}。请重新输入。",
                f"The current weight total is {total}; it must equal 100. Current weights: {_format_weights(weights)}. Enter the weights again.",
            )]
            return state
        custom["status"] = "validated"
        custom["weights"] = weights
        state["rpc_mode"] = "mixed"
        state["workload"] = {
            "confirmed": True,
            "choice": "custom_rpc",
            "methods": list(weights),
            "mixed_weights": weights,
            "replace_defaults": custom.get("scope") == "mixed_replace",
        }
        state["visible_response"] = list(state.get("visible_response") or []) + [_localized(
            language,
            f"自定义 RPC mixed workload 已确认：{_format_weights(weights)}；模板默认 methods {'会被替换' if custom.get('scope') == 'mixed_replace' else '会被保留并追加'}。",
            f"Custom RPC mixed workload confirmed: {_format_weights(weights)}; template default methods will be {'replaced' if custom.get('scope') == 'mixed_replace' else 'kept and appended'}.",
        )]
        state["pending_question"] = {}
        return state
    state["pending_question"] = {}
    return state


def _apply_compound_pending_answer(state: AgentGraphState, text: str, question: PendingQuestion) -> AgentGraphState | None:
    question_id = str(question.get("id") or "")
    if _looks_like_assignment_answer(text):
        return None
    if question_id not in {
        "new_chain_method_continue",
        "custom_rpc_continue",
        "new_chain_workload_scope",
        "custom_rpc_scope",
    }:
        return None
    selected = _compound_choice_hint(text, question_id)
    if not selected:
        choice = resolve_pending_choice(state, text, question)
        if not bool(choice.get("matched")) or str(choice.get("confidence") or "").lower() not in {"medium", "high"}:
            return None
        selected = choice.get("selected_value")
    if not _pending_option_value_exists(selected, question):
        return None
    state = _apply_pending_answer(state, str(selected), question)
    if question_id in {"new_chain_method_continue", "custom_rpc_continue"} and selected == "add_another":
        return _apply_inline_method_payload_after_add_another(state, text, question_id)
    if question_id in {"new_chain_method_continue", "custom_rpc_continue"} and selected == "finish":
        scope_question = _question_for_group(state, "endpoint_process")
        if not scope_question:
            return state
        scoped = _apply_inline_scope_payload(state, text, scope_question)
        if scoped:
            return scoped
        state["pending_question"] = scope_question
        state["visible_response"] = [_render_question(scope_question, state.get("language", "en"))]
        return state
    if question_id in {"new_chain_workload_scope", "custom_rpc_scope"}:
        return _apply_inline_weights_payload_after_scope(state, text)
    return state


def _apply_inline_method_payload_after_add_another(state: AgentGraphState, text: str, previous_question_id: str) -> AgentGraphState:
    question_id = "new_chain_method" if previous_question_id == "new_chain_method_continue" else "custom_rpc_method"
    question = _manual_question("endpoint_process", question_id, "", kind="manual_value")
    if _extract_json_object_or_array(text):
        return _apply_endpoint_answer(state, text, question)
    state["pending_question"] = question
    state["visible_response"] = [_render_question(question, state.get("language", "en"))]
    return state


def _apply_inline_scope_payload(state: AgentGraphState, text: str, scope_question: PendingQuestion) -> AgentGraphState | None:
    selected = _compound_choice_hint(text, str(scope_question.get("id") or ""))
    if not selected:
        choice = resolve_pending_choice(state, text, scope_question)
        if not bool(choice.get("matched")) or str(choice.get("confidence") or "").lower() not in {"medium", "high"}:
            return None
        selected = choice.get("selected_value")
    if not _pending_option_value_exists(selected, scope_question):
        return None
    scoped = _apply_pending_answer(state, str(selected), scope_question)
    return _apply_inline_weights_payload_after_scope(scoped, text)


def _apply_inline_weights_payload_after_scope(state: AgentGraphState, text: str) -> AgentGraphState:
    pending = _question_for_group(state, "endpoint_process")
    methods: list[str] = []
    if pending and str(pending.get("id") or "") == "new_chain_custom_weights":
        methods = _validated_new_chain_methods(state.get("chain_identity") or {})
    elif pending and str(pending.get("id") or "") == "custom_rpc_weights":
        methods = _validated_custom_methods(state.get("custom_rpc") or {})
    weights = _parse_weight_spec_for_methods(text, methods)
    if not weights:
        if pending:
            state["pending_question"] = pending
            state["visible_response"] = [_render_question(pending, state.get("language", "en"))]
        return state
    if pending and str(pending.get("id") or "") in {"new_chain_custom_weights", "custom_rpc_weights"}:
        return _apply_pending_answer(state, _format_weights(weights), pending)
    return state


def _custom_rpc_inline_workload_hint(text: str, methods: list[str]) -> dict[str, Any]:
    raw = str(text or "")
    lowered = raw.lower()
    if not raw.strip() or not methods:
        return {}
    scope = ""
    if "mixed" in lowered or "权重" in raw or "weight" in lowered or _parse_weight_spec_for_methods(raw, methods):
        scope = "mixed_replace"
    elif "single" in lowered or "单 method" in raw or "单一" in raw:
        scope = "single_replace"
    if not scope:
        return {}
    hint: dict[str, Any] = {"scope": scope}
    weights = _parse_weight_spec_for_methods(raw, methods)
    if weights:
        hint["weights"] = weights
    if any(token in lowered for token in ("only", "enough", "finish", "done")) or any(token in raw for token in ("只跑", "只用", "够了", "足够", "完成", "不用再加")):
        hint["finish"] = True
    elif scope == "mixed_replace" and weights and set(weights) == set(methods) and sum(weights.values()) == 100:
        hint["finish"] = True
    return hint


def _apply_custom_rpc_inline_workload_hint(state: AgentGraphState) -> bool:
    language = state.get("language", "en")
    custom = state.setdefault("custom_rpc", {})
    hint = custom.get("inline_workload_hint")
    if not isinstance(hint, dict) or not hint.get("finish"):
        return False
    methods = _validated_custom_methods(custom)
    if not methods:
        return False
    scope = str(hint.get("scope") or "")
    if scope == "single_replace":
        method = methods[0]
        custom["scope"] = "single_replace"
        custom["status"] = "validated"
        custom["methods"] = [method]
        state["rpc_mode"] = "single"
        state["workload"] = {"confirmed": True, "choice": "custom_rpc", "methods": [method], "replace_defaults": True}
        state["pending_question"] = {}
        state["visible_response"] = list(state.get("visible_response") or []) + [_localized(
            language,
            f"自定义 RPC single workload 已确认：{method}；模板默认 method 会被替换。",
            f"Custom RPC single workload confirmed: {method}; the template default method will be replaced.",
        )]
        return True
    if scope == "mixed_replace":
        weights = hint.get("weights")
        if not isinstance(weights, dict):
            return False
        weights = {str(k): int(v) for k, v in weights.items()}
        if set(weights) != set(methods) or sum(weights.values()) != 100:
            return False
        custom["scope"] = "mixed_replace"
        custom["status"] = "validated"
        custom["weights"] = weights
        state["rpc_mode"] = "mixed"
        state["workload"] = {
            "confirmed": True,
            "choice": "custom_rpc",
            "methods": list(weights),
            "mixed_weights": weights,
            "replace_defaults": True,
        }
        state["pending_question"] = {}
        state["visible_response"] = list(state.get("visible_response") or []) + [_localized(
            language,
            f"自定义 RPC mixed workload 已确认：{_format_weights(weights)}；模板默认 methods 会被替换。",
            f"Custom RPC mixed workload confirmed: {_format_weights(weights)}; template default methods will be replaced.",
        )]
        return True
    return False


def _compound_choice_hint(text: str, question_id: str) -> str:
    raw = str(text or "").strip().lower()
    if not raw:
        return ""
    if question_id in {"new_chain_method_continue", "custom_rpc_continue"}:
        if any(token in raw for token in ("继续加", "再加", "add another", "another method", "more method")):
            return "add_another"
        if any(token in raw for token in ("够了", "足够", "完成", "不用再加", "finish", "done", "enough")):
            return "finish"
    if question_id == "new_chain_workload_scope":
        if "mixed" in raw or "权重" in raw or _parse_weight_spec(text):
            return "mixed_replace"
        if "single" in raw or "单 method" in raw or "单一" in raw:
            return "single_replace"
    if question_id == "custom_rpc_scope":
        if "mixed" in raw or "权重" in raw or _parse_weight_spec(text):
            return "mixed_replace"
        if "single" in raw or "单 method" in raw or "单一" in raw:
            return "single_replace"
    return ""


def _handle_chain_candidate(state: AgentGraphState, raw: str, resolution: dict[str, Any] | None = None) -> AgentGraphState:
    known = set(repo_chain_names())
    canonical = canonicalize_chain_scalar(raw, known_chains=known)
    state["pending_question"] = {}
    if canonical:
        previous = state.get("chain_identity", {}).get("canonical")
        if previous and previous != canonical:
            _invalidate_for_chain_change(state)
        state["chain_identity"] = {"raw": raw, "canonical": canonical, "status": "confirmed", "case": "known"}
        state.setdefault("confirmed_config", {})["BLOCKCHAIN_NODE"] = canonical
        state["active_group"] = ""
        prefix = list(state.get("visible_response") or [])
        state["visible_response"] = prefix + [_localized(
            state.get("language", "en"),
            f"已确认链为 `{canonical}`。",
            f"Confirmed chain: `{canonical}`.",
        )]
        return state
    resolution = resolution or resolve_unknown_chain_identity(state, raw)
    adapter_family = str(resolution.get("adapter_family") or "unknown").strip()
    canonical_name = str(resolution.get("canonical_chain_name") or raw).strip()
    possible_known = canonicalize_chain_scalar(str(resolution.get("possible_known_chain") or ""), known_chains=known)
    if possible_known and possible_known != canonical:
        state["chain_identity"] = {
            "raw": raw,
            "canonical": possible_known,
            "status": "needs_known_chain_confirmation",
            "case": "known_candidate",
            "llm_resolution": resolution,
        }
        state["active_group"] = "chain_identity"
        state["pending_question"] = _option_question(
            "chain_identity",
            "unknown_chain_identity_confirm",
            _localized(
                state.get("language", "en"),
                f"`{raw}` 不在当前模板中。模型认为你可能想输入 `{possible_known}`。是否确认使用这个链？",
                f"`{raw}` is not a configured template. The model thinks you may mean `{possible_known}`. Use this chain?",
            ),
            [
                {"label": _localized(state.get("language", "en"), f"使用 `{possible_known}`", f"Use `{possible_known}`"), "value": "confirm_known_chain"},
                {"label": _localized(state.get("language", "en"), "不是，重新输入链名", "No, re-enter chain name"), "value": "reenter_chain"},
                {"label": _localized(state.get("language", "en"), "这是另一条真实链，继续确认协议", "This is another real chain; choose protocol"), "value": "choose_protocol"},
            ],
            field="unknown_chain_decision",
            kind="numbered_choice",
        )
        state["visible_response"] = [_render_question(state["pending_question"], state.get("language", "en"))]
        return state

    state["chain_identity"] = {
        "raw": raw,
        "canonical": canonical_name,
        "adapter_family": adapter_family,
        "status": "needs_identity_confirmation",
        "case": "unknown",
        "requires_llm_identity_resolution": True,
        "llm_resolution": resolution,
    }
    state["active_group"] = "chain_identity"
    chain_exists = resolution.get("chain_exists")
    if chain_exists and adapter_family in SUPPORTED_ADAPTER_FAMILIES:
        prompt = _localized(
            state.get("language", "en"),
            f"`{raw}` 不在当前 36 条已配置链中。模型认为它可能是 `{canonical_name}`，协议族为 `{adapter_family}`。是否确认按该协议继续 endpoint/RPC 验证？",
            f"`{raw}` is not one of the configured 36 chains. The model thinks it may be `{canonical_name}` with adapter family `{adapter_family}`. Continue endpoint/RPC validation with that family?",
        )
        options = [
            {"label": _localized(state.get("language", "en"), "确认，继续 endpoint/RPC 验证", "Yes, continue endpoint/RPC validation"), "value": "confirm_proposed_protocol"},
            {"label": _localized(state.get("language", "en"), "我来选择协议族", "I will choose the adapter family"), "value": "choose_protocol"},
            {"label": _localized(state.get("language", "en"), "我要重新输入链名", "I want to re-enter the chain name"), "value": "reenter_chain"},
        ]
    else:
        prompt = _localized(
            state.get("language", "en"),
            f"`{raw}` 不在当前 36 条已配置链或已知别名中。模型没有可靠确认它是已支持协议链。它是一个真实链名，还是你想更正输入？",
            f"`{raw}` is not one of the configured 36 chains or known aliases. The model could not reliably confirm it as a supported-family chain. Is it a real chain name, or do you want to correct it?",
        )
        options = [
            {"label": _localized(state.get("language", "en"), "真实链，继续确认协议", "Real chain; continue to protocol confirmation"), "value": "choose_protocol"},
            {"label": _localized(state.get("language", "en"), "我要重新输入链名", "I want to re-enter the chain name"), "value": "reenter_chain"},
        ]
    state["pending_question"] = _option_question(
        "chain_identity",
        "unknown_chain_identity_confirm",
        prompt,
        options,
        field="unknown_chain_decision",
        kind="numbered_choice",
    )
    state["visible_response"] = [_render_question(state["pending_question"], state.get("language", "en"))]
    return state


def _chain_ambiguity_question(state: AgentGraphState, text: str, primary_candidate: str) -> PendingQuestion | None:
    raw_text = str(text or "")
    candidate = _strip_scalar(primary_candidate)
    if not candidate:
        return None
    lowered = raw_text.lower()
    uncertainty = any(token in lowered for token in (" or ", "或", "还是", "可能", "maybe", "not sure", "不确定", "记错"))
    if not uncertainty:
        return None
    known = set(repo_chain_names())
    mentioned_known = []
    for chain in sorted(known, key=len, reverse=True):
        if re.search(rf"(?<![A-Za-z0-9_-]){re.escape(chain)}(?![A-Za-z0-9_-])", lowered):
            mentioned_known.append(chain)
    canonical = canonicalize_chain_scalar(candidate, known_chains=known)
    raw_candidates = _uncertain_chain_tokens(raw_text, candidate, known)
    if canonical and len(set(mentioned_known)) <= 1 and canonical in mentioned_known and not raw_candidates:
        return None
    options: list[dict[str, Any]] = []
    seen = set()
    for chain in mentioned_known[:4]:
        if chain in seen:
            continue
        seen.add(chain)
        options.append({
            "label": _localized(state.get("language", "en"), f"使用已支持链 `{chain}`", f"Use configured chain `{chain}`"),
            "value": {"chain_choice": chain},
        })
    for raw_candidate in raw_candidates[:4]:
        normalized = canonicalize_chain_scalar(raw_candidate, known_chains=known)
        if normalized:
            if normalized in seen:
                continue
            seen.add(normalized)
            options.append({
                "label": _localized(state.get("language", "en"), f"使用已支持链 `{normalized}`（来自 `{raw_candidate}`）", f"Use configured chain `{normalized}` from `{raw_candidate}`"),
                "value": {"chain_choice": normalized},
            })
            continue
        if raw_candidate.lower() in seen:
            continue
        seen.add(raw_candidate.lower())
        options.append({
            "label": _localized(state.get("language", "en"), f"`{raw_candidate}` 是另一条真实链，进入新链确认", f"`{raw_candidate}` is another real chain; enter new-chain confirmation"),
            "value": {"unknown_chain_choice": raw_candidate},
        })
    if not canonical and candidate.lower() not in seen:
        options.append({
            "label": _localized(state.get("language", "en"), f"`{candidate}` 是另一条真实链，进入新链确认", f"`{candidate}` is another real chain; enter new-chain confirmation"),
            "value": {"unknown_chain_choice": candidate},
        })
    options.append({"label": _localized(state.get("language", "en"), "我重新输入链名", "I will re-enter the chain name"), "value": "reenter_chain"})
    if len(options) < 2:
        return None
    return _option_question(
        "chain_identity",
        "chain_ambiguity_confirm",
        _localized(
            state.get("language", "en"),
            "你这句话里有链名不确定或多个候选。请先确认要测试哪条链。",
            "Your turn contains an uncertain or multiple chain candidates. Confirm which chain to test first.",
        ),
        options,
        field="chain_ambiguity_choice",
        kind="numbered_choice",
    )


def _drop_resolved_chain_actions(state: AgentGraphState) -> None:
    remaining = []
    for item in state.get("action_queue") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "") in {"choose_chain", "change_chain"}:
            continue
        remaining.append(item)
    state["action_queue"] = remaining


def _uncertain_chain_tokens(text: str, primary_candidate: str, known: set[str]) -> list[str]:
    """Extract user-mentioned uncertain chain-like tokens for confirmation only.

    This helper must never decide a chain by itself. It only prevents the
    Harness from silently accepting one LLM-selected candidate when the user
    explicitly wrote that the chain name might be one of several values.
    """

    lowered_primary = _strip_scalar(primary_candidate).lower()
    candidate_canonical = canonicalize_chain_scalar(lowered_primary, known_chains=known)
    tokens = re.findall(r"(?<![A-Za-z0-9_-])([A-Za-z][A-Za-z0-9_-]{1,31})(?![A-Za-z0-9_-])", text)
    ignored = {
        "agent",
        "benchmark",
        "fake",
        "fake-node",
        "fakenode",
        "real",
        "real-node",
        "realnode",
        "sync",
        "observe",
        "qps",
        "quick",
        "standard",
        "intensive",
        "grafana",
        "prometheus",
        "local",
        "rpc",
        "method",
        "methods",
        "node",
        "chain",
        "test",
        "testing",
    }
    output: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        lowered = token.lower()
        if lowered in ignored or lowered == lowered_primary:
            continue
        token_canonical = canonicalize_chain_scalar(lowered, known_chains=known)
        if candidate_canonical and token_canonical == candidate_canonical:
            continue
        looks_like_known_prefix = any(len(lowered) >= 3 and chain.startswith(lowered) for chain in known)
        if not token_canonical and not looks_like_known_prefix:
            continue
        key = token_canonical or lowered
        if key in seen:
            continue
        seen.add(key)
        output.append(token)
    return output


def _request_chain_change_confirmation(state: AgentGraphState, raw: str, action: dict[str, Any]) -> AgentGraphState:
    known = set(repo_chain_names())
    canonical = canonicalize_chain_scalar(raw, known_chains=known)
    previous = str((state.get("chain_identity") or {}).get("canonical") or "").strip()
    requested_mode = _normalized_target_mode(action.get("target_mode"))
    mode_changed = bool(requested_mode and requested_mode != state.get("target_mode"))
    if not previous:
        if requested_mode:
            state["target_mode"] = requested_mode
            state["workflow_mode"] = "sync_observe" if requested_mode == "sync-observe" else "rpc_benchmark"
            _invalidate_for_target_mode(state)
        return _handle_chain_candidate(state, raw, action)
    if canonical and canonical == previous and not mode_changed:
        identity = state.setdefault("chain_identity", {})
        if identity.get("status") != "confirmed":
            identity.update({"canonical": canonical, "status": "confirmed", "case": "known"})
            state.setdefault("confirmed_config", {})["BLOCKCHAIN_NODE"] = canonical
            state["active_group"] = "provider_deployment"
            state["pending_question"] = {}
        state["pending_question"] = {}
        state["visible_response"] = [_localized(
            state.get("language", "en"),
            f"当前链已经是 `{previous}`。我会继续当前配置流程。",
            f"The current chain is already `{previous}`. I will continue the current configuration flow.",
        )]
        return state
    if canonical:
        candidate_label = canonical
        resolution: dict[str, Any] | None = None
    else:
        has_resolution_hint = any(
            key in action and str(action.get(key) or "").strip()
            for key in ("canonical_chain_name", "adapter_family", "possible_known_chain")
        ) or isinstance(action.get("chain_exists"), bool)
        resolution = action if has_resolution_hint else resolve_unknown_chain_identity(state, raw)
        candidate_label = str(resolution.get("canonical_chain_name") or raw).strip()
    prompt = _localized(
        state.get("language", "en"),
        f"是否确认从 `{previous}` 切换到 `{candidate_label}`" + (f"，并使用 `{requested_mode}` 模式" if requested_mode else "") + "？",
        f"Confirm switching from `{previous}` to `{candidate_label}`" + (f" and using `{requested_mode}` mode" if requested_mode else "") + "?",
    )
    state.setdefault("chain_identity", {})["change_candidate"] = {
        "raw": raw,
        "canonical": canonical,
        "target_mode": requested_mode,
        "resolution": resolution or {},
        "interrupted_group": state.get("active_group") or "",
    }
    state["active_group"] = "chain_identity"
    if canonical:
        options = [{"label": "Y", "value": True}, {"label": "N", "value": False}]
        kind = "yes_no"
    else:
        possible_known = canonicalize_chain_scalar(str((resolution or {}).get("possible_known_chain") or ""), known_chains=known)
        adapter_family = str((resolution or {}).get("adapter_family") or "unknown").strip()
        chain_exists = (resolution or {}).get("chain_exists")
        if possible_known:
            if possible_known == previous:
                prompt = _localized(
                    state.get("language", "en"),
                    f"`{raw}` 不在当前模板中。模型认为你可能想输入当前链 `{possible_known}`。要保持当前链，还是把 `{raw}` 当作另一条真实链继续确认协议？",
                    f"`{raw}` is not a configured template. The model thinks you may mean the current chain `{possible_known}`. Keep the current chain, or treat `{raw}` as another real chain and choose protocol?",
                )
                keep_label = _localized(state.get("language", "en"), f"保持当前链 `{possible_known}`", f"Keep current chain `{possible_known}`")
            else:
                prompt = _localized(
                    state.get("language", "en"),
                    f"`{raw}` 不在当前模板中。模型认为你可能想输入 `{possible_known}`。你要从 `{previous}` 切换到 `{possible_known}`，还是把 `{raw}` 当作另一条真实链继续确认协议？",
                    f"`{raw}` is not a configured template. The model thinks you may mean `{possible_known}`. Switch from `{previous}` to `{possible_known}`, or treat `{raw}` as another real chain and choose protocol?",
                )
                keep_label = _localized(state.get("language", "en"), f"切换到 `{possible_known}`", f"Switch to `{possible_known}`")
            options = [
                {"label": keep_label, "value": "confirm_known_chain"},
                {"label": _localized(state.get("language", "en"), f"`{raw}` 是另一条真实链，继续确认协议", f"`{raw}` is another real chain; choose protocol"), "value": "choose_protocol"},
                {"label": "N", "value": False},
            ]
        elif chain_exists and adapter_family in SUPPORTED_ADAPTER_FAMILIES:
            prompt = _localized(
                state.get("language", "en"),
                f"`{raw}` 不在当前 36 条已配置链中。模型认为它可能是 `{candidate_label}`，协议族为 `{adapter_family}`。是否从 `{previous}` 切换并按该协议继续 endpoint/RPC 验证？",
                f"`{raw}` is not one of the configured 36 chains. The model thinks it may be `{candidate_label}` with adapter family `{adapter_family}`. Switch from `{previous}` and continue endpoint/RPC validation with that family?",
            )
            options = [
                {"label": _localized(state.get("language", "en"), "确认，继续 endpoint/RPC 验证", "Yes, continue endpoint/RPC validation"), "value": "confirm_proposed_protocol"},
                {"label": _localized(state.get("language", "en"), "我来选择协议族", "I will choose the adapter family"), "value": "choose_protocol"},
                {"label": "N", "value": False},
            ]
        else:
            prompt = _localized(
                state.get("language", "en"),
                f"`{raw}` 不在当前 36 条已配置链或已知别名中。模型没有可靠确认它是已支持协议链。要从 `{previous}` 切换到这条真实链并继续确认协议，还是取消？",
                f"`{raw}` is not one of the configured 36 chains or known aliases. The model could not reliably confirm it as a supported-family chain. Switch from `{previous}` to this real chain and choose protocol, or cancel?",
            )
            options = [
                {"label": _localized(state.get("language", "en"), "真实链，继续确认协议", "Real chain; continue to protocol confirmation"), "value": "choose_protocol"},
                {"label": "N", "value": False},
            ]
        kind = "numbered_choice"
    state["pending_question"] = _option_question(
        "chain_identity",
        "chain_change_confirm",
        prompt,
        options,
        field="chain_change_confirmed",
        kind=kind,
    )
    state["pending_question"]["interrupted_group"] = state.get("chain_identity", {}).get("change_candidate", {}).get("interrupted_group", "")
    state["visible_response"] = [_render_question(state["pending_question"], state.get("language", "en"))]
    return state


def _request_target_mode_change_confirmation(state: AgentGraphState, target_mode: str) -> AgentGraphState:
    current_mode = str(state.get("target_mode") or "").strip() or "<unset>"
    interrupted_group = str(state.get("active_group") or "").strip()
    state["target_mode_change_candidate"] = target_mode
    state["active_group"] = "target_mode"
    state["pending_question"] = _option_question(
        "target_mode",
        "target_mode_change_confirm",
        _localized(
            state.get("language", "en"),
            f"是否确认从 `{current_mode}` 切换到 `{target_mode}`？切换后 endpoint、workload、QPS、preflight/smoke 会重新确认。",
            f"Confirm switching from `{current_mode}` to `{target_mode}`? Endpoint, workload, QPS, and preflight/smoke will be re-confirmed.",
        ),
        [{"label": "Y", "value": True}, {"label": "N", "value": False}],
        field="target_mode_change_confirmed",
        kind="yes_no",
    )
    state["pending_question"]["interrupted_group"] = interrupted_group
    state["visible_response"] = [_render_question(state["pending_question"], state.get("language", "en"))]
    return state


def _apply_disk_answer(state: AgentGraphState, value: Any, question: PendingQuestion) -> None:
    field = str(question.get("field") or "")
    confirmed = state.setdefault("confirmed_config", {})
    inferred = state.setdefault("inferred_config", {})
    if field == "has_accounts_device":
        confirmed[field] = bool(value)
        if not value:
            for key in ("ACCOUNTS_DEVICE", "ACCOUNTS_VOL_TYPE", "ACCOUNTS_VOL_SIZE", "ACCOUNTS_VOL_MAX_IOPS", "ACCOUNTS_VOL_MAX_THROUGHPUT"):
                confirmed.pop(key, None)
        return
    if field:
        if value == "__manual__":
            inferred[f"{field}_manual_required"] = True
            return
        confirmed[field] = _strip_scalar(str(value))
        if field in {"LEDGER_DEVICE", "ACCOUNTS_DEVICE"}:
            size = _disk_size_for_device(state, str(value))
            size_key = "DATA_VOL_SIZE" if field == "LEDGER_DEVICE" else "ACCOUNTS_VOL_SIZE"
            if size:
                inferred[size_key] = size
                inferred.pop(f"{size_key}_manual_required", None)


def _disk_group_question(state: AgentGraphState, *, prefix: str, device_key: str, group: str) -> PendingQuestion | None:
    language = state.get("language", "en")
    confirmed = state.get("confirmed_config") or {}
    volume_type_key = f"{prefix}_VOL_TYPE"
    size_key = f"{prefix}_VOL_SIZE"
    iops_key = f"{prefix}_VOL_MAX_IOPS"
    throughput_key = f"{prefix}_VOL_MAX_THROUGHPUT"
    if not confirmed.get(device_key):
        candidates = _disk_options(state)
        return _choice_question(
            group,
            device_key,
            _localized(language, f"请选择 {device_key}，或直接输入设备名。", f"Choose {device_key}, or type the device name."),
            candidates,
            device_key,
            kind="device",
        )
    if not confirmed.get(volume_type_key):
        return _manual_question(group, volume_type_key, _localized(language, f"请输入 {device_key} 的磁盘类型，例如 hyperdisk-balanced、pd-ssd、ssd、nvme。", f"Confirm {device_key} volume type, for example hyperdisk-balanced, pd-ssd, ssd, or nvme."), kind="manual_value")
    if not confirmed.get(size_key):
        inferred = state.setdefault("inferred_config", {})
        size = "" if inferred.get(f"{size_key}_manual_required") else str(inferred.get(size_key) or _disk_size_for_device(state, str(confirmed.get(device_key))))
        if size:
            return _confirm_or_value_question(group, size_key, _localized(language, f"检测到 {size_key} 为 `{size}` GiB，是否使用？", f"Detected {size_key}: `{size}` GiB. Use it?"), field=size_key, default_value=size)
        return _manual_question(group, size_key, _localized(language, f"请输入 {size_key}，单位 GiB。", f"Confirm {size_key} in GiB."), kind="manual_value")
    if not confirmed.get(iops_key):
        return _manual_question(group, iops_key, _localized(language, f"请输入 {iops_key}。", f"Confirm {iops_key}."), kind="manual_value")
    if not confirmed.get(throughput_key):
        return _manual_question(group, throughput_key, _localized(language, f"请输入 {throughput_key}，单位 MiB/s。", f"Confirm {throughput_key} in MiB/s."), kind="manual_value")
    return None


def _opening_question(state: AgentGraphState, language: str) -> PendingQuestion:
    zh = str(language or "").startswith("zh")
    identity = state.get("chain_identity") or {}
    chain = identity.get("canonical") or identity.get("raw") or ""
    target_mode = str(state.get("target_mode") or "").strip()
    if chain or target_mode:
        prompt = _localized(
            language,
            f"已记录{f'链 `{chain}`' if chain else '当前目标'}{f'，当前模式 `{target_mode}`' if target_mode else ''}。请选择下一步。",
            f"Recorded {f'chain `{chain}`' if chain else 'the current target'}{f' with mode `{target_mode}`' if target_mode else ''}. Choose the next step.",
        )
    else:
        prompt = _localized(language, "你好，我是 AnyChain Benchmark Agent。你想让我帮你做什么？", "Hi, I am AnyChain Benchmark Agent. What would you like me to help with?")
    return _option_question(
        "opening",
        "opening_next_action",
        prompt,
        [
            {"label": "启动 fake-node 测试" if zh else "Start a fake-node benchmark", "value": "fake-node"},
            {"label": "启动 real-node 测试" if zh else "Start a real-node benchmark", "value": "real-node"},
            {"label": "观察节点同步" if zh else "Observe node sync behavior", "value": "sync-observe"},
            {"label": "了解支持的链、RPC method 和二次开发方式" if zh else "Learn supported chains, RPC methods, and extension paths", "value": "info"},
        ],
        field="target_mode",
        kind="numbered_choice",
    )


def _chain_question(state: AgentGraphState) -> PendingQuestion:
    target_mode = state.get("target_mode") or ("sync-observe" if state.get("workflow_mode") == "sync_observe" else "")
    language = state.get("language", "en")
    return _manual_question(
        "chain_identity",
        "chain",
        _localized(language, f"你想测试哪条链？当前目标模式：{target_mode or '未选择'}。", f"Which chain do you want to benchmark? Current target mode: {target_mode or 'not selected'}."),
        kind="chain",
    )


def _protocol_family_question(state: AgentGraphState) -> PendingQuestion:
    language = state.get("language", "en")
    return _option_question(
        "chain_identity",
        "adapter_family_confirm",
        _localized(language, "请确认该链属于哪个协议族。", "Confirm which adapter family this chain belongs to."),
        [
            {"label": "jsonrpc / EVM", "value": "jsonrpc"},
            {"label": "substrate", "value": "substrate"},
            {"label": "rest", "value": "rest"},
            {"label": "tendermint", "value": "tendermint"},
            {"label": "bitcoin_jsonrpc", "value": "bitcoin_jsonrpc"},
            {"label": "hedera_dual", "value": "hedera_dual"},
            {"label": "不属于以上协议族 / 不确定" if str(language or "").startswith("zh") else "None of the above / unsure", "value": "unsupported"},
        ],
        field="adapter_family",
        kind="numbered_choice",
    )


def _option_question(
    group: str,
    question_id: str,
    prompt: str,
    options: list[dict[str, Any]],
    *,
    field: str,
    kind: str,
    manual_input_allowed: bool = False,
) -> PendingQuestion:
    return {
        "id": question_id,
        "group": group,
        "subgroup": question_id,
        "kind": kind,
        "prompt": prompt,
        "options": options,
        "field": field,
        "manual_input_allowed": manual_input_allowed,
        "next_on_valid": "",
        "next_on_invalid": group,
    }


def _choice_question(group: str, question_id: str, prompt: str, options: list[dict[str, Any]], field: str, *, kind: str) -> PendingQuestion:
    return _option_question(group, question_id, prompt, options, field=field, kind=kind, manual_input_allowed=True)


def _manual_question(group: str, question_id: str, prompt: str, *, kind: str) -> PendingQuestion:
    return {
        "id": question_id,
        "group": group,
        "subgroup": question_id,
        "kind": kind,
        "prompt": prompt,
        "options": [],
        "field": question_id,
        "manual_input_allowed": True,
        "next_on_valid": "",
        "next_on_invalid": group,
    }


def _confirm_or_value_question(group: str, question_id: str, prompt: str, *, field: str, default_value: Any) -> PendingQuestion:
    return _option_question(
        group,
        question_id,
        prompt,
        [{"label": "Y", "value": default_value}, {"label": "N", "value": "__manual__"}],
        field=field,
        kind="confirm_or_value",
        manual_input_allowed=True,
    )


def _answer_fits_pending(text: str, question: PendingQuestion) -> bool:
    raw = _strip_scalar(text)
    if not raw:
        return False
    if question.get("group") != "chain_identity" and canonicalize_chain_scalar(raw, known_chains=set(repo_chain_names())):
        return False
    kind = question.get("kind", "")
    if kind == "yes_no":
        if str(question.get("field") or "") == "has_accounts_device" and (
            _text_mentions_no_accounts_disk(raw) or _text_mentions_yes_accounts_disk(raw)
        ):
            return True
        return raw.lower() in {"y", "yes", "n", "no"}
    if kind == "numbered_choice" and question.get("id") == "unknown_chain_identity_confirm":
        candidate = _chain_from_option_label(str((question.get("options") or [{}])[0].get("label") or ""))
        raw_chain = canonicalize_chain_scalar(raw, known_chains=set(repo_chain_names()))
        if candidate and raw_chain and candidate == raw_chain:
            return True
        if raw.lower() in {"y", "yes", "n", "no"}:
            return True
        if _adapter_family_hint_from_text(raw):
            return True
    if kind == "numbered_choice" and question.get("id") == "adapter_family_confirm":
        if _adapter_family_hint_from_text(raw):
            return True
    if kind == "confirm_or_value":
        if raw.lower() in {"y", "yes", "n", "no"}:
            return True
        question_id = str(question.get("id") or "")
        field = str(question.get("field") or "")
        if question_id == "MAINNET_RPC_URL_REVIEWED" or field == "MAINNET_RPC_URL_REVIEWED":
            return bool(_extract_url_candidate(raw))
        if field in {"DATA_VOL_SIZE", "ACCOUNTS_VOL_SIZE"}:
            return bool(re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", raw))
        return _is_plain_scalar_answer(raw) and not _looks_like_user_question(raw)
    if kind == "numbered_choice":
        return _matches_numbered_option(raw, question)
    if kind == "device" and raw.lower() in {"y", "yes"}:
        return len(question.get("options") or []) == 1
    if kind == "device" and raw.lower() in {"n", "no"}:
        return False
    if kind == "device" and raw.isdigit():
        return True
    if kind == "chain" and question.get("manual_input_allowed"):
        return _is_plain_chain_answer(raw)
    if kind == "url" and question.get("manual_input_allowed"):
        endpoint_question_ids = {"LOCAL_RPC_URL", "SYNC_OBSERVE_RPC_URL", "custom_rpc_endpoint", "new_chain_endpoint"}
        if str(question.get("id") or "") in endpoint_question_ids:
            if str(question.get("id") or "") in {"LOCAL_RPC_URL", "SYNC_OBSERVE_RPC_URL"}:
                return _is_bare_endpoint_answer(raw)
            return bool(_extract_url_candidate(raw))
        return _is_single_turn_value(raw)
    if kind == "evidence" and question.get("manual_input_allowed"):
        return bool(str(text or "").strip())
    if kind == "manual_value" and question.get("manual_input_allowed"):
        if str(question.get("id") or "") in {"new_chain_custom_weights", "custom_rpc_weights"}:
            return bool(_parse_weight_spec(str(text or "")) or _first_number_text(raw))
        if str(question.get("id") or "") in {"new_chain_method", "custom_rpc_method"}:
            return bool(str(text or "").strip()) and len(str(text or "")) <= 4000
        if raw.lower() in {"y", "yes", "n", "no"}:
            return False
        return _is_plain_scalar_answer(raw)
    if kind == "device" and question.get("manual_input_allowed"):
        return _is_plain_scalar_answer(raw)
    return False


def _coerce_answer(text: str, question: PendingQuestion) -> Any:
    raw = _strip_scalar(text)
    options = question.get("options") or []
    if raw.isdigit() and options:
        index = int(raw) - 1
        if 0 <= index < len(options):
            return options[index].get("value")
    lowered = raw.lower()
    if question.get("kind") == "device" and lowered in {"y", "yes"} and len(options) == 1:
        return options[0].get("value")
    if question.get("kind") == "numbered_choice" and question.get("id") == "unknown_chain_identity_confirm":
        candidate = _chain_from_option_label(str((options[0].get("label") if options else "") or ""))
        raw_chain = canonicalize_chain_scalar(raw, known_chains=set(repo_chain_names()))
        if candidate and raw_chain and candidate == raw_chain and options:
            return options[0].get("value")
        if lowered in {"y", "yes"} and options:
            return options[0].get("value")
        if lowered in {"n", "no"} and len(options) > 1:
            return options[1].get("value")
        family = _adapter_family_hint_from_text(raw)
        if family:
            return {"choose_protocol_family": family}
    if question.get("kind") == "numbered_choice" and question.get("id") == "adapter_family_confirm":
        family = _adapter_family_hint_from_text(raw)
        if family:
            return family
    if question.get("kind") == "yes_no":
        if str(question.get("field") or "") == "has_accounts_device":
            if _text_mentions_no_accounts_disk(raw):
                return False
            if _text_mentions_yes_accounts_disk(raw):
                return True
        if lowered in {"y", "yes"}:
            if options:
                return options[0].get("value")
            return True
        if lowered in {"n", "no"}:
            if len(options) > 1:
                return options[1].get("value")
            return False
    for option in options:
        if lowered in {
            str(option.get("value") or "").lower(),
            str(option.get("label") or "").lower(),
            str(option.get("id") or "").lower(),
        }:
            return option.get("value")
    return raw


def _chain_from_option_label(label: str) -> str:
    known = set(repo_chain_names())
    raw = str(label or "")
    for token in re.findall(r"`([^`]+)`|([A-Za-z0-9][A-Za-z0-9_-]{1,40})", raw):
        candidate_text = next((part for part in token if part), "")
        candidate = canonicalize_chain_scalar(candidate_text, known_chains=known)
        if candidate:
            return candidate
    return canonicalize_chain_scalar(raw, known_chains=known)


def _convert_known_candidate_to_unknown_chain(state: AgentGraphState) -> None:
    identity = state.setdefault("chain_identity", {})
    if identity.get("status") != "needs_known_chain_confirmation" and identity.get("case") != "known_candidate":
        return
    raw = str(identity.get("raw") or "").strip()
    if not raw:
        return
    resolution = identity.get("llm_resolution") if isinstance(identity.get("llm_resolution"), dict) else {}
    identity.clear()
    identity.update({
        "raw": raw,
        "canonical": raw,
        "adapter_family": str(resolution.get("adapter_family") or "unknown").strip(),
        "status": "needs_protocol_confirmation",
        "case": "unknown",
        "identity_confirmed": True,
        "llm_resolution": resolution,
    })


def _matches_numbered_option(raw: str, question: PendingQuestion) -> bool:
    options = question.get("options") or []
    if raw.isdigit():
        index = int(raw) - 1
        return 0 <= index < len(options)
    lowered = raw.lower()
    for option in options:
        values = {
            str(option.get("label") or "").strip().lower(),
            str(option.get("id") or "").strip().lower(),
        }
        option_value = option.get("value")
        if option_value not in {None, False}:
            values.add(str(option_value).strip().lower())
        elif option_value is False:
            values.add("false")
        if lowered in values:
            return True
    return False


def _pending_option_value_exists(value: Any, question: PendingQuestion) -> bool:
    return any(option.get("value") == value for option in question.get("options") or [])


def _workload_default_prompt(state: AgentGraphState, language: str) -> str:
    chain = str((state.get("chain_identity") or {}).get("canonical") or "selected chain")
    rpc_mode = str(state.get("rpc_mode") or "").strip() or "<not selected>"
    defaults = default_workload(chain)
    single = str(defaults.get("single") or "<none>")
    mixed_rows = defaults.get("mixed_weighted") or []
    if mixed_rows:
        mixed = ", ".join(f"{row.get('method')}={row.get('weight')}" for row in mixed_rows if row.get("method"))
    else:
        mixed = "<none>"
    if str(language or "").startswith("zh"):
        return (
            f"当前链 `{chain}` 的模板 workload：\n"
            f"- 当前 RPC 模式：`{rpc_mode}`\n"
            f"- single 默认 method：`{single}`\n"
            f"- mixed 默认权重：{mixed}\n"
            "请选择如何继续。"
        )
    return (
        f"Current chain template workload for `{chain}`:\n"
        f"- Current RPC mode: `{rpc_mode}`\n"
        f"- Default single method: `{single}`\n"
        f"- Default mixed weights: {mixed}\n"
        "Choose how to continue."
    )


def _qps_default_prompt(mode: str, language: str) -> str:
    mode = (mode or "quick").strip().lower()
    profiles = {
        "quick": {"INITIAL_QPS": "1000", "MAX_QPS": "1500", "QPS_STEP": "500", "DURATION": "60"},
        "standard": {"INITIAL_QPS": "2000", "MAX_QPS": "50000", "QPS_STEP": "500", "DURATION": "600"},
        "intensive": {"INITIAL_QPS": "50000", "MAX_QPS": "9999999", "QPS_STEP": "250", "DURATION": "600"},
    }
    values = profiles.get(mode, profiles["quick"])
    summary = ", ".join(f"{key}={value}" for key, value in values.items())
    if str(language or "").startswith("zh"):
        return (
            f"`{mode}` 模式默认 QPS 配置：{summary}。\n"
            "fake-node smoke 执行阶段会使用安全小流量覆盖来验证闭环；真实性能测试以最终 profile 为准。是否使用这些默认值？"
        )
    return (
        f"Default QPS profile for `{mode}`: {summary}.\n"
        "fake-node smoke uses a safe low-traffic execution override to validate the loop; real benchmarks use the final profile. Use these defaults?"
    )


def _render_question(question: PendingQuestion, language: str) -> str:
    lines = [str(question.get("prompt") or "").strip()]
    options = question.get("options") or []
    for index, option in enumerate(options, start=1):
        lines.append(f"{index}. {option.get('label') or option.get('value')}")
    if question.get("manual_input_allowed"):
        if options:
            lines.append(_localized(language, "你可以回复编号，也可以直接输入自定义值。", "Reply with a number, or type a custom value directly."))
        else:
            lines.append(_localized(language, "请直接输入值。", "Type the value directly."))
    elif options:
        lines.append(_localized(language, "请回复选项编号或选项名称。", "Reply with an option number or option name."))
    return "\n".join(line for line in lines if line)


def _disk_options(state: AgentGraphState) -> list[dict[str, Any]]:
    candidates = ((state.get("discovery") or {}).get("disks") or {}).get("candidates") or []
    options = []
    for item in candidates:
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        label = f"{name} ({item.get('size') or '<unknown>'}, {item.get('type') or 'disk'})"
        options.append({"label": label, "value": name})
    return options


def _usable_interfaces(interfaces: list[str], default: str = "") -> list[str]:
    blocked_exact = {"lo", "bonding_masters", "sit0", "tunl0", "gre0", "gretap0", "erspan0", "ip_vti0", "ip6_vti0", "ip6gre0", "ip6tnl0"}
    output = []
    for raw in interfaces:
        name = str(raw or "").strip()
        if not name:
            continue
        if name == default:
            output.append(name)
            continue
        if name in blocked_exact:
            continue
        output.append(name)
    if default and default not in output:
        output.insert(0, default)
    return output or interfaces


def _disk_size_for_device(state: AgentGraphState, device: str) -> str:
    device = device.strip().removeprefix("/dev/")
    candidates = ((state.get("discovery") or {}).get("disks") or {}).get("candidates") or []
    for item in candidates:
        if str(item.get("name") or "").strip().removeprefix("/dev/") == device:
            return _size_to_gib(item.get("size"))
    return ""


def _size_to_gib(value: Any) -> str:
    text = str(value or "").strip().upper()
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGT]?)(?:I?B)?", text)
    if not match:
        return ""
    number = float(match.group(1))
    unit = match.group(2)
    if unit == "T":
        number *= 1024
    elif unit == "M":
        number /= 1024
    elif unit == "K":
        number /= 1024 * 1024
    return str(int(number)) if number >= 1 else str(round(number, 3))


def _normalize_proposed_config_value(key: str, value: Any) -> tuple[str, Any]:
    """Normalize inferred config before it can enter confirmed state.

    This keeps LLM-extracted facts, structured paste, and direct KEY=value input
    on the same contract. It is intentionally limited to known AnyChain fields;
    workflow intent still belongs to the group router.
    """

    normalized_key = str(key or "").strip().upper()
    scalar = _strip_scalar(str(value)).strip().strip("\"'")
    if not normalized_key or not scalar:
        return normalized_key, ""
    if normalized_key == "ACCOUNTS_DEVICE" and _is_absence_value(scalar):
        return "HAS_ACCOUNTS_DEVICE", False
    if normalized_key == "HAS_ACCOUNTS_DEVICE":
        if _is_absence_value(scalar):
            return normalized_key, False
        if scalar.lower() in {"y", "yes", "true", "1", "有", "是"}:
            return normalized_key, True
        if scalar.lower() in {"n", "no", "false", "0", "没有", "无"}:
            return normalized_key, False
        return normalized_key, scalar
    if normalized_key in {"DATA_VOL_SIZE", "ACCOUNTS_VOL_SIZE"}:
        size = _size_to_gib(scalar)
        return normalized_key, size or _first_number_text(scalar)
    if normalized_key in {
        "DATA_VOL_MAX_IOPS",
        "DATA_VOL_MAX_THROUGHPUT",
        "ACCOUNTS_VOL_MAX_IOPS",
        "ACCOUNTS_VOL_MAX_THROUGHPUT",
        "NETWORK_MAX_BANDWIDTH_GBPS",
    }:
        return normalized_key, _first_number_text(scalar)
    if normalized_key in {"DATA_VOL_TYPE", "ACCOUNTS_VOL_TYPE"}:
        return normalized_key, scalar.lower()
    return normalized_key, scalar


def _is_absence_value(value: Any) -> bool:
    text = _strip_scalar(str(value or "")).strip().lower()
    return text in {
        "none",
        "<none>",
        "null",
        "nil",
        "n/a",
        "na",
        "false",
        "0",
        "no",
        "n",
        "none detected",
        "not detected",
        "no separate disk",
        "no accounts",
        "no accounts disk",
        "without accounts",
        "没有",
        "无",
        "没有 accounts",
        "没有 accounts 盘",
        "没有独立 accounts 盘",
        "不需要",
    }


def _text_mentions_no_accounts_disk(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    if "accounts" not in text and "account" not in text:
        return False
    return any(marker in text for marker in ("没有", "无", "no ", "without", "not have", "don't have", "does not have"))


def _text_mentions_yes_accounts_disk(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    if "accounts" not in text and "account" not in text:
        return False
    if _text_mentions_no_accounts_disk(text):
        return False
    return any(marker in text for marker in ("有", "独立", "separate", "dedicated", "with accounts", "has accounts"))


def _pending_answer_contains_extra_intent(text: str, question: PendingQuestion) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    if str(question.get("kind") or "") != "yes_no":
        return False
    if raw.lower() in {"y", "yes", "n", "no"} or raw.isdigit():
        return False
    if str(question.get("kind") or "") == "yes_no" and str(question.get("field") or "") == "has_accounts_device":
        compact = re.sub(r"\s+", " ", raw.lower()).strip(" ，,。.")
        no_markers = {
            "没有 accounts 盘",
            "没有独立 accounts 盘",
            "no accounts disk",
            "no separate accounts disk",
            "without accounts disk",
        }
        yes_markers = {
            "有 accounts 盘",
            "有独立 accounts 盘",
            "separate accounts disk",
            "dedicated accounts disk",
        }
        if compact in no_markers or compact in yes_markers:
            return False
    return any(token in raw.lower() for token in ("顺便", "然后", "另外", "先", "also", "and ", "then", "after that"))


def _pending_context_response(state: AgentGraphState, question: PendingQuestion, language: str) -> str:
    question_id = str(question.get("id") or "").strip()
    if question_id in {"new_chain_endpoint", "custom_rpc_endpoint"}:
        if question_id == "new_chain_endpoint":
            if str(language or "").startswith("zh"):
                return (
                    "这里是在验证一条未内置的新链是否能走当前已支持协议族。你需要准备：\n"
                    "1. 一个当前可访问的 HTTP RPC endpoint，用来做真实探测；\n"
                    "2. 至少一个要测试的 RPC method 名称、参数示例和期望 response 示例；\n"
                    "3. 如果有官方 RPC 文档，可以一起粘贴，我会结合 endpoint 的真实返回校验。\n"
                    "这个 endpoint 只作为新链/RPC method 验证证据；只有你后续选择 real-node benchmark 并确认时，才会作为最终被测 endpoint 使用。"
                )
            return (
                "This step validates whether a non-template chain can use an existing adapter family. Prepare:\n"
                "1. A currently reachable HTTP RPC endpoint for real probing;\n"
                "2. At least one RPC method name with parameter samples and expected response sample;\n"
                "3. Official RPC docs if available, so I can compare docs with live endpoint behavior.\n"
                "This endpoint is validation evidence only; it becomes the final benchmark endpoint only if you later choose and confirm a real-node benchmark."
            )
        if str(language or "").startswith("zh"):
            return (
                "这里是在验证自定义 RPC method。你需要准备：\n"
                "1. 一个可访问的 RPC endpoint，用来验证 method/params/response 是否真实可用；\n"
                "2. method 名称、params 示例，以及 response 示例或官方文档；\n"
                "3. 如果要 mixed workload，还需要给所有启用 method 的权重，总和必须为 100。\n"
                "验证用 endpoint 不会自动覆盖最终 `LOCAL_RPC_URL`，也不会修改 `config/chains` 原始模板。"
            )
        return (
            "This step validates a custom RPC method. Prepare:\n"
            "1. A reachable RPC endpoint to validate method/params/response behavior;\n"
            "2. Method name, params sample, and response sample or official docs;\n"
            "3. For mixed workload, weights for all enabled methods summing to 100.\n"
            "The validation endpoint will not automatically replace `LOCAL_RPC_URL`, and the original `config/chains` template is not modified."
        )
    return format_current_context(state, language)


def _pending_endpoint_context_question(question: PendingQuestion, text: str) -> bool:
    question_id = str(question.get("id") or "")
    if question_id not in {"LOCAL_RPC_URL", "SYNC_OBSERVE_RPC_URL", "custom_rpc_endpoint", "new_chain_endpoint"}:
        return False
    if _answer_fits_pending(text, question):
        return False
    return _looks_like_user_question(text)


def _post_pending_answer_context_requested(state: AgentGraphState, text: str, answered_question: PendingQuestion) -> bool:
    if not _looks_like_user_question(text):
        return False
    raw = _strip_scalar(text).lower()
    if raw in {"y", "yes", "n", "no"} or raw.isdigit():
        return False
    current_question = state.get("pending_question") or {}
    if not current_question:
        return False
    if str(current_question.get("id") or "") == str(answered_question.get("id") or ""):
        return False
    return True


def _looks_like_user_question(text: str) -> bool:
    raw = str(text or "").strip().lower()
    if not raw:
        return False
    if any(mark in raw for mark in ("?", "？", "什么", "为何", "为什么", "怎么", "如何", "是否", "能不能", "可以吗", "what", "why", "how", "which", "current")):
        return True
    return False


def _first_number_text(value: Any) -> str:
    match = re.search(r"[0-9]+(?:\.[0-9]+)?", str(value or ""))
    if not match:
        return ""
    number = float(match.group(0))
    return str(int(number)) if number.is_integer() else str(number)


def _looks_like_pasted_evidence(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("traceback", "runtimeerror", "error:", "exception", "agent>", "user>", "failed"))


def _looks_like_evidence_fragment(text: str, question: dict[str, Any]) -> bool:
    raw = str(text or "")
    lowered = raw.lower()
    question_id = str(question.get("id") or "")
    if question_id == "freeform_evidence":
        if _looks_like_pasted_evidence(raw):
            return True
        return bool(
            raw.startswith((" ", "\t"))
            or re.search(r'\bfile\s+"[^"]+",\s+line\s+\d+', lowered)
            or re.search(r"\b(?:caused by|during handling|traceback|stack trace)\b", lowered)
        )
    return any(
        marker in lowered
        for marker in (
            "curl ",
            "--data",
            "--header",
            "jsonrpc",
            '"method"',
            "'method'",
            '"params"',
            "'params'",
            "response:",
            "result",
            "http://",
            "https://",
        )
    )


def _looks_like_assignment_answer(text: str) -> bool:
    raw = _strip_scalar(text)
    if not raw or "\n" in raw:
        return False
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        return False
    return all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.:/-]*\s*=\s*[^=,]+", part) for part in parts)


def _chain_confirmed(state: AgentGraphState) -> bool:
    return (state.get("chain_identity") or {}).get("status") == "confirmed"


def _invalidate_for_chain_change(state: AgentGraphState) -> None:
    for key in ("rpc_mode", "workload", "custom_rpc", "endpoint_evidence", "fixture_evidence", "preflight", "smoke"):
        state[key] = {} if key != "rpc_mode" else ""
    state.setdefault("invalidated_groups", []).extend(["endpoint_process", "workload_rpc", "target_samples_fixtures", "preflight_smoke_execution"])


def _invalidate_for_target_mode(state: AgentGraphState) -> None:
    state.setdefault("invalidated_groups", []).extend(["endpoint_process", "workload_rpc", "qps_profile", "preflight_smoke_execution"])
    if state.get("workflow_mode") == "sync_observe":
        state["rpc_mode"] = ""
        state["workload"] = {}
        state["custom_rpc"] = {}
        state["fixture_evidence"] = {}


def _strip_scalar(value: str) -> str:
    text = str(value or "").strip()
    if len(text) >= 2 and text[0] in {"`", "'", '"'} and text[-1] in {"`", "'", '"'}:
        text = text[1:-1].strip()
    while text and text[-1] in {",", "，", ";", "；", "、"}:
        text = text[:-1].strip()
    return text


def _is_plain_scalar_answer(value: str) -> bool:
    text = str(value or "").strip()
    if not text or "\n" in text or "\r" in text:
        return False
    if len(text) > 180:
        return False
    if any(char in text for char in "。？！?"):
        return False
    if len(text.split()) > 1:
        return False
    return True


def _is_single_turn_value(value: str) -> bool:
    text = str(value or "").strip()
    if not text or "\n" in text or "\r" in text:
        return False
    return len(text) <= 4000


def _looks_like_url_value(value: str) -> bool:
    text = str(value or "").strip()
    return bool(
        re.match(r"^(https?|wss?)://\S+$", text, re.IGNORECASE)
        or re.match(r"^(localhost|127\.0\.0\.1|\[[0-9a-fA-F:]+\]|[A-Za-z0-9.-]+):[0-9]{2,5}(?:/.*)?$", text)
    )


def _is_bare_endpoint_answer(value: str) -> bool:
    text = str(value or "").strip()
    endpoint = _extract_url_candidate(text)
    return bool(endpoint) and text == endpoint


def _extract_url_candidate(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if _looks_like_url_value(text):
        return text
    match = re.search(r"\b(?:https?|wss?)://[^\s'\"`，。；;]+", text, flags=re.IGNORECASE)
    if match:
        return match.group(0).rstrip(".,;，。；")
    match = re.search(r"\b(?:localhost|127\.0\.0\.1|\[[0-9a-fA-F:]+\]|[A-Za-z0-9.-]+):[0-9]{2,5}(?:/[^\s'\"`，。；;]*)?", text)
    if match:
        return match.group(0).rstrip(".,;，。；")
    return ""


def _extract_json_object_or_array(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    for open_char, close_char in (("{", "}"), ("[", "]")):
        start = text.find(open_char)
        end = text.rfind(close_char)
        if start >= 0 and end > start:
            candidate = text[start : end + 1]
            try:
                json.loads(candidate)
            except json.JSONDecodeError:
                continue
            return candidate
    return ""


def _adapter_family_hint_from_text(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    if re.search(r"\b(evm|json[-_ ]?rpc|ethereum[-_ ]?compatible|eth_[a-z0-9_]+)\b", text):
        return "jsonrpc"
    if "substrate" in text or "polkadot" in text:
        return "substrate"
    if re.search(r"\b(rest|http api|rest api)\b", text):
        return "rest"
    if "tendermint" in text or "cosmos sdk" in text or "cometbft" in text:
        return "tendermint"
    if "bitcoin_jsonrpc" in text or "bitcoin json-rpc" in text or "bitcoin jsonrpc" in text:
        return "bitcoin_jsonrpc"
    if "hedera" in text:
        return "hedera_dual"
    return ""


def _looks_like_rest_path_or_doc_method(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if re.match(r"^(GET|POST|PUT|PATCH|DELETE)\s+/", text, re.IGNORECASE):
        return True
    if text.startswith("/"):
        return True
    lowered = text.lower()
    return any(token in lowered for token in (" path parameter", "query parameter", "get blocks by", "post ", "rest api"))


def _is_plain_chain_answer(value: str) -> bool:
    text = str(value or "").strip()
    if not text or "\n" in text or "\r" in text or len(text) > 80:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._ -]*", text))


def _normalized_target_mode(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    aliases = {
        "fake": "fake-node",
        "fake-node": "fake-node",
        "fakenode": "fake-node",
        "real": "real-node",
        "real-node": "real-node",
        "realnode": "real-node",
        "sync": "sync-observe",
        "sync-observe": "sync-observe",
        "syncobserve": "sync-observe",
    }
    return aliases.get(text, "")


def _target_mode_is_explicit_in_text(text: str, target_mode: str, action: dict[str, Any]) -> bool:
    if not target_mode:
        return False
    if action.get("target_mode_explicit") is not True:
        return False
    raw = str(text or "").strip().lower()
    if _text_is_mode_consultation(raw):
        return False
    if target_mode == "fake-node":
        evidence = ("fake-node", "fake node", "fakenode", "mock node", "mock-node", "模拟节点", "假节点")
    elif target_mode == "real-node":
        evidence = ("real-node", "real node", "realnode", "真实节点", "实际节点")
    elif target_mode == "sync-observe":
        evidence = ("sync-observe", "sync observe", "observe sync", "observe block", "block sync", "block import", "import observation", "观察同步", "节点同步", "同步观察", "观察追块", "追块")
    else:
        evidence = ()
    return any(item in raw for item in evidence)


def _text_is_mode_consultation(raw: str) -> bool:
    if not raw:
        return False
    choice_markers = (
        "let's",
        "let us",
        "i want to use",
        "i'll use",
        "i will use",
        "use ",
        "start ",
        "run ",
        "choose ",
        "select ",
        "switch to",
        "切换到",
        "换成",
        "选择",
        "我选",
        "我要用",
        "使用",
        "启动",
        "开始",
    )
    if any(marker in raw for marker in choice_markers):
        return False
    consultation_markers = (
        "?",
        "？",
        "why",
        "what",
        "which",
        "difference",
        "can ",
        "can't",
        "cannot",
        "does ",
        "do i",
        "should i",
        "what is",
        "how",
        "为什么",
        "是什么",
        "区别",
        "吗",
        "么",
        "能不能",
        "可以",
        "有什么用",
        "什么意义",
        "有意义",
        "哪个",
        "怎么",
    )
    return any(marker in raw for marker in consultation_markers)


def _qps_mode_is_explicit_in_text(text: str, qps_mode: str) -> bool:
    raw = str(text or "").strip().lower()
    return qps_mode in raw


def _chain_adapter_family(state: AgentGraphState) -> str:
    family = str((state.get("chain_identity") or {}).get("adapter_family") or "").strip().lower()
    if family:
        return family
    chain = str((state.get("chain_identity") or {}).get("canonical") or "").strip().lower()
    for row in load_framework_capabilities().get("chains", []):
        if str(row.get("chain") or "").strip().lower() == chain:
            return str(row.get("family") or row.get("adapter_family") or "").strip().lower()
    return ""


def _parse_json_params(value: str) -> Any | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, (list, dict)):
        return parsed
    return None


def _parse_rpc_params_or_request(value: str) -> tuple[str, Any | None]:
    text = str(value or "").strip()
    if not text:
        return "", None
    if _declares_no_rpc_params(text):
        return "", []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
        for candidate in _json_objects_or_arrays_in_text(text):
            if isinstance(candidate, dict) and "params" in candidate and ("method" in candidate or "jsonrpc" in candidate):
                parsed = candidate
                break
            if isinstance(candidate, list):
                parsed = candidate
                break
        if parsed is None:
            return "", None
    if isinstance(parsed, dict) and "params" in parsed and ("method" in parsed or "jsonrpc" in parsed):
        params = parsed.get("params")
        return str(parsed.get("method") or "").strip(), params if isinstance(params, (list, dict)) else None
    if isinstance(parsed, list):
        return "", parsed
    if isinstance(parsed, dict) and set(parsed).issubset({"params", "arguments", "args"}):
        params = parsed.get("params", parsed.get("arguments", parsed.get("args")))
        return "", params if isinstance(params, (list, dict)) else None
    return "", None


def _schema_evidence_from_turn_text(value: str, *, method_hint: str = "") -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    parsed_method, parsed = _parse_rpc_params_or_request(text)
    if parsed is not None:
        if parsed_method or not method_hint:
            return text
        return json.dumps({"jsonrpc": "2.0", "id": 1, "method": method_hint, "params": parsed}, ensure_ascii=False)
    if _declares_no_rpc_params(text) and method_hint:
        return json.dumps({"jsonrpc": "2.0", "id": 1, "method": method_hint, "params": []}, ensure_ascii=False)
    return ""


def _declares_no_rpc_params(value: str) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    return any(
        marker in text
        for marker in (
            "no parameters",
            "no params",
            "without parameters",
            "without params",
            "params: none",
            "params none",
            "没有参数",
            "无参数",
            "不需要参数",
            "参数为空",
        )
    )


def _is_direct_json_rpc_input(value: str) -> bool:
    text = str(value or "").strip()
    if not text or text[0] not in "[{":
        return False
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, (dict, list))


def _json_objects_or_arrays_in_text(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    output: list[Any] = []
    raw = str(text or "")
    for index, char in enumerate(raw):
        if char not in "{[":
            continue
        try:
            parsed, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, (dict, list)):
            output.append(parsed)
    return output


def _jsonrpc_schema_draft(method: str, params: Any) -> dict[str, Any]:
    return {
        "status": "draft",
        "evidence_kind": "jsonrpc_request",
        "transport": "jsonrpc",
        "method": str(method or "").strip(),
        "params": params if isinstance(params, list) else [],
        "params_json": params,
        "response_summary": "",
        "confidence": "high",
    }


def _schema_draft_has_params(draft: dict[str, Any]) -> bool:
    if not isinstance(draft, dict):
        return False
    if draft.get("status") == "failed":
        return False
    if "params_json" in draft and isinstance(draft.get("params_json"), (list, dict)):
        return True
    params = draft.get("params")
    return isinstance(params, list)


def _schema_params_from_draft(draft: dict[str, Any]) -> Any:
    if isinstance(draft.get("params_json"), (list, dict)):
        return draft["params_json"]
    params = draft.get("params")
    if not isinstance(params, list):
        return []
    output = []
    for item in params:
        if isinstance(item, dict) and "example" in item:
            output.append(item.get("example"))
        else:
            output.append(item)
    return output


def _schema_transport(draft: dict[str, Any]) -> str:
    return str(draft.get("transport") or draft.get("protocol") or "").strip().lower()


def _schema_indicates_rest(draft: dict[str, Any]) -> bool:
    transport = _schema_transport(draft)
    if transport in {"rest", "http"}:
        return True
    if str(draft.get("rest_path") or "").strip():
        return True
    if _looks_like_url_value(str(draft.get("endpoint_url") or "")):
        return True
    method = str(draft.get("method") or "").strip().upper()
    if method.startswith(("GET ", "POST ", "PUT ", "PATCH ", "DELETE ")):
        return True
    evidence_kind = str(draft.get("evidence_kind") or "").strip().lower()
    return evidence_kind in {"rest_endpoint", "rest_path"}


def _schema_indicates_jsonrpc(draft: dict[str, Any]) -> bool:
    transport = _schema_transport(draft)
    if transport == "jsonrpc":
        return True
    method = str(draft.get("method") or "").strip()
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+$", method))


def _new_chain_schema_conflict(state: AgentGraphState, draft: dict[str, Any]) -> AgentGraphState | None:
    adapter_family = _chain_adapter_family(state)
    if adapter_family == "jsonrpc" and _schema_indicates_rest(draft):
        identity = state.setdefault("chain_identity", {})
        identity["status"] = "needs_protocol_confirmation"
        identity["adapter_family"] = ""
        state.setdefault("endpoint_evidence", {})["candidate_endpoint_ready"] = False
        state["pending_question"] = _protocol_family_question(state)
        state["visible_response"] = [
            _localized(
                state.get("language", "en"),
                "你刚提供的证据更像 REST API（REST path、URL 或 REST 文档片段），但当前选择的是 `jsonrpc / EVM`。请先重新确认协议族；如果确认是 REST，我会重新验证 REST endpoint 和 method schema。",
                "The evidence looks like a REST API (REST path, URL, or REST docs excerpt), but the current adapter family is `jsonrpc / EVM`. Confirm the adapter family first; if it is REST, I will re-validate the REST endpoint and method schema.",
            ),
            _render_question(state["pending_question"], state.get("language", "en")),
        ]
        return state
    if adapter_family == "rest" and _schema_indicates_jsonrpc(draft):
        identity = state.setdefault("chain_identity", {})
        identity["status"] = "needs_protocol_confirmation"
        state["pending_question"] = _protocol_family_question(state)
        state["visible_response"] = [
            _localized(
                state.get("language", "en"),
                "你刚提供的证据更像 JSON-RPC request/method，但当前选择的是 `rest`。请先重新确认协议族。",
                "The evidence looks like a JSON-RPC request/method, but the current adapter family is `rest`. Confirm the adapter family first.",
            ),
            _render_question(state["pending_question"], state.get("language", "en")),
        ]
        return state
    return None


def _custom_rpc_schema_conflict(state: AgentGraphState, draft: dict[str, Any]) -> AgentGraphState | None:
    adapter_family = _chain_adapter_family(state)
    if adapter_family == "jsonrpc" and _schema_indicates_rest(draft):
        custom = state.setdefault("custom_rpc", {})
        custom["status"] = "needs_schema_evidence"
        state["pending_question"] = {}
        state["visible_response"] = [_localized(
            state.get("language", "en"),
            "你提供的证据更像 REST API，但当前已确认链的 adapter family 是 `jsonrpc`。请提供 JSON-RPC method/request/response 证据；如果你要测试 REST 协议链，请先切换链或进入新链 onboarding。",
            "The evidence looks like a REST API, but the confirmed chain adapter family is `jsonrpc`. Provide JSON-RPC method/request/response evidence; if you need a REST protocol chain, switch chain or enter new-chain onboarding first.",
        )]
        return state
    if adapter_family == "rest" and _schema_indicates_jsonrpc(draft):
        custom = state.setdefault("custom_rpc", {})
        custom["status"] = "needs_schema_evidence"
        state["pending_question"] = {}
        state["visible_response"] = [_localized(
            state.get("language", "en"),
            "你提供的证据更像 JSON-RPC，但当前链 adapter family 是 `rest`。请提供 REST path/request/response 证据，或切换到 JSON-RPC/EVM 链。",
            "The evidence looks like JSON-RPC, but the current chain adapter family is `rest`. Provide REST path/request/response evidence, or switch to a JSON-RPC/EVM chain.",
        )]
        return state
    return None


def _validated_custom_methods(custom: dict[str, Any]) -> list[str]:
    methods: list[str] = []
    for item in custom.get("validated_methods") or []:
        if isinstance(item, dict) and item.get("method"):
            methods.append(str(item["method"]))
    if custom.get("method"):
        methods.append(str(custom["method"]))
    return list(dict.fromkeys(method for method in methods if method))


def _validated_new_chain_methods(identity: dict[str, Any]) -> list[str]:
    methods: list[str] = []
    for item in identity.get("validated_methods") or []:
        if isinstance(item, dict) and item.get("method"):
            methods.append(str(item["method"]))
    if identity.get("candidate_method"):
        methods.append(str(identity["candidate_method"]))
    return list(dict.fromkeys(method for method in methods if method))


def _format_schema_confirmation_prompt(language: str, draft: dict[str, Any]) -> str:
    method = str(draft.get("method") or "<unknown>").strip()
    evidence_kind = str(draft.get("evidence_kind") or "unknown").strip()
    transport = str(draft.get("transport") or "unknown").strip()
    params = draft.get("params")
    if isinstance(params, list):
        param_lines = []
        for item in params:
            if isinstance(item, dict):
                name = item.get("name") or f"param[{item.get('index', '?')}]"
                param_lines.append(
                    f"- {name}: type={item.get('type') or 'unknown'}, meaning={item.get('meaning') or 'unknown'}, example={json.dumps(item.get('example'), ensure_ascii=False)}"
                )
            else:
                param_lines.append(f"- {json.dumps(item, ensure_ascii=False)}")
        params_text = "\n".join(param_lines) if param_lines else "- []"
    else:
        params_text = json.dumps(draft.get("params_json", []), ensure_ascii=False)
    response_summary = str(draft.get("response_summary") or "<unknown>").strip()
    confidence = str(draft.get("confidence") or "unknown").strip()
    conflicts = draft.get("conflicts") if isinstance(draft.get("conflicts"), list) else []
    conflicts_text = "; ".join(str(item) for item in conflicts) if conflicts else "<none>"
    return _localized(
        language,
        f"我从证据中提取到以下 RPC schema 草案，请确认是否正确。\nevidence: {evidence_kind} / transport={transport}\nmethod: `{method}`\nparams:\n{params_text}\nresponse: {response_summary}\nconfidence: {confidence}\nconflicts: {conflicts_text}\n回复 `Y` 确认并执行 endpoint probe；回复 `N` 重新提供证据。",
        f"I extracted this RPC schema draft from the evidence. Confirm whether it is correct.\nevidence: {evidence_kind} / transport={transport}\nmethod: `{method}`\nparams:\n{params_text}\nresponse: {response_summary}\nconfidence: {confidence}\nconflicts: {conflicts_text}\nReply `Y` to probe the endpoint with this schema, or `N` to provide corrected evidence.",
    )


def _sync_client_setup_prompt(state: AgentGraphState) -> str:
    language = state.get("language", "en")
    search_available = bool((state.get("web_research") or {}).get("google_search_available"))
    if search_available:
        return _localized(
            language,
            "可以进入真实节点客户端准备流程：我会优先查官方文档/镜像/metrics 启动参数，生成安装与启动计划；任何下载或启动命令都需要你确认后才执行。是否继续生成计划？",
            "I can enter real node client setup: I will use official docs/images/metrics flags to draft an install and run plan; any download or start command requires your confirmation. Generate the plan?",
        )
    return _localized(
        language,
        "当前模型没有 google_search 能力，不能可靠查找官方客户端下载地址或 metrics 参数。请提供官方安装文档、Docker image、二进制下载地址，或改为提供已运行节点的 RPC/metrics endpoint。是否先生成需要你提供资料的清单？",
        "The current model has no google_search capability, so it cannot reliably discover official client downloads or metrics flags. Provide official install docs, Docker image, binary URL, or an existing RPC/metrics endpoint. Generate the required-information checklist?",
    )


def _sync_client_setup_handoff_message(state: AgentGraphState) -> str:
    language = state.get("language", "en")
    chain = str((state.get("chain_identity") or {}).get("canonical") or (state.get("chain_identity") or {}).get("raw") or "<chain>")
    search_available = bool((state.get("web_research") or {}).get("google_search_available"))
    if search_available:
        return _localized(
            language,
            f"已进入 {chain} 真实节点客户端准备流程。下一步应使用 google_search 查询官方客户端、Docker image、metrics flags、启动参数和数据目录要求；生成计划后必须再次请求用户确认，不能静默下载或启动。",
            f"Entered {chain} real node client setup. Next, use google_search to verify official clients, Docker images, metrics flags, startup arguments, and data-dir requirements; ask for user approval before any download or start command.",
        )
    return _localized(
        language,
        f"已进入 {chain} 真实节点客户端准备流程，但当前没有 google_search。请提供官方安装文档、Docker image、二进制下载地址、metrics 开启方式，或直接提供已运行节点的 RPC/metrics endpoint；在此之前不会执行下载或启动。",
        f"Entered {chain} real node client setup, but google_search is unavailable. Provide official install docs, Docker image, binary URL, metrics flags, or an existing RPC/metrics endpoint; no download or start command will run until then.",
    )


def _validate_custom_rpc_schema(state: AgentGraphState, params: Any) -> AgentGraphState:
    language = state.get("language", "en")
    custom = state.setdefault("custom_rpc", {})
    custom["params"] = params
    draft = custom.get("schema_draft") if isinstance(custom.get("schema_draft"), dict) else {}
    draft_method = str(draft.get("method") or "").strip()
    if draft_method:
        custom["method"] = draft_method
    chain = (state.get("chain_identity") or {}).get("canonical") or ""
    endpoint = str(custom.get("endpoint") or "")
    method = str(custom.get("method") or "")
    result = validate_rpc_endpoint(
        chain=chain,
        endpoint=endpoint,
        methods=[method],
        adapter_family=(state.get("chain_identity") or {}).get("adapter_family") or "",
        method_params={method: params},
        timeout=3.0,
    )
    custom["method_probe"] = result
    state.setdefault("endpoint_evidence", {})["custom_rpc_method_probe"] = result
    if not result.get("ready"):
        custom["status"] = "needs_schema_evidence"
        state["pending_question"] = {}
        state["visible_response"] = [_localized(
            language,
            f"method/schema 验证失败：{result.get('error') or result.get('status')}。证据：{result.get('evidence_file') or '<none>'}。请修正 method、params 或证据。",
            f"Method/schema validation failed: {result.get('error') or result.get('status')}. Evidence: {result.get('evidence_file') or '<none>'}. Correct the method, params, or evidence.",
        )]
        return state
    custom.setdefault("validated_methods", []).append({
        "method": method,
        "params": params,
        "evidence_file": result.get("evidence_file") or "",
    })
    custom["status"] = "method_validated_next"
    if not isinstance(custom.get("inline_workload_hint"), dict):
        source_text = str(custom.get("source_turn_text") or state.get("last_user_input") or "")
        hint = _custom_rpc_inline_workload_hint(source_text, _validated_custom_methods(custom))
        if hint:
            custom["inline_workload_hint"] = hint
    state["visible_response"] = [_localized(
        language,
        f"method/schema 验证通过。证据：{result.get('evidence_file') or '<none>'}。验证 endpoint 仍只作为证据保存，最终测试 endpoint 会在 endpoint 配置组单独确认。",
        f"Method/schema validation passed. Evidence: {result.get('evidence_file') or '<none>'}. The validation endpoint remains evidence-only; the final benchmark endpoint will be confirmed separately in the endpoint configuration group.",
    )]
    if _apply_custom_rpc_inline_workload_hint(state):
        return state
    state["pending_question"] = _option_question(
        "endpoint_process",
        "custom_rpc_continue",
        _localized(
            language,
            "这个自定义 RPC method 已验证通过。下一步怎么处理？",
            "This custom RPC method has been validated. What should happen next?",
        ),
        [
            {"label": "继续添加另一个自定义 RPC method" if str(language or "").startswith("zh") else "Add another custom RPC method", "value": "add_another"},
            {"label": "当前 method 已够，继续配置 workload" if str(language or "").startswith("zh") else "This is enough; continue workload setup", "value": "finish"},
        ],
        field="custom_rpc_continue",
        kind="numbered_choice",
    )
    state["visible_response"] = list(state.get("visible_response") or []) + [_render_question(state["pending_question"], language)]
    return state


def _validate_new_chain_rpc_schema(state: AgentGraphState, params: Any) -> AgentGraphState:
    language = state.get("language", "en")
    identity = state.setdefault("chain_identity", {})
    identity["candidate_params"] = params
    draft = identity.get("schema_draft") if isinstance(identity.get("schema_draft"), dict) else {}
    draft_method = str(draft.get("method") or "").strip()
    if draft_method:
        identity["candidate_method"] = draft_method
    endpoint = str((state.get("endpoint_evidence") or {}).get("candidate_endpoint") or "")
    method = str(identity.get("candidate_method") or "")
    chain = str(identity.get("canonical") or identity.get("raw") or "").strip()
    result = validate_rpc_endpoint(
        chain=chain,
        endpoint=endpoint,
        methods=[method],
        adapter_family=_chain_adapter_family(state),
        method_params={method: params},
        timeout=3.0,
    )
    state.setdefault("endpoint_evidence", {})["new_chain_method_probe"] = result
    if not result.get("ready"):
        identity["status"] = "existing_family_needs_schema_evidence"
        state["pending_question"] = {}
        state["visible_response"] = [_localized(
            language,
            f"新链 method/schema 验证失败：{result.get('error') or result.get('status')}。证据：{result.get('evidence_file') or '<none>'}。请修正 method、params 或证据。",
            f"New-chain method/schema validation failed: {result.get('error') or result.get('status')}. Evidence: {result.get('evidence_file') or '<none>'}. Correct the method, params, or evidence.",
        )]
        return state
    identity.setdefault("validated_methods", []).append({
        "method": method,
        "params": params,
        "evidence_file": result.get("evidence_file") or "",
    })
    identity["status"] = "existing_family_method_validated_next"
    state["pending_question"] = _option_question(
        "endpoint_process",
        "new_chain_method_continue",
        _localized(
            language,
            "新链 RPC method 已验证通过。下一步怎么处理？",
            "The new-chain RPC method has been validated. What should happen next?",
        ),
        [
            {"label": "继续添加另一个 RPC method" if str(language or "").startswith("zh") else "Add another RPC method", "value": "add_another"},
            {"label": "当前 method 已够，继续后续流程" if str(language or "").startswith("zh") else "This is enough; continue the next workflow", "value": "finish"},
        ],
        field="new_chain_method_continue",
        kind="numbered_choice",
    )
    state["active_group"] = "endpoint_process"
    state["visible_response"] = [_localized(
        language,
        f"新链 endpoint 和 method/schema 已验证通过。证据：{result.get('evidence_file') or '<none>'}。",
        f"New-chain endpoint and method/schema validation passed. Evidence: {result.get('evidence_file') or '<none>'}.",
    ), _render_question(state["pending_question"], language)]
    return state


def _promote_new_chain_verified_endpoint_to_real_node(state: AgentGraphState) -> AgentGraphState:
    identity = state.setdefault("chain_identity", {})
    endpoint_evidence = state.setdefault("endpoint_evidence", {})
    confirmed = state.setdefault("confirmed_config", {})
    chain = str(identity.get("canonical") or identity.get("raw") or "").strip()
    method = str(identity.get("candidate_method") or "").strip()
    endpoint = str(endpoint_evidence.get("candidate_endpoint") or "").strip()
    identity["status"] = "confirmed"
    identity["case"] = "case2_runtime_override"
    confirmed["BLOCKCHAIN_NODE"] = chain
    if endpoint:
        confirmed["LOCAL_RPC_URL"] = endpoint
        endpoint_evidence["local_rpc_url_ready"] = True
    state["target_mode"] = "real-node"
    state["workflow_mode"] = "rpc_benchmark"
    if method and not state.get("workload", {}).get("confirmed"):
        state["rpc_mode"] = "single"
        state["workload"] = {
            "confirmed": True,
            "choice": "new_chain_verified_method",
            "methods": [method],
            "replace_defaults": True,
            "job_local_override": True,
        }
    state["preflight"] = {}
    state["smoke"] = {}
    state["pending_question"] = {}
    state["active_group"] = ""
    state["visible_response"] = [_localized(
        state.get("language", "en"),
        "已切换到 real-node 路径，并把已验证 endpoint/method 作为本次 job-local workload override 继续；不会修改 config/chains 原始模板。",
        "Switched to the real-node path and will continue with the verified endpoint/method as a job-local workload override; config/chains templates are not modified.",
    )]
    return state


def _new_chain_handoff_message(state: AgentGraphState) -> str:
    identity = state.get("chain_identity") or {}
    evidence = state.get("endpoint_evidence") or {}
    language = state.get("language", "en")
    return _localized(
        language,
        "已进入二次开发交接路径。需要基于已验证 endpoint/method/schema 生成 chain template、fake-node fixture、fixture coverage 和 smoke 验证任务；在这些 gate 通过前，不会声明该链已被生产支持。"
        f" 证据：endpoint={evidence.get('new_chain_endpoint_probe', {}).get('evidence_file') or '<none>'}, method={evidence.get('new_chain_method_probe', {}).get('evidence_file') or '<none>'}, chain={identity.get('canonical') or identity.get('raw') or '<unknown>'}。",
        "Entered the secondary-development handoff path. A chain template, fake-node fixture, fixture coverage, and smoke validation task must be generated from the verified endpoint/method/schema; this chain will not be claimed production-supported until those gates pass."
        f" Evidence: endpoint={evidence.get('new_chain_endpoint_probe', {}).get('evidence_file') or '<none>'}, method={evidence.get('new_chain_method_probe', {}).get('evidence_file') or '<none>'}, chain={identity.get('canonical') or identity.get('raw') or '<unknown>'}.",
    )


def _parse_weight_spec(value: str) -> dict[str, int]:
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        output = {}
        for key, item in parsed.items():
            method = str(key).strip()
            if method:
                try:
                    output[method] = int(item)
                except (TypeError, ValueError):
                    return {}
        return output
    output: dict[str, int] = {}
    pairs = re.findall(r"([A-Za-z][A-Za-z0-9_./:-]*)\s*=\s*([0-9]+)", text)
    if pairs:
        for method, weight in pairs:
            output[method.strip()] = int(weight)
        return output
    for chunk in re.split(r"[,，]\s*", text):
        if not chunk.strip():
            continue
        if "=" not in chunk:
            return {}
        method, weight = chunk.split("=", 1)
        method = method.strip()
        if not method:
            return {}
        try:
            output[method] = int(weight.strip())
        except ValueError:
            return {}
    return output


def _parse_weight_spec_for_methods(value: str, methods: list[str]) -> dict[str, int]:
    weights = _parse_weight_spec(value)
    if weights:
        return weights
    unique_methods = [method for method in dict.fromkeys(str(item).strip() for item in methods) if method]
    if len(unique_methods) != 1:
        return {}
    number = _single_method_weight_number_text(value)
    if not number:
        return {}
    try:
        return {unique_methods[0]: int(float(number))}
    except ValueError:
        return {}


def _single_method_weight_number_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", text):
        return text
    match = re.search(r"(?:weight|权重)\s*(?:is|为|=|:|：)?\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if match:
        return match.group(1)
    return ""


def _single_method_weight_example(methods: list[str]) -> str:
    unique_methods = [method for method in dict.fromkeys(str(item).strip() for item in methods) if method]
    if not unique_methods:
        return "method=100"
    if len(unique_methods) == 1:
        return f"{unique_methods[0]}=100"
    share = int(100 / len(unique_methods))
    return ",".join(f"{method}={share}" for method in unique_methods)


def _format_weights(weights: dict[str, int]) -> str:
    return ", ".join(f"{method}={weight}" for method, weight in weights.items())


def _enter_case_for_adapter_family(state: AgentGraphState, family: str) -> AgentGraphState:
    identity = state.setdefault("chain_identity", {})
    family = str(family or "").strip()
    identity["adapter_family"] = family
    if family not in SUPPORTED_ADAPTER_FAMILIES:
        identity["status"] = "unsupported_family_handoff"
        identity["case"] = "case3"
        state.setdefault("secondary_handoff", {})["status"] = "collecting_evidence"
        state["pending_question"] = {}
        state["visible_response"] = [_localized(
            state.get("language", "en"),
            "该链目前不属于已支持协议族。请提供官方协议/RPC 文档、endpoint 文档、request/response 示例；我会生成二次开发交接文档。",
            "This chain is outside the supported adapter families. Provide official protocol/RPC docs, endpoint docs, and request/response examples; I will generate a secondary-development handoff.",
        )]
        state["_stop_after_response"] = True
        return state
    identity["status"] = "existing_family_needs_endpoint"
    identity["case"] = "case2"
    identity["identity_confirmed"] = True
    state["active_group"] = "endpoint_process"
    state["pending_question"] = _manual_question(
        "endpoint_process",
        "new_chain_endpoint",
        _localized(state.get("language", "en"), "请提供可访问的 RPC endpoint，用于验证该新链和 RPC method。", "Provide a reachable RPC endpoint to validate this new chain and RPC methods."),
        kind="url",
    )
    state["visible_response"] = [_render_question(state["pending_question"], state.get("language", "en"))]
    if _consume_queued_new_chain_rpc_action(state):
        if state.get("pending_question") and state.get("action_queue"):
            state["pending_question"]["resume_action_queue"] = True
        return state
    return state


def _route_secondary_handoff_text(state: AgentGraphState, text: str) -> AgentGraphState | None:
    identity = state.get("chain_identity") or {}
    handoff = state.setdefault("secondary_handoff", {})
    if identity.get("status") != "unsupported_family_handoff" and handoff.get("status") != "collecting_evidence":
        return None
    if state.get("pending_question"):
        return None
    if _looks_like_handoff_navigation(text):
        return None
    stripped = str(text or "").strip()
    if not _looks_like_handoff_generation_request(stripped) and not _looks_like_handoff_evidence(stripped):
        return None
    evidence_items = handoff.setdefault("evidence", [])
    if stripped:
        evidence_items.append(stripped)
    handoff["status"] = "collecting_evidence"
    state["active_group"] = "chain_identity"
    state["pending_question"] = {}
    language = state.get("language", "en")
    if _looks_like_handoff_generation_request(stripped):
        state["visible_response"] = [_localized(
            language,
            _secondary_handoff_draft(state, evidence_items, language),
            _secondary_handoff_draft(state, evidence_items, language),
        )]
        state["_stop_after_response"] = True
        return state
    state["visible_response"] = [_localized(
        language,
        f"已记录第 {len(evidence_items)} 条二次开发证据。你可以继续粘贴官方协议/RPC 文档、endpoint 文档、request/response 示例；资料足够后说 `生成二次开发文档`。",
        f"Recorded secondary-development evidence item {len(evidence_items)}. Continue pasting official protocol/RPC docs, endpoint docs, and request/response samples; say `generate development handoff` when ready.",
    )]
    state["_stop_after_response"] = True
    return state


def _looks_like_handoff_generation_request(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    return any(token in lowered for token in ("生成", "交接", "handoff", "development doc", "coding doc"))


def _looks_like_handoff_navigation(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    return any(token in lowered for token in (
        "重新开始",
        "清空",
        "回到",
        "改回",
        "换成",
        "先别管",
        "最近",
        "报告",
        "日志",
        "switch to",
        "change to",
        "start over",
        "clear",
        "latest",
        "job",
        "report",
        "log",
        "status",
        "analyze",
    ))


def _looks_like_handoff_evidence(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return False
    evidence_tokens = (
        "protocol:",
        "rpc:",
        "rpc ",
        "transport:",
        "endpoint:",
        "endpoint ",
        "request:",
        "request ",
        "response:",
        "response ",
        "schema:",
        "method:",
        "params:",
        "curl ",
        "jsonrpc",
        "http",
        "docs",
        "documentation",
        "official",
        "文档",
        "官方",
        "协议",
        "请求示例",
        "响应示例",
        "参数",
    )
    return any(token in lowered for token in evidence_tokens)


def _secondary_handoff_draft(state: AgentGraphState, evidence_items: list[str], language: str) -> str:
    identity = state.get("chain_identity") or {}
    chain = str(identity.get("canonical") or identity.get("raw") or "<unknown>")
    evidence_preview = "\n".join(f"- {item}" for item in evidence_items[-5:]) or "- <none>"
    if str(language or "").startswith("zh"):
        return (
            f"二次开发交接草案：`{chain}` 当前不属于已支持协议族。\n"
            "目标：让另一个 AI 基于官方资料新增协议 adapter，并完成闭环验证。\n"
            "必须完成：\n"
            "1. 阅读并引用官方协议/RPC 文档，确认 transport、auth、request/response schema。\n"
            "2. 新增 adapter 或扩展现有 adapter，不得复用错误协议族。\n"
            "3. 新增 chain template，并定义可验证的 RPC method、参数、响应解析和错误处理。\n"
            "4. 基于真实 endpoint 录制 fixture，补充 fake-node 响应样本。\n"
            "5. 运行 schema validator、fixture coverage、preflight 和 smoke 测试。\n"
            "6. 更新中英文文档，确保 Agent 知识库和实际代码一致。\n"
            "已收集证据：\n"
            f"{evidence_preview}\n"
            "缺失资料请继续补充：官方文档 URL、可访问 endpoint、完整 request/response 示例、客户端/metrics 文档。"
        )
    return (
        f"Secondary-development handoff draft: `{chain}` is outside the supported adapter families.\n"
        "Goal: have another AI implement a protocol adapter from official materials and close the validation loop.\n"
        "Required work:\n"
        "1. Read and cite official protocol/RPC docs; confirm transport, auth, request/response schema.\n"
        "2. Add a new adapter or extend an existing one without reusing the wrong adapter family.\n"
        "3. Add a chain template with verifiable RPC methods, params, response parsing, and error handling.\n"
        "4. Record fixtures from a real endpoint and add fake-node response samples.\n"
        "5. Run schema validator, fixture coverage, preflight, and smoke tests.\n"
        "6. Update English/Chinese docs so Agent knowledge matches code behavior.\n"
        "Collected evidence:\n"
        f"{evidence_preview}\n"
        "Missing inputs to add next: official docs URL, reachable endpoint, full request/response samples, client/metrics docs."
    )


def _consume_queued_new_chain_rpc_action(state: AgentGraphState) -> bool:
    actions = list(state.get("action_queue") or [])
    action_index = next((idx for idx, item in enumerate(actions) if str(item.get("type") or "") == "start_custom_rpc"), None)
    if action_index is None:
        return False
    action = dict(actions.pop(action_index))
    state["action_queue"] = actions
    origin_text = str(action.get("_origin_text") or state.get("last_user_input") or "")
    endpoint = _extract_url_candidate(str(action.get("rpc_endpoint") or ""))
    method = _strip_scalar(str(action.get("rpc_method") or ""))
    evidence = str(action.get("rpc_schema_evidence") or "").strip()
    if not evidence:
        evidence = _schema_evidence_from_turn_text(origin_text, method_hint=method)
    changed = False
    if endpoint:
        state = _apply_endpoint_answer(
            state,
            endpoint,
            _manual_question("endpoint_process", "new_chain_endpoint", "", kind="url"),
        )
        changed = True
        if (state.get("chain_identity") or {}).get("status") == "existing_family_needs_endpoint":
            return changed
    if method and (state.get("chain_identity") or {}).get("status") == "existing_family_needs_method":
        state = _apply_endpoint_answer(
            state,
            method,
            _manual_question("endpoint_process", "new_chain_method", "", kind="manual_value"),
        )
        changed = True
    if evidence and (state.get("chain_identity") or {}).get("status") in {"existing_family_needs_schema_evidence", "existing_family_schema_needs_confirmation"}:
        if (state.get("chain_identity") or {}).get("status") == "existing_family_schema_needs_confirmation":
            return changed
        state = _apply_endpoint_answer(
            state,
            evidence,
            _manual_question("endpoint_process", "new_chain_schema_evidence", "", kind="evidence"),
        )
        changed = True
    if changed and state.get("pending_question") and state.get("action_queue"):
        state["pending_question"]["resume_action_queue"] = True
    return changed


def _opening_consultation_response(state: AgentGraphState, action: dict[str, Any], text: str) -> str:
    language = state.get("language", "en")
    topic = str(action.get("topic") or "").strip().lower()
    subject = str(action.get("subject") or "").strip()
    if topic in {"", "support"}:
        topic = "agent_capabilities"
    if topic == "capability":
        topic = "agent_capabilities"
    if topic in {"who", "who_are_you", "identity", "origin", "purpose", "destination", "direction"}:
        topic = "identity"
    if topic in {"what_can_you_do", "agent_capability", "agent_capabilities"}:
        topic = "agent_capabilities"
    if topic in {"capabilities", "supported", "supported_chains", "chains", "rpc_methods", "templates"}:
        topic = "supported_chains"
    if topic in {"current", "state", "settings", "current_settings"}:
        topic = "current_config"
    if topic in {"this", "current_prompt", "pending", "pending_question"}:
        topic = "current_context"
    if topic in {"reset", "restart", "start_over", "clear", "reset_help"}:
        topic = "reset_help"
    if topic in {"prepare", "prerequisites", "checklist"}:
        topic = "requirements"
    if topic in {"modes", "mode"}:
        topic = "mode_comparison"
    if topic in {"fake_node", "fake_node_usefulness", "fake-node", "fake-node-usefulness"}:
        topic = "mode_comparison"
    if topic in {"performance", "qps", "bottleneck", "throughput", "capacity", "performance_benchmark"}:
        topic = "performance_benchmark_guidance"
    if topic in {"confusion", "misunderstanding"}:
        topic = "correction"

    if topic == "identity":
        return _agent_identity_response(state)
    if topic in {"evidence_help", "log_help", "logs_help"}:
        return _evidence_help_response(state)
    if topic == "agent_capabilities":
        return _agent_capabilities_response(state)
    if topic == "supported_chains":
        return _framework_capability_summary(state)
    if topic == "current_context":
        return format_current_context(state, language)
    if topic == "current_config":
        return format_current_state(state, language)
    if topic == "reset_help":
        return _localized(
            language,
            (
                "可以。你有两种方式：\n"
                "1. 如果要完全重新开始，直接说“清空配置重新开始”，我会丢弃上次 Agent 配置并重新选择测试目标。\n"
                "2. 如果只想改一部分，直接说要改哪一组，例如“换一条链”、“改成 real-node”、“重新配置 QPS”或“修改磁盘”。\n"
                "如果你想重新测试别的链，请直接输入新链名；如果还没决定链名，我会先回到链选择。"
            ),
            (
                "Yes. You have two options:\n"
                "1. To restart completely, say 'clear the config and start over'; I will discard the previous Agent configuration and choose a new target.\n"
                "2. To change only one part, name the group, such as 'change chain', 'switch to real-node', 'reconfigure QPS', or 'change disk'.\n"
                "If you want to test another chain, type the new chain name; if you have not decided, I will return to chain selection."
            ),
        )
    if topic == "requirements":
        return _localized(
            language,
            (
                "如果你要开始一次测试，我会先做只读环境检查，然后逐组帮你确认这些信息：\n"
                "1. 测试类型：fake-node、real-node 或 sync-observe。\n"
                "2. 链和协议：已支持链直接选择；新链会先确认是否属于现有协议族。\n"
                "3. endpoint：fake-node 不需要真实被测节点；real-node 需要 LOCAL_RPC_URL；sync-observe 需要真实节点或可观测 endpoint。\n"
                "4. 环境元数据：CLOUD_REGION、CLOUD_ZONE、MACHINE_TYPE。\n"
                "5. 硬件：Ledger/data 磁盘、可选 accounts/state 磁盘、IOPS、吞吐、网络接口和带宽。\n"
                "6. workload：single/mixed、默认或自定义 RPC method、权重和样本/fixtures。\n"
                "7. QPS profile：quick、standard、intensive，以及是否调整具体参数。\n"
                "8. 可观测性：禁用、本地 Prometheus/Grafana，或只启动 exporter。\n"
                "最后会先做 preflight/smoke，确认通过后再运行正式测试。你可以直接说要测试的链和模式，我会缺什么问什么。"
            ),
            (
                "To start a test, I first run read-only diagnostics, then confirm these groups:\n"
                "1. Test type: fake-node, real-node, or sync-observe.\n"
                "2. Chain/protocol: supported chains are selected directly; new chains go through protocol confirmation.\n"
                "3. Endpoint: fake-node needs no real node; real-node needs LOCAL_RPC_URL; sync-observe needs a real node or observable endpoint.\n"
                "4. Environment metadata: CLOUD_REGION, CLOUD_ZONE, MACHINE_TYPE.\n"
                "5. Hardware: ledger/data disk, optional accounts/state disk, IOPS, throughput, network interface, and bandwidth.\n"
                "6. Workload: single/mixed, default or custom RPC methods, weights, samples/fixtures.\n"
                "7. QPS profile: quick, standard, intensive, and optional parameter changes.\n"
                "8. Observability: disabled, local Prometheus/Grafana, or exporter only.\n"
                "Then I run preflight/smoke before the final benchmark. You can state the chain and mode; I will ask only for missing pieces."
            ),
        )
    if topic == "workflow":
        return _localized(
            language,
            (
                "默认流程是：启动检查 -> 选择测试类型 -> 确认链/协议 -> 补齐环境和硬件信息 -> 配置 endpoint/process -> 配置 RPC workload -> 配置 QPS -> 配置可观测性 -> preflight/smoke -> 运行并跟踪日志/报告。"
                "你可以随时跳转、回退或修改某一组配置；完成后我会重新计算下一个缺失项，而不是从头开始。"
            ),
            (
                "Default flow: startup diagnostics -> choose test type -> confirm chain/protocol -> complete environment and hardware -> configure endpoint/process -> configure RPC workload -> configure QPS -> configure observability -> preflight/smoke -> run and follow logs/reports. "
                "You can jump, go back, or change any group; after that I recompute the next missing item instead of restarting."
            ),
        )
    if topic == "mode_comparison":
        return _localized(
            language,
            (
                "三种模式的区别：\n"
                "1. fake-node：使用预录 fixtures 验证框架闭环，不测真实节点性能。\n"
                "2. real-node：对真实 LOCAL_RPC_URL 做 RPC 压测，需要真实 endpoint 和节点进程信息。\n"
                "3. sync-observe：观察真实节点追块/同步状态、CPU、内存、磁盘、网络和可用的 MGas/s 指标，不走 vegeta 压测。\n"
                "所以 fake-node 的作用是低风险验证 Agent、配置、fixtures、执行、日志和 HTML 报告是否能闭环；它不能回答真实节点 QPS、同步速度或硬件瓶颈。\n"
                "如果目标是“能支持多少 QPS、瓶颈在哪里”，应使用 real-node benchmark；如果目标是“节点追块/import 过程表现”，才使用 sync-observe。"
            ),
            (
                "Mode differences:\n"
                "1. fake-node: uses recorded fixtures to validate the framework loop; it does not measure real node performance.\n"
                "2. real-node: runs RPC load tests against a real LOCAL_RPC_URL and needs endpoint/process details.\n"
                "3. sync-observe: observes real node sync progress, CPU, memory, disk, network, and available MGas/s metrics; it does not run vegeta load tests.\n"
                "So fake-node is useful for low-risk validation of the Agent, config, fixtures, execution, logs, and HTML report loop; it cannot answer real-node QPS, sync speed, or hardware bottlenecks.\n"
                "If the goal is QPS capacity or bottleneck discovery, use real-node benchmark; if the goal is sync/import observation, use sync-observe."
            ),
        )
    if topic == "performance_benchmark_guidance":
        return _localized(
            language,
            (
                "如果目标是观察一条链能支持多少 QPS、延迟如何变化、瓶颈在哪里，应该走 real-node benchmark：提供真实 `LOCAL_RPC_URL`，再配置 RPC workload、QPS profile、节点进程名和可观测性。\n"
                "fake-node 只能验证框架闭环和报告链路，不代表真实节点性能。\n"
                "sync-observe 观察真实节点追块/import、CPU、磁盘、网络和可用的 MGas/s，不会执行 vegeta QPS 压测。\n"
                "所以你刚才说的“支持多少 QPS、性能瓶颈”更适合 real-node benchmark。"
            ),
            (
                "If the goal is QPS capacity, latency behavior, and bottleneck discovery, use real-node benchmark: provide a real `LOCAL_RPC_URL`, then configure RPC workload, QPS profile, process name, and observability.\n"
                "fake-node only validates the framework/report loop and does not represent real node performance.\n"
                "sync-observe watches real node sync/import behavior, CPU, disk, network, and available MGas/s; it does not run vegeta QPS load tests.\n"
                "So a request for QPS capacity and bottlenecks belongs to real-node benchmark."
            ),
        )
    if topic == "recommendation":
        return _localized(
            language,
            (
                "如果你只是想快速确认 Agent 和框架能不能跑通，我建议先用 fake-node smoke：不需要真实节点，风险最低，能验证配置、fixtures、执行、日志和报告闭环。"
                "你可以选择一条已支持链，例如 `solana`、`ethereum` 或 `bsc`；如果你也不确定链，就先用 `solana` 做闭环验证。"
                "下一步你可以回复：`使用 solana fake-node`，或直接告诉我你想测的链。"
            ),
            (
                "If you only want to quickly verify that the Agent and framework run end to end, I recommend a fake-node smoke first: no real node is required, and it validates config, fixtures, execution, logs, and report generation with the lowest risk. "
                "Choose a supported chain such as `solana`, `ethereum`, or `bsc`; if you are unsure, use `solana` for the first closed-loop check. "
                "Next, reply `use solana fake-node`, or tell me the chain you want to test."
            ),
        )
    if topic == "recommend_start":
        return _opening_consultation_response(state, {**action, "topic": "recommendation"}, text)
    if topic == "extension":
        return _localized(
            language,
            (
                "扩展分三类：\n"
                "1. 已支持链 + 自定义 RPC method：需要 endpoint、request/response 或官方文档；我会抽取 schema、探测 endpoint、确认权重，再进入 smoke。\n"
                "2. 新链但属于现有协议族：先确认协议族，再用可访问 endpoint 和 method 证据生成运行时模板/fixtures，通过 smoke 后继续配置测试。\n"
                "3. 新链且不属于现有协议族：收集官方协议/RPC 文档和示例，生成二次开发交接文档，不会假装已经支持。"
            ),
            (
                "Extension paths have three cases:\n"
                "1. Supported chain + custom RPC method: provide endpoint, request/response, or official docs; I extract schema, probe endpoint, confirm weights, then smoke-test.\n"
                "2. New chain in an existing adapter family: confirm the family, validate endpoint/method evidence, create runtime template/fixtures, then continue after smoke.\n"
                "3. New chain outside supported families: collect official protocol/RPC docs and examples, then generate a secondary-development handoff instead of pretending support exists."
            ),
        )
    if topic == "config_explanation":
        label = subject or _localized(language, "这个配置项", "this config item")
        if not subject and state.get("pending_question"):
            return format_current_context(state, language)
        return _localized(
            language,
            f"{label} 用于生成测试元数据、资源归因或压测参数。如果你贴出当前问题或配置片段，我可以按字段解释它为什么需要、能否推断，以及是否必须手动确认。",
            f"{label} is used for test metadata, resource attribution, or benchmark parameters. Paste the current question or config snippet and I can explain why it is needed, whether it can be inferred, and whether it must be manually confirmed.",
        )
    if topic == "correction":
        return _localized(
            language,
            (
                "我理解，你不是只想看支持链列表，而是在问测试前需要准备什么、流程怎么走，或者我刚才是否答偏了。"
                "如果你的问题是“需要做什么/提供什么”，答案是：先选测试类型和链，然后确认 endpoint、环境元数据、磁盘/网络、RPC workload、QPS 和可观测性，最后做 preflight/smoke。"
                "你也可以直接说“开始 fake-node 测试”或“我有真实 endpoint”，我会按缺失项继续问。"
            ),
            (
                "I understand: you are not only asking for the supported-chain list; you are asking what you need to prepare, how the flow works, or whether my previous answer missed the point. "
                "If your question is what to do/provide, the answer is: choose test type and chain, then confirm endpoint, environment metadata, disk/network, RPC workload, QPS, and observability, followed by preflight/smoke. "
                "You can also say 'start fake-node' or 'I have a real endpoint' and I will ask for the missing pieces."
            ),
        )
    return _localized(
        language,
        "我可以回答框架能力、测试准备清单、执行流程、模式区别、配置项含义、扩展方式、错误日志和报告分析。你可以直接问具体问题，也可以说要开始测试。",
        "I can answer capabilities, benchmark requirements, workflow, mode differences, config-field meaning, extension paths, error logs, and report analysis. Ask a specific question or tell me to start a test.",
    )


def _agent_identity_response(state: AgentGraphState) -> str:
    language = state.get("language", "en")
    return _localized(
        language,
        (
            "我是 AnyChain Benchmark Agent，运行在你当前的 AnyChain Benchmark 工程环境里。"
            "我的职责是帮你完成区块链节点性能测试的配置、校验、执行和报告分析：从环境/磁盘/网络推断，到 fake-node 闭环验证、real-node RPC 压测、sync-observe 同步观察，再到错误日志和报告解读。"
            "你可以直接告诉我要测试哪条链、使用哪种模式，或者把配置片段/错误日志贴给我。"
        ),
        (
            "I am AnyChain Benchmark Agent, running inside your current AnyChain Benchmark project environment. "
            "My job is to help configure, validate, run, and analyze blockchain node benchmarks: environment/disk/network inference, fake-node closed-loop validation, real-node RPC load testing, sync-observe node-sync observation, error-log analysis, and report interpretation. "
            "You can tell me which chain and mode to use, or paste config snippets or error logs."
        ),
    )


def _agent_capabilities_response(state: AgentGraphState) -> str:
    language = state.get("language", "en")
    framework = state.get("framework_summary") or {}
    return _localized(
        language,
        (
            "我可以帮你做这些事：\n"
            "1. 引导配置 fake-node、real-node 或 sync-observe 测试。\n"
            "2. 根据当前环境推断并确认云区域、机器、磁盘、网络、QPS、RPC workload 和可观测性配置。\n"
            "3. 对真实 endpoint 做校验，帮助配置自定义 RPC method、权重和 fixtures。\n"
            "4. 对新链判断是否属于现有协议族；能支持就进入模板/endpoint/RPC 验证，不能支持就生成二次开发交接文档。\n"
            "5. 执行 preflight/smoke、跟踪 job、解释日志、报告和性能数据。\n"
            f"当前已加载 {framework.get('chain_count', '?')} 条链、{framework.get('family_count', '?')} 个协议族、{framework.get('unique_rpc_method_count', '?')} 个 RPC method。"
        ),
        (
            "I can help with:\n"
            "1. Guiding fake-node, real-node, or sync-observe tests.\n"
            "2. Inferring and confirming cloud region, machine, disk, network, QPS, RPC workload, and observability settings.\n"
            "3. Validating real endpoints and configuring custom RPC methods, weights, and fixtures.\n"
            "4. Classifying new chains against supported adapter families; supported-family chains enter template/endpoint/RPC validation, unsupported chains produce a secondary-development handoff.\n"
            "5. Running preflight/smoke, following jobs, and explaining logs, reports, and performance data.\n"
            f"Currently loaded: {framework.get('chain_count', '?')} chains, {framework.get('family_count', '?')} adapter families, and {framework.get('unique_rpc_method_count', '?')} RPC methods."
        ),
    )


def _evidence_help_response(state: AgentGraphState) -> str:
    language = state.get("language", "en")
    return _localized(
        language,
        (
            "可以。请把真实日志、错误栈、命令输出或 `logs <job_id>` 的内容粘贴给我；如果内容很多，可以分多行粘贴，最后输入 `END`。"
            "如果你想分析最近一次任务，也可以直接输入 `logs` 查看日志路径，或说“分析最近 job”。"
        ),
        (
            "Yes. Paste the actual log lines, stack trace, command output, or `logs <job_id>` content. For long content, paste multiple lines and type `END` when done. "
            "If you want to analyze the latest job, type `logs` to see the log path, or ask me to analyze the latest job."
        ),
    )


def _analyze_saved_evidence(state: AgentGraphState, text: str) -> AgentGraphState:
    language = state.get("language", "en")
    evidence_items = list(state.get("evidence_buffer") or [])
    latest = str((evidence_items[-1] if evidence_items else {}).get("text") or "").strip()
    state["active_group"] = "error_evidence_analysis"
    state["pending_question"] = {}
    state["visible_response"] = [_saved_evidence_analysis_response(language, latest)]
    state["_stop_after_response"] = True
    return state


def _saved_evidence_analysis_response(language: str, evidence: str) -> str:
    lowered = evidence.lower()
    findings: list[str] = []
    if "traceback" in lowered or "runtimeerror" in lowered or "exception" in lowered:
        findings.append(_localized(language, "这是一段程序异常栈，说明执行流程在运行时中断，不是正常 benchmark 结果。", "This is an exception stack trace, so the execution flow stopped at runtime and this is not a normal benchmark result."))
    if "endpoint" in lowered and ("failed" in lowered or "probe" in lowered):
        findings.append(_localized(language, "错误里出现 endpoint/probe failed，优先检查 RPC endpoint 是否可访问、协议是否匹配、method/params 是否真实可用。", "The error mentions endpoint/probe failure; first check whether the RPC endpoint is reachable, the protocol matches, and method/params are actually valid."))
    if "repl.py" in lowered or "agent/terminal" in lowered:
        findings.append(_localized(language, "栈里包含 Agent terminal/repl 路径，问题可能发生在交互层或 Harness 调用链，而不是底层压测工具本身。", "The stack includes Agent terminal/repl paths, so the issue may be in the interactive layer or Harness call path, not necessarily the benchmark tool itself."))
    if not findings:
        findings.append(_localized(language, "我已经读取了已保存证据，但这段内容没有明显的标准异常关键字；需要更多上下文才能判断根因。", "I read the saved evidence, but it does not contain obvious standard exception markers; more context is needed to identify the cause."))
    preview = "\n".join(evidence.splitlines()[:8])
    return _localized(
        language,
        "基于刚才保存的日志证据，我的初步判断：\n"
        + "\n".join(f"- {item}" for item in findings)
        + "\n建议下一步：\n"
        "- 如果这是 endpoint 验证失败，请先用同一个 endpoint 和同一个 RPC method 做一次最小 curl 验证。\n"
        "- 如果 curl 成功但 Agent 失败，请保留 evidence 文件路径和栈信息，让我继续定位 Harness/validator 调用链。\n"
        "- 如果 curl 也失败，需要更换 endpoint、修正协议族或修正 request/params。\n"
        f"证据预览：\n{preview}",
        "Based on the saved log evidence, my initial reading is:\n"
        + "\n".join(f"- {item}" for item in findings)
        + "\nSuggested next steps:\n"
        "- If this is endpoint validation failure, run a minimal curl check with the same endpoint and RPC method.\n"
        "- If curl succeeds but the Agent fails, keep the evidence file path and stack trace so I can inspect the Harness/validator call path.\n"
        "- If curl also fails, replace the endpoint, correct the adapter family, or correct request/params.\n"
        f"Evidence preview:\n{preview}",
    )


def _framework_capability_summary(state: AgentGraphState) -> str:
    framework = state.get("framework_summary") or {}
    language = state.get("language", "en")
    chains = framework.get("chains") or []
    chain_names: list[str] = []
    for item in chains:
        if isinstance(item, dict):
            name = str(item.get("chain") or "").strip()
        else:
            name = str(item or "").strip()
        if name:
            chain_names.append(name)
    chain_preview = ", ".join(chain_names[:36]) if chain_names else "<unknown>"
    families = framework.get("families") or {}
    if isinstance(families, dict) and families:
        family_preview = ", ".join(f"{name}: {count}" for name, count in sorted(families.items()))
    else:
        family_preview = "<unknown>"
    return _localized(
        language,
        (
            f"当前框架已加载 {framework.get('chain_count', '?')} 条链、"
            f"{framework.get('family_count', '?')} 个协议族、"
            f"{framework.get('unique_rpc_method_count', '?')} 个 RPC method。\n"
            f"协议族分布：{family_preview}。\n"
            f"已知链：{chain_preview}。\n"
            "你可以继续选择要测试的链，或询问某条链的默认 RPC workload / 二次开发方式。"
        ),
        (
            f"The framework has loaded {framework.get('chain_count', '?')} chains, "
            f"{framework.get('family_count', '?')} adapter families, and "
            f"{framework.get('unique_rpc_method_count', '?')} RPC methods.\n"
            f"Adapter family distribution: {family_preview}.\n"
            f"Known chains: {chain_preview}.\n"
            "You can continue by choosing a chain, or ask about a chain's default RPC workload / extension path."
        ),
    )


def _report_artifact_entry_response(state: AgentGraphState) -> str:
    language = state.get("language", "en")
    job_id = str(state.get("latest_job_id") or "").strip()
    if not job_id:
        try:
            jobs = list_jobs(limit=1)
            job_id = str(jobs[0].get("job_id") or "") if jobs else ""
        except Exception:
            job_id = ""
    if not job_id:
        return _localized(
            language,
            "没有找到历史 job。你可以先启动一次 fake-node smoke、real-node benchmark 或 sync-observe，再让我分析报告。",
            "No previous job was found. Start a fake-node smoke, real-node benchmark, or sync-observe run first, then ask me to analyze the report.",
        )
    try:
        summary = resume_job(job_id)
    except Exception as exc:
        return _localized(
            language,
            f"无法读取 job `{job_id}`：{type(exc).__name__}。你可以输入 `jobs` 查看可用任务，或提供具体 job_id。",
            f"Could not read job `{job_id}`: {type(exc).__name__}. Type `jobs` to list available jobs, or provide a specific job_id.",
        )
    artifact_index = str(summary.get("artifact_index") or "")
    run_dir = str(summary.get("run_dir") or "")
    runtime_env = str(summary.get("runtime_env_file") or "")
    next_actions = ", ".join(str(item) for item in summary.get("next_actions") or [])
    if str(language or "").startswith("zh"):
        return (
            f"最近任务 `{job_id}` 状态：{summary.get('status', 'unknown')}。\n"
            f"- 运行目录：{run_dir or '<unknown>'}\n"
            f"- artifact_index：{artifact_index or '<not found>'}\n"
            f"- runtime.env：{runtime_env or '<not found>'}\n"
            f"- 建议下一步：{next_actions or 'logs/status/artifact-qa'}\n"
            "你可以继续问：查看日志、解释报告图表、为什么某个指标为空、或分析瓶颈。"
        )
    return (
        f"Latest job `{job_id}` status: {summary.get('status', 'unknown')}.\n"
        f"- run_dir: {run_dir or '<unknown>'}\n"
        f"- artifact_index: {artifact_index or '<not found>'}\n"
        f"- runtime.env: {runtime_env or '<not found>'}\n"
        f"- suggested next actions: {next_actions or 'logs/status/artifact-qa'}\n"
        "You can ask me to inspect logs, explain report charts, investigate empty metrics, or analyze bottlenecks."
    )


def _localized(language: str, zh: str, en: str) -> str:
    return zh if str(language or "").startswith("zh") else en

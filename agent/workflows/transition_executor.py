"""Deterministic transitions after typed pending-question answers.

This module advances only structured workflow-state transitions. It does not
classify natural language and does not infer benchmark intent.
"""

from __future__ import annotations

from typing import Any

from knowledge.chain_identity import canonical_chain_aliases
from knowledge.framework_capabilities import load_framework_capabilities
from validators.config_contract import build_missing_config_questions
from workflows.conversation_state import DEFAULT_SESSION_ID
from workflows.conversation_state import answer_pending_question
from workflows.conversation_state import load_workflow_state
from workflows.conversation_state import update_workflow_state
from workflows.group_registry import is_post_config_question
from workflows.group_registry import is_setup_question_id
from workflows.group_registry import group_for_question_id
from workflows.group_registry import GROUP_ORDER
from workflows.group_registry import question_keys_for_group


def advance_after_pending_answer(
    answer_result: dict[str, Any],
    *,
    discovery: dict[str, Any] | None = None,
    language: str = "en",
    session_id: str = DEFAULT_SESSION_ID,
) -> dict[str, Any]:
    """Advance the next deterministic gate after a short typed answer.

    Fixed replies such as ``1``, ``Y``, disk names, URLs, or ``quick`` are
    already bound to the active ``pending_question`` before this function runs.
    The only allowed follow-up here is applying the next structured transition
    or registering the next blocking question.
    """
    if not answer_result.get("applied"):
        return {"advanced": False, "message": "", "pending_question": {}}

    state = load_workflow_state(session_id=session_id)
    pending = state.get("pending_question") or {}
    if pending:
        return {
            "advanced": True,
            "message": render_pending_question(pending, language=language),
            "pending_question": pending,
        }

    transition = answer_result.get("transition") or {}
    allowed_actions = list(state.get("allowed_next_actions") or [])
    if _should_ask_chain_selection(state, transition, allowed_actions):
        return _register_chain_selection_question(
            state,
            language=language,
            session_id=session_id,
        )
    if _should_ask_target_mode(state, transition, allowed_actions):
        return _register_target_mode_question(
            language=language,
            session_id=session_id,
        )
    if _should_ask_qps_adjust_item(state, transition, allowed_actions):
        return _register_qps_adjust_item_question(
            state,
            language=language,
            session_id=session_id,
        )
    if _should_ask_qps_adjust_value(state, transition, allowed_actions):
        return _register_qps_adjust_value_question(
            state,
            language=language,
            session_id=session_id,
        )
    if _should_ask_custom_rpc_endpoint(state, transition, allowed_actions):
        return _register_custom_rpc_endpoint_question(
            state,
            language=language,
            session_id=session_id,
        )
    if _should_ask_mixed_weights(state, transition, allowed_actions):
        return _register_mixed_weights_question(
            state,
            language=language,
            session_id=session_id,
        )

    if _should_delegate_to_adk_after_pending_answer(state, transition):
        return {
            "advanced": True,
            "message": "",
            "pending_question": {},
            "delegate_to_adk": True,
        }

    if _is_benchmark_setup_state(state):
        return _register_next_config_question(
            state,
            discovery=discovery or {},
            language=language,
            session_id=session_id,
            answered_question_id=str(answer_result.get("question_id") or ""),
        )

    message = _terminal_message_for_non_setup_transition(state, transition, language)
    return {
        "advanced": bool(message),
        "message": message,
        "pending_question": {},
    }


def ensure_next_benchmark_setup_question(
    state: dict[str, Any],
    *,
    discovery: dict[str, Any] | None = None,
    language: str = "en",
    session_id: str = DEFAULT_SESSION_ID,
) -> dict[str, Any]:
    """Ensure benchmark setup never pauses without a typed pending question.

    ADK owns intent inference, but once it has entered ``benchmark_setup`` the
    next blocking user decision must be represented as ``pending_question``.
    This function restores valid workflow shape by registering the next
    deterministic gate from structured state, not by parsing user prose.
    """
    if state.get("pending_question"):
        advanced = _advance_pending_question_from_structured_state(
            state,
            language=language,
            session_id=session_id,
        )
        if advanced.get("advanced"):
            state = load_workflow_state(session_id=session_id)
            if state.get("pending_question"):
                return advanced
        elif advanced.get("blocked"):
            return advanced
        if state.get("pending_question"):
            recomputed = _recompute_pending_question_from_group_state(
                state,
                discovery=discovery or {},
                language=language,
                session_id=session_id,
            )
            if recomputed.get("pending_question"):
                return recomputed
            return {"advanced": False, "message": "", "pending_question": state.get("pending_question") or {}}
    if not _is_benchmark_setup_state(state):
        return {"advanced": False, "message": "", "pending_question": {}}

    allowed_actions = list(state.get("allowed_next_actions") or [])
    if _should_ask_chain_selection(state, {}, allowed_actions):
        return _register_chain_selection_question(state, language=language, session_id=session_id)
    if _should_ask_target_mode(state, {}, allowed_actions):
        return _register_target_mode_question(language=language, session_id=session_id)
    if _should_ask_qps_adjust_item(state, {}, allowed_actions):
        return _register_qps_adjust_item_question(state, language=language, session_id=session_id)
    if _should_ask_qps_adjust_value(state, {}, allowed_actions):
        return _register_qps_adjust_value_question(state, language=language, session_id=session_id)
    if _should_ask_custom_rpc_endpoint(state, {}, allowed_actions):
        return _register_custom_rpc_endpoint_question(state, language=language, session_id=session_id)
    if _should_ask_mixed_weights(state, {}, allowed_actions):
        return _register_mixed_weights_question(state, language=language, session_id=session_id)
    return _register_next_config_question(
        state,
        discovery=discovery or {},
        language=language,
        session_id=session_id,
    )


def _advance_pending_question_from_structured_state(
    state: dict[str, Any],
    *,
    language: str,
    session_id: str,
) -> dict[str, Any]:
    """Advance a pending question already satisfied by structured state.

    ADK tools may set a confirmed field before the previous typed
    pending_question has been cleared. This is not transcript repair: no user
    prose is parsed. The current structured value is replayed through
    answer_pending_question so the same validators, normalizers, transitions,
    and history rules apply.
    """
    if not _is_benchmark_setup_state(state):
        return {"advanced": False}
    question = state.get("pending_question") or {}
    answer = _answer_from_structured_state(state, question)
    if answer is None:
        return {"advanced": False}
    result = answer_pending_question(
        answer,
        reason=f"transition_executor:advance_pending_question_from_structured_state:{question.get('id', '')}",
        session_id=session_id,
    )
    if not result.get("applied"):
        blockers = result.get("blockers") or []
        return {
            "advanced": False,
            "blocked": bool(blockers),
            "message": _structured_advance_blocked_message(blockers, question, language),
            "pending_question": load_workflow_state(session_id=session_id).get("pending_question") or question,
        }
    next_state = load_workflow_state(session_id=session_id)
    pending = next_state.get("pending_question") or {}
    return {
        "advanced": True,
        "message": render_pending_question(pending, language=language) if pending else "",
        "pending_question": pending,
    }


def _answer_from_structured_state(state: dict[str, Any], question: dict[str, Any]) -> str | None:
    field = str(question.get("field") or "").strip()
    fields = [field, *[str(item).strip() for item in list(question.get("fields") or []) if str(item).strip()]]
    for field_name in fields:
        value = _structured_value_for_field(state, field_name)
        if value not in (None, ""):
            return _answer_text_for_question_value(question, value)
    return None


def _structured_value_for_field(state: dict[str, Any], field_name: str) -> Any:
    if not field_name:
        return None
    if field_name in {"target_mode", "chain", "rpc_mode", "workflow_step", "active_intent", "active_workflow"}:
        return state.get(field_name)
    if "." in field_name:
        value: Any = state
        for part in field_name.split("."):
            if not isinstance(value, dict):
                return None
            value = value.get(part)
        return value
    confirmed = state.get("confirmed_config") if isinstance(state.get("confirmed_config"), dict) else {}
    if field_name in confirmed:
        return confirmed[field_name]
    aliases = _confirmed_config_aliases(field_name)
    for alias in aliases:
        if alias in confirmed:
            return confirmed[alias]
    return None


def _confirmed_config_aliases(field_name: str) -> tuple[str, ...]:
    aliases = {
        "LEDGER_DEVICE": ("ledger_device",),
        "ledger_device": ("LEDGER_DEVICE",),
        "ACCOUNTS_DEVICE": ("accounts_device",),
        "accounts_device": ("ACCOUNTS_DEVICE",),
        "DATA_VOL_TYPE": ("data_vol_type",),
        "DATA_VOL_SIZE": ("data_vol_size",),
        "DATA_VOL_MAX_IOPS": ("data_vol_max_iops",),
        "DATA_VOL_MAX_THROUGHPUT": ("data_vol_max_throughput",),
        "ACCOUNTS_VOL_TYPE": ("accounts_vol_type",),
        "ACCOUNTS_VOL_SIZE": ("accounts_vol_size",),
        "ACCOUNTS_VOL_MAX_IOPS": ("accounts_vol_max_iops",),
        "ACCOUNTS_VOL_MAX_THROUGHPUT": ("accounts_vol_max_throughput",),
        "NETWORK_INTERFACE": ("network_interface",),
        "NETWORK_MAX_BANDWIDTH_GBPS": ("network_max_bandwidth_gbps",),
        "BLOCKCHAIN_PROCESS_NAMES": ("blockchain_process_names", "BLOCKCHAIN_PROCESS_NAMES_STR"),
        "CLOUD_REGION": ("cloud_region",),
        "CLOUD_ZONE": ("cloud_zone",),
        "MACHINE_TYPE": ("machine_type",),
    }
    return aliases.get(field_name, ())


def _answer_text_for_question_value(question: dict[str, Any], value: Any) -> str:
    kind = str(question.get("kind") or question.get("expected_answer") or "").strip()
    if isinstance(value, bool):
        return "Y" if value else "N"
    if kind == "yes_no":
        lowered = str(value).strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return "Y"
        if lowered in {"false", "0", "no", "n"}:
            return "N"
    return str(value).strip()


def _structured_advance_blocked_message(blockers: list[Any], question: dict[str, Any], language: str) -> str:
    prompt = render_pending_question(question, language=language)
    blocker_text = "; ".join(str(item) for item in blockers if str(item).strip())
    if (language or "en").strip().lower().startswith("zh"):
        prefix = f"当前结构化状态无法通过校验：{blocker_text}" if blocker_text else "当前结构化状态无法通过校验。"
    else:
        prefix = f"The current structured state did not pass validation: {blocker_text}" if blocker_text else "The current structured state did not pass validation."
    return "\n".join(item for item in (prefix, prompt) if item).strip()


def _recompute_pending_question_from_group_state(
    state: dict[str, Any],
    *,
    discovery: dict[str, Any],
    language: str,
    session_id: str,
) -> dict[str, Any]:
    pending = state.get("pending_question") or {}
    if not _is_benchmark_setup_state(state) or not _is_setup_pending_question(pending):
        return {"advanced": False, "message": "", "pending_question": {}}
    confirmed = _confirmed_config_from_state(state)
    questions = build_missing_config_questions(
        str(state.get("target_mode") or ""),
        confirmed,
        discovery=discovery,
        preferred_group=state.get("next_blocking_group") or state.get("active_group") or "",
    )
    question = questions.get("next_question") or {}
    if not question or question.get("id") == pending.get("id"):
        return {"advanced": False, "message": "", "pending_question": {}}
    if not _should_replace_pending_question_from_group_state(state, pending, question):
        return {"advanced": False, "message": "", "pending_question": {}}
    update_workflow_state(
        {
            "confirmed_config": confirmed,
            "missing_fields": list(questions.get("missing") or []),
            "pending_question": question,
            "workflow_step": question.get("workflow_step") or question.get("id") or "",
            "allowed_next_actions": [],
        },
        reason="transition_executor:recompute_pending_question_from_group_state",
        session_id=session_id,
    )
    return {
        "advanced": True,
        "message": render_pending_question(question, language=language),
        "pending_question": question,
        "ready": False,
    }


def _should_replace_pending_question_from_group_state(
    state: dict[str, Any],
    pending: dict[str, Any],
    next_question: dict[str, Any],
) -> bool:
    pending_id = str(pending.get("id") or "").strip()
    next_id = str(next_question.get("id") or "").strip()
    if not pending_id or not next_id or pending_id == next_id:
        return False
    if _is_post_config_pending_question(pending):
        return True
    if pending_id not in {"chain", "chain_selection", "target_mode"}:
        if next_id in {"chain", "use_fake_node", "target_mode", "local_rpc_url", "real_node_local_rpc_url"}:
            return True
        if not str(state.get("chain") or "").strip():
            return True
        if str(state.get("target_mode") or "").strip() not in {"fake-node", "real-node"}:
            return True
    return False


def _is_setup_pending_question(question: dict[str, Any]) -> bool:
    qid = str(question.get("id") or "").strip()
    return is_setup_question_id(qid)


def _is_post_config_pending_question(question: dict[str, Any]) -> bool:
    branch = str(question.get("branch") or "").strip()
    qid = str(question.get("id") or "").strip()
    return is_post_config_question(qid, branch)


def render_pending_question(question: dict[str, Any], language: str = "en") -> str:
    prompt = _localized_prompt(question, language)
    options = list(question.get("options") or [])
    rendered = [prompt] if prompt else []
    if options and not _prompt_already_lists_options(prompt, options):
        for option in options:
            oid = str(option.get("id") or "").strip()
            label = str(option.get("label") or option.get("value") or "").strip()
            description = str(option.get("description") or "").strip()
            if not oid and not label:
                continue
            line = f"{oid}. {label}" if oid else label
            if description and description not in line:
                line = f"{line} — {description}"
            rendered.append(line)
    return "\n".join(item for item in rendered if item).strip()


def _should_ask_chain_selection(
    state: dict[str, Any],
    transition: dict[str, Any],
    allowed_actions: list[str],
) -> bool:
    if str(transition.get("next_question_id") or "") == "chain_selection":
        if str(transition.get("workflow_step") or "") == "chain_selection":
            return True
        return not str(state.get("chain") or "").strip()
    if "ask:chain_selection" in allowed_actions:
        return not str(state.get("chain") or "").strip()
    if str(state.get("workflow_step") or "") == "chain_selection" and not str(state.get("chain") or "").strip():
        return True
    return False


def _should_ask_target_mode(
    state: dict[str, Any],
    transition: dict[str, Any],
    allowed_actions: list[str],
) -> bool:
    if str(transition.get("next_question_id") or "") == "target_mode":
        return True
    if "ask:target_mode" in allowed_actions:
        return True
    if str(state.get("workflow_step") or "") == "target_mode_choice" and not str(state.get("target_mode") or "").strip():
        return True
    return False


def _should_ask_qps_adjust_item(
    state: dict[str, Any],
    transition: dict[str, Any],
    allowed_actions: list[str],
) -> bool:
    if str(transition.get("next_question_id") or "") == "benchmark_profile_adjust_item":
        return True
    return "ask:benchmark_profile_adjust_item" in allowed_actions


def _should_ask_qps_adjust_value(
    state: dict[str, Any],
    transition: dict[str, Any],
    allowed_actions: list[str],
) -> bool:
    if str(transition.get("next_question_id") or "") == "benchmark_profile_adjust_value":
        return True
    return "ask:benchmark_profile_adjust_value" in allowed_actions


def _should_ask_custom_rpc_endpoint(
    state: dict[str, Any],
    transition: dict[str, Any],
    allowed_actions: list[str],
) -> bool:
    if str(transition.get("next_question_id") or "") == "custom_rpc_endpoint_gate":
        return True
    return "ask:custom_rpc_endpoint_gate" in allowed_actions


def _should_ask_mixed_weights(
    state: dict[str, Any],
    transition: dict[str, Any],
    allowed_actions: list[str],
) -> bool:
    if str(transition.get("next_question_id") or "") == "mixed_weights_confirm":
        return True
    return "ask:mixed_weights_confirm" in allowed_actions


def _register_qps_adjust_item_question(
    state: dict[str, Any],
    *,
    language: str,
    session_id: str,
) -> dict[str, Any]:
    mode = _benchmark_mode_from_state(state)
    if (language or "en").strip().lower().startswith("zh"):
        prompt = f"请选择要调整的 {mode} QPS 参数。"
        descriptions = {
            "initial_qps": "起始 QPS",
            "max_qps": "最高 QPS",
            "qps_step": "每级递增 QPS",
            "duration_seconds": "每个 QPS 档位持续秒数",
        }
    else:
        prompt = f"Choose which {mode} QPS parameter to adjust."
        descriptions = {
            "initial_qps": "Starting QPS.",
            "max_qps": "Maximum QPS.",
            "qps_step": "QPS increment between levels.",
            "duration_seconds": "Seconds per QPS level.",
        }
    options = []
    for index, item in enumerate(("initial_qps", "max_qps", "qps_step", "duration_seconds"), start=1):
        options.append({
            "id": str(index),
            "value": item,
            "label": item,
            "description": descriptions[item],
            "state_patch": {"benchmark_profile": {"adjust_item": item}},
        })
    question = {
        "id": "benchmark_profile_adjust_item",
        "kind": "numbered_choice",
        "prompt": prompt,
        "source": "transition_executor",
        "source_tool": "advance_after_pending_answer",
        "branch": "benchmark_profile",
        "workflow_step": "benchmark_profile_adjust_item",
        "manual_input_allowed": True,
        "allow_manual_input": True,
        "blocks_execution": True,
        "blocking": True,
        "options": options,
        "next_on_choice": {
            "workflow_step": "benchmark_profile_adjust_value",
            "next_question_id": "benchmark_profile_adjust_value",
        },
    }
    update_workflow_state(
        {
            "pending_question": question,
            "workflow_step": "benchmark_profile_adjust_item",
            "allowed_next_actions": [],
        },
        reason="transition_executor:ask_qps_adjust_item",
        session_id=session_id,
    )
    return {
        "advanced": True,
        "message": render_pending_question(question, language=language),
        "pending_question": question,
    }


def _register_custom_rpc_endpoint_question(
    state: dict[str, Any],
    *,
    language: str,
    session_id: str,
) -> dict[str, Any]:
    chain = str(state.get("chain") or "selected-chain").strip() or "selected-chain"
    if (language or "en").strip().lower().startswith("zh"):
        prompt = (
            f"为 {chain} 添加自定义 RPC method 前，必须先验证一个可访问的 endpoint。\n"
            "请提供 LOCAL_RPC_URL 或 public RPC endpoint；如果暂时没有 endpoint，回复 `2` 生成 needs_review 开发交接。"
        )
    else:
        prompt = (
            f"Before adding a custom RPC method for {chain}, a reachable endpoint must be validated.\n"
            "Provide LOCAL_RPC_URL or a public RPC endpoint. If no endpoint is available, reply `2` for a needs_review handoff."
        )
    question = {
        "id": "custom_rpc_endpoint_gate",
        "kind": "url",
        "prompt": prompt,
        "field": "endpoint_validation.rpc_url",
        "source": "transition_executor",
        "source_tool": "advance_after_pending_answer",
        "branch": "custom_rpc",
        "workflow_step": "custom_rpc_endpoint_gate",
        "manual_input_allowed": True,
        "allow_manual_input": True,
        "blocks_execution": True,
        "blocking": True,
        "options": [
            {
                "id": "1",
                "label": "我有 endpoint，现在提供" if (language or "en").strip().lower().startswith("zh") else "I have an endpoint and will paste it now",
                "value": "provide_endpoint",
            },
            {
                "id": "2",
                "label": "暂时没有 endpoint，生成 needs_review 开发交接" if (language or "en").strip().lower().startswith("zh") else "No endpoint now; generate a needs_review development handoff",
                "value": "needs_review_handoff",
                "state_patch": {
                    "fixture_status": {"status": "needs_review", "reason": "no endpoint provided"},
                    "blockers": [f"custom RPC for {chain} requires endpoint/request/response evidence before fixtures or smoke can run"],
                },
                "transition": {"workflow_step": "custom_rpc_handoff_requested", "tool": "build_onboarding_handoff"},
            },
        ],
        "state_patch_on_valid": {"fixture_status": {"status": "needs_review"}},
        "next_on_manual": {
            "workflow_step": "validate_custom_rpc_endpoint",
            "tool": "validate_rpc_endpoint",
        },
    }
    update_workflow_state(
        {
            "active_intent": "custom_rpc_onboarding",
            "active_workflow": "custom_rpc_onboarding",
            "active_group": "target_samples_fixtures",
            "workflow_step": "custom_rpc_endpoint_gate",
            "pending_question": question,
            "fixture_status": {"status": "needs_endpoint"},
            "blockers": ["custom RPC requires a reachable endpoint before method validation or fixture recording"],
            "allowed_next_actions": [],
        },
        reason="transition_executor:ask_custom_rpc_endpoint",
        session_id=session_id,
    )
    return {
        "advanced": True,
        "message": render_pending_question(question, language=language),
        "pending_question": question,
    }


def _register_mixed_weights_question(
    state: dict[str, Any],
    *,
    language: str,
    session_id: str,
) -> dict[str, Any]:
    current = state.get("mixed_weights") if isinstance(state.get("mixed_weights"), dict) else {}
    current_total = sum(int(value) for value in current.values()) if current else 0
    if (language or "en").strip().lower().startswith("zh"):
        prompt = (
            "请输入 mixed RPC 权重，格式为 `method=weight`，总和必须为 100。\n"
            "例如：`eth_blockNumber=70, eth_getBalance=30`。"
        )
    else:
        prompt = (
            "Enter mixed RPC weights as `method=weight` pairs. The total must be 100.\n"
            "Example: `eth_blockNumber=70, eth_getBalance=30`."
        )
    if current:
        prompt = f"{prompt}\nCurrent weights: {_format_weights(current)}."
        if current_total != 100:
            if (language or "en").strip().lower().startswith("zh"):
                prompt = f"{prompt}\n当前总和为 {current_total}，必须调整为 100。"
            else:
                prompt = f"{prompt}\nCurrent total is {current_total}; adjust it to 100."
    question = {
        "id": "mixed_weights_confirm",
        "kind": "rpc_weight",
        "field": "mixed_weights",
        "prompt": prompt,
        "source": "transition_executor",
        "source_tool": "advance_after_pending_answer",
        "branch": "rpc_workload",
        "workflow_step": "mixed_weights_confirm",
        "manual_input_allowed": True,
        "allow_manual_input": True,
        "blocks_execution": True,
        "blocking": True,
        "state_patch_on_valid": {
            "confirmed_config": {
                "mixed_weights_confirmed": True,
                "rpc_workload_confirmed": True,
                "chain_template_reviewed": True,
                "rpc_param_samples_confirmed": True,
            },
            "blockers": [],
        },
        "next_on_manual": {
            "workflow_step": "mixed_weights_confirmed",
            "tool": "validate_rpc_workload",
        },
    }
    update_workflow_state(
        {
            "workflow_step": "mixed_weights_confirm",
            "pending_question": question,
            "allowed_next_actions": [],
        },
        reason="transition_executor:ask_mixed_weights",
        session_id=session_id,
    )
    return {
        "advanced": True,
        "message": render_pending_question(question, language=language),
        "pending_question": question,
    }


def _format_weights(weights: dict[str, Any]) -> str:
    return ", ".join(f"{method}={weight}" for method, weight in weights.items())


def _register_qps_adjust_value_question(
    state: dict[str, Any],
    *,
    language: str,
    session_id: str,
) -> dict[str, Any]:
    profile = state.get("benchmark_profile") if isinstance(state.get("benchmark_profile"), dict) else {}
    item = str(profile.get("adjust_item") or "").strip()
    if item not in {"initial_qps", "max_qps", "qps_step", "duration_seconds"}:
        return _register_qps_adjust_item_question(state, language=language, session_id=session_id)
    mode = _benchmark_mode_from_state(state)
    env_key = f"{_qps_env_prefix(mode)}_{_qps_env_suffix(item)}"
    if (language or "en").strip().lower().startswith("zh"):
        prompt = f"请输入 {env_key} 的值。"
    else:
        prompt = f"Enter the value for {env_key}."
    question = {
        "id": "benchmark_profile_adjust_value",
        "kind": "manual_value",
        "field": env_key,
        "prompt": prompt,
        "source": "transition_executor",
        "source_tool": "advance_after_pending_answer",
        "branch": "benchmark_profile",
        "workflow_step": "benchmark_profile_adjust_value",
        "manual_input_allowed": True,
        "allow_manual_input": True,
        "blocks_execution": True,
        "blocking": True,
        "state_patch_on_valid": {
            "confirmed_config": {"qps_profile_confirmed": True},
            "benchmark_profile": {"qps_profile_confirmed": True},
        },
        "next_on_manual": {
            "workflow_step": "benchmark_profile_adjusted",
            "tool": "build_missing_config_questions",
        },
    }
    update_workflow_state(
        {
            "pending_question": question,
            "workflow_step": "benchmark_profile_adjust_value",
            "allowed_next_actions": [],
        },
        reason="transition_executor:ask_qps_adjust_value",
        session_id=session_id,
    )
    return {
        "advanced": True,
        "message": render_pending_question(question, language=language),
        "pending_question": question,
    }


def _benchmark_mode_from_state(state: dict[str, Any]) -> str:
    profile = state.get("benchmark_profile") if isinstance(state.get("benchmark_profile"), dict) else {}
    mode = str(profile.get("mode") or profile.get("name") or state.get("benchmark_mode_confirmed") or "").strip().lower()
    return mode if mode in {"quick", "standard", "intensive"} else "quick"


def _qps_env_prefix(mode: str) -> str:
    normalized = mode if mode in {"quick", "standard", "intensive"} else "quick"
    return normalized.upper()


def _qps_env_suffix(item: str) -> str:
    return {
        "initial_qps": "INITIAL_QPS",
        "max_qps": "MAX_QPS",
        "qps_step": "QPS_STEP",
        "duration_seconds": "DURATION",
    }[item]


def _register_target_mode_question(
    *,
    language: str,
    session_id: str,
) -> dict[str, Any]:
    prompt, options = _target_mode_prompt_and_options(language)
    question = {
        "id": "target_mode",
        "kind": "numbered_choice",
        "prompt": prompt,
        "field": "target_mode",
        "source": "transition_executor",
        "source_tool": "advance_after_pending_answer",
        "branch": "target_selection",
        "workflow_step": "target_mode_choice",
        "manual_input_allowed": True,
        "allow_manual_input": True,
        "blocks_execution": True,
        "blocking": True,
        "options": options,
    }
    update_workflow_state(
        {
            "active_intent": "benchmark",
            "active_workflow": "benchmark_setup",
            "workflow_step": "target_mode_choice",
            "pending_question": question,
            "allowed_next_actions": [],
        },
        reason="transition_executor:ask_target_mode",
        session_id=session_id,
    )
    return {
        "advanced": True,
        "message": render_pending_question(question, language=language),
        "pending_question": question,
    }


def _register_chain_selection_question(
    state: dict[str, Any],
    *,
    language: str,
    session_id: str,
) -> dict[str, Any]:
    target_mode = str(state.get("target_mode") or "").strip()
    prompt = _chain_selection_prompt(target_mode, language)
    capabilities = load_framework_capabilities()
    question = {
        "id": "chain_selection",
        "kind": "chain",
        "prompt": prompt,
        "field": "chain",
        "known_chains": _known_chain_names(capabilities),
        "chain_aliases": _chain_aliases(),
        "source": "transition_executor",
        "source_tool": "advance_after_pending_answer",
        "branch": "chain_selection",
        "workflow_step": "chain_selection",
        "manual_input_allowed": True,
        "allow_manual_input": True,
        "blocks_execution": True,
        "blocking": True,
        "next_on_manual": {
            "workflow_step": "validate_chain_template",
            "tool": "validate_chain_template",
        },
    }
    update_workflow_state(
        {
            "active_intent": "benchmark",
            "active_workflow": "benchmark_setup",
            "workflow_step": "chain_selection",
            "pending_question": question,
        },
        reason="transition_executor:ask_chain_selection",
        session_id=session_id,
    )
    return {
        "advanced": True,
        "message": render_pending_question(question, language=language),
        "pending_question": question,
    }


def _known_chain_names(capabilities: dict[str, Any]) -> list[str]:
    names = []
    for item in list((capabilities or {}).get("chains") or []):
        if isinstance(item, dict):
            name = str(item.get("chain") or "").strip().lower()
            if name:
                names.append(name)
    return sorted(set(names))


def _chain_aliases() -> dict[str, str]:
    return canonical_chain_aliases()


def _register_next_config_question(
    state: dict[str, Any],
    *,
    discovery: dict[str, Any],
    language: str,
    session_id: str,
    answered_question_id: str = "",
) -> dict[str, Any]:
    confirmed = _confirmed_config_from_state(state)
    questions = build_missing_config_questions(
        str(state.get("target_mode") or ""),
        confirmed,
        discovery=discovery,
        preferred_group=state.get("next_blocking_group") or state.get("active_group") or "",
    )
    question = questions.get("next_question") or {}
    progress_patch = _group_progress_patch(
        state,
        answered_question_id=answered_question_id,
        next_question=question,
        missing_fields=list(questions.get("missing") or []),
    )
    if question:
        update_workflow_state(
            {
                "confirmed_config": confirmed,
                "missing_fields": list(questions.get("missing") or []),
                "pending_question": question,
                "workflow_step": question.get("workflow_step") or question.get("id") or "",
                "allowed_next_actions": [],
                **progress_patch,
            },
            reason="transition_executor:ask_next_config_question",
            session_id=session_id,
        )
        return {
            "advanced": True,
            "message": render_pending_question(question, language=language),
            "pending_question": question,
            "ready": False,
        }

    question = _preflight_smoke_question(language)
    ready_progress_patch = _group_progress_patch(
        state,
        answered_question_id=answered_question_id,
        next_question=question,
        missing_fields=[],
    )
    update_workflow_state(
        {
            "confirmed_config": confirmed,
            "missing_fields": [],
            "allowed_next_actions": [],
            "workflow_step": "config_ready_for_preflight",
            "pending_question": question,
            **ready_progress_patch,
        },
        reason="transition_executor:config_ready",
        session_id=session_id,
    )
    return {
        "advanced": True,
        "message": render_pending_question(question, language=language),
        "pending_question": question,
        "ready": True,
    }


def _group_progress_patch(
    state: dict[str, Any],
    *,
    answered_question_id: str,
    next_question: dict[str, Any],
    missing_fields: list[str],
) -> dict[str, Any]:
    progress = _normalized_progress(state.get("group_progress") or {})
    answered_group = group_for_question_id(answered_question_id)
    next_group = group_for_question_id(str(next_question.get("id") or ""))
    missing_set = {str(item or "").strip() for item in missing_fields if str(item or "").strip()}

    if answered_group and not _group_has_missing_question(answered_group, missing_set):
        progress[answered_group]["status"] = "complete"
        progress[answered_group]["confirmed_fields"] = sorted(
            set(progress[answered_group].get("confirmed_fields") or ()) | {answered_question_id}
        )
        progress[answered_group]["invalidated_fields"] = []

    if next_group:
        progress[next_group]["status"] = "in_progress"
        return {
            "group_progress": progress,
            "active_group": next_group,
            "next_blocking_group": next_group,
            "last_completed_group": answered_group if answered_group and progress[answered_group]["status"] == "complete" else "",
        }

    for group in GROUP_ORDER:
        if progress[group]["status"] not in {"complete"}:
            progress[group]["status"] = "complete"
    return {
        "group_progress": progress,
        "active_group": "preflight_smoke_execution",
        "next_blocking_group": "preflight_smoke_execution",
        "last_completed_group": answered_group or "",
    }


def _group_has_missing_question(group: str, missing_fields: set[str]) -> bool:
    questions = set(question_keys_for_group(group))
    return bool(questions & missing_fields)


def _normalized_progress(value: Any) -> dict[str, dict[str, Any]]:
    progress = {group: {"status": "pending", "confirmed_fields": [], "invalidated_fields": []} for group in GROUP_ORDER}
    if not isinstance(value, dict):
        return progress
    for raw_group, raw_data in value.items():
        group = str(raw_group or "").strip()
        if group not in progress or not isinstance(raw_data, dict):
            continue
        status = str(raw_data.get("status") or progress[group]["status"]).strip()
        if status in {"pending", "in_progress", "complete", "invalidated", "blocked"}:
            progress[group]["status"] = status
        progress[group]["confirmed_fields"] = [str(item) for item in list(raw_data.get("confirmed_fields") or []) if str(item)]
        progress[group]["invalidated_fields"] = [str(item) for item in list(raw_data.get("invalidated_fields") or []) if str(item)]
    return progress


def _confirmed_config_from_state(state: dict[str, Any]) -> dict[str, Any]:
    confirmed = dict(state.get("confirmed_config") or {})
    chain = str(state.get("chain") or "").strip()
    if chain:
        confirmed["chain"] = chain
        confirmed["BLOCKCHAIN_NODE"] = chain
    rpc_mode = str(state.get("rpc_mode") or "").strip()
    if rpc_mode:
        confirmed["rpc_mode"] = rpc_mode
        confirmed["RPC_MODE"] = rpc_mode
    target_mode = str(state.get("target_mode") or "").strip()
    if target_mode == "fake-node":
        confirmed["use_fake_node"] = True
        confirmed["TARGET_MODE"] = "fake-node"
        confirmed.setdefault("blockchain_process_names", ["fake-node"])
        confirmed.setdefault("BLOCKCHAIN_PROCESS_NAMES", ["fake-node"])
        confirmed.setdefault("BLOCKCHAIN_PROCESS_NAMES_STR", "fake-node")
    elif target_mode == "real-node":
        confirmed["use_fake_node"] = False
        confirmed["TARGET_MODE"] = "real-node"
        _clear_fake_node_process_names(confirmed)
    profile = state.get("benchmark_profile") or {}
    mode = str(profile.get("mode") or profile.get("name") or "").strip()
    if mode:
        confirmed.setdefault("benchmark_mode_confirmed", mode)
    observability = state.get("observability") or {}
    obs_mode = str(observability.get("mode") or "").strip()
    if obs_mode:
        confirmed.setdefault("observability_choice_confirmed", obs_mode)
        confirmed.setdefault("OBSERVABILITY_STACK_MODE", obs_mode)
    return confirmed


def _clear_fake_node_process_names(confirmed: dict[str, Any]) -> None:
    process_names = confirmed.get("blockchain_process_names") or confirmed.get("BLOCKCHAIN_PROCESS_NAMES")
    process_name_text = str(confirmed.get("BLOCKCHAIN_PROCESS_NAMES_STR") or "").strip()
    if process_names == ["fake-node"] or process_name_text == "fake-node":
        for key in ("blockchain_process_names", "BLOCKCHAIN_PROCESS_NAMES", "BLOCKCHAIN_PROCESS_NAMES_STR"):
            confirmed.pop(key, None)


def _is_benchmark_setup_state(state: dict[str, Any]) -> bool:
    active_workflow = str(state.get("active_workflow") or "").strip()
    if active_workflow == "benchmark_setup":
        return True
    if active_workflow and active_workflow != "benchmark_setup":
        return False
    if str(state.get("target_mode") or "") in {"fake-node", "real-node"}:
        return True
    return False


def _should_delegate_to_adk_after_pending_answer(
    state: dict[str, Any],
    transition: dict[str, Any],
) -> bool:
    tool = str(transition.get("tool") or "").strip()
    if tool in {
        "prepare_benchmark_run",
        "run_preflight",
        "run_fake_node_smoke_benchmark",
        "run_quick_assumed_fake_node_smoke",
        "validate_rpc_endpoint",
        "build_onboarding_handoff",
    }:
        return True
    allowed = set(state.get("allowed_next_actions") or [])
    return bool(allowed & {
        "tool:prepare_benchmark_run",
        "tool:run_preflight",
        "tool:run_fake_node_smoke_benchmark",
        "tool:run_quick_assumed_fake_node_smoke",
        "tool:validate_rpc_endpoint",
        "tool:build_onboarding_handoff",
    })


def _preflight_smoke_question(language: str) -> dict[str, Any]:
    if (language or "en").strip().lower().startswith("zh"):
        prompt = (
            "配置已完整。是否现在生成测试计划，并执行 preflight 和 fake-node smoke 闭环验证？\n"
            "回复 `Y` 开始；回复 `N` 暂停并调整配置。"
        )
    else:
        prompt = (
            "Configuration is complete. Generate the benchmark plan, then run preflight "
            "and fake-node smoke validation now?\n"
            "Reply `Y` to start, or `N` to pause and adjust configuration."
        )
    return {
        "id": "smoke_run_confirm",
        "kind": "yes_no",
        "field": "approval.smoke_run",
        "prompt": prompt,
        "source": "transition_executor",
        "source_tool": "advance_after_pending_answer",
        "branch": "preflight_smoke",
        "workflow_step": "config_ready_for_preflight",
        "blocks_execution": True,
        "blocking": True,
        "next_on_yes": {
            "workflow_step": "preflight_smoke_requested",
            "tool": "prepare_benchmark_run",
            "state_patch": {"approval": {"smoke_run": True}},
        },
        "next_on_no": {
            "workflow_step": "configuration_paused",
            "tool": "ask_configuration_change",
            "state_patch": {"approval": {"smoke_run": False}},
        },
    }


def _localized_prompt(question: dict[str, Any], language: str) -> str:
    prompt = str(question.get("prompt") or "").strip()
    inferred_prompt = _inferred_value_prompt(question, language)
    if inferred_prompt:
        return inferred_prompt
    if not (language or "en").strip().lower().startswith("zh"):
        return prompt
    qid = str(question.get("id") or "")
    zh = {
        "cloud_region": "请输入 CLOUD_REGION（云区域），例如 `asia-east1`。",
        "cloud_zone": "请输入 CLOUD_ZONE（可用区），例如 `asia-east1-c`。",
        "machine_type": "请输入 MACHINE_TYPE（机器或实例规格），用于报告元数据。",
        "disk_ledger_choice": "请选择 Ledger/data 磁盘，回复编号或直接输入设备名。",
        "data_vol_type": "请输入 Ledger/data 磁盘类型，例如 `hyperdisk-balanced`、`hyperdisk-extreme`、`pd-ssd`、`pd-balanced`、`local-ssd`、`ssd`、`nvme`。",
        "data_vol_size": "请输入 Ledger/data 磁盘容量，单位 GiB。",
        "data_vol_max_iops": "请输入 Ledger/data 磁盘最大 IOPS。",
        "data_vol_max_throughput": "请输入 Ledger/data 磁盘最大吞吐，单位 MiB/s。",
        "disk_accounts_exists": "这个节点是否有独立的 accounts/state 磁盘？回复 `Y` 或 `N`。",
        "has_accounts_device": "这个节点是否有独立的 accounts/state 磁盘？回复 `Y` 或 `N`。",
        "disk_accounts_choice": "请选择 accounts/state 磁盘，回复编号或直接输入设备名。",
        "accounts_vol_type": "请输入 accounts/state 磁盘类型，例如 `hyperdisk-balanced`、`hyperdisk-extreme`、`pd-ssd`、`pd-balanced`、`local-ssd`、`ssd`、`nvme`。",
        "accounts_vol_size": "请输入 accounts/state 磁盘容量，单位 GiB。",
        "accounts_vol_max_iops": "请输入 accounts/state 磁盘最大 IOPS。",
        "accounts_vol_max_throughput": "请输入 accounts/state 磁盘最大吞吐，单位 MiB/s。",
        "network_interface": "请输入节点使用的网卡名称，例如 `eth0`、`ens4`。",
        "network_max_bandwidth_gbps": "请输入网络带宽上限，单位 Gbps，用于瓶颈分析。",
        "blockchain_process_names": "请输入节点进程名或命令行关键字，用于进程归因统计。",
        "local_rpc_url": (
            "real-node benchmark 需要先验证被测节点 endpoint。\n"
            "请提供 LOCAL_RPC_URL，例如 `http://127.0.0.1:8899`。"
            "如果暂时没有 LOCAL_RPC_URL，请直接说明没有；系统会列出阻塞项，并可以帮你切换到 fake-node smoke。"
        ),
        "real_node_local_rpc_url": (
            "real-node benchmark 需要先验证被测节点 endpoint。\n"
            "请提供 LOCAL_RPC_URL，例如 `http://127.0.0.1:8899`。"
            "如果暂时没有 LOCAL_RPC_URL，请直接说明没有；系统会列出阻塞项，并可以帮你切换到 fake-node smoke。"
        ),
        "benchmark_profile_choice": "请选择 benchmark 模式：1 quick、2 standard、3 intensive。",
        "benchmark_profile_confirm": (
            "是否使用所选模式的默认 QPS 配置？回复 `Y` 保留，回复 `N` 调整 INITIAL_QPS、MAX_QPS、QPS_STEP 或 DURATION。\n"
            "说明：fake-node smoke 会在执行阶段使用安全小流量覆盖（1 QPS，10s）来验证完整闭环；"
            "这不是真实节点性能测试。real-node benchmark 才使用所选模式的完整 QPS profile。"
        ),
        "qps_profile_confirmed": (
            "是否使用所选模式的默认 QPS 配置？回复 `Y` 保留，回复 `N` 调整 INITIAL_QPS、MAX_QPS、QPS_STEP 或 DURATION。\n"
            "说明：fake-node smoke 会在执行阶段使用安全小流量覆盖（1 QPS，10s）来验证完整闭环；"
            "这不是真实节点性能测试。real-node benchmark 才使用所选模式的完整 QPS profile。"
        ),
        "chain_template_reviewed": "是否使用当前链模板中的默认 endpoint、TARGET_* sample 和 workload 继续？回复 `Y` 或 `N`。",
        "workload_customization_choice": (
            "请选择 RPC workload 下一步：\n"
            "1. 使用当前链模板默认 workload 继续\n"
            "2. 添加自定义 RPC method\n"
            "3. 调整 mixed 权重\n"
            "4. 更换链或目标模式\n"
            "回复 `1`、`2`、`3` 或 `4`，也可以直接说明要怎么改。"
        ),
        "rpc_mode": "请选择 RPC 模式：`single` 或 `mixed`。",
        "rpc_workload_confirmed": "是否使用当前链模板的默认 RPC method 和权重？回复 `Y` 使用默认值；回复 `N` 自定义 method 或 mixed 权重。",
        "rpc_param_samples_confirmed": "是否使用当前链模板中的 TARGET_* sample values？回复 `Y` 使用默认样本；回复 `N` 自定义样本。",
        "observability_mode_choice": "请选择可观测性模式：1 禁用，2 本地 Prometheus/Grafana，3 仅 exporter 接入已有 Prometheus。",
    }
    return zh.get(qid, prompt)


def _inferred_value_prompt(question: dict[str, Any], language: str) -> str:
    current_value = question.get("current_value")
    if current_value in (None, ""):
        return ""
    kind = str(question.get("kind") or question.get("expected_answer") or "").strip()
    if kind not in {"manual_value", "free_text", "device"}:
        return ""
    field = str(question.get("field") or question.get("id") or "value").strip()
    value = str(current_value).strip()
    if not value:
        return ""
    if (language or "en").strip().lower().startswith("zh"):
        return f"检测到 {field} 为 `{value}`。回复 `Y` 使用该值，或直接输入正确值。"
    return f"Detected {field}: `{value}`. Reply `Y` to use it, or type the correct value."


def _terminal_message_for_non_setup_transition(
    state: dict[str, Any],
    transition: dict[str, Any],
    language: str,
) -> str:
    tool = str(transition.get("tool") or "").strip()
    if tool == "latest_job":
        if language.startswith("zh"):
            return "已记录你的选择。请说 `status`、`logs` 或 `analyze latest job` 查看最近任务。"
        return "Choice recorded. Say `status`, `logs`, or `analyze latest job` to inspect the latest job."
    if tool == "load_framework_context":
        return _framework_capability_summary(language)
    return ""


def _framework_capability_summary(language: str) -> str:
    capabilities = load_framework_capabilities()
    chains = capabilities.get("chains") or []
    chain_count = capabilities.get("chain_count", len(chains))
    family_count = capabilities.get("family_count", "?")
    method_count = capabilities.get("unique_rpc_method_count", "?")
    if language.startswith("zh"):
        sample = ", ".join(str(item) for item in chains[:12])
        return (
            f"当前框架支持 {chain_count} 条链、{family_count} 个 adapter family、{method_count} 个 RPC method 名称。"
            f"\n示例链：{sample}。"
            "\n你可以继续问某条链支持哪些 RPC method，或直接说要新增链/自定义 RPC method，我会进入对应的引导流程。"
        )
    sample = ", ".join(str(item) for item in chains[:12])
    return (
        f"The framework currently supports {chain_count} chains, {family_count} adapter families, "
        f"and {method_count} RPC method names."
        f"\nExample chains: {sample}."
        "\nAsk for one chain's RPC methods, or say you want to add a chain/custom RPC method to enter the onboarding flow."
    )


def _chain_selection_prompt(target_mode: str, language: str) -> str:
    if (language or "en").strip().lower().startswith("zh"):
        mode_text = f"（当前目标模式：{target_mode}）" if target_mode else ""
        return (
            f"你想测试哪条链？{mode_text}\n"
            "请输入链名，例如 `solana`、`ethereum`、`bitcoin`。"
            "如果不是当前 36 条链，也可以直接输入链名，我会进入新增链/协议适配流程。"
        )
    mode_text = f" (current target mode: {target_mode})" if target_mode else ""
    return (
        f"Which chain do you want to benchmark?{mode_text}\n"
        "Type a chain name such as `solana`, `ethereum`, or `bitcoin`. "
        "If it is not one of the current 36 chains, type the chain name to enter the chain onboarding flow."
    )


def _target_mode_prompt_and_options(language: str) -> tuple[str, list[dict[str, Any]]]:
    if (language or "en").strip().lower().startswith("zh"):
        return (
            "这次要使用哪种测试目标？\n"
            "1. fake-node 模式 — 不需要真实节点，验证框架闭环流程\n"
            "2. real-node 模式 — 使用真实节点 RPC，测试真实节点性能\n"
            "回复 `1` 或 `2`。",
            [
                {
                    "id": "1",
                    "label": "fake-node 模式",
                    "value": "fake-node",
                    "state_patch": {"target_mode": "fake-node", "workflow_step": "benchmark_setup"},
                    "transition": {"workflow_step": "benchmark_setup", "next_question_id": "chain_selection"},
                },
                {
                    "id": "2",
                    "label": "real-node 模式",
                    "value": "real-node",
                    "state_patch": {"target_mode": "real-node", "workflow_step": "benchmark_setup"},
                    "transition": {"workflow_step": "benchmark_setup", "next_question_id": "chain_selection"},
                },
            ],
        )
    return (
        "Which target mode should this benchmark use?\n"
        "1. fake-node mode — no real node required; verify the framework closed loop\n"
        "2. real-node mode — use a real node RPC endpoint and benchmark real performance\n"
        "Reply `1` or `2`.",
        [
            {
                "id": "1",
                "label": "fake-node mode",
                "value": "fake-node",
                "state_patch": {"target_mode": "fake-node", "workflow_step": "benchmark_setup"},
                "transition": {"workflow_step": "benchmark_setup", "next_question_id": "chain_selection"},
            },
            {
                "id": "2",
                "label": "real-node mode",
                "value": "real-node",
                "state_patch": {"target_mode": "real-node", "workflow_step": "benchmark_setup"},
                "transition": {"workflow_step": "benchmark_setup", "next_question_id": "chain_selection"},
            },
        ],
    )


def _prompt_already_lists_options(prompt: str, options: list[dict[str, Any]]) -> bool:
    if not prompt or not options:
        return False
    for option in options[:3]:
        oid = str(option.get("id") or "").strip()
        if oid and (f"{oid}." in prompt or f"{oid}." in prompt or f"`{oid}`" in prompt):
            return True
    return False

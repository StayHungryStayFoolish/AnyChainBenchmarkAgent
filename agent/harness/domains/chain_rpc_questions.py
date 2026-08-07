"""Cohesive Chain/RPC component extracted from the product domain facade."""

from __future__ import annotations

import json
from functools import partial
from typing import Any

from ..contracts import TextRef
from ..input_values import (
    normalize_scalar,
    normalize_target_mode,
)
from ..questions import (
    choice_question as _choice_question,
    manual_question as _manual_question,
    question_text,
)
from ..state import AgentGraphState

from agent.onboarding.families import SUPPORTED_FAMILIES
from agent.validators.rpc_workload import default_workload
from .chain_rpc_support import _weight_example, active_rpc_onboarding_case
from .rpc_catalog import catalog_method_names, draft_view, next_parameter_to_confirm

choice_question = partial(_choice_question, owner="chain_rpc")
manual_question = partial(_manual_question, owner="chain_rpc")


def _endpoint_probe_completion() -> TextRef:
    """Describe the runtime-owned validation that follows endpoint intake."""

    return question_text("question.chain_rpc.endpoint_probe.completion")


def _method_identity_completion() -> TextRef:
    """Describe the typed continuation after a method identity is accepted."""

    return question_text("question.chain_rpc.method_identity.completion")


def _method_intake_question(
    state: AgentGraphState,
    *,
    case: str,
) -> dict[str, Any]:
    new_chain = case == "new_chain"
    question_id = "new_chain_method" if new_chain else "custom_rpc_method"
    prompt_id = (
        "question.chain_rpc.new_chain_method.prompt"
        if new_chain
        else "question.chain_rpc.custom_method.prompt"
    )
    options = []
    if catalog_method_names(state):
        options.append(
            _action_option(
                "finish",
                (
                    question_text("question.chain_rpc.option.finish_new_chain_methods")
                    if new_chain
                    else question_text("question.chain_rpc.option.finish_custom_methods")
                ),
                "finish",
                "rpc_catalog_command",
                {
                    (
                        "chain_identity.status"
                        if new_chain
                        else "custom_rpc.status"
                    ): (
                        "existing_family_needs_workload_scope"
                        if new_chain
                        else "needs_scope"
                    )
                },
                catalog_command="finish",
            )
        )
    if not options:
        return manual_question(
            "endpoint_process",
            question_id,
            question_text(prompt_id),
            field=question_id,
            accepted_action_types=("rpc_catalog_command",),
            manual_action={
                "type": "rpc_catalog_command",
                "catalog_command": "set_method",
                "value_argument": "rpc_method",
            },
            queue_barrier=True,
            barrier_policy="explicit_detour_only",
            validation={"input_mode": "rpc_method_or_schema_evidence"},
            completion_effect=_method_identity_completion(),
            domain_context={"rpc_case": case},
        )
    return choice_question(
        "endpoint_process",
        question_id,
        question_text(prompt_id),
        field=question_id,
        kind="manual_value",
        options=options,
        manual_input_allowed=True,
        accepted_action_types=("rpc_catalog_command",),
        manual_action={
            "type": "rpc_catalog_command",
            "catalog_command": "set_method",
            "value_argument": "rpc_method",
        },
        queue_barrier=True,
        barrier_policy="explicit_detour_only",
        validation={"input_mode": "rpc_method_or_schema_evidence"},
        completion_effect=_method_identity_completion(),
        domain_context={"rpc_case": case},
    )


def _method_conflict_question(
    state: AgentGraphState,
    *,
    case: str,
) -> dict[str, Any]:
    owner = (
        state.get("chain_identity") or {}
        if case == "new_chain"
        else state.get("custom_rpc") or {}
    )
    conflict = owner.get("method_conflict") or {}
    current = normalize_scalar(conflict.get("current_method"))
    incoming = normalize_scalar(conflict.get("incoming_method"))
    question_id = (
        "new_chain_method_conflict"
        if case == "new_chain"
        else "custom_rpc_method_conflict"
    )
    return choice_question(
        "endpoint_process",
        question_id,
        question_text(
            "question.chain_rpc.method_conflict.prompt",
            current_method=current,
            incoming_method=incoming,
        ),
        field=question_id,
        options=[
            _action_option(
                "keep_current",
                question_text(
                    "question.chain_rpc.option.keep_current_method",
                    method=current,
                ),
                "keep_current",
                "rpc_catalog_command",
                {
                    (
                        "chain_identity.status"
                        if case == "new_chain"
                        else "custom_rpc.status"
                    ): (
                        "existing_family_needs_schema_evidence"
                        if case == "new_chain"
                        else "needs_schema_evidence"
                    )
                },
                catalog_command="keep_current_method",
            ),
            _action_option(
                "replace",
                question_text(
                    "question.chain_rpc.option.replace_current_method",
                    method=incoming,
                ),
                "replace",
                "rpc_catalog_command",
                {"custom_rpc.catalog.draft.method": incoming},
                catalog_command="replace_current_method",
            ),
        ],
        accepted_action_types=("rpc_catalog_command",),
        queue_barrier=True,
        barrier_policy="exclusive_owner",
        domain_context={"rpc_case": case},
    )


def _endpoint_validation_question(state: AgentGraphState) -> dict[str, Any] | None:
    custom = state.get("custom_rpc") or {}
    identity = state.get("chain_identity") or {}
    active_case = active_rpc_onboarding_case(state)
    if active_case == "new_chain":
        custom = {}
    elif active_case == "custom_rpc":
        identity = {}
    if custom.get("status") == "needs_adapter_family_confirmation":
        return _adapter_family_question(state, custom=True)
    if custom.get("status") in {"needs_endpoint", "probe_failed"} and not custom.get("endpoint_ready"):
        return manual_question(
            "endpoint_process",
            "custom_rpc_endpoint",
            question_text("question.chain_rpc.custom_endpoint.prompt"),
            field="custom_rpc_endpoint",
            kind="url",
            accepted_action_types=("rpc_catalog_command",),
            manual_action={
                "type": "rpc_catalog_command",
                "catalog_command": "set_endpoint",
                "value_argument": "rpc_endpoint",
            },
            queue_barrier=True,
            barrier_policy="explicit_detour_only",
            requires_capabilities=("chain_identity",),
            evidence_path="custom_rpc.endpoint",
            completion_effect=_endpoint_probe_completion(),
            domain_context={
                "contract_type": "rpc_endpoint",
                "endpoint_role": "validation",
                "rpc_case": "custom_rpc",
                "config_field": "",
            },
        )
    if custom.get("status") == "needs_method" and not draft_view(state).get("method"):
        return _method_intake_question(state, case="custom_rpc")
    if custom.get("status") == "method_conflict":
        return _method_conflict_question(state, case="custom_rpc")
    if custom.get("status") == "needs_schema_evidence":
        method = normalize_scalar(
            draft_view(state).get("method") or custom.get("candidate_method")
        )
        return manual_question(
            "endpoint_process",
            "custom_rpc_schema_evidence",
            question_text(
                "question.chain_rpc.schema_evidence.prompt",
                method=method or "<current>",
            ),
            field="custom_rpc_schema_evidence",
            kind="evidence",
            accepted_action_types=("rpc_catalog_command",),
            manual_action={
                "type": "rpc_catalog_command",
                "catalog_command": "append_evidence",
                "value_argument": "rpc_schema_evidence",
            },
            queue_barrier=True,
            barrier_policy="explicit_detour_only",
            help_text=_schema_evidence_help(),
            completion_effect=_schema_evidence_completion(),
            structured_input_owner=True,
            domain_context={"rpc_case": "custom_rpc"},
        )
    if custom.get("status") == "schema_needs_confirmation":
        return _catalog_confirmation_question(state, "custom_rpc")
    if custom.get("status") == "response_needs_confirmation":
        return _response_confirmation_question(state, "custom_rpc")
    if custom.get("status") == "probe_failed":
        return _probe_confirmation_question(state, "custom_rpc", retry=True)
    if custom.get("status") == "method_validated_next":
        return _continue_question(state, "custom_rpc")
    if custom.get("status") == "needs_single_method":
        return _single_method_question(state, "custom_rpc")
    if custom.get("status") == "needs_scope":
        return _scope_question(state, "custom_rpc")
    if custom.get("status") == "needs_weights":
        return _weights_question(state, "custom_rpc")
    evidence = state.get("endpoint_evidence") or {}
    if identity.get("status") == "existing_family_needs_endpoint" and not evidence.get("candidate_endpoint_ready"):
        return manual_question(
            "endpoint_process",
            "new_chain_endpoint",
            question_text("question.chain_rpc.new_chain_endpoint.prompt"),
            field="new_chain_endpoint",
            kind="url",
            accepted_action_types=("rpc_catalog_command",),
            manual_action={
                "type": "rpc_catalog_command",
                "catalog_command": "set_endpoint",
                "value_argument": "rpc_endpoint",
            },
            queue_barrier=True,
            barrier_policy="explicit_detour_only",
            requires_capabilities=("chain_identity",),
            evidence_path="endpoint_evidence.candidate_endpoint",
            completion_effect=_endpoint_probe_completion(),
            domain_context={
                "contract_type": "rpc_endpoint",
                "endpoint_role": "validation",
                "rpc_case": "new_chain",
                "config_field": "",
            },
        )
    if identity.get("status") == "existing_family_needs_method":
        return _method_intake_question(state, case="new_chain")
    if identity.get("status") == "existing_family_method_conflict":
        return _method_conflict_question(state, case="new_chain")
    if identity.get("status") == "existing_family_needs_schema_evidence":
        method = normalize_scalar(draft_view(state).get("method"))
        return manual_question(
            "endpoint_process",
            "new_chain_schema_evidence",
            question_text(
                "question.chain_rpc.schema_evidence.prompt",
                method=method or "<current>",
            ),
            field="new_chain_schema_evidence",
            kind="evidence",
            accepted_action_types=("rpc_catalog_command",),
            manual_action={
                "type": "rpc_catalog_command",
                "catalog_command": "append_evidence",
                "value_argument": "rpc_schema_evidence",
            },
            queue_barrier=True,
            barrier_policy="explicit_detour_only",
            help_text=_schema_evidence_help(),
            completion_effect=_schema_evidence_completion(),
            structured_input_owner=True,
            domain_context={"rpc_case": "new_chain"},
        )
    if identity.get("status") == "existing_family_schema_needs_confirmation":
        return _catalog_confirmation_question(state, "new_chain")
    if identity.get("status") == "existing_family_response_needs_confirmation":
        return _response_confirmation_question(state, "new_chain")
    if identity.get("status") == "existing_family_method_validated_next":
        return _continue_question(state, "new_chain")
    if identity.get("status") == "existing_family_needs_workload_scope":
        return _scope_question(state, "new_chain")
    if identity.get("status") == "existing_family_needs_single_method":
        return _single_method_question(state, "new_chain")
    if identity.get("status") == "existing_family_needs_weights":
        return _weights_question(state, "new_chain")
    return None


def _schema_evidence_help() -> TextRef:
    return question_text("question.chain_rpc.schema_evidence.help")


def _schema_evidence_completion() -> TextRef:
    return question_text("question.chain_rpc.schema_evidence.completion")


def _mainnet_review_question(state: AgentGraphState, *, sync_observe: bool) -> dict[str, Any]:
    identity = state.get("chain_identity") or {}
    if identity.get("case") in {"case2", "case2_runtime_override"}:
        chain = normalize_scalar(identity.get("canonical") or identity.get("raw")) or "<unknown>"
        return _choice(
            "endpoint_process",
            "MAINNET_RPC_URL_REVIEWED",
            question_text(
                "question.chain_rpc.mainnet_review.case2.prompt",
                chain=chain,
            ),
            "MAINNET_RPC_URL_REVIEWED",
            [
                _answer_option(
                    "skip",
                    question_text("question.chain_rpc.option.skip_mainnet_comparison"),
                    False,
                    {"confirmed_config.MAINNET_RPC_URL_REVIEWED": True},
                ),
            ],
            kind="confirm_or_value",
            manual_input_allowed=True,
            validation={"value_type": "url"},
        )
    return _choice(
        "endpoint_process",
        "MAINNET_RPC_URL_REVIEWED",
        (
            question_text("question.chain_rpc.mainnet_review.sync_observe.prompt")
            if sync_observe
            else question_text("question.chain_rpc.mainnet_review.rpc_benchmark.prompt")
        ),
        "MAINNET_RPC_URL_REVIEWED",
        [
            _answer_option("yes", question_text("question.chain_rpc.option.yes"), True, {"confirmed_config.MAINNET_RPC_URL_REVIEWED": True}),
            _answer_option(
                "no",
                question_text("question.chain_rpc.option.no"),
                False,
                {
                    "confirmed_config.MAINNET_RPC_URL_REVIEWED": True,
                    "confirmed_config.MAINNET_RPC_URL_DISABLED": True,
                },
            ),
        ],
        kind="confirm_or_value",
        manual_input_allowed=True,
        validation={"value_type": "url"},
    )


def _chain_question(state: AgentGraphState) -> dict[str, Any]:
    target_mode = state.get("target_mode") or ("sync-observe" if state.get("workflow_mode") == "sync_observe" else "")
    return manual_question(
        "chain_identity",
        "chain",
        question_text(
            "question.chain_rpc.chain.prompt",
            target_mode=target_mode or "not-selected",
        ),
        field="chain",
        kind="chain",
        accepted_action_types=("choose_chain",),
        manual_action={
            "type": "choose_chain",
            "value_argument": "chain_text",
        },
        queue_barrier=True,
        evidence_path="chain_identity.canonical",
    )


def _chain_selection_question(
    state: AgentGraphState,
    *,
    candidates: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    current_chain = normalize_scalar(
        (state.get("chain_identity") or {}).get("canonical")
    )
    normalized_candidates = [
        normalize_scalar(item)
        for item in candidates
        if normalize_scalar(item)
    ]
    if current_chain:
        prompt = question_text(
            "question.chain_rpc.chain_change.current.prompt",
            current_chain=current_chain,
        )
    elif normalized_candidates:
        prompt = question_text(
            "question.chain_rpc.chain_change.candidates.prompt",
            candidates=", ".join(normalized_candidates),
        )
    else:
        prompt = question_text("question.chain_rpc.chain_change.prompt")
    return manual_question(
        "chain_identity",
        "chain_change_input",
        prompt,
        field="chain_change_input",
        accepted_action_types=("choose_chain",),
        manual_action={
            "type": "choose_chain",
            "value_argument": "chain_text",
        },
        queue_barrier=True,
        evidence_path="chain_identity.change_candidate.canonical",
    )


def _target_change_scope_question(state: AgentGraphState) -> dict[str, Any]:
    preserved = {
        "target_mode": state.get("target_mode") or "",
        "chain_identity.canonical": (state.get("chain_identity") or {}).get("canonical") or "",
        "rpc_mode": state.get("rpc_mode") or "",
        "workload.confirmed": bool((state.get("workload") or {}).get("confirmed")),
    }
    return _choice(
        "workload_rpc",
        "target_change_scope",
        question_text("question.chain_rpc.target_change_scope.prompt"),
        "target_change_scope",
        [
            _action_option("chain", question_text("question.chain_rpc.option.change_chain"), "chain", "request_chain_selection", {"pending_question.id": "chain_change_input"}),
            _action_option("target_mode", question_text("question.chain_rpc.option.change_target_mode"), "target_mode", "request_target_mode_selection", {"pending_question.id": "target_mode_select"}),
            _action_option(
                "cancel",
                question_text("question.chain_rpc.option.cancel_target_change"),
                "cancel",
                "cancel_target_change",
                preserved,
            ),
        ],
    )


def _target_mode_selection_question(state: AgentGraphState, *, include_current: bool = False) -> dict[str, Any]:
    current = normalize_target_mode(state.get("target_mode"))
    options = []
    for mode in ("fake-node", "real-node", "sync-observe"):
        if not include_current and mode == current:
            continue
        expected = {"pending_question.id": "target_mode_change_confirm"} if current and mode != current else {"target_mode": mode}
        options.append(_action_option(
            mode,
            question_text("question.chain_rpc.option.target_mode", mode=mode),
            mode,
            "choose_target_mode",
            expected,
            target_mode=mode,
            target_mode_explicit=True,
        ))
    question = _choice(
        "target_mode",
        "target_mode_select",
        (
            question_text("question.chain_rpc.target_mode.change.prompt")
            if current
            else question_text("question.chain_rpc.target_mode.select.prompt")
        ),
        "target_mode",
        options,
        queue_barrier=True,
    )
    question["barrier_policy"] = "exclusive_owner"
    return question


def _adapter_family_question(state: AgentGraphState, *, custom: bool = False) -> dict[str, Any]:
    options = [
        _answer_option(
            family,
            question_text(
                "question.chain_rpc.option.adapter_family",
                family="jsonrpc / EVM" if family == "jsonrpc" else family,
            ),
            family,
            {"chain_identity.adapter_family": family},
        )
        for family in SUPPORTED_FAMILIES
    ]
    options.append(
        _answer_option(
            "unsupported",
            question_text("question.chain_rpc.option.adapter_family_unsupported"),
            "unsupported",
            {"chain_identity.status": "unsupported_family_handoff"},
            return_policy="stop_after_response",
        )
    )
    return _choice(
        "endpoint_process" if custom else "chain_identity",
        "custom_rpc_adapter_family_confirm" if custom else "adapter_family_confirm",
        (
            question_text("question.chain_rpc.adapter_family.custom.prompt")
            if custom
            else question_text("question.chain_rpc.adapter_family.chain.prompt")
        ),
        "adapter_family",
        options,
        accepted_action_types=("choose_adapter_family",),
        queue_barrier=True,
        rpc_case="custom_rpc" if custom else "",
    )


def _case3_evidence_question(state: AgentGraphState) -> dict[str, Any]:
    evidence = list((state.get("secondary_handoff") or {}).get("evidence") or [])
    if evidence and (state.get("chain_identity") or {}).get("status") == "case3_collecting_evidence":
        return _choice(
            "chain_identity",
            "case3_evidence_next",
            question_text("question.chain_rpc.case3_evidence_next.prompt"),
            "case3_evidence_next",
            [
                _answer_option("add_more", question_text("question.chain_rpc.option.add_more_evidence"), "add_more", {"chain_identity.status": "case3_needs_evidence"}),
                _answer_option("generate", question_text("question.chain_rpc.option.generate_development_handoff"), "generate_handoff", {"secondary_handoff.status": "ready", "chain_identity.status": "needs_review_handoff"}, return_policy="stop_after_response"),
            ],
        )
    return manual_question(
        "chain_identity",
        "case3_protocol_evidence",
        question_text("question.chain_rpc.case3_protocol_evidence.prompt"),
        field="case3_protocol_evidence",
        kind="evidence",
        accepted_action_types=("secondary_handoff_command",),
        manual_action={
            "type": "secondary_handoff_command",
            "handoff_command": "append_evidence",
            "value_argument": "handoff_evidence",
        },
        evidence_path="secondary_handoff.evidence",
        structured_input_owner=True,
    )


def _schema_confirmation_question(state: AgentGraphState, case: str) -> dict[str, Any]:
    draft = draft_view(state)
    question_id = "new_chain_schema_confirm" if case == "new_chain" else "custom_rpc_schema_confirm"
    return _choice(
        "endpoint_process",
        question_id,
        _schema_confirmation_text(draft),
        question_id,
        [
            _answer_option("yes", question_text("question.chain_rpc.option.yes"), True, {"custom_rpc.catalog.draft.request_confirmed": True}),
            _answer_option("no", question_text("question.chain_rpc.option.no"), False, {f"{'chain_identity' if case == 'new_chain' else 'custom_rpc'}.status": "existing_family_needs_schema_evidence" if case == "new_chain" else "needs_schema_evidence"}),
        ],
        kind="yes_no",
        manual_input_allowed=True,
        accepted_action_types=("rpc_catalog_command",),
        manual_action={
            "type": "rpc_catalog_command",
            "catalog_command": "append_evidence",
            "value_argument": "rpc_schema_evidence",
        },
        queue_barrier=True,
        validation={"value_type": "evidence_contribution", "max_length": 65536},
        structured_input_owner=True,
        rpc_case=case,
    )


def _schema_confirmation_text(draft: dict[str, Any]) -> TextRef:
    params = draft.get("params")
    if isinstance(params, list):
        rows = []
        for item in params:
            if isinstance(item, dict):
                rows.append(
                    "{name}: json_type={json_type}, semantic_type={semantic_type}, "
                    "encoding={encoding}, meaning={meaning}, required={required}, "
                    "example={example}".format(
                        name=item.get("name") or f"param[{item.get('index', '?')}]",
                        json_type=item.get("json_type") or item.get("type") or "unknown",
                        semantic_type=item.get("semantic_type") or "unknown",
                        encoding=item.get("encoding") or "unknown",
                        meaning=item.get("meaning") or "unknown",
                        required=item.get("required", "unknown"),
                        example=json.dumps(item.get("example"), ensure_ascii=False),
                    )
                )
        params_summary = "; ".join(rows) if rows else "[]"
    else:
        params_summary = json.dumps(
            draft.get("params_json", []),
            ensure_ascii=False,
        )
    response_fields = draft.get("response_fields")
    field_rows = []
    if isinstance(response_fields, list):
        field_rows = [
            "{name}: type={field_type}, meaning={meaning}".format(
                name=item.get("name") or "<unnamed>",
                field_type=(
                    "/".join(
                        str(value)
                        for value in item.get("json_types") or ()
                        if str(value)
                    )
                    or item.get("json_type")
                    or item.get("type")
                    or "unknown"
                ),
                meaning=item.get("meaning") or "unknown",
            )
            for item in response_fields
            if isinstance(item, dict)
        ]
    conflicts = draft.get("conflicts")
    return question_text(
        "question.chain_rpc.schema_confirmation.prompt",
        evidence_kind=normalize_scalar(draft.get("evidence_kind")) or "unknown",
        transport=normalize_scalar(draft.get("transport")) or "unknown",
        method=normalize_scalar(draft.get("method")) or "<unknown>",
        params_summary=params_summary,
        response_summary=normalize_scalar(draft.get("response_summary")) or "<unknown>",
        response_fields="; ".join(field_rows) or "<unknown>",
        response_schema_status=(
            "truncated; additional response fields may exist"
            if draft.get("response_schema_truncated")
            else "complete for supplied response evidence"
            if draft.get("response_schema_complete")
            else "not established"
        ),
        confidence=normalize_scalar(draft.get("confidence")) or "unknown",
        conflicts="; ".join(str(item) for item in conflicts)
        if isinstance(conflicts, list) and conflicts
        else "<none>",
        evidence_summary=normalize_scalar(draft.get("evidence_summary")) or "<none>",
    )


def _catalog_confirmation_question(state: AgentGraphState, case: str) -> dict[str, Any]:
    draft = draft_view(state)
    phase = normalize_scalar(draft.get("phase"))
    parameter_index = next_parameter_to_confirm(state)
    if phase == "parameter_confirmation" and parameter_index is not None:
        return _parameter_confirmation_question(state, case, parameter_index)
    if phase == "response_confirmation":
        return _response_confirmation_question(state, case)
    if phase == "probe_confirmation":
        probe = draft.get("probe") if isinstance(draft.get("probe"), dict) else {}
        return _probe_confirmation_question(state, case, retry=bool(probe and not probe.get("ready")))
    return _schema_confirmation_question(state, case)


def _parameter_confirmation_question(state: AgentGraphState, case: str, index: int) -> dict[str, Any]:
    draft = draft_view(state)
    params = draft.get("params") if isinstance(draft.get("params"), list) else []
    parameter = params[index] if index < len(params) and isinstance(params[index], dict) else {}
    question_id = "new_chain_parameter_confirm" if case == "new_chain" else "custom_rpc_parameter_confirm"
    name = normalize_scalar(parameter.get("name")) or f"param[{index}]"
    prompt = question_text(
        "question.chain_rpc.parameter_confirmation.prompt",
        ordinal=index + 1,
        total=len(params),
        name=name,
        parameter_index=str(parameter.get("index", index)),
        json_type=normalize_scalar(parameter.get("json_type")) or "unknown",
        semantic_type=normalize_scalar(parameter.get("semantic_type")) or "unknown",
        encoding=normalize_scalar(parameter.get("encoding")) or "unknown",
        meaning=normalize_scalar(parameter.get("meaning")) or "unknown",
        requirement=normalize_scalar(parameter.get("required")) or "unknown",
        example=repr(parameter.get("example")),
    )
    owner_key = "chain_identity" if case == "new_chain" else "custom_rpc"
    return _choice(
        "endpoint_process",
        question_id,
        prompt,
        question_id,
        [
            _answer_option("yes", question_text("question.chain_rpc.option.yes"), True, {f"{owner_key}.status": "existing_family_schema_needs_confirmation" if case == "new_chain" else "schema_needs_confirmation"}),
            _answer_option("no", question_text("question.chain_rpc.option.no"), False, {f"{owner_key}.status": "existing_family_needs_schema_evidence" if case == "new_chain" else "needs_schema_evidence"}),
        ],
        kind="yes_no",
        manual_input_allowed=True,
        accepted_action_types=("rpc_catalog_command",),
        manual_action={
            "type": "rpc_catalog_command",
            "catalog_command": "append_evidence",
            "value_argument": "rpc_schema_evidence",
        },
        queue_barrier=True,
        validation={"value_type": "evidence_contribution", "max_length": 65536},
        structured_input_owner=True,
        rpc_case=case,
    )


def _probe_confirmation_question(state: AgentGraphState, case: str, *, retry: bool = False) -> dict[str, Any]:
    draft = draft_view(state)
    method = normalize_scalar(draft.get("method")) or "<unknown>"
    question_id = "new_chain_probe_confirm" if case == "new_chain" else "custom_rpc_probe_confirm"
    response_confirmed = bool(draft.get("response_confirmed"))
    prompt = question_text(
        "question.chain_rpc.probe_confirmation.prompt",
        prior_probe_status="failed" if retry else "not-run",
        contract_status="request-and-response-confirmed"
        if response_confirmed
        else "request-confirmed-response-pending",
        method=method,
    )
    return _choice(
        "endpoint_process",
        question_id,
        prompt,
        question_id,
        [
            _answer_option("yes", question_text("question.chain_rpc.option.yes"), True, {"custom_rpc.catalog.draft.request_confirmed": True}),
            _answer_option("no", question_text("question.chain_rpc.option.no"), False, {"custom_rpc.catalog.draft.request_confirmed": False}),
        ],
        kind="yes_no",
        manual_input_allowed=True,
        accepted_action_types=("rpc_catalog_command",),
        manual_action={
            "type": "rpc_catalog_command",
            "catalog_command": "append_evidence",
            "value_argument": "rpc_schema_evidence",
        },
        queue_barrier=True,
        validation={"value_type": "evidence_contribution", "max_length": 65536},
        structured_input_owner=True,
        rpc_case=case,
    )


def _response_confirmation_question(state: AgentGraphState, case: str) -> dict[str, Any]:
    draft = draft_view(state)
    observed = draft.get("observed_response") if isinstance(draft.get("observed_response"), dict) else {}
    response_hash = str(observed.get("shape_hash") or "<unknown>")
    response_sample = str(observed.get("sample") or "<unavailable>")
    if observed:
        prompt = question_text(
            "question.chain_rpc.response_confirmation.observed.prompt",
            response_hash=response_hash,
            response_sample=response_sample,
        )
    else:
        prompt = question_text(
            "question.chain_rpc.response_confirmation.expected.prompt",
            response_summary=normalize_scalar(draft.get("response_summary")) or "<unknown>",
            response_fields=json.dumps(
                draft.get("response_fields") or [],
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
    question_id = "new_chain_response_confirm" if case == "new_chain" else "custom_rpc_response_confirm"
    owner_key = "chain_identity" if case == "new_chain" else "custom_rpc"
    return _choice(
        "endpoint_process",
        question_id,
        prompt,
        question_id,
        [
            _answer_option("yes", question_text("question.chain_rpc.option.yes"), True, {"custom_rpc.catalog.draft.response_confirmed": True}),
            _answer_option("no", question_text("question.chain_rpc.option.no"), False, {f"{owner_key}.status": "existing_family_needs_schema_evidence" if case == "new_chain" else "needs_schema_evidence"}),
        ],
        kind="yes_no",
        manual_input_allowed=True,
        accepted_action_types=("rpc_catalog_command",),
        manual_action={
            "type": "rpc_catalog_command",
            "catalog_command": "append_evidence",
            "value_argument": "rpc_schema_evidence",
        },
        queue_barrier=True,
        validation={"value_type": "evidence_contribution", "max_length": 65536},
        structured_input_owner=True,
        rpc_case=case,
    )


def _continue_question(state: AgentGraphState, case: str) -> dict[str, Any]:
    new_chain = case == "new_chain"
    return _choice(
        "endpoint_process",
        "new_chain_method_continue" if new_chain else "custom_rpc_continue",
        (
            question_text("question.chain_rpc.continue.new_chain.prompt")
            if new_chain
            else question_text("question.chain_rpc.continue.custom_rpc.prompt")
        ),
        "new_chain_method_continue" if new_chain else "custom_rpc_continue",
        [
            _answer_option(
                "add_another",
                (
                    question_text("question.chain_rpc.option.add_another_new_chain_method")
                    if new_chain
                    else question_text("question.chain_rpc.option.add_another_custom_method")
                ),
                "add_another",
                {f"{'chain_identity' if new_chain else 'custom_rpc'}.status": "existing_family_needs_method" if new_chain else "needs_method"},
            ),
            _answer_option(
                "finish",
                (
                    question_text("question.chain_rpc.option.finish_new_chain_methods")
                    if new_chain
                    else question_text("question.chain_rpc.option.finish_custom_methods")
                ),
                "finish",
                {"custom_rpc.catalog.finished": True},
            ),
        ],
        queue_barrier=True,
        rpc_case=case,
    )


def _scope_question(state: AgentGraphState, case: str) -> dict[str, Any]:
    if case == "new_chain":
        return _choice(
            "endpoint_process",
            "new_chain_workload_scope",
            question_text("question.chain_rpc.scope.new_chain.prompt"),
            "new_chain_workload_scope",
            [
                _answer_option(
                    "single_replace",
                    question_text("question.chain_rpc.option.single_validated_method"),
                    "single",
                    {"chain_identity.workload_scope": "single_replace"},
                ),
                _answer_option(
                    "mixed_replace",
                    question_text("question.chain_rpc.option.mixed_validated_methods"),
                    "mixed",
                    {"chain_identity.status": "existing_family_needs_weights"},
                ),
            ],
            accepted_action_types=("rpc_workload_command",),
            queue_barrier=True,
            requires_capabilities=("target_mode", "chain_identity"),
            rpc_case=case,
        )
    return _choice(
        "endpoint_process",
        "custom_rpc_scope",
        question_text("question.chain_rpc.scope.custom_rpc.prompt"),
        "custom_rpc_scope",
        [
            _answer_option(
                "single_replace",
                question_text("question.chain_rpc.option.single_custom_method"),
                "single",
                {"custom_rpc.scope": "single_replace"},
            ),
            _answer_option(
                "mixed_replace",
                question_text("question.chain_rpc.option.mixed_custom_only"),
                "mixed_replace",
                {"custom_rpc.status": "needs_weights"},
            ),
            _answer_option(
                "mixed_add",
                question_text("question.chain_rpc.option.mixed_defaults_plus_custom"),
                "mixed_add",
                {"custom_rpc.status": "needs_weights"},
            ),
        ],
        accepted_action_types=("rpc_workload_command",),
        queue_barrier=True,
        requires_capabilities=("target_mode", "chain_identity"),
        rpc_case=case,
    )


def _single_method_question(state: AgentGraphState, case: str) -> dict[str, Any]:
    methods = catalog_method_names(state)
    question_id = "new_chain_single_method" if case == "new_chain" else "custom_rpc_single_method"
    return _choice(
        "endpoint_process",
        question_id,
        question_text("question.chain_rpc.single_method.prompt"),
        question_id,
        [
            _answer_option(
                method,
                question_text("question.chain_rpc.option.rpc_method", method=method),
                method,
                {"workload.methods": [method]},
            )
            for method in methods
        ],
        requires_capabilities=("target_mode", "chain_identity"),
        rpc_case=case,
    )


def _weights_question(state: AgentGraphState, case: str) -> dict[str, Any]:
    methods = catalog_method_names(state)
    if case == "custom_rpc" and not methods:
        chain = normalize_scalar((state.get("chain_identity") or {}).get("canonical"))
        methods = list(default_workload(chain).get("methods") or []) if chain else []
    example = _weight_example(methods)
    question_id = "new_chain_custom_weights" if case == "new_chain" else "custom_rpc_weights"
    return manual_question(
        "endpoint_process",
        question_id,
        (
            question_text(
                "question.chain_rpc.weights.new_chain.prompt",
                methods=", ".join(methods) or "<none>",
                example=example,
            )
            if case == "new_chain"
            else question_text(
                "question.chain_rpc.weights.custom_rpc.prompt",
                methods=", ".join(methods) or "<none>",
                example=example,
            )
        ),
        field=question_id,
        validation={"input_mode": "rpc_weights"},
        structured_input_owner=True,
        requires_capabilities=("target_mode", "chain_identity"),
        domain_context={"rpc_case": case},
        candidate_bindings=({
            "type": "rpc_workload_command",
            "value_argument": "rpc_weights",
        },),
    )


def _choice(
    group: str,
    question_id: str,
    prompt: TextRef,
    field: str,
    options: list[dict[str, Any]],
    *,
    kind: str = "numbered_choice",
    manual_input_allowed: bool = False,
    accepted_action_types: tuple[str, ...] = (),
    manual_action: dict[str, Any] | None = None,
    queue_barrier: bool = False,
    validation: dict[str, Any] | None = None,
    structured_input_owner: bool = False,
    requires_capabilities: tuple[str, ...] = (),
    rpc_case: str = "",
) -> dict[str, Any]:
    return choice_question(
        group,
        question_id,
        prompt,
        field=field,
        options=options,
        kind=kind,
        manual_input_allowed=manual_input_allowed,
        accepted_action_types=accepted_action_types,
        manual_action=manual_action,
        queue_barrier=queue_barrier,
        barrier_policy="exclusive_owner" if queue_barrier else "",
        validation=validation,
        structured_input_owner=structured_input_owner,
        requires_capabilities=requires_capabilities,
        domain_context={"rpc_case": rpc_case} if rpc_case else {},
    )


def _action_option(option_id: str, label: TextRef, value: Any, action_type: str, expected_patch: dict[str, Any], **arguments: Any) -> dict[str, Any]:
    return {"id": option_id, "label": label, "value": value, "action": {"type": action_type, **arguments}, "expected_patch": expected_patch}


def _answer_option(option_id: str, label: TextRef, value: Any, expected_patch: dict[str, Any], *, return_policy: str = "fallback") -> dict[str, Any]:
    return {"id": option_id, "label": label, "value": value, "action": {"type": "answer_pending", "answer": value}, "expected_patch": expected_patch, "return_policy": return_policy}

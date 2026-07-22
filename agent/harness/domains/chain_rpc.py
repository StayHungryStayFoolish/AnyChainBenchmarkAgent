"""Chain identity, endpoint, workload, and Case 1/2/3 domain behavior."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any, Mapping

from ..contracts import ActionProposal, HandlerResult
from ..input_values import (
    adapter_family_hint,
    extract_url_candidate,
    normalize_scalar,
    normalize_target_mode,
    schema_evidence_from_turn_text,
)
from ..intent import extract_chain_mention
from ..localization import localized
from ..questions import choice_question, manual_question, render_question
from ..routing import chain_auxiliary_fields_needed, chain_identity_confirmed
from ..state import AgentGraphState
from ..transitions import (
    clear_effective_custom_rpc_workload,
    invalidate_for_chain_change,
    invalidate_for_rpc_mode_change,
    invalidate_for_target_mode,
    mark_group_reconfigured,
    record_group_invalidations,
)

from agent.knowledge.chain_identity import canonicalize_chain_scalar, repo_chain_names
from agent.onboarding.families import SUPPORTED_FAMILIES
from agent.planners import question_prompts
from agent.validators.rpc_workload import default_workload
from agent.validators.endpoint_probe import health_probe_methods, validate_rpc_endpoint
from .rpc_catalog import catalog_method_names, draft_view
CHAIN_RPC_GROUPS = {
    "target_mode",
    "chain_identity",
    "endpoint_process",
    "chain_auxiliary_endpoints",
    "workload_rpc",
    "target_samples_fixtures",
}
CHAIN_RPC_ACTIONS = frozenset(
    {
        "choose_target_mode",
        "choose_chain",
        "change_chain",
        "choose_adapter_family",
        "set_rpc_mode",
        "rpc_catalog_command",
        "secondary_handoff_command",
        "rpc_workload_command",
        "use_default_workload",
        "configure_workload_weights",
        "request_target_change",
        "request_chain_selection",
        "request_target_mode_selection",
        "cancel_target_change",
    }
)
SUPPORTED_ADAPTER_FAMILIES = frozenset(SUPPORTED_FAMILIES)

__all__ = [
    "CHAIN_RPC_ACTIONS",
    "CHAIN_RPC_GROUPS",
    "apply_chain_rpc_action",
    "apply_chain_rpc_answer",
    "cancel_chain_rpc_question",
    "question_for_chain_rpc",
    "SUPPORTED_ADAPTER_FAMILIES",
]


from .chain_handoff import (_prepare_case2_handoff, _prepare_case3_handoff, _promote_case2_endpoint, _record_case3_evidence)
from .chain_identity import (_apply_chain_candidate, _apply_chain_change_decision, _apply_unknown_chain_decision, _chain_ambiguity_question, _confirm_custom_rpc_family, _enter_case_for_adapter_family, _identity_confirmation_question, _origin_text, _preserve_same_chain, _request_chain_change, _request_target_mode_change, _resolution_from_arguments, _target_mode_is_explicit)
from .chain_rpc_questions import (_action_option, _adapter_family_question, _answer_option, _case3_evidence_question, _chain_question, _choice, _endpoint_probe_completion, _endpoint_validation_question, _mainnet_review_question, _target_change_scope_question, _target_mode_selection_question)
from .chain_rpc_support import (_adapter_family, _answer_result, _chain_confirmed, _chain_rpc_draft, _handoff_stops, _invalidate_execution, _invalidate_groups, _next_group, _result, _set_control, _workload_default_prompt, is_existing_family_lifecycle)
from .rpc_endpoint import (
    _apply_endpoint_answer,
    _apply_method_answer,
    _apply_schema_evidence,
    _confirm_method_probe,
    _confirm_parameter_contract,
    _confirm_probe_response,
    _confirm_request_contract,
)
from .rpc_workload import (_apply_continue, _apply_requested_workload, _apply_scope, _apply_weights, _set_single_workload)
from .rpc_catalog import append_evidence, strict_method_identity

def question_for_chain_rpc(state: AgentGraphState, group: str) -> dict[str, Any] | None:
    """Return the next blocking question owned by a Chain/RPC group."""

    if group not in CHAIN_RPC_GROUPS:
        return None
    language = str(state.get("language") or "en")
    confirmed = state.get("confirmed_config") or {}
    identity = state.get("chain_identity") or {}
    if group == "target_mode":
        return _target_mode_selection_question(state, include_current=not bool(state.get("target_mode")))
    if group == "chain_identity":
        identity_question = _identity_confirmation_question(state)
        if identity_question:
            return identity_question
        if identity.get("status") in {"needs_protocol_confirmation", "needs_adapter_family_confirmation"}:
            return _adapter_family_question(state)
        if identity.get("status") in {
            "unsupported_family_handoff",
            "case3_collecting_evidence",
            "case3_needs_evidence",
        }:
            return _case3_evidence_question(state)
        if identity.get("canonical") and identity.get("status") == "confirmed":
            return None
        return _chain_question(state)
    if group == "endpoint_process":
        validation = _endpoint_validation_question(state)
        if validation:
            return validation
        evidence = state.get("endpoint_evidence") or {}
        sync = state.get("sync_observe") or {}
        if state.get("target_mode") == "real-node" and not evidence.get("local_rpc_url_ready"):
            return manual_question(
                group,
                "LOCAL_RPC_URL",
                localized(
                    language,
                    "请提供被测真实节点的 LOCAL_RPC_URL。该 endpoint 会先被探测验证。",
                    "Provide the LOCAL_RPC_URL for the real node under test. The endpoint will be probed before it is trusted.",
                ),
                field="LOCAL_RPC_URL",
                kind="url",
                evidence_path="endpoint_evidence.local_rpc_url_ready",
                rejection_evidence_value=False,
                completion_effect=_endpoint_probe_completion(language),
            )
        if state.get("target_mode") == "real-node" and not confirmed.get("BLOCKCHAIN_PROCESS_NAMES"):
            return manual_question(
                group,
                "BLOCKCHAIN_PROCESS_NAMES",
                localized(
                    language,
                    "请输入真实节点进程名或命令行片段，用于资源归因。",
                    "Enter the real node process name or command-line fragment for resource attribution.",
                ),
                field="BLOCKCHAIN_PROCESS_NAMES",
                validation={"value_type": "bounded_text", "max_length": 512},
                evidence_path="confirmed_config.BLOCKCHAIN_PROCESS_NAMES",
            )
        if state.get("target_mode") == "real-node" and not confirmed.get("MAINNET_RPC_URL_REVIEWED"):
            return _mainnet_review_question(state, sync_observe=False)
        if (
            state.get("workflow_mode") == "sync_observe"
            and sync.get("source") in {"existing_local_node", "endpoint_only"}
            and not evidence.get("sync_rpc_url_ready")
        ):
            return manual_question(
                group,
                "SYNC_OBSERVE_RPC_URL",
                localized(
                    language,
                    "已选择的本地节点进程用于 CPU、线程和资源归因；还需要一个可访问的真实节点 RPC endpoint，用于观测同步高度和健康状态。fake-node 不提供真实同步/import metrics，因此该 endpoint 必须验证。",
                    "The selected local node process is used for CPU, thread, and resource attribution. A reachable real-node RPC endpoint is still required to observe sync height and health. fake-node does not provide real sync/import metrics, so the endpoint must be probed.",
                ),
                field="SYNC_OBSERVE_RPC_URL",
                kind="url",
                evidence_path="endpoint_evidence.sync_rpc_url_ready",
                rejection_evidence_value=False,
                completion_effect=_endpoint_probe_completion(language),
            )
        if (
            state.get("workflow_mode") == "sync_observe"
            and sync.get("source") == "existing_local_node"
            and not confirmed.get("BLOCKCHAIN_PROCESS_NAMES")
        ):
            return manual_question(
                group,
                "BLOCKCHAIN_PROCESS_NAMES",
                localized(
                    language,
                    "请输入节点进程名或命令行片段，用于 sync-observe 的 CPU/线程归因。",
                    "Enter the node process name or command-line fragment for sync-observe CPU/thread attribution.",
                ),
                field="BLOCKCHAIN_PROCESS_NAMES",
                validation={"value_type": "bounded_text", "max_length": 512},
                evidence_path="confirmed_config.BLOCKCHAIN_PROCESS_NAMES",
            )
        if (
            state.get("workflow_mode") == "sync_observe"
            and sync.get("source") in {"existing_local_node", "endpoint_only"}
            and not confirmed.get("MAINNET_RPC_URL_REVIEWED")
        ):
            return _mainnet_review_question(state, sync_observe=True)
        return None
    if group == "chain_auxiliary_endpoints":
        chain = normalize_scalar(identity.get("canonical"))
        for field in chain_auxiliary_fields_needed(chain):
            if not confirmed.get(field):
                return choice_question(
                    group,
                    field,
                    question_prompts.chain_auxiliary_field_prompt(chain, field, language=language),
                    field=field,
                    kind="manual_value",
                    manual_input_allowed=True,
                    options=[{
                        "id": "skip",
                        "label": "跳过（未配置）" if language.startswith("zh") else "Skip (not configured)",
                        "value": "none",
                        "expected_patch": {f"confirmed_config.{field}": "none"},
                    }],
                )
        return None
    if group == "workload_rpc":
        if not chain_identity_confirmed(state):
            return None
        if not state.get("rpc_mode"):
            return _choice(
                group,
                "rpc_mode",
                localized(language, "请选择 RPC 模式。", "Choose RPC mode."),
                "rpc_mode",
                [
                    _action_option("single", "single", "single", "set_rpc_mode", {"rpc_mode": "single"}, rpc_mode="single", mutation_explicit=True),
                    _action_option("mixed", "mixed", "mixed", "set_rpc_mode", {"rpc_mode": "mixed"}, rpc_mode="mixed", mutation_explicit=True),
                ],
                queue_barrier=True,
            )
        if not (state.get("workload") or {}).get("confirmed"):
            zh = language.startswith("zh")
            options = [
                _action_option(
                    "default",
                    "使用默认值" if zh else "Use defaults",
                    "default",
                    "use_default_workload",
                    {"workload.confirmed": True},
                ),
                _action_option(
                    "custom_rpc",
                    "添加自定义 RPC method" if zh else "Add custom RPC method",
                    "custom_rpc",
                    "rpc_catalog_command",
                    {"custom_rpc.status": "needs_endpoint"},
                    catalog_command="enter",
                ),
            ]
            if state.get("rpc_mode") == "mixed":
                options.append(
                    _action_option(
                        "weights",
                        "调整 mixed 权重" if zh else "Adjust mixed weights",
                        "weights",
                        "configure_workload_weights",
                        {"custom_rpc.status": "needs_weights"},
                    )
                )
            options.append(
                _action_option(
                    "change_target",
                    "更换链或目标模式" if zh else "Change chain or target mode",
                    "change_target",
                    "request_target_change",
                    {"pending_question.id": "target_change_scope"},
                )
            )
            return _choice(
                group,
                "workload_confirm",
                _workload_default_prompt(state),
                "workload_choice",
                options,
                queue_barrier=True,
            )
        return None
    if (
        group == "target_samples_fixtures"
        and state.get("target_mode") == "fake-node"
        and identity.get("status") == "existing_family_runtime_choice"
    ):
        zh = language.startswith("zh")
        return _choice(
            group,
            "new_chain_runtime_choice",
            localized(
                language,
                "新链 endpoint 和 RPC schema 已验证。当前 fake-node 还没有该链的 fixture，不能直接声称 fake-node 可运行。请选择下一步。",
                "The new-chain endpoint and RPC schema are validated. fake-node has no fixture for this chain yet, so it cannot be claimed runnable as fake-node. Choose the next step.",
            ),
            "new_chain_runtime_choice",
            [
                _answer_option(
                    "real_node",
                    "使用已验证 endpoint 切到 real-node 路径继续" if zh else "Use the verified endpoint and continue as real-node",
                    "use_verified_endpoint_real_node",
                    {"target_mode": "real-node", "chain_identity.status": "confirmed"},
                ),
                _answer_option(
                    "handoff",
                    "生成 chain template / fixture / smoke 二次开发交接" if zh else "Generate chain-template / fixture / smoke development handoff",
                    "generate_handoff",
                    {"chain_identity.status": "needs_review_handoff", "secondary_handoff.status": "ready"},
                    return_policy="stop_after_response",
                ),
            ],
        )
    if group == "target_samples_fixtures" and (state.get("fixture_evidence") or {}).get("status") == "missing":
        zh = language.startswith("zh")
        missing = [str(item.get("method") or "") for item in (state.get("fixture_evidence") or {}).get("missing", [])]
        return _choice(
            group,
            "custom_rpc_fixture_choice",
            localized(
                language,
                f"当前 fake-node 缺少本次自定义 workload 的 fixture：{', '.join(filter(None, missing)) or '<unknown>'}。endpoint/schema 验证不能替代 fixture。请选择下一步。",
                f"The current fake-node lacks fixtures for this custom workload: {', '.join(filter(None, missing)) or '<unknown>'}. Endpoint/schema validation does not replace fixture evidence. Choose the next step.",
            ),
            "custom_rpc_fixture_choice",
            [
                _answer_option("defaults", "改用当前链模板默认 workload" if zh else "Use the chain-template default workload", "use_template_defaults", {"workload.choice": "default"}),
                _answer_option("real_node", "切换到 real-node，并单独验证最终 LOCAL_RPC_URL" if zh else "Switch to real-node and separately validate the final LOCAL_RPC_URL", "switch_real_node", {"target_mode": "real-node"}),
                _answer_option("handoff", "生成 fixture 录制与验证交接" if zh else "Generate a fixture recording and validation handoff", "generate_fixture_handoff", {"secondary_handoff.status": "ready"}, return_policy="stop_after_response"),
            ],
            queue_barrier=True,
        )
    return None


def _candidate_identity_from_action(
    state: AgentGraphState,
    raw: str,
    arguments: Mapping[str, Any],
) -> str:
    """Separate a chain identity from a planner-supplied evidence sentence."""

    if arguments.get("chain_candidates"):
        return raw
    evidence = normalize_scalar(arguments.get("source_evidence"))
    origin = normalize_scalar(_origin_text(state, arguments))
    if not origin or normalize_scalar(raw) != origin:
        return raw
    extraction_text = evidence if evidence and evidence.casefold() in origin.casefold() else origin
    mention = extract_chain_mention(state, extraction_text)
    if mention.get("found") is not True:
        return raw
    if normalize_scalar(mention.get("confidence")).casefold() not in {"medium", "high"}:
        return raw
    candidate = normalize_scalar(mention.get("chain_text"))
    if not candidate or candidate.casefold() not in origin.casefold():
        return raw
    return candidate


def apply_chain_rpc_action(state: AgentGraphState, action: ActionProposal) -> HandlerResult:
    """Validate and apply one registered Chain/RPC action."""

    if action.action_type not in CHAIN_RPC_ACTIONS:
        return HandlerResult(blocker=f"unsupported chain/RPC action: {action.action_type}")
    next_state = _chain_rpc_draft(state)
    arguments = dict(action.arguments)
    action_type = action.action_type
    if action_type == "choose_target_mode":
        mode = normalize_target_mode(arguments.get("target_mode"))
        if not mode:
            return HandlerResult(blocker="target_mode must be fake-node, real-node, or sync-observe")
        if not next_state.get("target_mode") and not _target_mode_is_explicit(next_state, mode, arguments):
            next_state.setdefault("action_errors", []).append(
                {
                    "action": {
                        "type": action_type,
                        **deepcopy(arguments),
                    },
                    "error": "target_mode_not_explicit",
                }
            )
            return _result(state, next_state, action, completion="unchanged")
        previous = normalize_target_mode(next_state.get("target_mode"))
        if previous and previous != mode:
            _request_target_mode_change(next_state, mode)
            return _result(state, next_state, action, completion="blocked")
        if previous == mode:
            next_state["workflow_mode"] = "sync_observe" if mode == "sync-observe" else "rpc_benchmark"
            return _result(state, next_state, action, completion="unchanged")
        next_state["target_mode"] = mode
        next_state["workflow_mode"] = "sync_observe" if mode == "sync-observe" else "rpc_benchmark"
        invalidate_for_target_mode(next_state, previous_mode=previous)
        _set_control(next_state, 'active_group', "chain_identity" if not _chain_confirmed(next_state) else _next_group(next_state))
        _set_control(next_state, 'pending_question', {})
        mark_group_reconfigured(next_state, "target_mode")
        return _result(state, next_state, action)
    if action_type in {"choose_chain", "change_chain"}:
        raw = normalize_scalar(arguments.get("chain_text"))
        candidate_mode = normalize_target_mode(arguments.get("target_mode"))
        if candidate_mode and _target_mode_is_explicit(next_state, candidate_mode, arguments) and not next_state.get("target_mode"):
            next_state["target_mode"] = candidate_mode
            next_state["workflow_mode"] = "sync_observe" if candidate_mode == "sync-observe" else "rpc_benchmark"
            invalidate_for_target_mode(next_state)
        if not raw and candidate_mode:
            mention = extract_chain_mention(next_state, _origin_text(next_state, arguments))
            if bool(mention.get("found")) and normalize_scalar(mention.get("confidence")).casefold() in {"medium", "high"}:
                raw = normalize_scalar(mention.get("chain_text"))
        if not raw:
            _set_control(next_state, 'active_group', "chain_identity")
            _set_control(next_state, 'pending_question', _chain_question(next_state))
            _set_control(next_state, 'visible_response', [render_question(next_state["pending_question"], next_state.get("language", "en"))])
            return _result(state, next_state, action, completion="blocked")
        raw = _candidate_identity_from_action(next_state, raw, arguments)
        current = canonicalize_chain_scalar(
            normalize_scalar((next_state.get("chain_identity") or {}).get("canonical")),
            known_chains=set(repo_chain_names()),
        )
        canonical = canonicalize_chain_scalar(raw, known_chains=set(repo_chain_names()))
        if current and canonical == current:
            _preserve_same_chain(next_state, current)
            return _result(state, next_state, action, completion="unchanged")
        ambiguity = _chain_ambiguity_question(next_state, raw, arguments)
        if ambiguity:
            _set_control(next_state, 'active_group', "chain_identity")
            _set_control(next_state, 'pending_question', ambiguity)
            _set_control(next_state, 'visible_response', [render_question(ambiguity, next_state.get("language", "en"))])
            return _result(state, next_state, action, completion="blocked")
        resolution = _resolution_from_arguments(arguments)
        if action_type == "change_chain" or current:
            _request_chain_change(next_state, raw, arguments, resolution=resolution)
            return _result(state, next_state, action, completion="blocked" if next_state.get("pending_question") else "completed")
        _apply_chain_candidate(next_state, raw, resolution=resolution)
        return _result(
            state,
            next_state,
            action,
            completion="blocked" if next_state.get("pending_question") else "completed",
            stop=_handoff_stops(next_state),
        )
    if action_type == "set_rpc_mode":
        if not chain_identity_confirmed(next_state):
            return HandlerResult(blocker="RPC mode requires a confirmed chain identity")
        mode = normalize_scalar(arguments.get("rpc_mode")).casefold()
        if mode not in {"single", "mixed"}:
            return HandlerResult(blocker="rpc_mode must be single or mixed")
        if next_state.get("rpc_mode") != mode:
            invalidate_for_rpc_mode_change(next_state)
        next_state["rpc_mode"] = mode
        _set_control(next_state, 'active_group', "workload_rpc")
        _set_control(next_state, 'pending_question', question_for_chain_rpc(next_state, "workload_rpc") or {})
        _set_control(next_state, 'visible_response', [render_question(next_state["pending_question"], next_state.get("language", "en"))] if next_state["pending_question"] else [])
        return _result(state, next_state, action, completion="blocked" if next_state.get("pending_question") else "completed")
    if action_type == "use_default_workload":
        chain = normalize_scalar((next_state.get("chain_identity") or {}).get("canonical"))
        rpc_mode = normalize_scalar(next_state.get("rpc_mode"))
        if not chain_identity_confirmed(next_state) or not chain or rpc_mode not in {"single", "mixed"}:
            return HandlerResult(blocker="a confirmed chain and rpc_mode are required before accepting defaults")
        defaults = default_workload(chain)
        if not defaults.get("exists"):
            return HandlerResult(blocker=f"no default workload exists for {chain}")
        if rpc_mode == "single":
            methods = [normalize_scalar(defaults.get("single"))]
            methods = [method for method in methods if method]
            weights: dict[str, int] = {}
        else:
            rows = [row for row in defaults.get("mixed_weighted") or [] if isinstance(row, dict)]
            methods = [normalize_scalar(row.get("method")) for row in rows if normalize_scalar(row.get("method"))]
            weights = {normalize_scalar(row.get("method")): int(row.get("weight") or 0) for row in rows if normalize_scalar(row.get("method"))}
            if not methods or sum(weights.values()) != 100 or set(methods) != set(weights):
                return HandlerResult(blocker=f"default mixed workload for {chain} must have exact weights totaling 100")
        next_state["workload"] = {
            "confirmed": True,
            "choice": "default",
            "methods": methods,
            "weights": weights,
            "replace_defaults": False,
            "job_local_override": False,
        }
        clear_effective_custom_rpc_workload(next_state)
        next_state["fixture_evidence"] = {}
        _invalidate_execution(next_state)
        _set_control(next_state, 'active_group', "workload_rpc")
        _set_control(next_state, 'pending_question', {})
        mark_group_reconfigured(next_state, "workload_rpc")
        return _result(state, next_state, action)
    if action_type == "configure_workload_weights":
        if not chain_identity_confirmed(next_state):
            return HandlerResult(blocker="workload weights require a confirmed chain identity")
        if next_state.get("rpc_mode") != "mixed":
            return HandlerResult(blocker="mixed workload weights are only available in mixed mode")
        next_state.setdefault("custom_rpc", {})["status"] = "needs_weights"
        next_state["custom_rpc"]["job_local_override"] = True
        next_state.setdefault("workload", {})["choice"] = "weights"
        _invalidate_execution(next_state)
        _set_control(next_state, 'active_group', "endpoint_process")
        _set_control(next_state, 'pending_question', {})
        return _result(state, next_state, action, completion="in_progress")
    if action_type == "request_target_change":
        _set_control(next_state, 'active_group', "workload_rpc")
        _set_control(next_state, 'pending_question', _target_change_scope_question(next_state))
        _set_control(next_state, 'visible_response', [render_question(next_state["pending_question"], next_state.get("language", "en"))])
        return _result(state, next_state, action, completion="blocked")
    if action_type == "request_chain_selection":
        _set_control(next_state, 'active_group', "chain_identity")
        current_chain = normalize_scalar((next_state.get("chain_identity") or {}).get("canonical"))
        candidates = [
            normalize_scalar(item)
            for item in arguments.get("chain_candidates") or []
            if normalize_scalar(item)
        ]
        if current_chain:
            prompt = localized(
                next_state.get("language", "en"),
                f"请输入要切换到的链名。当前链 `{current_chain}` 会保留到你确认新链为止。",
                f"Enter the replacement chain name. Current chain `{current_chain}` is retained until you confirm the new chain.",
            )
        elif candidates:
            rendered_candidates = ", ".join(f"`{item}`" for item in candidates)
            prompt = localized(
                next_state.get("language", "en"),
                f"你提到了可能的链 {rendered_candidates}，但尚未确认测试目标。请输入最终要测试的链名。",
                f"You mentioned possible chain targets {rendered_candidates}, but none is confirmed. Enter the final chain name to test.",
            )
        else:
            prompt = localized(
                next_state.get("language", "en"),
                "请输入要测试的链名。",
                "Enter the chain name to test.",
            )
        manual_action_type = "change_chain" if current_chain else "choose_chain"
        _set_control(next_state, 'pending_question', manual_question(
            "chain_identity",
            "chain_change_input",
            prompt,
            field="chain_change_input",
            accepted_action_types=("choose_chain", "change_chain"),
            manual_action={
                "type": manual_action_type,
                "value_argument": "chain_text",
            },
            queue_barrier=True,
            evidence_path="chain_identity.change_candidate.canonical",
        ))
        _set_control(next_state, 'visible_response', [render_question(next_state["pending_question"], next_state.get("language", "en"))])
        return _result(state, next_state, action, completion="blocked")
    if action_type == "request_target_mode_selection":
        _set_control(next_state, 'active_group', "target_mode")
        _set_control(next_state, 'pending_question', _target_mode_selection_question(next_state))
        _set_control(next_state, 'visible_response', [render_question(next_state["pending_question"], next_state.get("language", "en"))])
        return _result(state, next_state, action, completion="blocked")
    if action_type == "choose_adapter_family":
        family = adapter_family_hint(str(arguments.get("adapter_family") or ""))
        identity = next_state.setdefault("chain_identity", {})
        if family not in SUPPORTED_ADAPTER_FAMILIES:
            return HandlerResult(blocker="adapter_family must be one supported adapter family")
        if identity.get("status") not in {"needs_identity_confirmation", "needs_protocol_confirmation"}:
            return HandlerResult(blocker="adapter family can only be confirmed for an unresolved chain identity")
        _enter_case_for_adapter_family(next_state, family)
        return _result(state, next_state, action, completion="in_progress")
    if action_type == "cancel_target_change":
        _set_control(next_state, 'active_group', "workload_rpc")
        _set_control(next_state, 'pending_question', question_for_chain_rpc(next_state, "workload_rpc") or {})
        _set_control(next_state, 'visible_response', [render_question(next_state["pending_question"], next_state.get("language", "en"))] if next_state["pending_question"] else [])
        return _result(state, next_state, action, completion="blocked" if next_state.get("pending_question") else "completed")
    if action_type == "secondary_handoff_command":
        command = normalize_scalar(arguments.get("handoff_command"))
        evidence = str(arguments.get("handoff_evidence") or "").strip()
        identity = next_state.get("chain_identity") or {}
        handoff = next_state.get("secondary_handoff") or {}
        if command != "append_evidence":
            return HandlerResult(blocker="handoff_command must be append_evidence")
        if identity.get("case") != "case3" or handoff.get("status") != "collecting_evidence":
            return HandlerResult(blocker="protocol evidence can be appended only to an active Case 3 handoff")
        if not evidence:
            return HandlerResult(blocker="protocol handoff evidence is empty")
        _record_case3_evidence(next_state, evidence)
        return _result(state, next_state, action, completion="in_progress", stop=True)
    if action_type == "rpc_catalog_command":
        identity = next_state.get("chain_identity") or {}
        if identity.get("case") == "case3" or identity.get("adapter_family") == "unsupported":
            return HandlerResult(blocker="RPC catalog mutations require a supported adapter family or an explicit Case 2 transition")
        command = normalize_scalar(arguments.get("catalog_command"))
        command_arguments: dict[str, Any] = {}
        if command == "set_endpoint":
            command_arguments["rpc_endpoint"] = arguments.get("rpc_endpoint")
        elif command == "set_method":
            command_arguments["rpc_method"] = arguments.get("rpc_method")
        elif command == "append_evidence":
            command_arguments["rpc_schema_evidence"] = arguments.get("rpc_schema_evidence")
        elif command != "enter":
            return HandlerResult(blocker="unsupported rpc_catalog_command")
        arguments = command_arguments
    elif action_type == "rpc_workload_command":
        if not chain_identity_confirmed(next_state):
            return HandlerResult(blocker="custom RPC workload selection requires a confirmed chain identity")
        arguments = {
            key: value
            for key, value in arguments.items()
            if key in {"workload_scope", "rpc_weights", "finish_methods"}
        }
    identity_name = normalize_scalar(
        (next_state.get("chain_identity") or {}).get("canonical")
        or (next_state.get("chain_identity") or {}).get("raw")
    )
    if not identity_name:
        return HandlerResult(blocker="custom RPC setup requires an identified chain")
    method = normalize_scalar(arguments.get("rpc_method"))
    endpoint = extract_url_candidate(arguments.get("rpc_endpoint"))
    origin_text = _origin_text(next_state, arguments)
    evidence = str(arguments.get("rpc_schema_evidence") or "").strip()
    if not evidence:
        evidence = schema_evidence_from_turn_text(origin_text, method_hint=method)
    identity = next_state.setdefault("chain_identity", {})
    if is_existing_family_lifecycle(identity):
        previous_pending = deepcopy(next_state.get("pending_question") or {})
        _set_control(next_state, 'pending_question', {})
        if endpoint and identity.get("status") == "existing_family_needs_endpoint":
            _apply_endpoint_answer(next_state, "new_chain_endpoint", arguments.get("rpc_endpoint") or endpoint)
        if method and identity.get("status") == "existing_family_needs_method":
            _apply_method_answer(next_state, "new_chain_method", method)
        if evidence and identity.get("status") in {"existing_family_needs_schema_evidence", "existing_family_schema_needs_confirmation"}:
            if not _apply_schema_evidence(next_state, case="new_chain", evidence=evidence):
                _set_control(next_state, "pending_question", previous_pending)
        return _result(state, next_state, action, completion="in_progress")
    custom = next_state.setdefault("custom_rpc", {})
    custom["source_turn_text"] = origin_text
    custom["job_local_override"] = True
    if method:
        strict_method = strict_method_identity(method, adapter_family=_adapter_family(next_state))
        if strict_method:
            append_evidence(
                next_state,
                content=strict_method,
                source="typed_action",
                kind="method_identity",
                method=strict_method,
            )
            method = strict_method
        else:
            method = ""
            _set_control(next_state, 'visible_response', [localized(
                next_state.get("language", "en"),
                "typed custom-RPC action 中的 method 不符合当前协议族 grammar，未写入 catalog。请提供精确 method token 或完整 protocol request。",
                "The method in the typed custom-RPC action does not match the current adapter-family grammar and was not written to the catalog. Provide the exact method token or a complete protocol request.",
            )])
    requested_scope = normalize_scalar(arguments.get("workload_scope"))
    requested_weights = arguments.get("rpc_weights")
    if requested_scope in {"single_replace", "mixed_replace", "mixed_add"}:
        custom["requested_workload"] = {
            "scope": requested_scope,
            "weights": dict(requested_weights) if isinstance(requested_weights, Mapping) else {},
            "finish_methods": bool(arguments.get("finish_methods")),
        }
        if (
            custom["requested_workload"]["finish_methods"]
            and catalog_method_names(next_state)
            and not any((method, endpoint, evidence))
            and _apply_requested_workload(next_state)
        ):
            return _result(state, next_state, action, completion="completed")
    if endpoint:
        custom["status"] = "needs_endpoint"
    elif custom.get("endpoint_ready"):
        resumable_statuses = {
            "needs_method",
            "needs_schema_evidence",
            "schema_needs_confirmation",
            "method_validated_next",
            "needs_scope",
            "needs_single_method",
            "needs_weights",
        }
        if method:
            custom["status"] = "needs_schema_evidence"
        elif custom.get("status") not in resumable_statuses:
            custom["status"] = "needs_schema_evidence" if draft_view(next_state).get("method") else "needs_method"
    else:
        custom["status"] = "needs_endpoint"
    next_state.setdefault("workload", {})["choice"] = "custom_rpc"
    _set_control(next_state, 'active_group', "endpoint_process")
    previous_pending = deepcopy(next_state.get("pending_question") or {})
    _set_control(next_state, 'pending_question', {})
    record_group_invalidations(next_state, "endpoint_process", "workload_rpc")
    if endpoint:
        _apply_endpoint_answer(next_state, "custom_rpc_endpoint", arguments.get("rpc_endpoint") or endpoint)
        if custom.get("status") == "probe_failed":
            return _result(state, next_state, action, completion="in_progress")
    if evidence and custom.get("endpoint_ready") and custom.get("status") in {"needs_schema_evidence", "needs_method", "schema_needs_confirmation"}:
        if not draft_view(next_state).get("method"):
            _apply_method_answer(next_state, "custom_rpc_method", evidence)
        else:
            if not _apply_schema_evidence(next_state, case="custom_rpc", evidence=evidence):
                _set_control(next_state, "pending_question", previous_pending)
    return _result(state, next_state, action, completion="in_progress")


def _install_chain_rpc_next_question(state: AgentGraphState, group: str) -> None:
    question = question_for_chain_rpc(state, group)
    if question:
        _set_control(state, 'pending_question', question)
        _set_control(state, 'active_group', str(question.get("group") or group))


def apply_chain_rpc_answer(
    state: AgentGraphState,
    question: dict[str, Any],
    value: Any,
    user_text: str,
) -> HandlerResult:
    """Apply one already-coerced answer to a Chain/RPC question."""

    group = normalize_scalar(question.get("group"))
    if group not in CHAIN_RPC_GROUPS:
        return HandlerResult(blocker=f"question is not owned by chain/RPC: {group or '<missing>'}")
    next_state = _chain_rpc_draft(state)
    question_id = normalize_scalar(question.get("id"))
    _set_control(next_state, 'active_group', group)
    _set_control(next_state, 'pending_question', {})
    if question_id == "target_mode_change_confirm":
        requested = normalize_target_mode(next_state.get("target_mode_change_candidate"))
        next_state["target_mode_change_candidate"] = ""
        if value is False:
            _set_control(next_state, 'active_group', normalize_scalar(question.get("interrupted_group")) or group)
            return _answer_result(state, next_state)
        if not requested:
            return HandlerResult(blocker="target mode change candidate is missing")
        previous = normalize_target_mode(question.get("previous_mode") or next_state.get("target_mode"))
        next_state["target_mode"] = requested
        next_state["workflow_mode"] = "sync_observe" if requested == "sync-observe" else "rpc_benchmark"
        invalidate_for_target_mode(next_state, previous_mode=previous)
        mark_group_reconfigured(next_state, "target_mode")
        _set_control(next_state, 'active_group', "chain_identity" if not _chain_confirmed(next_state) else _next_group(next_state))
        _set_control(
            next_state,
            "visible_response",
            [localized(next_state.get("language", "en"), f"已切换到 `{requested}` 模式。", f"Switched to `{requested}` mode.")],
        )
        return _answer_result(state, next_state)
    if question_id == "target_mode_select":
        return apply_chain_rpc_action(
            next_state,
            ActionProposal(
                "target_mode:answer",
                "choose_target_mode",
                {"target_mode": value, "selection_contract_verified": True},
                "high",
            ),
        )
    if question_id in {"chain", "chain_change_input"}:
        raw = normalize_scalar(value or user_text)
        raw = _candidate_identity_from_action(
            next_state,
            raw,
            {"source_evidence": user_text},
        )
        if question_id == "chain_change_input" and (next_state.get("chain_identity") or {}).get("canonical"):
            _request_chain_change(next_state, raw, {})
        else:
            _apply_chain_candidate(next_state, raw)
        return _answer_result(state, next_state, completion="in_progress" if next_state.get("pending_question") else "completed", stop=_handoff_stops(next_state))
    if question_id == "chain_ambiguity_confirm":
        if value == "reenter_chain":
            next_state["chain_identity"] = {}
        elif isinstance(value, dict) and value.get("chain_choice"):
            chain = canonicalize_chain_scalar(str(value.get("chain_choice") or ""), known_chains=set(repo_chain_names()))
            if chain:
                previous = normalize_scalar((next_state.get("chain_identity") or {}).get("canonical"))
                if previous and previous != chain:
                    invalidate_for_chain_change(next_state)
                next_state["chain_identity"] = {"raw": str(value.get("chain_choice") or ""), "canonical": chain, "status": "confirmed", "case": "known"}
                next_state.setdefault("confirmed_config", {})["BLOCKCHAIN_NODE"] = chain
                _set_control(
                    next_state,
                    "visible_response",
                    [localized(next_state.get("language", "en"), f"已确认链为 `{chain}`。", f"Confirmed chain: `{chain}`.")],
                )
        elif isinstance(value, dict) and value.get("unknown_chain_choice"):
            _apply_chain_candidate(next_state, str(value.get("unknown_chain_choice") or ""))
        return _answer_result(state, next_state, completion="in_progress" if next_state.get("pending_question") else "completed")
    if question_id == "chain_change_confirm":
        _apply_chain_change_decision(next_state, question, value)
        return _answer_result(state, next_state, completion="in_progress" if next_state.get("pending_question") else "completed", stop=_handoff_stops(next_state))
    if question_id == "unknown_chain_identity_confirm":
        _apply_unknown_chain_decision(next_state, value, user_text)
        return _answer_result(state, next_state, completion="in_progress", stop=_handoff_stops(next_state))
    if question_id in {"adapter_family_confirm", "custom_rpc_adapter_family_confirm"}:
        family = adapter_family_hint(str(value or user_text)) or normalize_scalar(value)
        if question_id == "custom_rpc_adapter_family_confirm":
            _confirm_custom_rpc_family(next_state, family)
        else:
            _enter_case_for_adapter_family(next_state, family)
        return _answer_result(state, next_state, completion="in_progress", stop=_handoff_stops(next_state))
    if question_id in {"case3_protocol_evidence", "case3_evidence_input"}:
        _record_case3_evidence(next_state, str(user_text or value).strip())
        return _answer_result(state, next_state, completion="in_progress", stop=True)
    if question_id == "case3_evidence_next":
        if value == "add_more":
            next_state.setdefault("chain_identity", {})["status"] = "case3_needs_evidence"
            next_state.setdefault("secondary_handoff", {})["status"] = "collecting_evidence"
        else:
            _prepare_case3_handoff(next_state)
            return _answer_result(state, next_state, stop=True)
        return _answer_result(state, next_state, completion="in_progress")
    if question_id in {"LOCAL_RPC_URL", "SYNC_OBSERVE_RPC_URL", "custom_rpc_endpoint", "new_chain_endpoint"}:
        raw = user_text if question_id in {"custom_rpc_endpoint", "new_chain_endpoint"} and user_text else value
        _apply_endpoint_answer(next_state, question_id, raw)
        next_question = question_for_chain_rpc(next_state, "endpoint_process")
        if next_question:
            _set_control(next_state, 'pending_question', next_question)
            _set_control(next_state, 'active_group', "endpoint_process")
        return _answer_result(state, next_state, completion="in_progress")
    if question_id == "BLOCKCHAIN_PROCESS_NAMES":
        next_state.setdefault("confirmed_config", {})["BLOCKCHAIN_PROCESS_NAMES"] = normalize_scalar(value)
        return _answer_result(state, next_state)
    if question_id == "MAINNET_RPC_URL_REVIEWED":
        confirmed = next_state.setdefault("confirmed_config", {})
        if value is False:
            next_state.setdefault("endpoint_evidence", {})["mainnet_review_declined"] = True
        elif value is not True:
            endpoint = extract_url_candidate(value)
            if not endpoint:
                return HandlerResult(blocker="MAINNET_RPC_URL must be a valid endpoint")
            identity = next_state.get("chain_identity") or {}
            chain = normalize_scalar(identity.get("canonical") or identity.get("raw"))
            family = _adapter_family(next_state)
            methods, params = health_probe_methods(chain, family)
            probe = validate_rpc_endpoint(
                chain=chain,
                endpoint=endpoint,
                methods=methods,
                adapter_family=family,
                method_params=params,
                timeout=3.0,
            )
            evidence = next_state.setdefault("endpoint_evidence", {})
            evidence["mainnet_rpc_url_probe"] = probe
            if not probe.get("ready"):
                evidence["mainnet_rpc_url_ready"] = False
                return HandlerResult(
                    blocker=f"MAINNET_RPC_URL validation failed: {probe.get('error') or probe.get('status')}"
                )
            evidence["mainnet_rpc_url_ready"] = True
            confirmed["MAINNET_RPC_URL"] = endpoint
        confirmed["MAINNET_RPC_URL_REVIEWED"] = True
        return _answer_result(state, next_state)
    if question_id in {"custom_rpc_method", "new_chain_method"}:
        _apply_method_answer(next_state, question_id, user_text or value)
        _install_chain_rpc_next_question(next_state, "endpoint_process")
        return _answer_result(state, next_state, completion="in_progress")
    if question_id in {"custom_rpc_schema_evidence", "new_chain_schema_evidence"}:
        _apply_schema_evidence(next_state, case="new_chain" if question_id.startswith("new_chain") else "custom_rpc", evidence=str(user_text or value))
        return _answer_result(state, next_state, completion="in_progress")
    if question_id in {"custom_rpc_parameter_confirm", "new_chain_parameter_confirm"}:
        case = "new_chain" if question_id.startswith("new_chain") else "custom_rpc"
        _confirm_parameter_contract(next_state, case, bool(value))
        _install_chain_rpc_next_question(next_state, "endpoint_process")
        return _answer_result(state, next_state, completion="in_progress")
    if question_id in {"custom_rpc_schema_confirm", "new_chain_schema_confirm"}:
        case = "new_chain" if question_id.startswith("new_chain") else "custom_rpc"
        _confirm_request_contract(next_state, case, bool(value))
        if value is False:
            _set_control(next_state, "visible_response", [localized(next_state.get("language", "en"), "请提供修正后的 request/parameter 证据或直接输入 params JSON。", "Provide corrected request/parameter evidence or direct params JSON.")])
        _install_chain_rpc_next_question(next_state, "endpoint_process")
        return _answer_result(state, next_state, completion="in_progress")
    if question_id in {"custom_rpc_response_confirm", "new_chain_response_confirm"}:
        case = "new_chain" if question_id.startswith("new_chain") else "custom_rpc"
        _confirm_probe_response(next_state, case, bool(value))
        _install_chain_rpc_next_question(next_state, "endpoint_process")
        return _answer_result(state, next_state, completion="in_progress")
    if question_id in {"custom_rpc_probe_confirm", "new_chain_probe_confirm"}:
        case = "new_chain" if question_id.startswith("new_chain") else "custom_rpc"
        _confirm_method_probe(next_state, case, bool(value))
        _install_chain_rpc_next_question(next_state, "endpoint_process")
        return _answer_result(state, next_state, completion="in_progress")
    if question_id in {"custom_rpc_continue", "new_chain_method_continue"}:
        _apply_continue(next_state, question_id, value)
        _install_chain_rpc_next_question(next_state, "endpoint_process")
        return _answer_result(state, next_state, completion="in_progress")
    if question_id in {"custom_rpc_scope", "new_chain_workload_scope"}:
        _apply_scope(next_state, question_id, value)
        _install_chain_rpc_next_question(next_state, "endpoint_process")
        return _answer_result(state, next_state, completion="in_progress")
    if question_id in {"custom_rpc_single_method", "new_chain_single_method"}:
        _set_single_workload(next_state, normalize_scalar(value), "new_chain" if question_id.startswith("new_chain") else "custom_rpc")
        return _answer_result(state, next_state)
    if question_id in {"custom_rpc_weights", "new_chain_custom_weights"}:
        _apply_weights(next_state, question_id, value)
        return _answer_result(state, next_state, completion="in_progress" if not (next_state.get("workload") or {}).get("confirmed") else "completed")
    if question_id == "new_chain_runtime_choice":
        if value == "use_verified_endpoint_real_node":
            _promote_case2_endpoint(next_state)
            return _answer_result(state, next_state)
        if value == "generate_handoff":
            _prepare_case2_handoff(next_state)
            return _answer_result(state, next_state, stop=True)
    if question_id == "custom_rpc_fixture_choice":
        if value == "use_template_defaults":
            return apply_chain_rpc_action(next_state, ActionProposal("fixture:defaults", "use_default_workload", {}, "high"))
        if value == "switch_real_node":
            previous = normalize_target_mode(next_state.get("target_mode"))
            next_state["target_mode"] = "real-node"
            next_state["workflow_mode"] = "rpc_benchmark"
            invalidate_for_target_mode(next_state, previous_mode=previous)
            next_state["fixture_evidence"] = {}
            _set_control(next_state, 'active_group', _next_group(next_state))
            _set_control(
                next_state,
                "visible_response",
                [localized(next_state.get("language", "en"), "已保留自定义 workload 并切换到 real-node。下一步会单独验证最终压测使用的 LOCAL_RPC_URL。", "The custom workload was preserved and the flow switched to real-node. The final benchmark LOCAL_RPC_URL will be validated separately next.")],
            )
            return _answer_result(state, next_state)
        if value == "generate_fixture_handoff":
            evidence = next_state.get("fixture_evidence") or {}
            next_state["secondary_handoff"] = {
                "status": "ready",
                "kind": "custom_rpc_fixture_recording",
                "chain": (next_state.get("chain_identity") or {}).get("canonical", ""),
                "workload": deepcopy(next_state.get("workload") or {}),
                "endpoint_evidence": deepcopy(next_state.get("endpoint_evidence") or {}),
                "missing_fixtures": deepcopy(evidence.get("missing") or []),
                "requirements": ["record real endpoint response", "validate fixture authenticity", "validate fixture coverage", "rerun preflight and smoke"],
            }
            _set_control(
                next_state,
                "visible_response",
                [localized(next_state.get("language", "en"), "已生成自定义 RPC fixture 录制交接。在真实响应被录制并通过真实性、覆盖率和 smoke gate 前，不会提交 fake-node job。", "Generated the custom-RPC fixture recording handoff. No fake-node job will be submitted until a real response is recorded and passes authenticity, coverage, and smoke gates.")],
            )
            return _answer_result(state, next_state, stop=True)
    if question_id == "rpc_mode":
        return apply_chain_rpc_action(next_state, ActionProposal("rpc_mode:answer", "set_rpc_mode", {"rpc_mode": value}, "high"))
    if question_id == "workload_confirm":
        mapping = {
            "default": "use_default_workload",
            "custom_rpc": "rpc_catalog_command",
            "weights": "configure_workload_weights",
            "change_target": "request_target_change",
        }
        action_type = mapping.get(str(value))
        if action_type:
            arguments = {"catalog_command": "enter"} if action_type == "rpc_catalog_command" else {}
            return apply_chain_rpc_action(next_state, ActionProposal(f"workload:{value}", action_type, arguments, "high"))
    if question_id == "target_change_scope":
        mapping = {"chain": "request_chain_selection", "target_mode": "request_target_mode_selection", "cancel": "cancel_target_change"}
        action_type = mapping.get(str(value))
        if action_type:
            return apply_chain_rpc_action(next_state, ActionProposal(f"target_change:{value}", action_type, {}, "high"))
    if group == "chain_auxiliary_endpoints":
        field = normalize_scalar(question.get("field"))
        if field:
            next_state.setdefault("confirmed_config", {})[field] = normalize_scalar(value)
            mark_group_reconfigured(next_state, group)
            record_group_invalidations(next_state, group)
            return _answer_result(state, next_state)
    return HandlerResult(blocker=f"unsupported chain/RPC question: {question_id}")


def cancel_chain_rpc_question(state: AgentGraphState, question: dict[str, Any]) -> HandlerResult:
    """Cancel domain-local transient work without changing shared routing state."""

    group = normalize_scalar(question.get("group"))
    if group not in CHAIN_RPC_GROUPS:
        return HandlerResult(blocker=f"question is not owned by chain/RPC: {group or '<missing>'}")
    next_state = _chain_rpc_draft(state)
    question_id = normalize_scalar(question.get("id"))
    resume_group = ""
    if question_id == "target_mode_change_confirm":
        next_state["target_mode_change_candidate"] = ""
        resume_group = normalize_scalar(question.get("interrupted_group")) or group
    elif question_id in {"chain_change_confirm", "chain_change_input"}:
        candidate = (next_state.get("chain_identity") or {}).get("change_candidate") or {}
        next_state.setdefault("chain_identity", {}).pop("change_candidate", None)
        resume_group = normalize_scalar(candidate.get("interrupted_group")) or group
    elif question_id.startswith("custom_rpc_"):
        next_state["custom_rpc"] = {}
        if (next_state.get("workload") or {}).get("choice") in {"custom_rpc", "weights"}:
            next_state["workload"] = {}
        resume_group = "workload_rpc"
    elif question_id.startswith("new_chain_"):
        identity = next_state.setdefault("chain_identity", {})
        for key in ("candidate_method", "candidate_params", "schema_draft", "schema_evidence", "weights", "workload_scope"):
            identity.pop(key, None)
        if identity.get("case") == "case2":
            identity["status"] = "needs_protocol_confirmation"
        resume_group = "chain_identity"
    elif question_id == "SYNC_OBSERVE_RPC_URL":
        evidence = next_state.setdefault("endpoint_evidence", {})
        for key in ("sync_rpc_url_ready", "sync_rpc_url_probe", "candidate_endpoint"):
            evidence.pop(key, None)
        resume_group = "sync_observe"
    _set_control(next_state, 'pending_question', {})
    result = _answer_result(state, next_state, completion="unchanged")
    return replace(
        result,
        next_group="",
        navigation_resume_group=resume_group,
        followup_actions=(
            ({"type": "clear_sync_observe_source"},)
            if question_id == "SYNC_OBSERVE_RPC_URL"
            else ()
        ),
    )

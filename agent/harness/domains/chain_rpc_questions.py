"""Cohesive Chain/RPC component extracted from the product domain facade."""

from __future__ import annotations

from typing import Any

from ..input_values import (
    normalize_scalar,
    normalize_target_mode,
)
from ..localization import localized
from ..questions import choice_question, manual_question
from ..state import AgentGraphState

from agent.onboarding.families import SUPPORTED_FAMILIES
from agent.planners import question_prompts
from agent.validators.rpc_workload import default_workload
from .chain_rpc_support import _case_dict, _schema_confirmation_prompt, _weight_example
from .rpc_catalog import catalog_method_names, draft_view, next_parameter_to_confirm


def _endpoint_probe_completion(language: str) -> str:
    """Describe the runtime-owned validation that follows endpoint intake."""

    return localized(
        language,
        "提交后，Agent 会先对该 endpoint 执行低速探测并记录验证证据；验证成功后才继续，失败时会保留失败证据并要求更正。",
        "After submission, the Agent will run a low-rate probe and record validation evidence for this endpoint. It continues only after validation succeeds; a failure is retained as evidence and the endpoint must be corrected.",
    )


def _method_identity_completion(language: str) -> str:
    """Describe the typed continuation after a method identity is accepted."""

    return localized(
        language,
        "接受 method 名称后，Agent 会保存带来源的 job-local draft，并继续收集该 method 的参数、request、response 或官方文档证据；暂时缺少后续证据不会撤销已经确认的 method。",
        "After accepting the method name, the Agent stores a source-attributed job-local draft and continues with parameter, request, response, or official-document evidence. Missing later evidence does not undo the accepted method.",
    )


def _endpoint_validation_question(state: AgentGraphState) -> dict[str, Any] | None:
    language = str(state.get("language") or "en")
    custom = state.get("custom_rpc") or {}
    identity = state.get("chain_identity") or {}
    if custom.get("status") == "needs_adapter_family_confirmation":
        return _adapter_family_question(state, custom=True)
    if custom.get("status") in {"needs_endpoint", "probe_failed"} and not custom.get("endpoint_ready"):
        return manual_question(
            "endpoint_process",
            "custom_rpc_endpoint",
            localized(
                language,
                "请提供可访问的 RPC endpoint，用于验证自定义 RPC method。不会修改 config/chains 原始模板，也不会自动作为最终压测的 LOCAL_RPC_URL。",
                "Provide a reachable RPC endpoint to validate the custom RPC method. The original config/chains template will not be modified, and this will not automatically become the final benchmark LOCAL_RPC_URL.",
            ),
            field="custom_rpc_endpoint",
            kind="url",
            accepted_action_types=("rpc_catalog_command",),
            queue_barrier=True,
            requires_capabilities=("chain_identity",),
            evidence_path="custom_rpc.endpoint",
            completion_effect=_endpoint_probe_completion(language),
        )
    if custom.get("status") == "needs_method" and not draft_view(state).get("method"):
        return manual_question(
            "endpoint_process",
            "custom_rpc_method",
            localized(language, "请输入要验证的自定义 RPC method 名称。", "Enter the custom RPC method name to validate."),
            field="custom_rpc_method",
            accepted_action_types=("rpc_catalog_command",),
            queue_barrier=True,
            validation={"input_mode": "rpc_method_or_schema_evidence"},
            completion_effect=_method_identity_completion(language),
        )
    if custom.get("status") == "needs_schema_evidence":
        method = normalize_scalar(
            draft_view(state).get("method") or custom.get("candidate_method")
        )
        method_label = (
            f"`{method}`"
            if method
            else localized(language, "当前 RPC method", "the current RPC method")
        )
        return manual_question(
            "endpoint_process",
            "custom_rpc_schema_evidence",
            localized(
                language,
                f"请提供 {method_label} 的 schema 证据：可以粘贴 params JSON、curl/request、response 示例或官方文档片段。没有参数时可直接输入 `[]`。",
                f"Provide schema evidence for {method_label}: params JSON, curl/request, response sample, or docs excerpt. Use `[]` when there are no params.",
            ),
            field="custom_rpc_schema_evidence",
            kind="evidence",
            accepted_action_types=("rpc_catalog_command",),
            queue_barrier=True,
            help_text=_schema_evidence_help(language),
            completion_effect=_schema_evidence_completion(language),
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
            localized(
                language,
                "请提供可访问的 RPC endpoint，用于验证该新链和 RPC method。",
                "Provide a reachable RPC endpoint to validate this new chain and RPC methods.",
            ),
            field="new_chain_endpoint",
            kind="url",
            accepted_action_types=("rpc_catalog_command",),
            queue_barrier=True,
            requires_capabilities=("chain_identity",),
            evidence_path="endpoint_evidence.candidate_endpoint",
            completion_effect=_endpoint_probe_completion(language),
        )
    if identity.get("status") == "existing_family_needs_method":
        return manual_question(
            "endpoint_process",
            "new_chain_method",
            localized(language, "请输入要验证的 RPC method 名称。", "Enter the RPC method name to validate."),
            field="new_chain_method",
            accepted_action_types=("rpc_catalog_command",),
            queue_barrier=True,
            validation={"input_mode": "rpc_method_or_schema_evidence"},
            completion_effect=_method_identity_completion(language),
        )
    if identity.get("status") == "existing_family_needs_schema_evidence":
        method = normalize_scalar(draft_view(state).get("method"))
        method_label = (
            f"`{method}`"
            if method
            else localized(language, "当前 RPC method", "the current RPC method")
        )
        return manual_question(
            "endpoint_process",
            "new_chain_schema_evidence",
            localized(
                language,
                f"请提供 {method_label} 的 schema 证据：params JSON、curl/request、response 示例或官方文档片段。没有参数时可直接输入 `[]`。",
                f"Provide schema evidence for {method_label}: params JSON, curl/request, response sample, or docs excerpt. Use `[]` when there are no params.",
            ),
            field="new_chain_schema_evidence",
            kind="evidence",
            accepted_action_types=("rpc_catalog_command",),
            queue_barrier=True,
            help_text=_schema_evidence_help(language),
            completion_effect=_schema_evidence_completion(language),
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


def _schema_evidence_help(language: str) -> str:
    return localized(
        language,
        "schema 证据用于建立该 RPC method 的真实 wire contract：method 名、参数顺序、每个参数的 JSON 类型/链语义/编码/示例，以及 response 结构。可以提供 params JSON、完整 request/curl、response 示例或官方文档；模型抽取结果不会直接被信任。",
        "Schema evidence establishes the RPC method's real wire contract: method identity, parameter order, each parameter's JSON type/blockchain meaning/encoding/example, and the response shape. You may provide params JSON, a complete request/curl, a response sample, or official docs; model extraction is not trusted by itself.",
    )


def _schema_evidence_completion(language: str) -> str:
    return localized(
        language,
        "提供后，我会逐项让你确认参数语义和 request contract，再用已验证 endpoint 执行低速 probe，并让你确认观察到的 response；通过后才写入运行时 method catalog，然后继续选择 single/mixed、method 范围和总和为 100 的权重。这个流程不会修改 `config/chains` 原始模板，也不会把验证 endpoint 自动当作最终 `LOCAL_RPC_URL`。",
        "After you provide it, I will confirm parameter semantics and the request contract item by item, run a low-rate probe against the validated endpoint, and ask you to confirm the observed response. Only then is the method added to the runtime catalog before single/mixed scope and weights totaling 100 are selected. This never modifies the original `config/chains` template or automatically promotes the validation endpoint to the final `LOCAL_RPC_URL`.",
    )


def _mainnet_review_question(state: AgentGraphState, *, sync_observe: bool) -> dict[str, Any]:
    language = str(state.get("language") or "en")
    identity = state.get("chain_identity") or {}
    if identity.get("case") in {"case2", "case2_runtime_override"}:
        chain = normalize_scalar(identity.get("canonical") or identity.get("raw")) or "<unknown>"
        prompt = localized(
            language,
            f"`{chain}` 没有已配置的 chain template mainnet endpoint。请输入自定义对比 endpoint，或选择不使用 mainnet 高度对比继续。",
            f"`{chain}` has no configured chain-template mainnet endpoint. Type a custom comparison endpoint, or continue without mainnet height comparison.",
        )
        return _choice(
            "endpoint_process",
            "MAINNET_RPC_URL_REVIEWED",
            prompt,
            "MAINNET_RPC_URL_REVIEWED",
            [
                _answer_option(
                    "skip",
                    "不使用 mainnet 对比继续" if language.startswith("zh") else "Continue without mainnet comparison",
                    False,
                    {"confirmed_config.MAINNET_RPC_URL_REVIEWED": True},
                ),
            ],
            kind="confirm_or_value",
            manual_input_allowed=True,
        )
    if sync_observe:
        prompt = localized(language, "是否使用当前链模板的 sync-health / MAINNET_RPC_URL 逻辑进行高度对比？如果你有自定义 mainnet endpoint，可以直接输入 URL。", "Use the current chain template sync-health / MAINNET_RPC_URL behavior for height comparison? If you have a custom mainnet endpoint, type the URL directly.")
    else:
        prompt = localized(language, "是否使用当前链模板的 MAINNET_RPC_URL / sync-health 逻辑进行对比？如果你有自定义 mainnet endpoint，可以直接输入 URL。", "Use the current chain template MAINNET_RPC_URL / sync-health behavior for comparison? If you have a custom mainnet endpoint, type the URL directly.")
    return _choice(
        "endpoint_process",
        "MAINNET_RPC_URL_REVIEWED",
        prompt,
        "MAINNET_RPC_URL_REVIEWED",
        [
            _answer_option("yes", "Y", True, {"confirmed_config.MAINNET_RPC_URL_REVIEWED": True}),
            _answer_option("no", "N", False, {"confirmed_config.MAINNET_RPC_URL_REVIEWED": True}),
        ],
        kind="confirm_or_value",
        manual_input_allowed=True,
    )


def _chain_question(state: AgentGraphState) -> dict[str, Any]:
    target_mode = state.get("target_mode") or ("sync-observe" if state.get("workflow_mode") == "sync_observe" else "")
    return manual_question(
        "chain_identity",
        "chain",
        question_prompts.chain_prompt(target_mode=target_mode, language=state.get("language", "en")),
        field="chain",
        kind="chain",
        accepted_action_types=("choose_chain", "change_chain"),
        queue_barrier=True,
        evidence_path="chain_identity.canonical",
    )


def _target_change_scope_question(state: AgentGraphState) -> dict[str, Any]:
    language = str(state.get("language") or "en")
    zh = language.startswith("zh")
    preserved = {
        "target_mode": state.get("target_mode") or "",
        "chain_identity.canonical": (state.get("chain_identity") or {}).get("canonical") or "",
        "rpc_mode": state.get("rpc_mode") or "",
        "workload.confirmed": bool((state.get("workload") or {}).get("confirmed")),
    }
    return _choice(
        "workload_rpc",
        "target_change_scope",
        localized(language, "你要更换哪一项？当前配置会保留到新选择被确认。", "What do you want to change? Current configuration is retained until the replacement is confirmed."),
        "target_change_scope",
        [
            _action_option("chain", "更换链" if zh else "Change chain", "chain", "request_chain_selection", {"pending_question.id": "chain_change_input"}),
            _action_option("target_mode", "更换 fake-node / real-node / sync-observe 模式" if zh else "Change fake-node / real-node / sync-observe mode", "target_mode", "request_target_mode_selection", {"pending_question.id": "target_mode_select"}),
            _action_option(
                "cancel",
                "取消更改，保留当前 workload 并继续" if zh else "Cancel the change, keep the current workload, and continue",
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
        options.append(_action_option(mode, mode, mode, "choose_target_mode", expected, target_mode=mode, target_mode_explicit=True))
    return _choice(
        "target_mode",
        "target_mode_select",
        localized(
            state.get("language", "en"),
            "请选择新的目标模式。选择后会再次确认切换及受影响配置。" if current else "请选择目标模式。",
            "Choose the new target mode. The switch and affected configuration will be confirmed next." if current else "Choose the target mode.",
        ),
        "target_mode",
        options,
        queue_barrier=True,
    )


def _adapter_family_question(state: AgentGraphState, *, custom: bool = False) -> dict[str, Any]:
    language = str(state.get("language") or "en")
    options = [
        _answer_option(
            family,
            "jsonrpc / EVM" if family == "jsonrpc" else family,
            family,
            {"chain_identity.adapter_family": family},
        )
        for family in SUPPORTED_FAMILIES
    ]
    options.append(
        _answer_option(
            "unsupported",
            "不属于以上协议族 / 不确定" if language.startswith("zh") else "None of the above / unsure",
            "unsupported",
            {"chain_identity.status": "unsupported_family_handoff"},
            return_policy="stop_after_response",
        )
    )
    return _choice(
        "endpoint_process" if custom else "chain_identity",
        "custom_rpc_adapter_family_confirm" if custom else "adapter_family_confirm",
        localized(language, "请确认该链的协议族。" if custom else "请确认该链属于哪个协议族。", "Confirm this chain's adapter family." if custom else "Confirm which adapter family this chain belongs to."),
        "adapter_family",
        options,
        accepted_action_types=("choose_adapter_family",),
        queue_barrier=True,
    )


def _case3_evidence_question(state: AgentGraphState) -> dict[str, Any]:
    language = str(state.get("language") or "en")
    evidence = list((state.get("secondary_handoff") or {}).get("evidence") or [])
    if evidence and (state.get("chain_identity") or {}).get("status") == "case3_collecting_evidence":
        return _choice(
            "chain_identity",
            "case3_evidence_next",
            localized(language, "已记录协议证据。继续补充，还是生成二次开发交接？", "Protocol evidence is recorded. Add more, or generate the development handoff?"),
            "case3_evidence_next",
            [
                _answer_option("add_more", "继续补充证据" if language.startswith("zh") else "Add more evidence", "add_more", {"chain_identity.status": "case3_needs_evidence"}),
                _answer_option("generate", "生成二次开发交接" if language.startswith("zh") else "Generate development handoff", "generate_handoff", {"secondary_handoff.status": "ready", "chain_identity.status": "needs_review_handoff"}, return_policy="stop_after_response"),
            ],
        )
    return manual_question(
        "chain_identity",
        "case3_protocol_evidence",
        localized(language, "请提供官方协议/RPC 文档、endpoint 文档或完整 request/response 示例。", "Provide official protocol/RPC docs, endpoint docs, or a complete request/response example."),
        field="case3_protocol_evidence",
        kind="evidence",
        accepted_action_types=("secondary_handoff_command",),
        evidence_path="secondary_handoff.evidence",
    )


def _schema_confirmation_question(state: AgentGraphState, case: str) -> dict[str, Any]:
    draft = draft_view(state)
    question_id = "new_chain_schema_confirm" if case == "new_chain" else "custom_rpc_schema_confirm"
    return _choice(
        "endpoint_process",
        question_id,
        _schema_confirmation_prompt(str(state.get("language") or "en"), draft),
        question_id,
        [
            _answer_option("yes", "Y", True, {"custom_rpc.catalog.draft.request_confirmed": True}),
            _answer_option("no", "N", False, {f"{'chain_identity' if case == 'new_chain' else 'custom_rpc'}.status": "existing_family_needs_schema_evidence" if case == "new_chain" else "needs_schema_evidence"}),
        ],
        kind="yes_no",
        queue_barrier=True,
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
    language = str(state.get("language") or "en")
    question_id = "new_chain_parameter_confirm" if case == "new_chain" else "custom_rpc_parameter_confirm"
    name = normalize_scalar(parameter.get("name")) or f"param[{index}]"
    prompt = localized(
        language,
        f"请单独确认参数 {index + 1}/{len(params)} `{name}` 的 contract：index={parameter.get('index', index)}，JSON wire type={parameter.get('json_type') or 'unknown'}，blockchain semantic type={parameter.get('semantic_type') or 'unknown'}，encoding={parameter.get('encoding') or 'unknown'}，meaning={parameter.get('meaning') or 'unknown'}，required/optional={parameter.get('required', 'unknown')}，example={parameter.get('example')!r}。回复 `Y` 确认；回复 `N` 补充或修正该参数证据。",
        f"Confirm parameter {index + 1}/{len(params)} `{name}` separately: index={parameter.get('index', index)}, JSON wire type={parameter.get('json_type') or 'unknown'}, blockchain semantic type={parameter.get('semantic_type') or 'unknown'}, encoding={parameter.get('encoding') or 'unknown'}, meaning={parameter.get('meaning') or 'unknown'}, required/optional={parameter.get('required', 'unknown')}, example={parameter.get('example')!r}. Reply `Y` to confirm it, or `N` to add/correct parameter evidence.",
    )
    owner_key = "chain_identity" if case == "new_chain" else "custom_rpc"
    return _choice(
        "endpoint_process",
        question_id,
        prompt,
        question_id,
        [
            _answer_option("yes", "Y", True, {f"{owner_key}.status": "existing_family_schema_needs_confirmation" if case == "new_chain" else "schema_needs_confirmation"}),
            _answer_option("no", "N", False, {f"{owner_key}.status": "existing_family_needs_schema_evidence" if case == "new_chain" else "needs_schema_evidence"}),
        ],
        kind="yes_no",
        queue_barrier=True,
    )


def _probe_confirmation_question(state: AgentGraphState, case: str, *, retry: bool = False) -> dict[str, Any]:
    draft = draft_view(state)
    method = normalize_scalar(draft.get("method")) or "<unknown>"
    endpoint = normalize_scalar(draft.get("validation_endpoint"))
    language = str(state.get("language") or "en")
    question_id = "new_chain_probe_confirm" if case == "new_chain" else "custom_rpc_probe_confirm"
    response_confirmed = bool(draft.get("response_confirmed"))
    contract_status = localized(
        language,
        "request 和 response contract 已分别确认。" if response_confirmed else "request contract 已确认；尚无已确认的 response contract，probe 成功后会单独展示实际响应结构供你确认。",
        "The request and response contracts are confirmed separately. " if response_confirmed else "The request contract is confirmed; no response contract is confirmed yet. A successful probe will capture the observed response shape for separate confirmation. ",
    )
    prompt = localized(
        language,
        f"{'上一次 probe 失败。' if retry else ''}{contract_status}是否现在对验证 endpoint `{endpoint or '<missing>'}` 执行 method `{method}` probe？`Y` 只授权这次 probe；`N` 返回证据修正。",
        f"{'The previous probe failed. ' if retry else ''}{contract_status}Probe method `{method}` against validation endpoint `{endpoint or '<missing>'}` now? `Y` authorizes only this probe; `N` returns to evidence correction.",
    )
    return _choice(
        "endpoint_process",
        question_id,
        prompt,
        question_id,
        [
            _answer_option("yes", "Y", True, {"custom_rpc.catalog.draft.request_confirmed": True}),
            _answer_option("no", "N", False, {"custom_rpc.catalog.draft.request_confirmed": False}),
        ],
        kind="yes_no",
        queue_barrier=True,
    )


def _response_confirmation_question(state: AgentGraphState, case: str) -> dict[str, Any]:
    draft = draft_view(state)
    observed = draft.get("observed_response") if isinstance(draft.get("observed_response"), dict) else {}
    response_hash = str(observed.get("shape_hash") or "<unknown>")
    response_sample = str(observed.get("sample") or "<unavailable>")
    language = str(state.get("language") or "en")
    if observed:
        prompt = localized(
            language,
            f"method probe 已成功，但之前没有可靠的 response contract。请单独确认实际响应是否可作为该 method 的样本。\nresponse shape hash: `{response_hash}`\nresponse sample: `{response_sample}`\n回复 `Y` 接受实际响应结构；回复 `N` 补充或修正 response/docs 证据。",
            f"The method probe succeeded, but no reliable response contract was supplied earlier. Confirm the observed response separately as this method's sample.\nresponse shape hash: `{response_hash}`\nresponse sample: `{response_sample}`\nReply `Y` to accept the observed response shape, or `N` to add/correct response or docs evidence.",
        )
    else:
        prompt = localized(
            language,
            f"请单独确认 evidence 中的 expected response contract。\nresponse summary: `{normalize_scalar(draft.get('response_summary')) or '<unknown>'}`\nresponse fields: `{draft.get('response_fields') or []}`\n回复 `Y` 确认 response contract；回复 `N` 补充或修正 response/docs 证据。确认不会执行 probe。",
            f"Confirm the expected response contract from the evidence separately.\nresponse summary: `{normalize_scalar(draft.get('response_summary')) or '<unknown>'}`\nresponse fields: `{draft.get('response_fields') or []}`\nReply `Y` to confirm the response contract, or `N` to add/correct response or docs evidence. Confirmation does not run a probe.",
        )
    question_id = "new_chain_response_confirm" if case == "new_chain" else "custom_rpc_response_confirm"
    owner_key = "chain_identity" if case == "new_chain" else "custom_rpc"
    return _choice(
        "endpoint_process",
        question_id,
        prompt,
        question_id,
        [
            _answer_option("yes", "Y", True, {"custom_rpc.catalog.draft.response_confirmed": True}),
            _answer_option("no", "N", False, {f"{owner_key}.status": "existing_family_needs_schema_evidence" if case == "new_chain" else "needs_schema_evidence"}),
        ],
        kind="yes_no",
        queue_barrier=True,
    )


def _continue_question(state: AgentGraphState, case: str) -> dict[str, Any]:
    language = str(state.get("language") or "en")
    new_chain = case == "new_chain"
    return _choice(
        "endpoint_process",
        "new_chain_method_continue" if new_chain else "custom_rpc_continue",
        localized(language, "新链 RPC method 已验证通过。下一步怎么处理？" if new_chain else "这个自定义 RPC method 已验证通过。下一步怎么处理？", "The new-chain RPC method has been validated. What should happen next?" if new_chain else "This custom RPC method has been validated. What should happen next?"),
        "new_chain_method_continue" if new_chain else "custom_rpc_continue",
        [
            _answer_option("add_another", "继续添加另一个 RPC method" if new_chain and language.startswith("zh") else "继续添加另一个自定义 RPC method" if language.startswith("zh") else "Add another RPC method" if new_chain else "Add another custom RPC method", "add_another", {f"{'chain_identity' if new_chain else 'custom_rpc'}.status": "existing_family_needs_method" if new_chain else "needs_method"}),
            _answer_option("finish", "当前 method 已够，继续后续流程" if new_chain and language.startswith("zh") else "当前 method 已够，继续配置 workload" if language.startswith("zh") else "This is enough; continue the next workflow" if new_chain else "This is enough; continue workload setup", "finish", {"custom_rpc.catalog.finished": True}),
        ],
        queue_barrier=True,
    )


def _scope_question(state: AgentGraphState, case: str) -> dict[str, Any]:
    language = str(state.get("language") or "en")
    zh = language.startswith("zh")
    if case == "new_chain":
        return _choice(
            "endpoint_process",
            "new_chain_workload_scope",
            localized(language, "请选择新链已验证 RPC methods 如何作为本次 workload 使用。", "Choose how to use the validated new-chain RPC methods for this workload."),
            "new_chain_workload_scope",
            [
                _answer_option("single_replace", "single 中使用一个已验证 method" if zh else "Use one validated method as single", "single_replace", {"chain_identity.workload_scope": "single_replace"}),
                _answer_option("mixed_replace", "mixed 中只使用这些已验证 methods，并配置权重" if zh else "Use only these validated methods in mixed and configure weights", "mixed_replace", {"chain_identity.status": "existing_family_needs_weights"}),
            ],
        )
    return _choice(
        "endpoint_process",
        "custom_rpc_scope",
        localized(language, "请选择自定义 RPC workload 如何应用。", "Choose how to apply this custom RPC workload."),
        "custom_rpc_scope",
        [
            _answer_option("single_replace", "仅使用这个 method 作为 single workload" if zh else "Use this method as single workload only", "single_replace", {"custom_rpc.scope": "single_replace"}),
            _answer_option("mixed_replace", "mixed 中只使用我提供的自定义 methods" if zh else "Use only my custom methods in mixed", "mixed_replace", {"custom_rpc.status": "needs_weights"}),
            _answer_option("mixed_add", "mixed 中保留模板默认 methods，并追加自定义 method" if zh else "Keep template defaults in mixed and add this method", "mixed_add", {"custom_rpc.status": "needs_weights"}),
        ],
    )


def _single_method_question(state: AgentGraphState, case: str) -> dict[str, Any]:
    methods = catalog_method_names(state)
    question_id = "new_chain_single_method" if case == "new_chain" else "custom_rpc_single_method"
    return _choice(
        "endpoint_process",
        question_id,
        localized(state.get("language", "en"), "请选择 single workload 使用哪个已验证 method。", "Choose which validated method to use as the single workload."),
        question_id,
        [_answer_option(method, method, method, {"workload.methods": [method]}) for method in methods],
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
        localized(
            state.get("language", "en"),
            f"请输入{'新链 ' if case == 'new_chain' else ''}mixed 权重，总和必须为 100。{'已验证' if case == 'new_chain' else '可用'} methods：{', '.join(methods) or '<none>'}。格式示例：`{example}`。",
            f"Enter mixed weights{' for the new chain' if case == 'new_chain' else ''}. The total must be 100. {'Validated' if case == 'new_chain' else 'Available'} methods: {', '.join(methods) or '<none>'}. Example: `{example}`.",
        ),
        field=question_id,
        validation={"input_mode": "rpc_weights"},
    )


def _choice(
    group: str,
    question_id: str,
    prompt: str,
    field: str,
    options: list[dict[str, Any]],
    *,
    kind: str = "numbered_choice",
    manual_input_allowed: bool = False,
    accepted_action_types: tuple[str, ...] = (),
    queue_barrier: bool = False,
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
        queue_barrier=queue_barrier,
    )


def _action_option(option_id: str, label: str, value: Any, action_type: str, expected_patch: dict[str, Any], **arguments: Any) -> dict[str, Any]:
    return {"id": option_id, "label": label, "value": value, "action": {"type": action_type, **arguments}, "expected_patch": expected_patch}


def _answer_option(option_id: str, label: str, value: Any, expected_patch: dict[str, Any], *, return_policy: str = "fallback") -> dict[str, Any]:
    return {"id": option_id, "label": label, "value": value, "action": {"type": "answer_pending", "answer": value}, "expected_patch": expected_patch, "return_policy": return_policy}

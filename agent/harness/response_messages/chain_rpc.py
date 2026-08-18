"""Chain/RPC response definitions.

Domains publish only these semantic identifiers and typed arguments. Localized
product copy is deliberately centralized here.
"""

_MESSAGE = {"message"}
_EVIDENCE = {"evidence"}
_ERROR = {"error"}

MESSAGES = {
    "chain_rpc.response.chain_confirmed": {
        "en": "Confirmed chain: `{chain}`.",
        "zh": "已确认链为 `{chain}`。",
        "arguments": {"chain": "string"},
        "kinds": _MESSAGE,
    },
    "chain_rpc.response.same_chain": {
        "en": "The current chain is already `{chain}`. I will continue the current configuration flow.",
        "zh": "当前链已经是 `{chain}`。我会继续当前配置流程。",
        "arguments": {"chain": "string"},
        "kinds": _MESSAGE,
    },
    "chain_rpc.response.chain_kept": {
        "en": "Keeping the current chain `{chain}`; confirmed chain-dependent configuration is unchanged.",
        "zh": "保持当前链 `{chain}`，已确认的链相关配置未更改。",
        "arguments": {"chain": "string"},
        "kinds": _MESSAGE,
    },
    "chain_rpc.response.target_mode_switched": {
        "en": "Switched to `{target_mode}` mode.",
        "zh": "已切换到 `{target_mode}` 模式。",
        "arguments": {"target_mode": "string"},
        "kinds": _MESSAGE,
    },
    "chain_rpc.response.adapter_family_updated": {
        "en": "Adapter family updated to `{adapter_family}`. Provide a reachable RPC endpoint to validate again.",
        "zh": "已更新协议族为 `{adapter_family}`。请重新提供可访问的 RPC endpoint 用于验证。",
        "arguments": {"adapter_family": "string"},
        "kinds": _MESSAGE,
    },
    "chain_rpc.response.unsupported_family_handoff": {
        "en": "This chain is outside the supported adapter families. Provide official protocol/RPC docs, endpoint docs, and request/response examples; I will generate a secondary-development handoff.",
        "zh": "该链目前不属于已支持协议族。请提供官方协议/RPC 文档、endpoint 文档、request/response 示例；我会生成二次开发交接文档。",
        "arguments": {},
        "kinds": _MESSAGE,
    },
    "chain_rpc.response.method_grammar_rejected": {
        "en": "The method in the typed custom-RPC action does not match the current adapter-family grammar and was not written to the catalog. Provide the exact method token or a complete protocol request.",
        "zh": "typed custom-RPC action 中的 method 不符合当前协议族 grammar，未写入 catalog。请提供精确 method token 或完整 protocol request。",
        "arguments": {},
        "kinds": _MESSAGE,
    },
    "chain_rpc.response.request_correction_required": {
        "en": "Provide corrected request/parameter evidence or direct params JSON.",
        "zh": "请提供修正后的 request/parameter 证据或直接输入 params JSON。",
        "arguments": {},
        "kinds": _MESSAGE,
    },
    "chain_rpc.response.response_evidence_required": {
        "en": "Provide the correct response sample, response schema, or official documentation. Confirmed request evidence is preserved.",
        "zh": "请补充正确的 response sample、response schema 或官方文档；已确认的 request 证据会保留。",
        "arguments": {},
        "kinds": _MESSAGE,
    },
    "chain_rpc.response.custom_workload_switched_real_node": {
        "en": "The custom workload was preserved and the flow switched to real-node. The final benchmark LOCAL_RPC_URL will be validated separately in the endpoint configuration group.",
        "zh": "已保留自定义 workload 并切换到 real-node。最终压测使用的 LOCAL_RPC_URL 会在 endpoint 配置组单独验证。",
        "arguments": {},
        "kinds": _MESSAGE,
    },
    "chain_rpc.response.fixture_handoff_created": {
        "en": "Generated the custom-RPC fixture recording handoff. No fake-node job will be submitted until a real response is recorded and passes authenticity, coverage, and smoke gates.",
        "zh": "已生成自定义 RPC fixture 录制交接。在真实响应被录制并通过真实性、覆盖率和 smoke gate 前，不会提交 fake-node job。",
        "arguments": {},
        "kinds": _MESSAGE,
    },
    "chain_rpc.response.endpoint_validation_failed": {
        "en": "{endpoint_role} validation failed: {reason}. Evidence: {evidence}. Provide a reachable endpoint.",
        "zh": "{endpoint_role} 验证失败：{reason}。证据：{evidence}。请提供可访问的 endpoint。",
        "arguments": {
            "endpoint_role": "string",
            "reason": "string",
            "evidence": "string",
        },
        "kinds": _MESSAGE,
    },
    "chain_rpc.response.endpoint_validation_passed": {
        "en": "{endpoint_role} validation passed. Evidence: {evidence}.",
        "zh": "{endpoint_role} 验证通过。证据：{evidence}。",
        "arguments": {"endpoint_role": "string", "evidence": "string"},
        "kinds": _MESSAGE,
    },
    "chain_rpc.response.custom_endpoint_validation_passed": {
        "en": "Endpoint validation passed. Evidence: {evidence}. This endpoint is stored only as custom RPC method/schema validation evidence and will not automatically become the final benchmark LOCAL_RPC_URL.",
        "zh": "endpoint 验证通过。证据：{evidence}。这个 endpoint 只作为自定义 RPC method/schema 验证证据，不会自动作为最终压测的 LOCAL_RPC_URL。",
        "arguments": {"evidence": "string"},
        "kinds": _MESSAGE,
    },
    "chain_rpc.response.method_identity_required": {
        "en": "The supplied value is not a directly verifiable RPC method for the current adapter family. Provide an exact method token or protocol request; change the chain or adapter family explicitly when that is the intended operation.",
        "zh": "提供的值不是当前协议族可直接验证的 RPC method。请提供精确 method token 或协议 request；如果要更换链或协议族，请明确提出。",
        "arguments": {},
        "kinds": _MESSAGE,
    },
    "chain_rpc.response.rpc_evidence_rejected": {
        "en": "This turn did not contribute an attributable RPC request, parameter, response, or documentation fact, so nothing was written to the method catalog. The current draft and pending question are unchanged. Paste request/response/docs, or ask explicitly for the state you want to inspect.",
        "zh": "本轮内容没有贡献可归因的 RPC request、参数、response 或官方文档事实，因此没有写入 method catalog。当前 draft 和待确认问题保持不变。请直接粘贴 request/response/docs；如果你是在询问当前配置，请明确提出要查看的状态。",
        "arguments": {},
        "kinds": _MESSAGE,
    },
    "chain_rpc.response.rpc_method_conflict": {
        "en": "Evidence method `{incoming_method}` conflicts with deterministic method `{current_method}`. The fragment was not written. Choose whether to keep the current draft or explicitly replace it.",
        "zh": "证据中的 method `{incoming_method}` 与当前 deterministic method `{current_method}` 冲突。该 fragment 未被写入；请选择保留当前 draft，或明确替换它。",
        "arguments": {"incoming_method": "string", "current_method": "string"},
        "kinds": _MESSAGE,
    },
    "chain_rpc.response.rpc_exchange_uncorrelated": {
        "en": "The request/response evidence could not be correlated: {reason}. Nothing from this fragment was written. Provide a request and response with matching JSON-RPC `id` values, or provide the request first.",
        "zh": "request/response 证据无法关联：{reason}。本 fragment 未写入任何内容。请提供具有相同 JSON-RPC `id` 的 request 和 response，或先提供 request。",
        "arguments": {"reason": "string"},
        "kinds": _MESSAGE,
    },
    "chain_rpc.response.parameter_semantics_missing": {
        "en": "Parameter semantics are incomplete: {details}. Provide official documentation or correct the parameter contract before validation.",
        "zh": "参数语义不完整：{details}。请提供官方文档或修正参数契约后再验证。",
        "arguments": {"details": "string"},
        "kinds": _MESSAGE,
    },
    "chain_rpc.response.search_grounding": {
        "en": "Google Search grounding completed for `{query}`.\nSummary: {summary}\nCitations: {citations}\nConfirm these researched facts together with the supplied endpoint/request/response evidence before they affect the RPC contract.",
        "zh": "已完成针对 `{query}` 的 Google Search 资料核实。\n摘要：{summary}\n引用：{citations}\n这些搜索事实必须与用户提供的 endpoint/request/response 证据一起确认，之后才能影响 RPC contract。",
        "arguments": {
            "query": "string",
            "summary": "string",
            "citations": "string",
        },
        "kinds": _EVIDENCE,
    },
    "chain_rpc.response.params_evidence_required": {
        "en": "I could not extract verifiable params from the evidence. Provide clearer request/response/docs, or enter params JSON directly; use `[]` when there are no params.",
        "zh": "没有从证据中提取到可验证的 params。请提供更完整的 request/response/docs，或直接输入 params JSON；没有参数时输入 `[]`。",
        "arguments": {},
        "kinds": _MESSAGE,
    },
    "chain_rpc.response.request_confirmation_required": {
        "en": "The request contract must be confirmed before a probe can run.",
        "zh": "必须先确认 request contract，才能执行 probe。",
        "arguments": {},
        "kinds": _MESSAGE,
    },
    "chain_rpc.response.method_probe_failed": {
        "en": "{case_label}method probe failed: {reason}. Evidence: {evidence}. The confirmed request/response contract and endpoint provenance are preserved; retry the probe or return to evidence correction.",
        "zh": "{case_label}method probe 失败：{reason}。证据：{evidence}。已确认的 request/response contract 和 endpoint provenance 均已保留，可重试 probe 或返回证据修正。",
        "arguments": {"case_label": "string", "reason": "string", "evidence": "string"},
        "kinds": _MESSAGE,
    },
    "chain_rpc.response.method_schema_validated": {
        "en": "Method/schema validation passed. Evidence: {evidence}. The validation endpoint remains evidence-only; the final benchmark endpoint will be confirmed separately in the endpoint configuration group.",
        "zh": "method/schema 验证通过。证据：{evidence}。验证 endpoint 仍只作为证据保存，最终测试 endpoint 会在 endpoint 配置组单独确认。",
        "arguments": {"evidence": "string"},
        "kinds": _MESSAGE,
    },
    "chain_rpc.response.new_chain_method_validated": {
        "en": "New-chain endpoint and method/schema validation passed. Evidence: {evidence}.",
        "zh": "新链 endpoint 和 method/schema 已验证通过。证据：{evidence}。",
        "arguments": {"evidence": "string"},
        "kinds": _MESSAGE,
    },
    "chain_rpc.response.schema_protocol_conflict": {
        "en": "The supplied evidence conflicts with the selected adapter family `{adapter_family}`. Confirm the adapter family before continuing endpoint and method validation.",
        "zh": "提供的证据与已选择的协议族 `{adapter_family}` 冲突。请先重新确认协议族，再继续 endpoint 和 method 验证。",
        "arguments": {"adapter_family": "string"},
        "kinds": _MESSAGE,
    },
    "chain_rpc.response.weights_unrecognized": {
        "en": "No unique valid weight mapping was found. Use a JSON/YAML mapping or `method=weight,method2=weight2`, and do not provide conflicting mappings.",
        "zh": "没有识别到唯一且有效的权重映射。请使用 JSON/YAML 映射或 `method=weight,method2=weight2`，并确保没有相互冲突的多份映射。",
        "arguments": {},
        "kinds": _MESSAGE,
    },
    "chain_rpc.response.weights_invalid": {
        "en": "{case_label} mixed weights need adjustment. Missing methods: {missing}; unknown methods: {unknown}; invalid weights: {invalid}; total: {total}. Current weights: {weights}. Enter the weights again.",
        "zh": "{case_label} mixed 权重需要调整。缺少 method：{missing}；未知 method：{unknown}；无效权重：{invalid}；总和：{total}。当前配置：{weights}。请重新输入。",
        "arguments": {
            "case_label": "string",
            "missing": "string",
            "unknown": "string",
            "invalid": "string",
            "total": "integer",
            "weights": "string",
        },
        "kinds": _MESSAGE,
    },
    "chain_rpc.response.custom_mixed_workload_replaced": {
        "en": "Custom RPC mixed workload confirmed: {weights}; template default methods were replaced.",
        "zh": "自定义 RPC mixed workload 已确认：{weights}；模板默认 methods 已被替换。",
        "arguments": {"weights": "string"},
        "kinds": _MESSAGE,
    },
    "chain_rpc.response.custom_mixed_workload_merged": {
        "en": "Custom RPC mixed workload confirmed: {weights}; template default methods were kept and merged into the final weights.",
        "zh": "自定义 RPC mixed workload 已确认：{weights}；模板默认 methods 已保留并合并到最终权重。",
        "arguments": {"weights": "string"},
        "kinds": _MESSAGE,
    },
    "chain_rpc.response.custom_single_workload_confirmed": {
        "en": "Custom RPC single workload confirmed: {method}; the template default method will be replaced.",
        "zh": "自定义 RPC single workload 已确认：{method}；模板默认 method 会被替换。",
        "arguments": {"method": "string"},
        "kinds": _MESSAGE,
    },
    "chain_rpc.response.case2_promotion_blocked": {
        "en": "Cannot promote the new-chain configuration to real-node: the endpoint probe is not bound to the current chain/family/address, or no validated RPC method exists. Complete the corresponding validation again.",
        "zh": "无法把新链配置提升到 real-node：endpoint 探测证据与当前链/协议/地址不一致，或还没有已验证的 RPC method。请重新完成对应验证。",
        "arguments": {},
        "kinds": _MESSAGE,
    },
    "chain_rpc.response.case2_promoted": {
        "en": "Switched to the real-node path and will continue with the verified endpoint/method as a job-local workload override; config/chains templates are not modified.",
        "zh": "已切换到 real-node 路径，并把已验证 endpoint/method 作为本次 job-local workload override 继续；不会修改 config/chains 原始模板。",
        "arguments": {},
        "kinds": _MESSAGE,
    },
    "chain_rpc.response.case2_handoff_ready": {
        "en": "Entered the secondary-development handoff path. A chain template, fake-node fixture, fixture coverage, and smoke validation task must be generated from the verified endpoint/method/schema; this chain will not be claimed production-supported until those gates pass. Evidence: endpoint={endpoint_evidence}, method={method_evidence}, chain={chain}.",
        "zh": "已进入二次开发交接路径。需要基于已验证 endpoint/method/schema 生成 chain template、fake-node fixture、fixture coverage 和 smoke 验证任务；在这些 gate 通过前，不会声明该链已被生产支持。证据：endpoint={endpoint_evidence}, method={method_evidence}, chain={chain}。",
        "arguments": {
            "endpoint_evidence": "string",
            "method_evidence": "string",
            "chain": "string",
        },
        "kinds": _MESSAGE,
    },
    "chain_rpc.response.case3_evidence_recorded": {
        "en": "Recorded protocol-development evidence item {count}.",
        "zh": "已记录第 {count} 条协议开发证据。",
        "arguments": {"count": "integer"},
        "kinds": _MESSAGE,
    },
    "chain_rpc.response.case3_handoff_ready": {
        "en": "Secondary-development handoff draft: `{chain}` is outside the supported adapter families.\nGoal: have another AI implement a protocol adapter from official materials and close the validation loop.\nRequired work:\n1. Read and cite official protocol/RPC docs; confirm transport, auth, request/response schema.\n2. Add a new adapter or extend an existing one without reusing the wrong adapter family.\n3. Add a chain template with verifiable RPC methods, params, response parsing, and error handling.\n4. Record fixtures from a real endpoint and add fake-node response samples.\n5. Run schema validator, fixture coverage, preflight, and smoke tests.\n6. Update English/Chinese docs so Agent knowledge matches code behavior.\nCollected evidence:\n{evidence_preview}",
        "zh": "二次开发交接草案：`{chain}` 当前不属于已支持协议族。\n目标：让另一个 AI 基于官方资料新增协议 adapter，并完成闭环验证。\n必须完成：\n1. 阅读并引用官方协议/RPC 文档，确认 transport、auth、request/response schema。\n2. 新增 adapter 或扩展现有 adapter，不得复用错误协议族。\n3. 新增 chain template，并定义可验证的 RPC method、参数、响应解析和错误处理。\n4. 基于真实 endpoint 录制 fixture，补充 fake-node 响应样本。\n5. 运行 schema validator、fixture coverage、preflight 和 smoke 测试。\n6. 更新中英文文档，确保 Agent 知识库和实际代码一致。\n已收集证据：\n{evidence_preview}",
        "arguments": {"chain": "string", "evidence_preview": "string"},
        "kinds": _MESSAGE,
    },
    "chain_rpc.failure.unsupported_action": {
        "en": "Unsupported Chain/RPC action: {action_type}.",
        "zh": "不支持的 Chain/RPC action：{action_type}。",
        "arguments": {"action_type": "string"},
        "kinds": _ERROR,
    },
    "chain_rpc.failure.invalid_target_mode": {
        "en": "Target mode must be fake-node, real-node, or sync-observe.",
        "zh": "目标模式必须是 fake-node、real-node 或 sync-observe。",
        "arguments": {},
        "kinds": _ERROR,
    },
    "chain_rpc.failure.invalid_operation": {
        "en": "The Chain/RPC operation `{operation}` is not valid in the current state.",
        "zh": "Chain/RPC 操作 `{operation}` 在当前状态下无效。",
        "arguments": {"operation": "string"},
        "kinds": _ERROR,
    },
    "chain_rpc.failure.missing_required_value": {
        "en": "The Chain/RPC operation requires `{field}`.",
        "zh": "Chain/RPC 操作缺少必填值 `{field}`。",
        "arguments": {"field": "string"},
        "kinds": _ERROR,
    },
    "chain_rpc.failure.invalid_default_workload": {
        "en": "The default workload for `{chain}` is unavailable or invalid.",
        "zh": "`{chain}` 的默认 workload 不存在或无效。",
        "arguments": {"chain": "string"},
        "kinds": _ERROR,
    },
    "chain_rpc.failure.invalid_question_contract": {
        "en": "The Chain/RPC question contract is invalid: {question_id}.",
        "zh": "Chain/RPC 问题契约无效：{question_id}。",
        "arguments": {"question_id": "string"},
        "kinds": _ERROR,
    },
    "chain_rpc.failure.endpoint_invalid": {
        "en": "The endpoint for `{field}` is invalid.",
        "zh": "`{field}` 的 endpoint 无效。",
        "arguments": {"field": "string"},
        "kinds": _ERROR,
    },
}

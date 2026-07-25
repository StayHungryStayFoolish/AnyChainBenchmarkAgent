"""Sync-observe response definitions."""

MESSAGES = {
    "harness.sync_observe.client_setup_without_search": {
        "en": "Prepare and start a real node, then return to select its local process or real endpoint. The Agent does not auto-download a client or claim performance data without a real source. The current model has no google_search; provide official client documentation, or install it and provide the endpoint.",
        "zh": "准备并启动真实节点后，回来选择本机进程或真实 endpoint。Agent 不会自动下载客户端，也不会在没有真实数据源时声称获得了性能数据。当前模型没有 google_search；请提供客户端官方文档，或自行安装后提供 endpoint。",
        "arguments": {},
        "kinds": {"message"},
    },
    "harness.sync_observe.client_setup_search_unavailable": {
        "en": "Prepare and start a real node, then return to select its local process or real endpoint. The Agent does not auto-download a client or claim performance data without a real source. google_search did not return verifiable official setup material; provide official documentation or the installed endpoint.",
        "zh": "准备并启动真实节点后，回来选择本机进程或真实 endpoint。Agent 不会自动下载客户端，也不会在没有真实数据源时声称获得了性能数据。google_search 没有返回可验证的官方准备资料；请提供官方文档或安装后的 endpoint。",
        "arguments": {},
        "kinds": {"warning"},
    },
    "harness.sync_observe.client_setup_grounded": {
        "en": "Prepare and start a real node, then return to select its local process or real endpoint. The Agent does not auto-download a client or claim performance data without a real source.\n{evidence}",
        "zh": "准备并启动真实节点后，回来选择本机进程或真实 endpoint。Agent 不会自动下载客户端，也不会在没有真实数据源时声称获得了性能数据。\n{evidence}",
        "arguments": {"evidence": "string"},
        "kinds": {"evidence"},
    },
    "harness.sync_observe.failure.invalid_duration": {
        "en": "Invalid observation duration; enter a positive integer.",
        "zh": "观察时长无效，请输入正整数。",
        "arguments": {},
        "kinds": {"error"},
    },
    "harness.sync_observe.failure.duration_condition_mismatch": {
        "en": "A sync-observe duration requires stop_condition=duration.",
        "zh": "设置 sync-observe 时长时，停止条件必须是 duration。",
        "arguments": {},
        "kinds": {"error"},
    },
    "harness.sync_observe.failure.unsupported_action": {
        "en": "The sync-observe owner does not support action `{action_type}`.",
        "zh": "sync-observe owner 不支持动作 `{action_type}`。",
        "arguments": {"action_type": "string"},
        "kinds": {"error"},
    },
    "harness.sync_observe.failure.real_source_required": {
        "en": "sync-observe requires a real source or client setup guidance.",
        "zh": "sync-observe 需要真实数据源或客户端准备说明。",
        "arguments": {},
        "kinds": {"error"},
    },
    "harness.sync_observe.failure.mode_required": {
        "en": "A sync-observe source can be selected only in sync-observe mode.",
        "zh": "只有在 sync-observe 模式下才能选择 sync-observe 数据源。",
        "arguments": {},
        "kinds": {"error"},
    },
}

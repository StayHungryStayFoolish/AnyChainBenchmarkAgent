"""Question text contracts for orientation, execution, and recovery owners."""

_PROMPT = {"question_prompt"}
_LABEL = {"option_label"}
_DESCRIPTION = {"option_description"}
_COMPLETION = {"completion_effect"}


MESSAGES = {
    "question.control.option.yes": {
        "en": "Y",
        "zh": "Y",
        "arguments": {},
        "kinds": _LABEL,
    },
    "question.control.option.no": {
        "en": "N",
        "zh": "N",
        "arguments": {},
        "kinds": _LABEL,
    },
    "question.orientation.resume_quarantined.prompt": {
        "en": (
            "An incomplete or inconsistent legacy configuration was quarantined "
            "and has not been resumed. Choose whether to inspect its safe fields, "
            "retain only safe values, or clear it and start over."
        ),
        "zh": (
            "检测到不完整或不一致的旧配置，已隔离且尚未恢复。请选择查看安全字段、"
            "仅保留安全值，或清空后重新开始。"
        ),
        "arguments": {},
        "kinds": _PROMPT,
    },
    "question.orientation.resume.prompt": {
        "en": (
            "Found a previous Agent configuration session:\n"
            "- target_mode: {target_mode}\n"
            "- workflow_mode: {workflow_mode}\n"
            "- chain: {chain}\n"
            "- rpc_mode: {rpc_mode}\n"
            "- qps: {qps}\n"
            "- observability: {observability}\n"
            "- confirmed fields: {confirmed_fields}\n"
            "- deferred request count: {deferred_request_count}\n"
            "- saved workflow goals: {saved_workflow_goals}\n"
            "Choose whether to continue, keep confirmed values and modify, or "
            "clear the configuration and start over."
        ),
        "zh": (
            "检测到之前的 Agent 配置会话：\n"
            "- target_mode: {target_mode}\n"
            "- workflow_mode: {workflow_mode}\n"
            "- chain: {chain}\n"
            "- rpc_mode: {rpc_mode}\n"
            "- qps: {qps}\n"
            "- observability: {observability}\n"
            "- 已确认字段：{confirmed_fields}\n"
            "- 延后请求数量：{deferred_request_count}\n"
            "- 已保存 workflow 目标：{saved_workflow_goals}\n"
            "请选择继续、保留已确认值并修改，或清空配置后重新开始。"
        ),
        "arguments": {
            "target_mode": "string",
            "workflow_mode": "string",
            "chain": "string",
            "rpc_mode": "string",
            "qps": "string",
            "observability": "string",
            "confirmed_fields": "string",
            "deferred_request_count": "integer",
            "saved_workflow_goals": "string",
        },
        "kinds": _PROMPT,
    },
    "question.orientation.resume.option.inspect_quarantine": {
        "en": "Inspect the quarantine reason and safe fields",
        "zh": "查看隔离原因和安全字段",
        "arguments": {},
        "kinds": _LABEL,
    },
    "question.orientation.resume.option.retain_safe": {
        "en": "Retain only safely identified confirmed values",
        "zh": "仅保留可安全识别的确认值",
        "arguments": {},
        "kinds": _LABEL,
    },
    "question.orientation.resume.option.continue": {
        "en": "Continue the previous configuration",
        "zh": "继续之前的配置",
        "arguments": {},
        "kinds": _LABEL,
    },
    "question.orientation.resume.option.modify": {
        "en": "Keep confirmed values and choose what to modify",
        "zh": "保留已确认值并选择要修改的配置",
        "arguments": {},
        "kinds": _LABEL,
    },
    "question.orientation.resume.option.reset": {
        "en": "Clear the configuration and start over",
        "zh": "清空配置并重新开始",
        "arguments": {},
        "kinds": _LABEL,
    },
    "question.orientation.resume_modify.prompt": {
        "en": (
            "Confirmed values were kept. Choose the configuration group to "
            "modify, or describe the change in natural language."
        ),
        "zh": "已保留确认值。请选择要修改的配置组，也可以用自然语言说明要修改什么。",
        "arguments": {},
        "kinds": _PROMPT,
    },
    "question.orientation.group.option": {
        "en": "{group}",
        "zh": "{group}",
        "arguments": {"group": "string"},
        "kinds": _LABEL,
    },
    "question.orientation.group.description": {
        "en": "Configurable fields: {fields}",
        "zh": "可配置字段：{fields}",
        "arguments": {"fields": "string"},
        "kinds": _DESCRIPTION,
    },
    "question.orientation.group.completion": {
        "en": "Navigate through the registered `{group}` workflow owner.",
        "zh": "通过已注册的 `{group}` workflow owner 进入该配置组。",
        "arguments": {"group": "string"},
        "kinds": _COMPLETION,
    },
    "question.orientation.opening.prompt": {
        "en": (
            "I am AnyChain Benchmark Agent. What would you like help with? "
            "State a test goal or choose the next step."
        ),
        "zh": (
            "我是 AnyChain Benchmark Agent。你想让我帮你做什么？"
            "请直接说明测试目标，或选择下一步。"
        ),
        "arguments": {},
        "kinds": _PROMPT,
    },
    "question.orientation.opening.fake_node.label": {
        "en": "Start a fake-node benchmark",
        "zh": "启动 fake-node 测试",
        "arguments": {},
        "kinds": _LABEL,
    },
    "question.orientation.opening.fake_node.description": {
        "en": (
            "Fast, low-risk framework validation with recorded fixtures; "
            "it does not measure real-node performance."
        ),
        "zh": "使用预录 fixtures 做最快的低风险框架闭环验证；不测量真实节点性能。",
        "arguments": {},
        "kinds": _DESCRIPTION,
    },
    "question.orientation.opening.real_node.label": {
        "en": "Start a real-node benchmark",
        "zh": "启动 real-node 测试",
        "arguments": {},
        "kinds": _LABEL,
    },
    "question.orientation.opening.real_node.description": {
        "en": (
            "Load-test a reachable real-node RPC endpoint and analyze "
            "performance bottlenecks."
        ),
        "zh": "对可访问的真实节点 RPC endpoint 进行负载测试并分析性能瓶颈。",
        "arguments": {},
        "kinds": _DESCRIPTION,
    },
    "question.orientation.opening.sync_observe.label": {
        "en": "Start sync-observe (real-node synchronization)",
        "zh": "启动 sync-observe（观察真实节点同步）",
        "arguments": {},
        "kinds": _LABEL,
    },
    "question.orientation.opening.sync_observe.description": {
        "en": (
            "Observe real-node synchronization, resources, and available "
            "MGas/s metrics without Vegeta load."
        ),
        "zh": "观察真实节点追块、资源与可用 MGas/s 指标；不运行 Vegeta 压测。",
        "arguments": {},
        "kinds": _DESCRIPTION,
    },
    "question.orientation.opening.info.label": {
        "en": "Learn supported chains, RPC methods, and extension paths",
        "zh": "了解支持的链、RPC method 和二次开发方式",
        "arguments": {},
        "kinds": _LABEL,
    },
    "question.orientation.opening.info.description": {
        "en": (
            "Read-only guidance about capabilities, default workloads, and "
            "extension paths without changing test configuration."
        ),
        "zh": "只读查看框架能力、默认 workload 与扩展路径，不修改测试配置。",
        "arguments": {},
        "kinds": _DESCRIPTION,
    },
    "question.orientation.recommendation.prompt": {
        "en": "Start with fake-node smoke, then choose the chain to validate?",
        "zh": "是否先进入 fake-node smoke，再选择要验证的链？",
        "arguments": {},
        "kinds": _PROMPT,
    },
    "question.execution.real_node_smoke.prompt": {
        "en": (
            "Preflight passed, but no isolated real-node smoke has run. "
            "Revalidate and submit the safe low-traffic smoke now?"
        ),
        "zh": (
            "preflight 已通过，但还没有执行隔离的 real-node smoke。"
            "现在重新校验并提交安全小流量 smoke？"
        ),
        "arguments": {},
        "kinds": _PROMPT,
    },
    "question.execution.real_node_smoke.approve.completion": {
        "en": (
            "Revalidate the confirmed endpoint, custom RPC method/schema, and "
            "execution prerequisites, then submit one isolated safe low-traffic "
            "real-node smoke."
        ),
        "zh": (
            "重新校验已确认的 endpoint、自定义 RPC method/schema 和执行前置条件，"
            "然后提交一次隔离的安全小流量 real-node smoke。"
        ),
        "arguments": {},
        "kinds": _COMPLETION,
    },
    "question.execution.real_node_smoke.reject.completion": {
        "en": (
            "Do not revalidate or submit the real-node smoke; return to "
            "configuration."
        ),
        "zh": "不重新校验或提交 real-node smoke，并返回配置流程。",
        "arguments": {},
        "kinds": _COMPLETION,
    },
    "question.execution.final_benchmark.prompt": {
        "en": (
            "The isolated real-node smoke completed successfully. Submit the "
            "final benchmark with the confirmed QPS profile?"
        ),
        "zh": (
            "隔离的 real-node smoke 已成功完成。"
            "是否按已确认的 QPS profile 提交正式 benchmark？"
        ),
        "arguments": {},
        "kinds": _PROMPT,
    },
    "question.execution.final_benchmark.approve.completion": {
        "en": "Submit the final benchmark with the already confirmed QPS profile.",
        "zh": "提交正式 benchmark，并沿用已经确认的 QPS profile。",
        "arguments": {},
        "kinds": _COMPLETION,
    },
    "question.execution.final_benchmark.reject.completion": {
        "en": (
            "Do not submit the final benchmark and preserve the successful "
            "smoke evidence."
        ),
        "zh": "不提交正式 benchmark，并保留已经成功完成的 smoke 证据。",
        "arguments": {},
        "kinds": _COMPLETION,
    },
    "question.execution.preflight_smoke.prompt": {
        "en": "Configuration is collected. Run preflight and smoke?",
        "zh": "配置已收集。是否运行 preflight 和 smoke？",
        "arguments": {},
        "kinds": _PROMPT,
    },
    "question.recovery.action.prompt": {
        "en": (
            "Choose the next step. After correction, the Agent will rerun the "
            "relevant validation and resume the normal configuration flow."
        ),
        "zh": (
            "请选择下一步。修正完成后，Agent 会重新运行相关校验并回到正常配置流程。"
        ),
        "arguments": {},
        "kinds": _PROMPT,
    },
    "question.recovery.action_with_summary.prompt": {
        "en": (
            "Failure {failure_id}: severity={severity}, code={code}.\n"
            "Observed facts: {facts}\n"
            "Evidence paths: {evidence_paths}\n"
            "Choose the next step. After correction, the Agent will rerun the "
            "relevant validation and resume the normal configuration flow."
        ),
        "zh": (
            "故障 {failure_id}：severity={severity}，code={code}。\n"
            "已观察事实：{facts}\n"
            "证据路径：{evidence_paths}\n"
            "请选择下一步。修正完成后，Agent 会重新运行相关校验并回到正常配置流程。"
        ),
        "arguments": {
            "severity": "string",
            "code": "string",
            "failure_id": "string",
            "facts": "string",
            "evidence_paths": "string",
        },
        "kinds": _PROMPT,
    },
    "question.recovery.option.correct": {
        "en": "Correct the affected configuration and revalidate",
        "zh": "修正受影响的配置并重新验证",
        "arguments": {},
        "kinds": _LABEL,
    },
    "question.recovery.option.inspect": {
        "en": "Inspect failure evidence and diagnostics",
        "zh": "查看错误证据和诊断信息",
        "arguments": {},
        "kinds": _LABEL,
    },
    "question.recovery.option.retry": {
        "en": "Retry the current request after restoring the external service",
        "zh": "恢复外部服务后重试当前请求",
        "arguments": {},
        "kinds": _LABEL,
    },
    "question.recovery.option.cancel": {
        "en": "Pause recovery and preserve evidence/configuration",
        "zh": "暂不修复，保留证据和配置",
        "arguments": {},
        "kinds": _LABEL,
    },
}

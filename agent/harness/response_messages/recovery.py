"""Recovery response definitions."""

MESSAGES = {
    "recovery.response.paused": {
        "en": "Recovery is paused; failure evidence and the current configuration remain available. You can inspect the job/logs or resume correction later.",
        "zh": "已暂停恢复；错误证据和当前配置仍保留。你可以稍后查看 job、日志或重新开始修正。",
        "arguments": {},
        "kinds": {"status"},
    },
    "recovery.response.retry_ready": {
        "en": "The external-service failure state is cleared. Retry your previous request; this action did not change benchmark configuration or resubmit a job.",
        "zh": "已解除本次外部服务错误状态。请重新提交刚才的请求；本次操作没有修改 benchmark 配置或重复提交 job。",
        "arguments": {},
        "kinds": {"status"},
    },
    "recovery.failure.activation_record_required": {
        "en": "Recovery activation requires a typed failure record.",
        "zh": "恢复流程激活需要 typed failure record。",
        "arguments": {},
        "kinds": {"error"},
    },
    "recovery.failure.contract_unavailable": {
        "en": "The failure record has no executable recovery contract.",
        "zh": "failure record 没有可执行的恢复契约。",
        "arguments": {},
        "kinds": {"error"},
    },
    "recovery.failure.record_required": {
        "en": "No active failure record is available.",
        "zh": "当前没有 active failure record。",
        "arguments": {},
        "kinds": {"error"},
    },
    "recovery.failure.action_not_allowed": {
        "en": "Recovery action `{action_type}` is not allowed for failure `{code}`.",
        "zh": "failure `{code}` 不允许恢复动作 `{action_type}`。",
        "arguments": {"action_type": "string", "code": "string"},
        "kinds": {"error"},
    },
    "recovery.failure.correction_not_supported": {
        "en": "Failure `{code}` does not support configuration correction.",
        "zh": "failure `{code}` 不支持配置修正。",
        "arguments": {"code": "string"},
        "kinds": {"error"},
    },
    "recovery.failure.unsupported_action": {
        "en": "The recovery owner does not support action `{action_type}`.",
        "zh": "恢复 owner 不支持动作 `{action_type}`。",
        "arguments": {"action_type": "string"},
        "kinds": {"error"},
    },
}

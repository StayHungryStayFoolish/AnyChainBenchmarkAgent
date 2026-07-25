"""Control-plane response definitions."""

MESSAGES = {
    "harness.response.configuration_loaded": {
        "en": "Configuration loaded.",
        "zh": "配置已加载。",
        "arguments": {},
        "kinds": {"status"},
    },
    "harness.response.probe_evidence_saved": {
        "en": "Probe evidence {receipt} was saved.",
        "zh": "探测证据 {receipt} 已保存。",
        "arguments": {"receipt": "string"},
        "kinds": {"evidence"},
    },
    "harness.failure.internal_contract_violation": {
        "en": "The Agent response contract was violated: {reason}",
        "zh": "Agent 响应契约校验失败：{reason}",
        "arguments": {"reason": "string"},
        "kinds": {"error"},
    },
    "harness.response.workflow_goal_saved": {
        "en": "Saved a later goal: enter `{target_mode}` after the current workflow ({goal}).",
        "zh": "已保存后续目标：完成当前流程后进入 `{target_mode}`（{goal}）。",
        "arguments": {"target_mode": "string", "goal": "string"},
        "kinds": {"status"},
    },
    "harness.response.workflow_goal_removed": {
        "en": "Removed the oldest saved workflow goal.",
        "zh": "已移除最早保存的后续测试目标。",
        "arguments": {},
        "kinds": {"status"},
    },
    "harness.response.workflow_goal_activated": {
        "en": "Activating the saved workflow goal: {goal}.",
        "zh": "正在切换到已保存的后续目标：{goal}。",
        "arguments": {"goal": "string"},
        "kinds": {"status"},
    },
    "harness.response.pending_answer_not_admitted": {
        "en": (
            "That reply does not satisfy the active question. The question "
            "remains active; answer it or name the configuration area to change."
        ),
        "zh": (
            "这条回复不符合当前问题契约。当前问题保持不变；你可以回答该问题，"
            "或说明要切换的配置项。"
        ),
        "arguments": {},
        "kinds": {"warning"},
    },
    "harness.response.no_previous_group": {
        "en": (
            "There is no previous configuration group. Name the area to revisit, "
            "such as RPC, QPS, disk, or observability."
        ),
        "zh": (
            "当前没有可回退的配置组。请说明要返回的配置项，例如 RPC、QPS、"
            "磁盘或可观测性。"
        ),
        "arguments": {},
        "kinds": {"message"},
    },
    "harness.response.no_blocking_configuration": {
        "en": (
            "No blocking configuration item remains. Next group: `{next_group}`; "
            "execution status: `{execution_status}`."
        ),
        "zh": "当前配置没有新的阻塞项。下一组：`{next_group}`；执行状态：`{execution_status}`。",
        "arguments": {"next_group": "string", "execution_status": "string"},
        "kinds": {"status"},
    },
    "harness.response.group_has_no_blocker": {
        "en": "Entered `{group}`. This configuration group has no blocking item.",
        "zh": "已进入 `{group}`。该配置组当前没有阻塞项。",
        "arguments": {"group": "string"},
        "kinds": {"status"},
    },
    "harness.failure.coordinator.no_active_question": {
        "en": "No active typed question is available for `{operation}`.",
        "zh": "当前没有可供 `{operation}` 使用的 typed question。",
        "arguments": {"operation": "string"},
        "kinds": {"error"},
    },
    "harness.failure.coordinator.field_not_registered": {
        "en": "Field `{field}` is not registered for typed reconfiguration.",
        "zh": "字段 `{field}` 未注册 typed reconfiguration。",
        "arguments": {"field": "string"},
        "kinds": {"error"},
    },
    "harness.failure.coordinator.field_question_unavailable": {
        "en": "The registered question for field `{field}` is unavailable.",
        "zh": "字段 `{field}` 已注册的问题当前不可用。",
        "arguments": {"field": "string"},
        "kinds": {"error"},
    },
    "harness.failure.coordinator.group_not_navigable": {
        "en": "Group `{group}` is not a user-navigable destination.",
        "zh": "配置组 `{group}` 不是用户可导航目标。",
        "arguments": {"group": "string"},
        "kinds": {"error"},
    },
    "harness.failure.coordinator.workflow_goal_invalid": {
        "en": "The queued workflow goal is missing target mode, goal, or source evidence.",
        "zh": "待保存的 workflow 目标缺少 target mode、goal 或 source evidence。",
        "arguments": {},
        "kinds": {"error"},
    },
    "harness.failure.coordinator.workflow_goal_missing": {
        "en": "No queued workflow goal is available.",
        "zh": "当前没有已保存的后续 workflow 目标。",
        "arguments": {},
        "kinds": {"error"},
    },
    "harness.failure.coordinator.pending_dispatch_required": {
        "en": "The pending answer must be dispatched by the pending-question coordinator.",
        "zh": "pending answer 必须由 pending-question coordinator 分发。",
        "arguments": {},
        "kinds": {"error"},
    },
    "harness.failure.coordinator.unsupported_action": {
        "en": "Coordinator action `{action_type}` is not supported.",
        "zh": "coordinator 不支持 action `{action_type}`。",
        "arguments": {"action_type": "string"},
        "kinds": {"error"},
    },
    "harness.failure.coordinator.pending_answer_empty": {
        "en": "The answer for pending question `{question_id}` is empty.",
        "zh": "pending question `{question_id}` 的答案为空。",
        "arguments": {"question_id": "string"},
        "kinds": {"error"},
    },
    "harness.failure.coordinator.pending_value_invalid": {
        "en": "The model-derived value does not satisfy pending question `{question_id}`.",
        "zh": "模型解析出的值不符合 pending question `{question_id}` 的契约。",
        "arguments": {"question_id": "string"},
        "kinds": {"error"},
    },
    "harness.failure.coordinator.pending_option_unmapped": {
        "en": "The reply was not mapped to a declared option for pending question `{question_id}`.",
        "zh": "该回复未映射到 pending question `{question_id}` 的已声明选项。",
        "arguments": {"question_id": "string"},
        "kinds": {"error"},
    },
    "harness.failure.coordinator.answer_handler_missing": {
        "en": "No answer handler owns `{owner}` / `{group}` / `{question_id}`.",
        "zh": "没有 answer handler 负责 `{owner}` / `{group}` / `{question_id}`。",
        "arguments": {
            "owner": "string",
            "group": "string",
            "question_id": "string",
        },
        "kinds": {"error"},
    },
    "harness.failure.coordinator.model_provider_unavailable": {
        "en": (
            "The model provider could not resolve this turn (`{error_type}`). "
            "No configuration or workflow state was changed."
        ),
        "zh": "模型服务无法解析本轮请求（`{error_type}`）。配置和 workflow 状态均未修改。",
        "arguments": {"error_type": "string"},
        "kinds": {"error"},
    },
    "harness.failure.recovery_summary": {
        "en": (
            "Execution recovery: {severity} / {code} "
            "(diagnostic id: `{failure_id}`).\n"
            "- Observed facts: {facts}\n"
            "- Preserved config keys: {preserved_config_keys}\n"
            "- Evidence paths: {evidence_paths}"
        ),
        "zh": (
            "执行恢复：{severity} / {code}（诊断 ID：`{failure_id}`）。\n"
            "- 已观察事实：{facts}\n"
            "- 保留配置字段：{preserved_config_keys}\n"
            "- 证据路径：{evidence_paths}"
        ),
        "arguments": {
            "severity": "string",
            "code": "string",
            "failure_id": "string",
            "facts": "string",
            "preserved_config_keys": "string",
            "evidence_paths": "string",
        },
        "kinds": {"error"},
    },
    "harness.failure.recovery_action_summary": {
        "en": (
            "Execution recovery: {severity} / {code} "
            "(diagnostic id: `{failure_id}`).\n"
            "- Observed facts: {facts}\n"
            "- Preserved config keys: {preserved_config_keys}\n"
            "- Evidence paths: {evidence_paths}\n"
            "- Failed action: `{action_type}` (`{action_id}`)\n"
            "- Recovery choices: {recovery_choices}"
        ),
        "zh": (
            "执行恢复：{severity} / {code}（诊断 ID：`{failure_id}`）。\n"
            "- 已观察事实：{facts}\n"
            "- 保留配置字段：{preserved_config_keys}\n"
            "- 证据路径：{evidence_paths}\n"
            "- 失败 action：`{action_type}`（`{action_id}`）\n"
            "- 恢复选择：{recovery_choices}"
        ),
        "arguments": {
            "severity": "string",
            "code": "string",
            "failure_id": "string",
            "facts": "string",
            "preserved_config_keys": "string",
            "evidence_paths": "string",
            "action_type": "string",
            "action_id": "string",
            "recovery_choices": "string",
        },
        "kinds": {"error"},
    },
}

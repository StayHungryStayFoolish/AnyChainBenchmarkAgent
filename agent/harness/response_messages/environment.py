"""Environment response definitions."""

MESSAGES = {
    "harness.environment.inferred_config_discarded": {
        "en": "Discarded these inferred candidate values. Paste corrected information, or name the configuration group to change.",
        "zh": "已丢弃这次推断的候选配置。你可以重新粘贴更完整的信息，或直接说明要修改哪个配置组。",
        "arguments": {},
        "kinds": {"message"},
    },
    "harness.environment.inferred_config_applied": {
        "en": "Applied confirmed candidate values: {details}",
        "zh": "已应用确认的配置候选值：{details}",
        "arguments": {"details": "string"},
        "kinds": {"message"},
    },
    "harness.environment.endpoint_candidates_saved": {
        "en": "Saved endpoint candidate values for mandatory validation later: {details}",
        "zh": "已保存 endpoint 候选值，后续仍会进行真实性验证：{details}",
        "arguments": {"details": "string"},
        "kinds": {"message"},
    },
    "harness.environment.invalid_candidates_ignored": {
        "en": "These candidate values were invalid and were not applied: {details}",
        "zh": "以下候选值无效，未写入：{details}",
        "arguments": {"details": "string"},
        "kinds": {"warning"},
    },
    "harness.environment.unmapped_values_retained": {
        "en": "Unmapped values were retained as evidence and were not applied automatically.",
        "zh": "未映射字段已作为证据保留，不会自动写入配置。",
        "arguments": {},
        "kinds": {"evidence"},
    },
    "harness.environment.positive_number_required": {
        "en": "Invalid input: enter a number greater than zero.",
        "zh": "输入无效：请输入大于 0 的数值。",
        "arguments": {},
        "kinds": {"warning"},
    },
    "harness.environment.failure.accounts_presence_missing": {
        "en": "The accounts-disk presence decision is missing.",
        "zh": "缺少 accounts 磁盘存在性决策。",
        "arguments": {},
        "kinds": {"error"},
    },
    "harness.environment.failure.unsupported_action": {
        "en": "The environment owner does not support action `{action_type}`.",
        "zh": "环境配置 owner 不支持动作 `{action_type}`。",
        "arguments": {"action_type": "string"},
        "kinds": {"error"},
    },
}

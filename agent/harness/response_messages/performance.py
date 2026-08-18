"""Performance response definitions."""

MESSAGES = {
    "harness.performance.invalid_qps_value": {
        "en": "Invalid QPS value: {error}. Please re-enter.",
        "zh": "QPS 数值无效：{error}。请重新输入。",
        "arguments": {"error": "string"},
        "kinds": {"warning"},
    },
    "harness.performance.qps_overrides_applied": {
        "en": "Applied QPS profile overrides: {details}",
        "zh": "已应用 QPS profile 覆盖值：{details}",
        "arguments": {"details": "string"},
        "kinds": {"message"},
    },
    "harness.performance.exporter_selected": {
        "en": "Exporter-only observability is selected. Configure the existing Prometheus to scrape `http://<benchmark-host>:9108/metrics`.",
        "zh": "已选择 exporter-only；请让已有 Prometheus 抓取 `http://<benchmark-host>:9108/metrics`。",
        "arguments": {},
        "kinds": {"message"},
    },
    "harness.performance.local_observability_selected": {
        "en": "Local observability is selected: exporter:9108, Prometheus:9091, and Grafana:3001.",
        "zh": "已选择本地可观测性：exporter:9108、Prometheus:9091 和 Grafana:3001。",
        "arguments": {},
        "kinds": {"message"},
    },
    "harness.performance.observability_disabled": {
        "en": "Local observability components are disabled.",
        "zh": "已禁用本地可观测性组件。",
        "arguments": {},
        "kinds": {"message"},
    },
    "harness.performance.invalid_advanced_value": {
        "en": "Invalid value: enter a positive number; percentage fields must not exceed 100.",
        "zh": "数值无效：请输入正数；百分比字段不得超过 100。",
        "arguments": {},
        "kinds": {"warning"},
    },
    "harness.performance.failure.invalid_qps_mode": {
        "en": "QPS mode must be quick, standard, or intensive.",
        "zh": "QPS 模式必须是 quick、standard 或 intensive。",
        "arguments": {},
        "kinds": {"error"},
    },
    "harness.performance.failure.qps_overrides_required": {
        "en": "At least one QPS override is required.",
        "zh": "至少需要提供一个 QPS 覆盖值。",
        "arguments": {},
        "kinds": {"error"},
    },
    "harness.performance.failure.invalid_qps_profile": {
        "en": "The QPS profile is invalid: {error}",
        "zh": "QPS profile 无效：{error}",
        "arguments": {"error": "string"},
        "kinds": {"error"},
    },
    "harness.performance.failure.invalid_observability_mode": {
        "en": "Observability mode must be disabled, local, or exporter.",
        "zh": "可观测性模式必须是 disabled、local 或 exporter。",
        "arguments": {},
        "kinds": {"error"},
    },
    "harness.performance.failure.unsupported_action": {
        "en": "The performance owner does not support action `{action_type}`.",
        "zh": "性能配置 owner 不支持动作 `{action_type}`。",
        "arguments": {"action_type": "string"},
        "kinds": {"error"},
    },
}

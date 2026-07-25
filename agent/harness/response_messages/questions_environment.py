"""Question text contracts for environment, performance, and sync-observe."""

_PROMPT = {"question_prompt"}
_LABEL = {"option_label"}
_HELP = {"question_help"}
_COMPLETION = {"completion_effect"}


MESSAGES = {
    "question.common.option.yes": {
        "en": "Y",
        "zh": "Y",
        "arguments": {},
        "kinds": _LABEL,
    },
    "question.common.option.no": {
        "en": "N",
        "zh": "N",
        "arguments": {},
        "kinds": _LABEL,
    },
    "question.environment.inferred_config_review.prompt": {
        "en": (
            "Review the configuration inferred from the pasted content.\n"
            "Mapped candidates:\n{mapped_values}\n"
            "Unmapped evidence (not applied automatically):\n{unmapped_values}\n"
            "Conflicts requiring review:\n{conflicts}\n"
            "Inference reason: {reason}\n"
            "Apply the mapped candidates? Endpoint values remain unverified "
            "candidates until endpoint probing succeeds."
        ),
        "zh": (
            "请核对从粘贴内容中推断出的配置。\n"
            "已映射候选值：\n{mapped_values}\n"
            "未映射证据（不会自动写入）：\n{unmapped_values}\n"
            "需要核对的冲突：\n{conflicts}\n"
            "推断依据：{reason}\n"
            "是否应用已映射候选值？endpoint 值在探测成功前仍只是未验证候选。"
        ),
        "arguments": {
            "mapped_values": "string",
            "unmapped_values": "string",
            "conflicts": "string",
            "reason": "string",
        },
        "kinds": _PROMPT,
    },
    "question.environment.detected_value.prompt": {
        "en": "Detected {field}: `{value}`. Use this value?",
        "zh": "检测到 {field} 为 `{value}`，是否使用？",
        "arguments": {"field": "string", "value": "string"},
        "kinds": _PROMPT,
    },
    "question.environment.cloud_region.prompt": {
        "en": "Confirm CLOUD_REGION; use the detected value or enter a custom region.",
        "zh": "请输入 CLOUD_REGION（云区域），可以使用检测值或输入自定义值。",
        "arguments": {},
        "kinds": _PROMPT,
    },
    "question.environment.cloud_zone.prompt": {
        "en": "Confirm CLOUD_ZONE; use the detected value or enter a custom zone.",
        "zh": "请输入 CLOUD_ZONE（可用区），可以使用检测值或输入自定义值。",
        "arguments": {},
        "kinds": _PROMPT,
    },
    "question.environment.machine_type.prompt": {
        "en": "Confirm MACHINE_TYPE or instance type for report metadata.",
        "zh": "请输入 MACHINE_TYPE（机器或实例规格），用于报告元数据。",
        "arguments": {},
        "kinds": _PROMPT,
    },
    "question.environment.has_accounts_device.prompt": {
        "en": "Does this node have a separate accounts/state disk?",
        "zh": "这个节点是否有独立的 accounts/state 磁盘？",
        "arguments": {},
        "kinds": _PROMPT,
    },
    "question.environment.network_interface.prompt": {
        "en": "Choose the network interface, or type the interface name.",
        "zh": "请选择网络接口，或直接输入接口名。",
        "arguments": {},
        "kinds": _PROMPT,
    },
    "question.environment.network_interface.option": {
        "en": "{interface}",
        "zh": "{interface}",
        "arguments": {"interface": "string"},
        "kinds": _LABEL,
    },
    "question.environment.network_interface.option_default": {
        "en": "{interface} (default)",
        "zh": "{interface}（默认）",
        "arguments": {"interface": "string"},
        "kinds": _LABEL,
    },
    "question.environment.network_max_bandwidth_gbps.prompt": {
        "en": "Confirm NETWORK_MAX_BANDWIDTH_GBPS for saturation analysis.",
        "zh": "请输入 NETWORK_MAX_BANDWIDTH_GBPS，用于饱和度分析。",
        "arguments": {},
        "kinds": _PROMPT,
    },
    "question.environment.reconfiguration_value.prompt": {
        "en": "Enter the new {field}.",
        "zh": "请输入新的 {field}。",
        "arguments": {"field": "string"},
        "kinds": _PROMPT,
    },
    "question.environment.disk_device.prompt": {
        "en": "Choose {device_key}, or type the device name.",
        "zh": "请选择 {device_key}，或直接输入设备名。",
        "arguments": {"device_key": "string"},
        "kinds": _PROMPT,
    },
    "question.environment.disk_device.option": {
        "en": "{name} ({size}, {device_type})",
        "zh": "{name}（{size}，{device_type}）",
        "arguments": {
            "name": "string",
            "size": "string",
            "device_type": "string",
        },
        "kinds": _LABEL,
    },
    "question.environment.disk_device_manual.prompt": {
        "en": "Enter {device_key}.",
        "zh": "请输入 {device_key}。",
        "arguments": {"device_key": "string"},
        "kinds": _PROMPT,
    },
    "question.environment.detected_disk_size.prompt": {
        "en": "Detected {field}: `{value}` GiB. Use this value?",
        "zh": "检测到 {field} 为 `{value}` GiB，是否使用？",
        "arguments": {"field": "string", "value": "string"},
        "kinds": _PROMPT,
    },
    "question.environment.data_vol_type.prompt": {
        "en": (
            "Confirm DATA_VOL_TYPE for the ledger/data disk, for example "
            "hyperdisk-balanced, hyperdisk-extreme, pd-ssd, pd-balanced, "
            "local-ssd, ssd, or nvme."
        ),
        "zh": (
            "请输入 Ledger/data 磁盘的 DATA_VOL_TYPE，例如 "
            "hyperdisk-balanced、hyperdisk-extreme、pd-ssd、pd-balanced、"
            "local-ssd、ssd 或 nvme。"
        ),
        "arguments": {},
        "kinds": _PROMPT,
    },
    "question.environment.data_vol_size.prompt": {
        "en": "Confirm DATA_VOL_SIZE in GiB for the ledger/data disk.",
        "zh": "请输入 Ledger/data 磁盘的 DATA_VOL_SIZE（GiB）。",
        "arguments": {},
        "kinds": _PROMPT,
    },
    "question.environment.data_vol_max_iops.prompt": {
        "en": "Confirm DATA_VOL_MAX_IOPS for the ledger/data disk.",
        "zh": "请输入 Ledger/data 磁盘的 DATA_VOL_MAX_IOPS。",
        "arguments": {},
        "kinds": _PROMPT,
    },
    "question.environment.data_vol_max_throughput.prompt": {
        "en": "Confirm DATA_VOL_MAX_THROUGHPUT in MiB/s for the ledger/data disk.",
        "zh": "请输入 Ledger/data 磁盘的 DATA_VOL_MAX_THROUGHPUT（MiB/s）。",
        "arguments": {},
        "kinds": _PROMPT,
    },
    "question.environment.accounts_vol_type.prompt": {
        "en": (
            "Confirm ACCOUNTS_VOL_TYPE for the accounts/state disk, for example "
            "hyperdisk-balanced, hyperdisk-extreme, pd-ssd, pd-balanced, "
            "local-ssd, ssd, or nvme."
        ),
        "zh": (
            "请输入 accounts/state 磁盘的 ACCOUNTS_VOL_TYPE，例如 "
            "hyperdisk-balanced、hyperdisk-extreme、pd-ssd、pd-balanced、"
            "local-ssd、ssd 或 nvme。"
        ),
        "arguments": {},
        "kinds": _PROMPT,
    },
    "question.environment.accounts_vol_size.prompt": {
        "en": "Confirm ACCOUNTS_VOL_SIZE in GiB for the accounts/state disk.",
        "zh": "请输入 accounts/state 磁盘的 ACCOUNTS_VOL_SIZE（GiB）。",
        "arguments": {},
        "kinds": _PROMPT,
    },
    "question.environment.accounts_vol_max_iops.prompt": {
        "en": "Confirm ACCOUNTS_VOL_MAX_IOPS for the accounts/state disk.",
        "zh": "请输入 accounts/state 磁盘的 ACCOUNTS_VOL_MAX_IOPS。",
        "arguments": {},
        "kinds": _PROMPT,
    },
    "question.environment.accounts_vol_max_throughput.prompt": {
        "en": (
            "Confirm ACCOUNTS_VOL_MAX_THROUGHPUT in MiB/s for the "
            "accounts/state disk."
        ),
        "zh": (
            "请输入 accounts/state 磁盘的 ACCOUNTS_VOL_MAX_THROUGHPUT"
            "（MiB/s）。"
        ),
        "arguments": {},
        "kinds": _PROMPT,
    },
    "question.performance.benchmark_mode.prompt": {
        "en": "Choose benchmark mode.",
        "zh": "请选择 benchmark 模式。",
        "arguments": {},
        "kinds": _PROMPT,
    },
    "question.performance.qps_mode.quick": {
        "en": "quick",
        "zh": "quick",
        "arguments": {},
        "kinds": _LABEL,
    },
    "question.performance.qps_mode.standard": {
        "en": "standard",
        "zh": "standard",
        "arguments": {},
        "kinds": _LABEL,
    },
    "question.performance.qps_mode.intensive": {
        "en": "intensive",
        "zh": "intensive",
        "arguments": {},
        "kinds": _LABEL,
    },
    "question.performance.qps_profile_confirm.prompt": {
        "en": "Default QPS profile for `{mode}`: {profile}. Use these defaults?",
        "zh": "`{mode}` 模式默认 QPS 配置：{profile}。是否使用这些默认值？",
        "arguments": {"mode": "string", "profile": "string"},
        "kinds": _PROMPT,
    },
    "question.performance.qps_profile_confirm_fake.prompt": {
        "en": (
            "Default QPS profile for `{mode}`: {profile}. fake-node smoke uses "
            "a safe low-traffic execution override to validate the loop; real "
            "benchmarks use the final profile. Use these defaults?"
        ),
        "zh": (
            "`{mode}` 模式默认 QPS 配置：{profile}。fake-node smoke 会使用"
            "安全小流量执行覆盖来验证闭环；真实压测使用最终 profile。"
            "是否使用这些默认值？"
        ),
        "arguments": {"mode": "string", "profile": "string"},
        "kinds": _PROMPT,
    },
    "question.performance.qps_field.initial_qps": {
        "en": "INITIAL_QPS / initial QPS",
        "zh": "INITIAL_QPS / 起始 QPS",
        "arguments": {},
        "kinds": _LABEL,
    },
    "question.performance.qps_field.max_qps": {
        "en": "MAX_QPS / maximum QPS",
        "zh": "MAX_QPS / 最高 QPS",
        "arguments": {},
        "kinds": _LABEL,
    },
    "question.performance.qps_field.qps_step": {
        "en": "QPS_STEP / increment",
        "zh": "QPS_STEP / 每级递增",
        "arguments": {},
        "kinds": _LABEL,
    },
    "question.performance.qps_field.duration": {
        "en": "DURATION / seconds per step",
        "zh": "DURATION / 每档持续秒数",
        "arguments": {},
        "kinds": _LABEL,
    },
    "question.performance.qps_adjustments.finish": {
        "en": "Finish QPS adjustments",
        "zh": "完成 QPS 调整",
        "arguments": {},
        "kinds": _LABEL,
    },
    "question.performance.qps_adjust_field.prompt": {
        "en": "Choose the `{mode}` QPS parameter to adjust.",
        "zh": "请选择要调整的 `{mode}` QPS 参数。",
        "arguments": {"mode": "string"},
        "kinds": _PROMPT,
    },
    "question.performance.qps_adjust_value.prompt": {
        "en": "Enter the value for {field}.",
        "zh": "请输入 {field} 的值。",
        "arguments": {"field": "string"},
        "kinds": _PROMPT,
    },
    "question.performance.observability_mode.prompt": {
        "en": "Choose observability mode.",
        "zh": "请选择可观测性模式。",
        "arguments": {},
        "kinds": _PROMPT,
    },
    "question.performance.observability.disabled": {
        "en": "Disabled",
        "zh": "禁用",
        "arguments": {},
        "kinds": _LABEL,
    },
    "question.performance.observability.local": {
        "en": "Local Prometheus/Grafana",
        "zh": "本地 Prometheus/Grafana",
        "arguments": {},
        "kinds": _LABEL,
    },
    "question.performance.observability.exporter": {
        "en": "Exporter only for an existing Prometheus",
        "zh": "仅 exporter，对接已有 Prometheus",
        "arguments": {},
        "kinds": _LABEL,
    },
    "question.performance.advanced_tuning_confirm.prompt": {
        "en": (
            "Use the default advanced monitoring and bottleneck thresholds? "
            "Choose N to adjust individual values: {fields}."
        ),
        "zh": "是否使用默认高级监控和瓶颈阈值？选择 N 可逐项调整：{fields}。",
        "arguments": {"fields": "string"},
        "kinds": _PROMPT,
    },
    "question.performance.advanced_field.option": {
        "en": "{field}",
        "zh": "{field}",
        "arguments": {"field": "string"},
        "kinds": _LABEL,
    },
    "question.performance.advanced_adjustments.finish": {
        "en": "Finish adjustments",
        "zh": "完成调整",
        "arguments": {},
        "kinds": _LABEL,
    },
    "question.performance.advanced_tuning_adjust_field.prompt": {
        "en": "Choose the advanced tuning parameter to adjust.",
        "zh": "请选择要调整的高级调优参数。",
        "arguments": {},
        "kinds": _PROMPT,
    },
    "question.performance.advanced_tuning_adjust_value.prompt": {
        "en": "Enter the value for {field}.",
        "zh": "请输入 {field} 的值。",
        "arguments": {"field": "string"},
        "kinds": _PROMPT,
    },
    "question.sync_observe.source.prompt": {
        "en": (
            "Choose the real data source for sync-observe. This mode does not "
            "run Vegeta or use fake-node as a performance source."
        ),
        "zh": (
            "请选择 sync-observe 的真实数据来源。该模式不运行 Vegeta，"
            "也不把 fake-node 当作性能数据源。"
        ),
        "arguments": {},
        "kinds": _PROMPT,
    },
    "question.sync_observe.source.local_process": {
        "en": "Existing local real-node process",
        "zh": "本机真实节点进程",
        "arguments": {},
        "kinds": _LABEL,
    },
    "question.sync_observe.source.endpoint": {
        "en": "Real RPC/metrics endpoint",
        "zh": "真实 RPC/metrics endpoint",
        "arguments": {},
        "kinds": _LABEL,
    },
    "question.sync_observe.source.client_setup": {
        "en": "Generate real-node client setup guidance",
        "zh": "生成节点客户端准备说明",
        "arguments": {},
        "kinds": _LABEL,
    },
    "question.sync_observe.after_client_setup.prompt": {
        "en": (
            "After preparing a real-node client, select a local process or real "
            "endpoint before observation can start."
        ),
        "zh": "准备真实节点客户端后，必须选择本机进程或真实 endpoint 才能开始观察。",
        "arguments": {},
        "kinds": _PROMPT,
    },
    "question.sync_observe.stop_condition.prompt": {
        "en": "Choose a stop condition. The default is run until stopped.",
        "zh": "请选择停止条件。默认是一直运行直到用户停止。",
        "arguments": {},
        "kinds": _PROMPT,
    },
    "question.sync_observe.stop_condition.until_stopped": {
        "en": "Run until stopped",
        "zh": "一直运行直到停止",
        "arguments": {},
        "kinds": _LABEL,
    },
    "question.sync_observe.stop_condition.duration": {
        "en": "Fixed duration",
        "zh": "固定时长",
        "arguments": {},
        "kinds": _LABEL,
    },
    "question.sync_observe.stop_condition.until_synced": {
        "en": "Stop when synced",
        "zh": "同步完成后停止",
        "arguments": {},
        "kinds": _LABEL,
    },
    "question.sync_observe.duration_seconds.prompt": {
        "en": "Enter observation duration in seconds as a positive integer.",
        "zh": "请输入观察时长（秒），必须是正整数。",
        "arguments": {},
        "kinds": _PROMPT,
    },
}

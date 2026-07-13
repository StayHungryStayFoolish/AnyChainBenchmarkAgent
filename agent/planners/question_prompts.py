"""Canonical prompt wording for configuration questions.

`agent/validators/config_contract.py` and `agent/planners/config_questions.py`
used to author the same question wording independently and could (and did)
drift apart for the same missing field (architecture audit Finding C1). This
module is the single place that wording lives; both files call into it
instead of hand-typing prompt strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PromptText:
    en: str
    zh: str = ""

    def text(self, language: str = "en") -> str:
        if str(language or "").startswith("zh") and self.zh:
            return self.zh
        return self.en


FIELD_PROMPTS: dict[str, PromptText] = {
    "cloud_region": PromptText(
        "Confirm CLOUD_REGION; use the detected value or enter a custom region.",
        "请输入 CLOUD_REGION（云区域），可以使用检测值或输入自定义值。",
    ),
    "cloud_zone": PromptText(
        "Confirm CLOUD_ZONE; use the detected value or enter a custom zone.",
        "请输入 CLOUD_ZONE（可用区），可以使用检测值或输入自定义值。",
    ),
    "machine_type": PromptText(
        "Confirm MACHINE_TYPE or instance type for report metadata.",
        "请输入 MACHINE_TYPE（机器或实例规格），用于报告元数据。",
    ),
    "blockchain_process_names": PromptText(
        "Provide node process names or command-line fragments used for attribution.",
    ),
    "ledger_device": PromptText("Confirm the ledger/data disk device."),
    "accounts_device": PromptText("Confirm the accounts/state disk device."),
    "data_vol_type": PromptText(
        "Confirm DATA_VOL_TYPE for the ledger/data disk, for example hyperdisk-balanced, "
        "hyperdisk-extreme, pd-ssd, pd-balanced, local-ssd, ssd, or nvme."
    ),
    "data_vol_size": PromptText("Confirm DATA_VOL_SIZE in GiB for the ledger/data disk."),
    "data_vol_max_iops": PromptText("Confirm DATA_VOL_MAX_IOPS for the ledger/data disk."),
    "data_vol_max_throughput": PromptText("Confirm DATA_VOL_MAX_THROUGHPUT in MiB/s for the ledger/data disk."),
    "accounts_vol_type": PromptText(
        "Confirm ACCOUNTS_VOL_TYPE for the accounts/state disk, for example hyperdisk-balanced, "
        "hyperdisk-extreme, pd-ssd, pd-balanced, local-ssd, ssd, or nvme."
    ),
    "accounts_vol_size": PromptText("Confirm ACCOUNTS_VOL_SIZE in GiB for the accounts/state disk."),
    "accounts_vol_max_iops": PromptText("Confirm ACCOUNTS_VOL_MAX_IOPS for the accounts/state disk."),
    "accounts_vol_max_throughput": PromptText("Confirm ACCOUNTS_VOL_MAX_THROUGHPUT in MiB/s for the accounts/state disk."),
    "network_interface": PromptText(
        "Choose the network interface, or type the interface name.",
        "请选择网络接口，或直接输入接口名。",
    ),
    "network_max_bandwidth_gbps": PromptText(
        "Confirm NETWORK_MAX_BANDWIDTH_GBPS for saturation analysis.",
        "请输入 NETWORK_MAX_BANDWIDTH_GBPS。",
    ),
    "rpc_mode": PromptText("Choose RPC mode: single or mixed."),
    "rpc_workload_confirmed": PromptText("Confirm the RPC methods and workload weights."),
    "mixed_weights_confirmed": PromptText("Confirm that mixed RPC method weights are explicit and sum to 100%."),
    "rpc_param_samples_confirmed": PromptText("Confirm TARGET_* sample values for the selected RPC methods."),
    "chain_template_reviewed": PromptText(
        "Review selected chain template endpoints, TARGET_* sample variables, and default workload."
    ),
    "use_fake_node": PromptText("Choose fake-node closed-loop testing or real-node testing."),
    "local_rpc_url": PromptText("Provide LOCAL_RPC_URL for the node under test."),
    "sync_observe_rpc_url": PromptText(
        "Provide the real node RPC endpoint for sync-observe.",
        "请提供用于 sync-observe 的真实节点 RPC endpoint。",
    ),
    "mainnet_rpc_url_reviewed": PromptText("Provide MAINNET_RPC_URL or confirm template/default sync-health handling."),
    "has_accounts_device": PromptText(
        "Does this node have a separate accounts/state disk?",
        "这个节点是否有独立的 accounts/state 磁盘？",
    ),
    "node_prometheus_metrics_url": PromptText(
        "Confirm NODE_PROMETHEUS_METRICS_URL if the node exposes Prometheus metrics, or leave it unavailable."
    ),
    "node_process_identity": PromptText(
        "Confirm the node PID or process-name/command-line fragment for CPU/thread attribution."
    ),
}


def text_for(key: str, *, language: str = "en", **ctx: Any) -> str:
    """Resolve prompt text for `key`, trying the simple registry then the

    parameterized functions below by name. Falls back to a generic prompt
    for any key with no registered wording.
    """

    prompt = FIELD_PROMPTS.get(key)
    if prompt is not None:
        return prompt.text(language)
    parameterized = _PARAMETERIZED_PROMPTS.get(key)
    if parameterized is not None:
        return parameterized(language=language, **ctx)
    return f"Provide required value: {key}"


def chain_prompt(*, target_mode: str = "", language: str = "en", **_ignored: Any) -> str:
    if str(language or "").startswith("zh"):
        return f"你想测试哪条链？当前目标模式：{target_mode or '未选择'}。"
    return f"Which chain do you want to benchmark? Current target mode: {target_mode or 'not selected'}."


def device_prompt(device_key: str, *, language: str = "en") -> str:
    if str(language or "").startswith("zh"):
        return f"请选择 {device_key}，或直接输入设备名。"
    return f"Choose {device_key}, or type the device name."


BENCHMARK_MODE_CANDIDATES: list[dict[str, str]] = [
    {"id": "quick", "description": "Short smoke/sanity run."},
    {"id": "standard", "description": "Normal benchmark run."},
    {"id": "intensive", "description": "Long bottleneck discovery run."},
]


def benchmark_mode_prompt(*, language: str = "en", **_ignored: Any) -> str:
    if str(language or "").startswith("zh"):
        return "选择 benchmark 模式：1 quick，2 standard，或 3 intensive。"
    return "Choose benchmark mode: 1 quick, 2 standard, or 3 intensive."


# Numeric QPS profile defaults, keyed by benchmark mode. These correspond to
# `agent.planners.strategy_planner.DEFAULT_QPS`'s per-strategy values
# (quick=smoke, standard=baseline/ramp, intensive=stress) — kept as a
# separate literal here because the key spaces differ (benchmark mode vs.
# plan "strategy") and a translation table would add more risk than the
# duplication it removes.
_QPS_PROFILES: dict[str, dict[str, str]] = {
    "quick": {"INITIAL_QPS": "1000", "MAX_QPS": "1500", "QPS_STEP": "500", "DURATION": "60"},
    "standard": {"INITIAL_QPS": "2000", "MAX_QPS": "50000", "QPS_STEP": "500", "DURATION": "600"},
    "intensive": {"INITIAL_QPS": "50000", "MAX_QPS": "9999999", "QPS_STEP": "250", "DURATION": "600"},
}


def qps_profile_defaults(mode: str = "") -> dict[str, str]:
    """Return a copy of the numeric QPS defaults for `mode` (quick if unknown)."""

    normalized = (mode or "").strip().lower()
    return dict(_QPS_PROFILES.get(normalized, _QPS_PROFILES["quick"]))


def qps_profile_prompt(mode: str = "", *, fake_node: bool = False, language: str = "en", **_ignored: Any) -> str:
    normalized = (mode or "").strip().lower()
    values = _QPS_PROFILES.get(normalized, _QPS_PROFILES["quick"])
    if not normalized:
        mode = "quick"
    elif normalized in _QPS_PROFILES:
        mode = normalized
    else:
        # An unrecognized mode must not be echoed back next to numbers that
        # were never actually chosen for it — display "selected" instead.
        mode = "selected"
    summary = ", ".join(f"{key}={value}" for key, value in values.items())
    zh = str(language or "").startswith("zh")
    if fake_node:
        if zh:
            return (
                f"`{mode}` 模式默认 QPS 配置：{summary}。\n"
                "fake-node smoke 执行阶段会使用安全小流量覆盖来验证闭环；真实性能测试以最终 profile 为准。是否使用这些默认值？"
            )
        return (
            f"Default QPS profile for `{mode}`: {summary}.\n"
            "fake-node smoke uses a safe low-traffic execution override to validate the loop; "
            "real benchmarks use the final profile. Use these defaults?"
        )
    if zh:
        return f"`{mode}` 模式默认 QPS 配置：{summary}。是否使用这些默认值？"
    return f"Default QPS profile for `{mode}`: {summary}. Use these defaults?"


OBSERVABILITY_CANDIDATES: list[dict[str, str]] = [
    {"id": "disabled", "description": "Do not start observability stack."},
    {"id": "local", "description": "Start exporter, local Prometheus, and local Grafana."},
    {"id": "exporter", "description": "Start only exporter for an existing Prometheus/Grafana environment."},
]


def observability_mode_prompt(*, language: str = "en", **_ignored: Any) -> str:
    if str(language or "").startswith("zh"):
        return "选择可观测性模式：1 禁用，2 本地 Prometheus/Grafana，或 3 仅 exporter。"
    return "Choose observability mode: 1 disabled, 2 local Prometheus/Grafana, or 3 exporter-only."


SYNC_OBSERVE_STOP_CANDIDATES: list[dict[str, str]] = [
    {"id": "until_stopped", "description": "Run until the user stops it; default for long sync observation."},
    {"id": "duration", "description": "Run for a fixed duration in seconds."},
    {"id": "until_synced", "description": "Run until the chain sync-health model reports synced."},
]


def sync_observe_stop_condition_prompt(*, language: str = "en", **_ignored: Any) -> str:
    if str(language or "").startswith("zh"):
        return "选择 sync-observe 停止条件：1 手动停止，2 固定时长，或 3 同步完成为止。"
    return "Choose sync-observe stop condition: 1 until stopped, 2 fixed duration, or 3 until synced."


def workload_defaults_summary(chain: str, rpc_mode: str, single: str, mixed: str, *, language: str = "en") -> str:
    if str(language or "").startswith("zh"):
        return (
            f"当前链 `{chain}` 的模板 workload：\n"
            f"- 当前 RPC 模式：`{rpc_mode}`\n"
            f"- single 默认 method：`{single}`\n"
            f"- mixed 默认权重：{mixed}"
        )
    return (
        f"Current chain template workload for `{chain}`:\n"
        f"- Current RPC mode: `{rpc_mode}`\n"
        f"- Default single method: `{single}`\n"
        f"- Default mixed weights: {mixed}"
    )


def workload_customization_prompt(chain: str, rpc_mode: str, *, defaults_summary: str = "", language: str = "en") -> str:
    zh = str(language or "").startswith("zh")
    if zh:
        header = f"请查看 {chain} {rpc_mode} RPC workload。"
        if defaults_summary:
            header = f"{header}\n{defaults_summary}"
        return (
            f"{header}\n请选择下一步：\n"
            "1. 使用链模板默认值\n"
            "2. 添加自定义 RPC method\n"
            "3. 调整 mixed 权重\n"
            "4. 更换链或目标模式\n"
            "回复 `1`、`2`、`3` 或 `4`，或直接描述你想要的修改。"
        )
    header = f"Review the {chain} {rpc_mode} RPC workload."
    if defaults_summary:
        header = f"{header}\n{defaults_summary}"
    return (
        f"{header} Choose the next step:\n"
        "1. Continue with the chain template defaults\n"
        "2. Add a custom RPC method\n"
        "3. Adjust mixed weights\n"
        "4. Change chain or target mode\n"
        "Reply with `1`, `2`, `3`, or `4`, or describe the change manually."
    )


def workload_customization_options() -> list[dict[str, Any]]:
    return [
        {
            "id": "1",
            "value": "use_defaults",
            "label": "continue with defaults",
            "state_patch": {
                "workflow_step": "workload_default_confirmed",
                "confirmed_config": {
                    "chain_template_reviewed": True,
                    "rpc_workload_confirmed": True,
                    "rpc_param_samples_confirmed": True,
                    "mixed_weights_confirmed": True,
                },
            },
            "transition": {
                "workflow_step": "workload_default_confirmed",
                "tool": "validate_rpc_workload",
            },
        },
        {
            "id": "2",
            "value": "add_custom_rpc",
            "label": "add a custom RPC method",
            "state_patch": {"workflow_step": "custom_rpc_requested", "fixture_status": {"status": "needs_endpoint"}},
            "transition": {
                "workflow_step": "custom_rpc_endpoint_gate",
                "next_question_id": "custom_rpc_endpoint_gate",
            },
        },
        {
            "id": "3",
            "value": "adjust_weights",
            "label": "adjust mixed weights",
            "state_patch": {"workflow_step": "mixed_weight_adjustment_requested"},
            "transition": {
                "workflow_step": "mixed_weights_adjust",
                "next_question_id": "mixed_weights_confirm",
            },
        },
        {
            "id": "4",
            "value": "change_chain_or_mode",
            "label": "change chain or target mode",
            "state_patch": {"workflow_step": "chain_selection"},
            "transition": {
                "workflow_step": "chain_selection",
                "next_question_id": "chain_selection",
            },
        },
    ]


_ADVANCED_TUNING_DEFAULTS: dict[str, str] = {
    "MONITOR_INTERVAL": "5",
    "DISK_MONITOR_RATE": "1",
    "SUCCESS_RATE_THRESHOLD": "95",
    "MAX_LATENCY_THRESHOLD": "1000",
    "BOTTLENECK_CPU_THRESHOLD": "85",
    "BOTTLENECK_MEMORY_THRESHOLD": "90",
    "BOTTLENECK_DISK_UTIL_THRESHOLD": "90",
    "BOTTLENECK_DISK_LATENCY_THRESHOLD": "50",
    "BOTTLENECK_NETWORK_THRESHOLD": "80",
    "BOTTLENECK_ERROR_RATE_THRESHOLD": "5",
    "BOTTLENECK_DISK_IOPS_THRESHOLD": "90",
    "BOTTLENECK_DISK_THROUGHPUT_THRESHOLD": "90",
}


def advanced_tuning_default_prompt(*, language: str = "en") -> str:
    summary = ", ".join(f"{key}={value}" for key, value in _ADVANCED_TUNING_DEFAULTS.items())
    if str(language or "").startswith("zh"):
        return (
            "高级调优参数使用框架默认值：\n"
            f"{summary}\n"
            "这些值控制监控采样间隔、磁盘监控频率，以及 CPU/内存/磁盘/网络/错误率的瓶颈判定阈值。"
            "大多数用户不需要修改。是否使用默认值？"
        )
    return (
        "Advanced tuning parameters use the framework defaults:\n"
        f"{summary}\n"
        "These control monitoring sample intervals, disk monitor rate, and CPU/memory/disk/network/error-rate "
        "bottleneck detection thresholds. Most users do not need to change them. Use these defaults?"
    )


def chain_auxiliary_field_prompt(chain: str, field: str, *, language: str = "en") -> str:
    if str(language or "").startswith("zh"):
        return (
            f"当前链 `{chain}` 需要 {field}。这是链模板运行时会替换的可选 endpoint/凭据变量，不会修改 config/chains 原始模板。"
        )
    return (
        f"Chain `{chain}` needs {field}. This is a chain-template-driven optional endpoint/credential variable "
        "substituted at runtime; the config/chains template is not modified."
    )


_PARAMETERIZED_PROMPTS = {
    "chain": chain_prompt,
    "benchmark_mode_confirmed": benchmark_mode_prompt,
    "observability_choice_confirmed": observability_mode_prompt,
    "sync_observe_stop_condition": sync_observe_stop_condition_prompt,
    "qps_profile_confirmed": qps_profile_prompt,
}

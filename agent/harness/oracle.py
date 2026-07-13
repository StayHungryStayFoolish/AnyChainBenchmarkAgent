"""Pure next-action oracle for the AnyChain Agent Harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .routing import next_group_and_reason


@dataclass(frozen=True)
class NextAction:
    config_status: str
    next_blocking_group: str
    next_blocking_question_id: str
    next_blocking_reason: str
    execution_status: str
    recommended_next_action: str
    blockers: tuple[str, ...] = field(default_factory=tuple)
    latest_job_id: str = ""


def compute_next_action(state: dict[str, Any]) -> NextAction:
    """Compute the product-facing next action without mutating state."""

    pending = state.get("pending_question") or {}
    pending_id = str(pending.get("id") or "").strip()
    pending_group = str(pending.get("group") or state.get("active_group") or "").strip()
    latest_job_id = str(state.get("latest_job_id") or (state.get("job") or {}).get("job_id") or "").strip()
    execution_status = _execution_status(state)

    if pending_id:
        return NextAction(
            config_status="in_progress",
            next_blocking_group=pending_group,
            next_blocking_question_id=pending_id,
            next_blocking_reason=_pending_reason(pending),
            execution_status=execution_status,
            recommended_next_action=_answer_pending_action(pending),
            blockers=(pending_id,),
            latest_job_id=latest_job_id,
        )

    group, reason = _next_group_and_reason(state)
    config_status = "complete" if group in {"job_monitoring", ""} else "incomplete"
    if execution_status in {"job_running", "job_completed", "job_failed"}:
        config_status = "complete"
    action = _recommended_action_for_group(group, reason, execution_status)
    blockers = () if config_status == "complete" else (reason or group,)
    return NextAction(
        config_status=config_status,
        next_blocking_group=group,
        next_blocking_question_id="",
        next_blocking_reason=reason,
        execution_status=execution_status,
        recommended_next_action=action,
        blockers=tuple(item for item in blockers if item),
        latest_job_id=latest_job_id,
    )


def format_current_state(state: dict[str, Any], language: str) -> str:
    action = compute_next_action(state)
    identity = state.get("chain_identity") or {}
    qps = state.get("qps_profile") or {}
    observability = state.get("observability") or {}
    confirmed = state.get("confirmed_config") or {}
    chain = identity.get("canonical") or identity.get("raw") or _localized(language, "<未选择>", "<not selected>")
    target_mode = state.get("target_mode") or _localized(language, "<未选择>", "<not selected>")
    workflow_mode = state.get("workflow_mode") or _localized(language, "<未选择>", "<not selected>")
    rpc_mode = state.get("rpc_mode") or _localized(language, "<未选择>", "<not selected>")
    qps_mode = qps.get("mode") or _localized(language, "<未选择>", "<not selected>")
    obs_mode = observability.get("mode") or _localized(language, "<未选择>", "<not selected>")
    confirmed_keys = ", ".join(sorted(str(key) for key in confirmed.keys())) or _localized(language, "<无>", "<none>")
    if str(language or "").startswith("zh"):
        return (
            "当前状态：\n"
            f"- chain: `{chain}`\n"
            f"- target_mode: `{target_mode}`\n"
            f"- workflow_mode: `{workflow_mode}`\n"
            f"- rpc_mode: `{rpc_mode}`\n"
            f"- qps: `{qps_mode}`\n"
            f"- observability: `{obs_mode}`\n"
            f"- 已确认字段：{confirmed_keys}\n"
            f"- 配置状态：{action.config_status}\n"
            f"- 执行状态：{action.execution_status}\n"
            f"- 下一个阻塞项：{action.next_blocking_reason or '<none>'}\n"
            f"- 建议下一步：{format_recommended_next_action(action, language)}"
        )
    return (
        "Current state:\n"
        f"- chain: `{chain}`\n"
        f"- target_mode: `{target_mode}`\n"
        f"- workflow_mode: `{workflow_mode}`\n"
        f"- rpc_mode: `{rpc_mode}`\n"
        f"- qps: `{qps_mode}`\n"
        f"- observability: `{obs_mode}`\n"
        f"- confirmed fields: {confirmed_keys}\n"
        f"- config status: {action.config_status}\n"
        f"- execution status: {action.execution_status}\n"
        f"- next blocker: {action.next_blocking_reason or '<none>'}\n"
        f"- recommended next action: {format_recommended_next_action(action, language)}"
    )


def format_current_context(state: dict[str, Any], language: str) -> str:
    pending = state.get("pending_question") or {}
    if pending:
        pending_id = str(pending.get("id") or "").strip()
        prompt = str(pending.get("prompt") or "").strip()
        if pending_id == "qps_profile_confirm":
            qps = state.get("qps_profile") or {}
            mode = qps.get("mode") or _localized(language, "当前模式", "the selected mode")
            if str(language or "").startswith("zh"):
                return (
                    f"这里是在确认 `{mode}` 的 QPS profile 是否使用默认值。"
                    "如果回复 `Y`，我会保留默认 INITIAL_QPS、MAX_QPS、QPS_STEP 和 DURATION；"
                    "如果回复 `N`，我会让你逐项调整这些参数。"
                    "fake-node smoke 仍会用安全小流量覆盖来验证闭环，real-node 才会按最终 profile 做真实性能测试。"
                )
            return (
                f"This is asking whether to keep the default QPS profile for `{mode}`. "
                "Reply `Y` to keep INITIAL_QPS, MAX_QPS, QPS_STEP, and DURATION; "
                "reply `N` to adjust those parameters. "
                "fake-node smoke still uses a safe low-traffic override for loop validation; real-node uses the final profile for performance testing."
            )
        if pending_id == "opening_next_action":
            if str(language or "").startswith("zh"):
                return "这里是在让你选择下一步：开始 fake-node、real-node、sync-observe，或先了解支持的链/RPC method/扩展方式。你也可以不选编号，直接描述目标。"
            return "This asks for the next step: start fake-node, real-node, sync-observe, or learn supported chains/RPC methods/extension paths. You can also describe the goal instead of choosing a number."
        if pending_id == "rpc_mode":
            if str(language or "").startswith("zh"):
                return (
                    "这里是在选择 RPC workload 的模式：`single` 表示只压测一个 RPC method；"
                    "`mixed` 表示按权重组合多个 RPC method。选择后我会展示当前链模板的默认 method/权重，"
                    "并让你决定使用默认值、添加自定义 RPC method，或调整 mixed 权重。"
                )
            return (
                "This asks for the RPC workload mode: `single` benchmarks one RPC method; "
                "`mixed` combines multiple RPC methods by weights. After the choice I will show the chain template defaults "
                "and let you keep defaults, add a custom RPC method, or adjust mixed weights."
            )
        if pending_id == "workload_confirm":
            if str(language or "").startswith("zh"):
                return (
                    "这里是在确认当前链模板的默认 RPC workload。你可以使用默认值，也可以添加自定义 RPC method，"
                    "或调整 mixed 权重。自定义 RPC method 需要 endpoint、request/response 或官方文档，并且验证通过后才会进入后续 smoke。"
                )
            return (
                "This confirms the current chain template's default RPC workload. You can keep defaults, add a custom RPC method, "
                "or adjust mixed weights. A custom RPC method needs an endpoint, request/response, or official docs and must be validated before smoke."
            )
        if pending_id == "inferred_config_review":
            if str(language or "").startswith("zh"):
                return "这里是在确认是否应用我从你粘贴内容中推断出的配置候选值。回复 `Y` 才会写入已确认配置；回复 `N` 会丢弃候选值。endpoint 类值仍会在后续 probe 中验证。"
            return "This asks whether to apply config candidates inferred from your pasted content. Reply `Y` to save them as confirmed config; reply `N` to discard them. Endpoint-like values are still validated later by probing."
        if str(language or "").startswith("zh"):
            return (
                f"当前待确认问题是 `{pending_id}`。{prompt or '它用于补齐当前流程缺失的信息。'}\n"
                "你可以回答这个问题，也可以直接说要跳转到链、模式、磁盘、网络、RPC、QPS、可观测性、preflight/smoke、日志或报告分析。"
            )
        return (
            f"The current pending question is `{pending_id}`. {prompt or 'It fills a missing item in the current flow.'}\n"
            "You can answer it, or ask to jump to chain, mode, disk, network, RPC, QPS, observability, preflight/smoke, logs, or report analysis."
        )
    action = compute_next_action(state)
    if str(language or "").startswith("zh"):
        return f"当前没有待确认问题。{format_recommended_next_action(action, language)}"
    return f"There is no pending question. {format_recommended_next_action(action, language)}"


_STARTUP_DISCOVERY_CONFIRMABLE_KEYS = (
    "CLOUD_REGION",
    "CLOUD_ZONE",
    "MACHINE_TYPE",
    "LEDGER_DEVICE",
    "DATA_VOL_TYPE",
    "DATA_VOL_SIZE",
    "DATA_VOL_MAX_IOPS",
    "DATA_VOL_MAX_THROUGHPUT",
    "has_accounts_device",
    "NETWORK_INTERFACE",
    "NETWORK_MAX_BANDWIDTH_GBPS",
)


def format_startup_discovery(state: dict[str, Any], language: str) -> str:
    """Explain read-only startup environment inference.

    `discovery` is populated once at Agent startup and is independent of
    `confirmed_config`. Clearing user configuration must not be confused
    with clearing this data, and the two must be explained separately.
    """

    discovery = state.get("discovery") or {}
    if not discovery:
        return _localized(
            language,
            "启动时的环境推断数据当前不存在，可能是首次运行或诊断尚未完成。输入 `doctor` 可以重新执行只读环境检查。",
            "No startup environment inference data is available yet; this may be the first run or diagnostics have not completed. Type `doctor` to re-run read-only environment diagnostics.",
        )
    cloud = discovery.get("cloud") or {}
    deployment = discovery.get("deployment") or {}
    host = discovery.get("host") or {}
    network = discovery.get("network") or {}
    disks = discovery.get("disks") or {}
    confirmed = state.get("confirmed_config") or {}
    memory = host.get("memory_gib")
    inferred_lines = [
        f"- CLOUD_PROVIDER: {cloud.get('provider') or '<unknown>'}",
        f"- deployment: {cloud.get('platform') or deployment.get('type') or '<unknown>'}",
        f"- CPU: {host.get('cpu_count') or '<unknown>'}",
        f"- Memory: {memory} GiB" if memory not in (None, "") else "- Memory: <unknown>",
        f"- NETWORK_INTERFACE candidate: {network.get('default_interface') or '<none>'}",
        f"- LEDGER_DEVICE candidate: {disks.get('proposed_ledger_device') or '<none>'}",
        f"- ACCOUNTS_DEVICE candidate: {disks.get('proposed_accounts_device') or '<none detected>'}",
        f"- dependency status: {discovery.get('dependencies') or '<unknown>'}",
    ]
    needs_confirmation = [key for key in _STARTUP_DISCOVERY_CONFIRMABLE_KEYS if key not in confirmed]
    if confirmed:
        confirmed_note = _localized(
            language,
            f"已确认配置字段：{', '.join(sorted(str(key) for key in confirmed.keys()))}。",
            f"Confirmed config fields: {', '.join(sorted(str(key) for key in confirmed.keys()))}.",
        )
    else:
        confirmed_note = _localized(
            language,
            "当前没有已确认的配置（如果刚清空过配置，这是正常现象，不代表启动推断也被清空）。",
            "There is no confirmed configuration right now (this is expected right after clearing config; it does not mean startup inference was also cleared).",
        )
    if str(language or "").startswith("zh"):
        return (
            "启动时的只读环境推断仍然存在，可作为后续配置参考：\n"
            + "\n".join(inferred_lines)
            + f"\n尚未推断/需要你确认的项：{', '.join(needs_confirmation) if needs_confirmation else '<none>'}。\n"
            + confirmed_note
        )
    return (
        "Startup read-only environment inference is still available for later configuration:\n"
        + "\n".join(inferred_lines)
        + f"\nNot yet inferred / needs confirmation: {', '.join(needs_confirmation) if needs_confirmation else '<none>'}.\n"
        + confirmed_note
    )


def format_recommended_next_action(action: NextAction, language: str) -> str:
    """Render a user-facing next action without leaking internal group ids."""

    zh = str(language or "").startswith("zh")
    group = str(action.next_blocking_group or "").strip()
    reason = str(action.next_blocking_reason or "").strip()
    execution_status = str(action.execution_status or "").strip()

    if group == "job_monitoring":
        if execution_status.startswith("job_"):
            return _localized(
                language,
                "可以查看 job 状态、跟踪日志，或分析报告产物。",
                "You can check job status, follow logs, or analyze report artifacts.",
            )
        return _localized(language, "配置已完成，可以提交或监控 benchmark job。", "Configuration is complete; you can run or monitor the benchmark job.")

    if group == "preflight_smoke_execution":
        return _localized(
            language,
            "配置已收集，下一步应确认并运行 preflight/smoke。回复 `Y` 继续，或直接说明要修改哪一组配置。",
            "Configuration is collected. Next, approve and run preflight/smoke. Reply `Y` to continue, or describe which configuration group to change.",
        )

    if group:
        group_label = _group_label(group, language)
        reason_text = _reason_label(reason, language)
        if zh:
            return f"下一步需要继续确认{group_label}：{reason_text}。你也可以直接说明要跳转或修改哪一项配置。"
        return f"Next, continue {group_label}: {reason_text}. You can also describe which item to change or jump to."

    return _localized(language, "没有新的阻塞项。", "No blocking configuration item remains.")


def _group_label(group: str, language: str) -> str:
    labels = {
        "opening": ("入口选择", "opening choice"),
        "chain_identity": ("链和协议", "chain and protocol"),
        "provider_deployment": ("云区域和机器信息", "cloud region and machine metadata"),
        "ledger_disk": ("Ledger/data 磁盘", "ledger/data disk"),
        "accounts_disk": ("accounts/state 磁盘", "accounts/state disk"),
        "network": ("网络配置", "network configuration"),
        "endpoint_process": ("endpoint 和节点进程", "endpoint and node process"),
        "chain_auxiliary_endpoints": ("链辅助 endpoint", "chain auxiliary endpoints"),
        "workload_rpc": ("RPC workload", "RPC workload"),
        "target_samples_fixtures": ("target samples / fixtures", "target samples / fixtures"),
        "qps_profile": ("QPS profile", "QPS profile"),
        "sync_observe": ("sync-observe", "sync-observe"),
        "observability": ("Prometheus/Grafana 可观测性", "Prometheus/Grafana observability"),
        "advanced_tuning": ("高级调优参数", "advanced tuning settings"),
        "preflight_smoke_execution": ("preflight/smoke", "preflight/smoke"),
        "job_monitoring": ("job 监控", "job monitoring"),
    }
    zh, en = labels.get(group, (group, group))
    return _localized(language, zh, en)


def _reason_label(reason: str, language: str) -> str:
    if not reason:
        return _localized(language, "补齐当前缺失信息", "fill the missing information")
    if reason.startswith("confirm "):
        field = reason.removeprefix("confirm ").strip()
        return _localized(language, f"确认 `{field}`", f"confirm `{field}`")
    # `routing.next_group_and_reason` emits these two as
    # f"continue ...: {internal_status_enum}" — without a label, the raw
    # internal status (e.g. "needs_schema_evidence") leaked verbatim into a
    # user-facing sentence.
    if reason.startswith("continue new-chain validation:"):
        return _localized(language, "继续验证新链", "continue verifying the new chain")
    if reason.startswith("continue custom RPC workflow:"):
        return _localized(language, "继续自定义 RPC 配置", "continue the custom RPC setup")
    mapping = {
        "choose target mode": ("选择测试模式", "choose target mode"),
        "confirm chain identity": ("确认链名和协议", "confirm chain identity"),
        "choose RPC mode": ("选择 single 或 mixed RPC 模式", "choose single or mixed RPC mode"),
        "confirm RPC workload": ("确认 RPC method、权重和 workload", "confirm RPC methods, weights, and workload"),
        "choose benchmark QPS mode": ("选择 quick、standard 或 intensive", "choose quick, standard, or intensive"),
        "confirm or adjust QPS profile": ("确认或调整 QPS 参数", "confirm or adjust the QPS profile"),
        "choose observability mode": ("选择是否开启或接入 Prometheus/Grafana", "choose observability mode"),
        "review advanced tuning settings": ("确认是否调整高级调优参数", "review advanced tuning settings"),
        "approve preflight/smoke": ("确认执行 preflight/smoke", "approve preflight/smoke"),
        "choose sync-observe data source": ("选择真实同步观测数据来源", "choose the sync-observe data source"),
        "validate real sync-observe RPC endpoint": ("验证真实 sync-observe RPC endpoint", "validate the real sync-observe RPC endpoint"),
        "choose sync-observe stop condition": ("选择 sync-observe 停止条件", "choose the sync-observe stop condition"),
        "validate LOCAL_RPC_URL": ("验证 `LOCAL_RPC_URL`", "validate `LOCAL_RPC_URL`"),
        "confirm BLOCKCHAIN_PROCESS_NAMES": ("确认节点进程名", "confirm node process names"),
        "confirm MAINNET_RPC_URL / sync-health behavior": ("确认 `MAINNET_RPC_URL` 或同步健康检查方式", "confirm `MAINNET_RPC_URL` or sync-health behavior"),
        "confirm sync-health / MAINNET_RPC_URL behavior": ("确认同步健康检查或 `MAINNET_RPC_URL`", "confirm sync-health or `MAINNET_RPC_URL` behavior"),
    }
    zh, en = mapping.get(reason, (reason, reason))
    return _localized(language, zh, en)


def _next_group_and_reason(state: dict[str, Any]) -> tuple[str, str]:
    """Delegate to `routing.next_group_and_reason`.

    This used to be an independent reimplementation of the same
    precondition chain `groups._next_group` uses to drive live turn
    routing. Two hand-maintained copies could (and did, in at least one
    case) disagree; see architecture audit Finding B1.
    """

    return next_group_and_reason(state)


def _execution_status(state: dict[str, Any]) -> str:
    job = state.get("job") or {}
    smoke = state.get("smoke") or {}
    preflight = state.get("preflight") or {}
    if job.get("status"):
        status = str(job.get("status"))
        if status in {"running", "submitted", "completed", "failed"}:
            return f"job_{status}" if status != "submitted" else "job_submitted"
        return status
    if smoke.get("status"):
        return f"smoke_{smoke.get('status')}"
    if preflight.get("status"):
        return f"preflight_{preflight.get('status')}"
    if preflight.get("approved"):
        return "approval_recorded"
    pending = state.get("pending_question") or {}
    if pending.get("id") == "preflight_smoke_confirm":
        return "approval_pending"
    return "not_requested"


def _pending_reason(pending: dict[str, Any]) -> str:
    return str(pending.get("prompt") or pending.get("id") or "answer current pending question").strip()


def _answer_pending_action(pending: dict[str, Any]) -> str:
    qid = str(pending.get("id") or "").strip()
    if qid:
        return f"Answer `{qid}` or ask to jump to another configuration group."
    return "Answer the current pending question or describe what to change."


def _recommended_action_for_group(group: str, reason: str, execution_status: str) -> str:
    if group == "job_monitoring":
        if execution_status.startswith("job_"):
            return "Check job status, follow logs, or analyze report artifacts."
        return "Run or monitor the benchmark job."
    if group:
        return f"Continue with `{group}`: {reason}."
    return "No blocking configuration remains."


def _localized(language: str, zh: str, en: str) -> str:
    return zh if str(language or "").startswith("zh") else en

"""Pure next-action oracle for the AnyChain Agent Harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .routing import next_group_and_reason
from .state import AgentGraphState

from agent.runners.job_manager import get_job
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
    latest_job_id = str((state.get("job") or {}).get("job_id") or "").strip()
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
    # `group in {"job_monitoring", ""}` (from `routing.next_group_and_reason`,
    # the single source of truth for "is this workflow's config complete")
    # is the only correct test here. A prior version also forced
    # config_status to "complete" whenever `execution_status` showed any job
    # status at all -- but the workflow-owned `job` receipt deliberately survives a full
    # reset (`RESET_PRESERVED_KEYS`) so `analyze_report`/`status` keep
    # working for the last completed job, so that shortcut falsely reported
    # a brand-new, still-in-progress workflow as "complete" whenever any
    # unrelated past job happened to exist in state (reproduced live: a
    # fresh sync-observe setup reported `config_status: complete` with
    # `execution_status: job_failed` from an unrelated earlier job, while
    # still correctly naming an unmet next blocking question in the same
    # response).
    config_status = "complete" if group in {"job_monitoring", ""} else "incomplete"
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
    qps_overrides = qps.get("overrides") if isinstance(qps.get("overrides"), dict) else {}
    qps_values = ", ".join(f"{key}={value}" for key, value in qps_overrides.items()) or _localized(language, "使用模式默认值", "mode defaults")
    obs_mode = observability.get("mode") or _localized(language, "<未选择>", "<not selected>")
    confirmed_keys = ", ".join(sorted(str(key) for key in confirmed.keys())) or _localized(language, "<无>", "<none>")
    workload = state.get("workload") or {}
    methods = [str(item) for item in workload.get("methods") or [] if str(item).strip()]
    method_text = ", ".join(methods) or _localized(language, "<未确认>", "<not confirmed>")
    weights = workload.get("weights") if isinstance(workload.get("weights"), dict) else {}
    weight_text = ", ".join(f"{method}={weight}" for method, weight in weights.items()) or _localized(language, "<不适用>", "<not applicable>")
    custom_rpc = state.get("custom_rpc") or {}
    custom_status = str(custom_rpc.get("status") or _localized(language, "<未启用>", "<not active>"))
    validation_endpoint = str(custom_rpc.get("validation_endpoint") or custom_rpc.get("endpoint") or "")
    queued_requests = queued_configuration_summary(state, language)
    workflow_goals = state.get("workflow_goals") or []
    workflow_goal_text = ", ".join(
        f"{item.get('target_mode')}: {item.get('goal')}"
        for item in workflow_goals
        if isinstance(item, dict)
    ) or _localized(language, "<无>", "<none>")
    if str(language or "").startswith("zh"):
        endpoint_line = f"- 自定义 RPC 验证 endpoint：`{validation_endpoint}`（仅验证证据，不覆盖最终 LOCAL_RPC_URL）\n" if validation_endpoint else ""
        return (
            "当前状态：\n"
            f"- chain: `{chain}`\n"
            f"- target_mode: `{target_mode}`\n"
            f"- workflow_mode: `{workflow_mode}`\n"
            f"- rpc_mode: `{rpc_mode}`\n"
            f"- 实际 RPC methods：{method_text}\n"
            f"- mixed 权重：{weight_text}\n"
            f"- 自定义 RPC 状态：`{custom_status}`\n"
            + endpoint_line
            +
            "- config/chains 原始模板：未修改；自定义 RPC 仅保存在本次运行时配置与验证证据中\n"
            f"- 尚待执行的用户配置请求：{queued_requests}\n"
            f"- 已保存的后续 workflow 目标：{workflow_goal_text}\n"
            f"- qps: `{qps_mode}`\n"
            f"- QPS 生效参数：{qps_values}\n"
            f"- observability: `{obs_mode}`\n"
            f"- 已确认字段：{confirmed_keys}\n"
            f"- 配置状态：{action.config_status}\n"
            f"- 执行状态：{action.execution_status}\n"
            f"- 下一个阻塞项：{action.next_blocking_reason or '<none>'}\n"
            f"- 建议下一步：{format_recommended_next_action(action, language)}"
        )
    endpoint_line = f"- custom RPC validation endpoint: `{validation_endpoint}` (validation evidence only; it does not replace final LOCAL_RPC_URL)\n" if validation_endpoint else ""
    return (
        "Current state:\n"
        f"- chain: `{chain}`\n"
        f"- target_mode: `{target_mode}`\n"
        f"- workflow_mode: `{workflow_mode}`\n"
        f"- rpc_mode: `{rpc_mode}`\n"
        f"- effective RPC methods: {method_text}\n"
        f"- mixed weights: {weight_text}\n"
        f"- custom RPC status: `{custom_status}`\n"
        + endpoint_line
        +
        "- original config/chains template: unchanged; custom RPC stays in runtime configuration and validation evidence\n"
        f"- deferred user configuration requests: {queued_requests}\n"
        f"- saved later workflow goals: {workflow_goal_text}\n"
        f"- qps: `{qps_mode}`\n"
        f"- effective QPS values: {qps_values}\n"
        f"- observability: `{obs_mode}`\n"
        f"- confirmed fields: {confirmed_keys}\n"
        f"- config status: {action.config_status}\n"
        f"- execution status: {action.execution_status}\n"
        f"- next blocker: {action.next_blocking_reason or '<none>'}\n"
        f"- recommended next action: {format_recommended_next_action(action, language)}"
    )


def queued_configuration_summary(state: dict[str, Any], language: str) -> str:
    """Return user-visible deferred mutations from the durable action queue."""
    labels: list[str] = []
    for action in state.get("action_queue") or []:
        action_type = str((action or {}).get("type") or "")
        if action_type == "set_qps_mode":
            labels.append(f"QPS={action.get('qps_mode')}")
        elif action_type == "set_observability":
            labels.append(f"observability={action.get('observability_mode')}")
        elif action_type == "set_rpc_mode":
            labels.append(f"RPC mode={action.get('rpc_mode')}")
        elif action_type in {"choose_chain", "change_chain"}:
            labels.append(f"chain={action.get('chain_text')}")
        elif action_type == "choose_target_mode":
            labels.append(f"target mode={action.get('target_mode')}")
    return ", ".join(labels) or _localized(language, "<无>", "<none>")


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
            context = _pending_workflow_context(state)
            return (
                f"当前待确认问题是 `{pending_id}`。{prompt or '它用于补齐当前流程缺失的信息。'}\n"
                f"{context['zh']}\n"
                "之所以回到这个问题，是因为显式跳转/配置动作完成后，fallback 会选择当前状态中最早的未完成配置组；纯咨询不会改变它。\n"
                "你可以回答这个问题，也可以直接说要跳转到链、模式、磁盘、网络、RPC、QPS、可观测性、preflight/smoke、日志或报告分析。"
            )
        context = _pending_workflow_context(state)
        return (
            f"The current pending question is `{pending_id}`. {prompt or 'It fills a missing item in the current flow.'}\n"
            f"{context['en']}\n"
            "It is current because, after explicit detours or configuration actions, fallback selects the earliest incomplete relevant group; a consultation does not change it.\n"
            "You can answer it, or ask to jump to chain, mode, disk, network, RPC, QPS, observability, preflight/smoke, logs, or report analysis."
        )
    action = compute_next_action(state)
    if str(language or "").startswith("zh"):
        return f"当前没有待确认问题。{format_recommended_next_action(action, language)}"
    return f"There is no pending question. {format_recommended_next_action(action, language)}"


def _pending_workflow_context(state: AgentGraphState) -> dict[str, str]:
    identity = state.get("chain_identity") or {}
    queued = [
        str(item.get("type") or "")
        for item in state.get("action_queue") or []
        if str(item.get("type") or "") not in {"greeting", "ask_capabilities", "answer_opening_question"}
    ]
    queued_text = ", ".join(dict.fromkeys(queued)) or "<none>"
    target_mode = str(state.get("target_mode") or "<not selected>")
    chain = str(identity.get("canonical") or identity.get("raw") or "<not selected>")
    return {
        "zh": f"当前 target mode：`{target_mode}`；链：`{chain}`；尚未执行的已请求动作：{queued_text}。",
        "en": f"Current target mode: `{target_mode}`; chain: `{chain}`; requested actions not yet applied: {queued_text}.",
    }


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
        "target_mode": ("测试模式", "target mode"),
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
        "failure_recovery": ("执行失败恢复", "execution failure recovery"),
    }
    zh, en = labels.get(group, (group, group))
    return _localized(language, zh, en)


def _reason_label(reason: str, language: str) -> str:
    if not reason:
        return _localized(language, "补齐当前缺失信息", "fill the missing information")
    if reason.startswith("confirm "):
        field = reason.removeprefix("confirm ").strip()
        field_labels = {
            "target_mode": ("选择 fake-node、real-node 或 sync-observe", "choose fake-node, real-node, or sync-observe"),
            "chain": ("确认链名", "confirm the chain"),
            "rpc_mode": ("选择 single 或 mixed", "choose single or mixed"),
        }
        if field in field_labels:
            zh, en = field_labels[field]
            return _localized(language, zh, en)
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
        "resolve the current execution failure": ("查看证据并选择可执行的恢复方式", "inspect evidence and choose an executable recovery action"),
        "choose sync-observe data source": ("选择真实同步观测数据来源", "choose the sync-observe data source"),
        "acknowledge real client setup handoff": ("确认真实节点客户端准备交接", "acknowledge the real node client setup handoff"),
        "choose sync-observe data source after client setup": ("选择真实客户端准备完成后的数据来源", "choose the sync-observe data source after client setup"),
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
    precondition chain `routing.next_group_and_reason` uses to drive live turn
    routing. Two hand-maintained copies could (and did, in at least one
    case) disagree; see architecture audit Finding B1.
    """

    return next_group_and_reason(state)


def _execution_status(state: dict[str, Any]) -> str:
    job = state.get("job") or {}
    smoke = state.get("smoke") or {}
    preflight = state.get("preflight") or {}
    if job.get("status"):
        # `state["job"]` is a one-time snapshot written at submission time
        # (`agent/harness/domains/execution_runtime.py`) and never refreshed -- a
        # `current_config`/status-dump response could claim `job_running` for
        # a job that has actually long since finished, contradicting the
        # deterministic `status`/`jobs`/`logs` commands (which already read
        # correctly from disk via `job_manager`). Prefer the live on-disk
        # status by `job_id` when available; fall back to the snapshot only
        # if the job can no longer be read (e.g. its directory was removed).
        job_id = str(job.get("job_id") or "").strip()
        status = str(job.get("status"))
        if job_id:
            try:
                status = str(get_job(job_id).get("status") or status)
            except Exception:
                pass
        if status in {"running", "submitted", "completed", "failed", "partial"}:
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
    # The full prompt is presentation, not state-summary metadata. Embedding it
    # here causes current-state consultations to repeat the blocking question
    # in the blocker line, the recommendation, and the canonical renderer.
    field = str(pending.get("field") or "").strip()
    question_id = str(pending.get("id") or "").strip()
    subject = field or question_id
    return f"confirm {subject}" if subject else "answer current pending question"


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

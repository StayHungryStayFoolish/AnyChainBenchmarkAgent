"""Orientation and consultation behavior.

This domain answers questions from product facts and current state. It never
mutates benchmark configuration.
"""

from __future__ import annotations

from typing import Any

from agent.knowledge.entry_contract import ALL_RUNTIME_FIELDS
from agent.knowledge.chain_identity import canonicalize_chain_scalar, repo_chain_names
from agent.validators.rpc_workload import default_workload
from agent.runners.job_manager import get_job, list_jobs
from ..action_registry import canonical_consultation_topic
from ..contracts import ActionProposal, CheckpointCommand, HandlerResult, StateDelta
from ..failures import render_failure_summary, unresolved_recovery
from ..localization import localized
from ..oracle import format_current_context, format_current_state, format_startup_discovery, queued_configuration_summary
from ..questions import choice_question
from ..state import AgentGraphState
from .chain_identity import research_chain_identity


def has_resumable_configuration(state: AgentGraphState) -> bool:
    pending = state.get("pending_question") or {}
    pending_id = str(pending.get("id") or "")
    identity = state.get("chain_identity") or {}
    return any(
        (
            bool(state.get("target_mode")),
            bool(identity.get("canonical") or identity.get("raw")),
            bool(state.get("confirmed_config")),
            bool(state.get("rpc_mode")),
            bool(state.get("qps_profile")),
            bool(state.get("observability")),
            bool(state.get("sync_observe")),
            bool(state.get("workflow_goals")),
            bool(pending_id and pending_id not in {"opening_next_action", "resume_harness_session"}),
            bool((state.get("checkpoint_recovery") or {}).get("status") == "quarantined"),
        )
    )


def resume_question(state: AgentGraphState) -> dict[str, Any]:
    language = str(state.get("language") or "en")
    summary = resume_summary(state)
    recovery = state.get("checkpoint_recovery") or {}
    if recovery.get("status") == "quarantined":
        prompt = localized(
            language,
            "检测到不完整或不一致的旧配置，已隔离，尚未恢复到当前会话。\n1. 查看隔离原因和安全字段\n2. 仅保留可安全识别的确认值\n3. 清空旧配置重新开始",
            "An incomplete or inconsistent legacy configuration was quarantined and has not been resumed.\n1. Inspect the reason and safe fields\n2. Retain only safely identified confirmed values\n3. Clear the legacy configuration and start over",
        )
        values = ("inspect_quarantine", "retain_safe", "reset")
    else:
        prompt = localized(
        language,
        f"检测到之前的 Agent 配置会话：\n{summary}\n1. 继续之前的配置\n2. 保留已确认值并修改\n3. 清空配置重新开始",
        f"Found a previous Agent configuration session:\n{summary}\n1. Continue previous configuration\n2. Keep confirmed values and modify\n3. Clear configuration and start over",
        )
        values = ("continue", "modify", "reset")
    question = choice_question(
        "opening",
        "resume_harness_session",
        prompt,
        field="resume_harness_session",
        options=[
            {
                "id": "1",
                "label": "1",
                "value": values[0],
                **(
                    {"semantic_action": "continue_current_flow"}
                    if values[0] == "continue"
                    else {}
                ),
                "expected_patch": {"checkpoint_recovery.status": "quarantined"}
                if values[0] == "inspect_quarantine"
                else {"resume_context": {}},
                "return_policy": "stop_after_response" if values[0] == "inspect_quarantine" else "fallback",
            },
            {"id": "2", "label": "2", "value": values[1], "expected_patch": {"resume_context": {}}},
            {"id": "3", "label": "3", "value": values[2], "expected_patch": {"confirmed_config": {}}},
        ],
        queue_barrier=True,
    )
    if state.get("action_queue"):
        question["resume_action_queue"] = True
    return question


def apply_orientation_action(state: AgentGraphState, action: ActionProposal) -> HandlerResult:
    """Apply an orientation action without stealing workflow ownership."""

    action_type = action.action_type
    if action_type == "set_response_language":
        language = str(action.arguments.get("language") or "").strip().lower()
        if language not in {"zh", "en"}:
            return HandlerResult(blocker="response language must be zh or en")
        return HandlerResult(
            delta=StateDelta.set_values({"language": language}),
            consumed_action_ids=(action.action_id,),
            completion="completed",
        )
    if action_type == "greeting":
        language = str(state.get("language") or "en")
        question = None
        if not state.get("pending_question") and not has_resumable_configuration(state):
            question = opening_question(state)
        return HandlerResult(
            visible_result=localized(
                language,
                "你好，我是 AnyChain Benchmark Agent。你可以直接说明测试目标、询问当前状态，或粘贴配置、日志和报告。",
                "Hi, I am AnyChain Benchmark Agent. State a test goal, ask about current state, or paste configuration, logs, or reports.",
            ),
            pending_question=question,
            next_group="opening" if question else "",
            consumed_action_ids=(action.action_id,),
            stop_after_response=True,
        )
    if action_type == "clarify_unresolved":
        clauses = [
            str(item).strip()
            for item in action.arguments.get("clauses") or []
            if str(item).strip()
        ]
        if not clauses:
            return HandlerResult(blocker="clarify_unresolved requires unresolved clauses")
        language = str(state.get("language") or "en")
        details = "\n".join(f"- {item}" for item in clauses)
        return HandlerResult(
            visible_result=localized(
                language,
                f"为了避免只应用你这一轮的部分需求，请先澄清以下尚未安全映射的内容：\n{details}",
                f"To avoid applying only part of this turn, clarify the following items that were not mapped safely:\n{details}",
            ),
            pending_question=dict(state.get("pending_question") or {}) or None,
            consumed_action_ids=(action.action_id,),
            stop_after_response=True,
            completion="unchanged",
        )
    if action_type == "reset_session":
        return HandlerResult(
            checkpoint_command=CheckpointCommand("reset"),
            consumed_action_ids=(action.action_id,),
            visible_result=localized(
                state.get("language", "en"),
                "已清空之前的 Agent 配置。启动环境推断和历史 job 已保留。请告诉我这次要测试什么。",
                "Cleared the previous Agent configuration. Startup discovery and historical jobs were preserved. Tell me what to test.",
            ),
            completion="completed",
            stop_after_response=True,
        )
    if action_type in {"ask_capabilities", "answer_opening_question"}:
        raw = {"topic": "capabilities"} if action_type == "ask_capabilities" else dict(action.arguments)
        topic = canonical_consultation_topic(raw.get("topic"))
        pending = dict(state.get("pending_question") or {})
        if topic == "recommendation" and not pending and not state.get("target_mode"):
            pending = recommendation_question(state)
        return HandlerResult(
            visible_result=answer_consultation(state, raw),
            pending_question=pending or None,
            consumed_action_ids=(action.action_id,),
            stop_after_response=True,
        )
    if action_type == "answer_pending":
        return HandlerResult(
            consumed_action_ids=(action.action_id,),
            blocker="answer_pending must be resolved against the active question contract",
        )
    if action_type == "unknown":
        return HandlerResult(
            consumed_action_ids=(action.action_id,),
            blocker="no safe typed action was resolved",
        )
    return HandlerResult(blocker=f"unsupported orientation action: {action_type}")


def apply_orientation_answer(
    state: AgentGraphState,
    question: dict[str, Any],
    value: Any,
    user_text: str,
) -> HandlerResult:
    """Apply opening/session answers without performing global routing."""

    question_id = str(question.get("id") or "")
    language = str(state.get("language") or "en")
    if question_id == "resume_harness_session":
        recovery = state.get("checkpoint_recovery") or {}
        if value == "inspect_quarantine":
            safe = recovery.get("safe_confirmed_config") or {}
            return HandlerResult(
                visible_result=localized(
                    language,
                    f"隔离原因：{recovery.get('error_type') or 'unknown'}。可安全识别的确认值：{safe or '<无>'}。请选择保留安全值或清空重新开始。",
                    f"Quarantine reason: {recovery.get('error_type') or 'unknown'}. Safely identified confirmed values: {safe or '<none>'}. Choose retain-safe or clear-and-restart.",
                ),
                pending_question=question,
                completion="blocked",
                stop_after_response=True,
            )
        if value == "retain_safe":
            return HandlerResult(
                checkpoint_command=CheckpointCommand(
                    "retain_safe",
                    confirmed_config=dict(recovery.get("safe_confirmed_config") or {}),
                ),
                visible_result=localized(
                    language,
                    "仅保留了可安全识别的确认值。请说明要继续或修改什么。",
                    "Retained only safely identified confirmed values. Say what to continue or modify.",
                ),
                clear_pending=True,
                next_group="opening",
                completion="completed",
                stop_after_response=True,
            )
        if value == "reset":
            return HandlerResult(
                checkpoint_command=CheckpointCommand("reset"),
                visible_result=localized(
                    language,
                    "已清空之前的 Agent 配置。启动环境推断和历史 job 已保留。请告诉我这次要测试什么。",
                    "Cleared the previous Agent configuration. Startup discovery and job history were preserved. Tell me what to test.",
                ),
                completion="completed",
                stop_after_response=True,
            )
        if value == "modify":
            return HandlerResult(
                delta=StateDelta.set_values({"resume_context": {}}),
                visible_result=localized(
                    language,
                    "已保留确认值。请直接说明要修改链、模式、磁盘、网络、RPC、QPS、可观测性或其他配置组。",
                    "Confirmed values were kept. Name the chain, mode, disk, network, RPC, QPS, observability, or other group to modify.",
                ),
                clear_pending=True,
                next_group="opening",
                completion="completed",
                stop_after_response=True,
            )
        resume_context = dict(state.get("resume_context") or {})
        restored_pending = dict(resume_context.get("pending_question") or {})
        return HandlerResult(
            delta=StateDelta.set_values({"resume_context": {}}),
            pending_question=restored_pending or None,
            clear_pending=not restored_pending,
            next_group=str(resume_context.get("active_group") or restored_pending.get("group") or "opening"),
            visible_result=localized(language, "已继续之前的配置。", "Continuing the previous configuration."),
            completion="completed",
        )
    if question_id == "accept_recommendation":
        if not value:
            return HandlerResult(
                visible_result=localized(
                    language,
                    "好的，不按推荐。你可以直接说要测哪条链、用哪种模式（fake-node / real-node / sync-observe）。",
                    "OK, not using the recommendation. Tell me which chain and mode (fake-node / real-node / sync-observe) you want.",
                ),
                clear_pending=True,
                next_group="opening",
                completion="completed",
            )
        recommended = question.get("recommended_setup") if isinstance(question.get("recommended_setup"), dict) else {}
        mode = str(recommended.get("target_mode") or "fake-node")
        return HandlerResult(
            followup_actions=(
                {
                    "type": "choose_target_mode",
                    "target_mode": mode,
                    "target_mode_explicit": True,
                    "source_evidence": str(user_text or "").strip(),
                    "selection_contract_verified": True,
                    "confidence": "high",
                },
            ),
            visible_result=localized(
                language,
                f"好的，按推荐进入 `{mode}`。下一步请选择要验证的链，我会逐项确认缺失配置。",
                f"Starting the recommended `{mode}` path. Next choose the chain to validate; I will confirm the remaining configuration item by item.",
            ),
            clear_pending=True,
            completion="completed",
        )
    return HandlerResult(blocker=f"unsupported orientation question: {question_id}")


def opening_question(state: AgentGraphState) -> dict[str, Any]:
    language = str(state.get("language") or "en")
    zh = language.startswith("zh")
    return choice_question(
        "opening",
        "opening_next_action",
        localized(language, "你想让我帮你做什么？", "What would you like me to help with?"),
        field="target_mode",
        options=[
            {
                "label": "启动 fake-node 测试" if zh else "Start a fake-node benchmark",
                "description": localized(
                    language,
                    "使用预录 fixtures 做最快的低风险框架闭环验证；不测量真实节点性能。",
                    "Fast, low-risk framework validation with recorded fixtures; it does not measure real-node performance.",
                ),
                "value": "fake-node",
                "action": {"type": "choose_target_mode", "target_mode": "fake-node", "target_mode_explicit": True},
                "expected_patch": {"target_mode": "fake-node", "workflow_mode": "rpc_benchmark"},
            },
            {
                "label": "启动 real-node 测试" if zh else "Start a real-node benchmark",
                "description": localized(
                    language,
                    "对可访问的真实节点 RPC endpoint 进行负载测试并分析性能瓶颈。",
                    "Load-test a reachable real-node RPC endpoint and analyze performance bottlenecks.",
                ),
                "value": "real-node",
                "action": {"type": "choose_target_mode", "target_mode": "real-node", "target_mode_explicit": True},
                "expected_patch": {"target_mode": "real-node", "workflow_mode": "rpc_benchmark"},
            },
            {
                "label": "启动 sync-observe（观察真实节点同步）" if zh else "Start sync-observe (real-node synchronization)",
                "description": localized(
                    language,
                    "观察真实节点追块、资源与可用 MGas/s 指标；不运行 Vegeta 压测。",
                    "Observe real-node synchronization, resources, and available MGas/s metrics without Vegeta load.",
                ),
                "value": "sync-observe",
                "action": {"type": "choose_target_mode", "target_mode": "sync-observe", "target_mode_explicit": True},
                "expected_patch": {"target_mode": "sync-observe", "workflow_mode": "sync_observe"},
            },
            {
                "label": "了解支持的链、RPC method 和二次开发方式" if zh else "Learn supported chains, RPC methods, and extension paths",
                "description": localized(
                    language,
                    "只读查看框架能力、默认 workload 与扩展路径，不修改测试配置。",
                    "Read-only guidance about capabilities, default workloads, and extension paths without changing test configuration.",
                ),
                "value": "info",
                "action": {"type": "answer_opening_question", "topic": "capabilities"},
                "expected_patch": {},
                "return_policy": "stop_after_response",
            },
        ],
    )


def recommendation_question(state: AgentGraphState) -> dict[str, Any]:
    """Create the executable contract for the low-risk validation recommendation."""

    language = str(state.get("language") or "en")
    question = choice_question(
        "opening",
        "accept_recommendation",
        localized(
            language,
            "是否先进入 fake-node smoke，再选择要验证的链？",
            "Start with fake-node smoke, then choose the chain to validate?",
        ),
        field="accept_recommendation",
        kind="yes_no",
        options=[
            {
                "label": "Y",
                "value": True,
                "expected_patch": {"active_group": "opening"},
            },
            {
                "label": "N",
                "value": False,
                "expected_patch": {"active_group": "opening"},
            },
        ],
        queue_barrier=True,
    )
    question["recommended_setup"] = {"target_mode": "fake-node"}
    return question


def resume_summary(state: AgentGraphState) -> str:
    identity = state.get("chain_identity") or {}
    confirmed = state.get("confirmed_config") or {}
    qps = state.get("qps_profile") or {}
    observability = state.get("observability") or {}
    language = str(state.get("language") or "en")
    return "\n".join(
        (
            f"- target_mode: {state.get('target_mode') or '<not selected>'}",
            f"- workflow_mode: {state.get('workflow_mode') or '<not selected>'}",
            f"- chain: {identity.get('canonical') or identity.get('raw') or '<not selected>'}",
            f"- rpc_mode: {state.get('rpc_mode') or '<not selected>'}",
            f"- qps: {qps.get('mode') or '<not selected>'}",
            f"- observability: {observability.get('mode') or '<not selected>'}",
            f"- confirmed fields: {', '.join(sorted(confirmed)) if confirmed else '<none>'}",
            f"- deferred requests: {queued_configuration_summary(state, language)}",
        )
    )


def answer_consultation(state: AgentGraphState, action: dict[str, Any]) -> str:
    language = str(state.get("language") or "en")
    topic = canonical_consultation_topic(action.get("topic"))
    subject = str(action.get("subject") or "").strip()

    if topic == "identity":
        return localized(
            language,
            "我是 AnyChain Benchmark Agent，运行在当前 AnyChain Benchmark 工程中，负责区块链节点测试的配置、校验、执行和报告分析。你可以直接提出测试目标、询问当前状态，或粘贴配置、日志和报告。",
            "I am AnyChain Benchmark Agent, running in the current AnyChain Benchmark project. I configure, validate, execute, and analyze blockchain-node benchmarks and reports. State a goal, ask about current state, or paste configuration, logs, or reports.",
        )
    if topic in {"capabilities", "agent_capability"}:
        return _capabilities(state)
    if topic == "supported_chains":
        return _supported_chains(state, subject)
    if topic == "extension":
        current_chain = str((state.get("chain_identity") or {}).get("canonical") or "")
        chain = canonicalize_chain_scalar(subject or current_chain, known_chains=set(repo_chain_names())) or ""
        defaults = ""
        if chain:
            defaults = ", ".join(default_workload(chain).get("methods") or []) or "<none>"
        return localized(
            language,
            (
                "扩展分三类：已支持链的自定义 RPC、现有协议族的新链，以及未支持协议族的二次开发。\n"
                f"已支持链 `{chain}` 的自定义 RPC 流程：提供用于真实验证的 endpoint，再提供 method 名、params/request、response 或官方文档。"
                f"我会抽取并校验 schema，让你确认是保留、追加还是替换默认 methods，然后确认 single/mixed 和总和为 100 的权重。"
                f"模板默认 methods: {defaults or '<select a supported chain>'}。运行时配置不会修改 `config/chains` 默认模板。"
            ),
            (
                "There are three extension cases: custom RPC for a supported chain, a new chain in an existing adapter family, and secondary development for an unsupported family. "
                f"Custom RPC flow for supported chain `{chain}`: provide the endpoint selected for real validation, then the method name, params/request, response, or official documentation. "
                f"I extract and validate the schema, confirm whether defaults are retained, augmented, or replaced, then confirm single/mixed mode and weights totaling 100. "
                f"Template defaults: {defaults or '<select a supported chain>'}. Runtime configuration never mutates `config/chains` defaults."
            ),
        )
    if topic == "config_explanation":
        explanation = config_field_explanation(state, subject)
        if explanation:
            return _with_active_failure_context(state, explanation)
        pending = state.get("pending_question") or {}
        if pending and subject in {
            str(pending.get("id") or ""),
            str(pending.get("field") or ""),
        }:
            return _with_active_failure_context(
                state,
                pending_context_response(state, pending),
            )
        if subject:
            return _with_active_failure_context(
                state,
                pending_context_response(state, {"id": subject}),
            )
        if state.get("pending_question"):
            return _with_active_failure_context(
                state,
                pending_context_response(state, state.get("pending_question") or {}),
            )
    if topic == "current_config":
        return _with_active_failure_context(state, format_current_state(state, language))
    if topic == "workload_config":
        return _workload_config(state, language)
    if topic in {"current_context", "next_action"}:
        return _with_active_failure_context(state, format_current_context(state, language))
    if topic in {"startup_discovery", "environment_inference"}:
        return format_startup_discovery(state, language)
    if topic == "environment_readiness":
        discovery = state.get("discovery") or {}
        host = discovery.get("host") or {}
        dependencies = discovery.get("dependencies") or {}
        machine = (
            (discovery.get("cloud") or {}).get("machine_type")
            or host.get("machine_type")
            or host.get("hostname")
            or "<unknown>"
        )
        status = "ready" if not dependencies.get("missing_required") else "blocked"
        return localized(
            language,
            f"环境就绪状态：{status}。检测到的机器/host：{machine}。\n" + format_startup_discovery(state, language),
            f"Environment readiness: {status}. Detected machine/host: {machine}.\n" + format_startup_discovery(state, language),
        )
    if topic in {"requirements", "workflow"}:
        return _requirements(state, language)
    if topic in {"mode_comparison", "performance_benchmark_guidance"}:
        return _mode_comparison(language, performance_goal=topic == "performance_benchmark_guidance")
    if topic == "execution_preflight_smoke":
        return localized(
            language,
            "preflight（预检）在执行前校验依赖、配置、endpoint 和运行前置条件；smoke（冒烟测试）以安全的小流量执行完整链路，验证请求、日志、产物和报告能够闭环。smoke 不是正式性能结论。",
            "Preflight validates dependencies, configuration, endpoints, and execution prerequisites. Smoke runs the complete path at safe low traffic to verify requests, logs, artifacts, and reports. Smoke is not a production performance conclusion.",
        )
    if topic == "recommendation":
        return localized(
            language,
            "如果只想快速确认 Agent 和工具链能否闭环，建议先用 fake-node smoke；它不需要真实节点，但不代表真实节点性能。可以从 `solana` 开始，也可以直接说出目标链。",
            "For a quick end-to-end Agent/toolchain check, start with fake-node smoke. It needs no real node but does not represent real-node performance. Start with `solana`, or name the target chain.",
        )
    if topic in {"reset_help", "reset"}:
        return localized(
            language,
            "可以完全清空当前配置并完全重新开始，也可以保留已确认值并修改任意配置组。重置不会删除启动环境推断和历史 job。请直接说明“清空配置”或要修改的项目。",
            "You can clear the current workflow and start over, or keep confirmed values and modify any configuration group. Reset preserves startup discovery and job history. Say `clear configuration` or name the item to change.",
        )
    if topic in {"evidence_help", "log_help"}:
        return localized(
            language,
            "可以。粘贴日志、错误栈或命令输出；多行内容会作为一个证据块接收。也可以指定 job id，让我读取已有日志和产物后分析。",
            "Yes. Paste logs, a stack trace, or command output; multiline input is collected as one evidence block. You may also name a job id so I can read its existing logs and artifacts.",
        )
    if topic in {"correction", "clarification"}:
        failure_context = _active_failure_context(state)
        if failure_context:
            return failure_context
        return localized(
            language,
            "我理解，刚才的回答没有解决你的问题。请继续原问题；我会结合当前状态回答，不会把它当作配置确认或推进流程。",
            "I understand: my previous answer did not resolve your question. Continue with the original question; I will answer from current state without treating it as configuration confirmation or advancing the workflow.",
        )
    if topic in {"current_job", "job_status", "execution_status"}:
        return _job_status(state, language)
    return localized(
        language,
        "我可以解释产品能力、测试模式、配置要求、当前状态、历史 job、错误日志和报告，也可以继续当前配置流程。请直接说明问题。",
        "I can explain product capabilities, test modes, requirements, current state, historical jobs, errors, and reports, or continue the current setup. Ask the question directly.",
    )


def _workload_config(state: AgentGraphState, language: str) -> str:
    identity = state.get("chain_identity") or {}
    chain = str(identity.get("canonical") or identity.get("raw") or "").strip()
    rpc_mode = str(state.get("rpc_mode") or "").strip()
    effective = state.get("workload") or {}
    if effective.get("confirmed") and effective.get("methods"):
        methods = ", ".join(str(item) for item in effective.get("methods") or [])
        weights = effective.get("weights") if isinstance(effective.get("weights"), dict) else {}
        weight_text = ", ".join(f"{method}={weight}" for method, weight in weights.items()) or "<not applicable>"
        return localized(
            language,
            f"当前有效 RPC workload：链 `{chain or '<未选择>'}`，模式 `{rpc_mode or '<未选择>'}`，methods：{methods}，mixed 权重：{weight_text}。",
            f"Effective RPC workload: chain `{chain or '<not selected>'}`, mode `{rpc_mode or '<not selected>'}`, methods: {methods}, mixed weights: {weight_text}.",
        )
    defaults = default_workload(chain) if chain else {}
    single = str(defaults.get("single") or "<none>")
    mixed = ", ".join(
        f"{item.get('method')}={item.get('weight')}"
        for item in defaults.get("mixed_weighted") or []
        if isinstance(item, dict) and item.get("method")
    ) or "<none>"
    return localized(
        language,
        f"链 `{chain or '<未选择>'}` 的模板 workload：single 默认 method 为 `{single}`；mixed 默认权重为 {mixed}。这些值尚未确认，不会修改原始 chain template。",
        f"Template workload for `{chain or '<not selected>'}`: default single method `{single}`; default mixed weights: {mixed}. These values are not confirmed yet and do not modify the original chain template.",
    )


def _active_failure_context(state: AgentGraphState) -> str:
    recovery = state.get("failure_recovery") or {}
    record = recovery.get("record") if unresolved_recovery(recovery) else None
    if not isinstance(record, dict):
        endpoint_record = (state.get("endpoint_evidence") or {}).get("last_failure_record")
        record = endpoint_record if isinstance(endpoint_record, dict) else None
    if not record:
        return ""
    language = str(state.get("language") or "en")
    pending = state.get("pending_question") or {}
    field = str(pending.get("field") or pending.get("id") or "").strip()
    if field:
        correction = localized(
            language,
            f"- 当前应修正：在 `{record.get('affected_group') or '<unknown>'}` 组重新提供 `{field}`；其他已确认配置保持不变。",
            f"- Correct now: provide `{field}` again in `{record.get('affected_group') or '<unknown>'}`; other confirmed configuration remains unchanged.",
        )
        return f"{render_failure_summary(record, language)}\n{correction}"
    return render_failure_summary(record, language)


def _with_active_failure_context(state: AgentGraphState, response: str) -> str:
    context = _active_failure_context(state)
    return f"{response}\n{context}" if context else response


def pending_context_response(state: AgentGraphState, question: dict[str, Any]) -> str:
    """Explain the active question without advancing workflow state."""

    language = str(state.get("language") or "en")
    question_id = str(question.get("id") or "").strip()
    help_text = str(question.get("help_text") or "").strip()
    completion_effect = str(question.get("completion_effect") or "").strip()
    if help_text:
        if completion_effect:
            return f"{help_text}\n{completion_effect}"
        return help_text
    if question_id in {"LOCAL_RPC_URL", "SYNC_OBSERVE_RPC_URL"}:
        explanation = config_field_explanation(state, question_id, language)
        if explanation:
            return explanation
    if question_id == "new_chain_endpoint":
        return localized(
            language,
            "这里是在验证未内置的新链能否使用现有协议族。请提供可访问的 HTTP RPC endpoint、至少一个 RPC method 及参数/响应示例；官方文档也可以作为证据。该 endpoint 只作为验证证据，不会自动成为最终被测 endpoint。",
            "This validates whether a non-template chain can use an existing adapter family. Provide a reachable HTTP RPC endpoint, at least one RPC method with params/response samples, and official docs when available. The validation endpoint does not automatically become the final benchmark endpoint.",
        )
    if question_id == "custom_rpc_endpoint":
        return localized(
            language,
            "这里是在验证自定义 RPC method。请提供可访问的验证 endpoint，以及 method、params、response 示例或官方文档；mixed workload 的启用 method 权重最终必须合计 100。验证 endpoint 不会覆盖最终 `LOCAL_RPC_URL`，也不会修改原始 chain template。",
            "This validates a custom RPC method. Provide a reachable validation endpoint plus method, params, response samples, or official docs; enabled mixed-workload weights must ultimately total 100. The validation endpoint does not replace the final `LOCAL_RPC_URL` or modify the original chain template.",
        )
    if question_id == "custom_rpc_fixture_choice":
        workload = state.get("workload") or {}
        methods = ", ".join(str(item) for item in workload.get("methods") or []) or "<unknown>"
        preserved = ", ".join(sorted((state.get("confirmed_config") or {}).keys())) or "<none>"
        return localized(
            language,
            (
                f"自定义 workload `{methods}` 的 endpoint 和 schema 已验证，但 fake-node 缺少对应 fixture，因此不能提交 smoke。"
                "你可以改用链模板默认 workload、保留自定义 workload 并切换到 real-node，或生成 fixture 录制交接。"
                f"无论选择哪条路径，已确认的硬件、网络和 QPS 配置都会保留：{preserved}。"
            ),
            (
                f"The endpoint and schema for custom workload `{methods}` are validated, but fake-node lacks the required fixture, so smoke cannot be submitted. "
                "You can use the chain-template defaults, preserve the custom workload and switch to real-node, or generate a fixture-recording handoff. "
                f"Confirmed hardware, network, and QPS configuration is preserved across these choices: {preserved}."
            ),
        )
    return format_current_context(state, language)


def capability_summary(state: AgentGraphState) -> str:
    return _capabilities(state)


def completed_group_status(state: AgentGraphState, group: str) -> str:
    """Explain an already-complete group without changing workflow state."""

    language = state.get("language", "en")
    if group == "observability":
        mode = str((state.get("observability") or {}).get("mode") or "").strip()
        if mode:
            return localized(
                language,
                f"当前可观测性模式已设置为 `{mode}`。如需修改，请选择本地 Prometheus/Grafana、exporter-only 或 disabled。",
                f"Observability is set to `{mode}`. To change it, choose local Prometheus/Grafana, exporter-only, or disabled.",
            )
    if group == "qps_profile":
        qps = state.get("qps_profile") or {}
        if qps.get("mode") and qps.get("confirmed"):
            overrides = qps.get("overrides") or {}
            detail = ", ".join(f"{key}={value}" for key, value in sorted(overrides.items())) or "default profile"
            return localized(
                language,
                f"当前 QPS profile 为 `{qps.get('mode')}`，配置：{detail}。可修改 INITIAL_QPS、MAX_QPS、QPS_STEP 或 DURATION。",
                f"The current QPS profile is `{qps.get('mode')}` with {detail}. You can change INITIAL_QPS, MAX_QPS, QPS_STEP, or DURATION.",
            )
    if group == "workload_rpc":
        workload = state.get("workload") or {}
        if state.get("rpc_mode") and workload.get("confirmed"):
            return localized(
                language,
                f"当前 RPC workload 已确认：mode=`{state.get('rpc_mode')}`。可切换 single/mixed、添加自定义 RPC method 或调整 mixed 权重。",
                f"The RPC workload is confirmed with mode=`{state.get('rpc_mode')}`. You can switch single/mixed, add a custom RPC method, or adjust mixed weights.",
            )
    return ""


def config_field_explanation(state: AgentGraphState, subject: str, language: str = "") -> str:
    language = str(language or state.get("language") or "en")
    identifier = subject or str((state.get("pending_question") or {}).get("field") or "")

    def normalize(value: str) -> str:
        return value.strip().lower().replace("_", "").replace("-", "").replace(" ", "").rstrip("s")

    wanted = normalize(identifier)
    if not wanted:
        return ""
    field = next(
        (
            candidate
            for candidate in ALL_RUNTIME_FIELDS
            if wanted in {normalize(candidate.env), normalize(candidate.key), normalize(candidate.label)}
        ),
        None,
    )
    if field is None:
        return ""
    display_id = field.env or field.key
    modes = ", ".join(mode.replace("_", "-") for mode in field.applies_to)
    return localized(
        language,
        f"`{display_id}`（{field.label}）\n作用：{field.description or field.reason}\n属性：{'必填' if field.required else '可选'}；适用模式：{modes}。\n如果启动检测给出候选值，可以确认该值；否则需要手动输入。",
        f"`{display_id}` ({field.label})\nPurpose: {field.description or field.reason}\nIt is {'required' if field.required else 'optional'} and applies to: {modes}.\nAccept a detected candidate when available; otherwise enter it manually.",
    )


def _capabilities(state: AgentGraphState) -> str:
    language = str(state.get("language") or "en")
    facts = state.get("framework_summary") or {}
    prefix = localized(
        language,
        "我是 AnyChain Benchmark Agent。我支持 fake-node 闭环验证、real-node RPC 压测、sync-observe 同步观察、自定义 RPC、新链协议适配、preflight/smoke、job 跟踪，以及日志和报告分析。",
        "I am AnyChain Benchmark Agent. I support fake-node closed-loop validation, real-node RPC load tests, sync-observe, custom RPC methods, new-chain onboarding, preflight/smoke, job tracking, and log/report analysis.",
    )
    return (
        f"{prefix}\n"
        f"{localized(language, '当前框架事实', 'Current framework facts')}: "
        f"{facts.get('chain_count', '?')} chains, {facts.get('family_count', '?')} adapter families, "
        f"{facts.get('unique_rpc_method_count', '?')} RPC methods."
    )


def _supported_chains(state: AgentGraphState, subject: str) -> str:
    language = str(state.get("language") or "en")
    facts = state.get("framework_summary") or {}
    chains = []
    for item in facts.get("chains") or []:
        value = str(item.get("chain") if isinstance(item, dict) else item).strip()
        if value:
            chains.append(value)
    if subject:
        chain = canonicalize_chain_scalar(subject, known_chains=set(repo_chain_names())) or ""
        if chain:
            workload = default_workload(chain)
            if workload.get("exists"):
                single = str(workload.get("single") or "<none>")
                mixed = ", ".join(
                    f"{row.get('method')}={row.get('weight')}"
                    for row in workload.get("mixed_weighted") or []
                    if row.get("method")
                ) or "<none>"
                methods = ", ".join(workload.get("methods") or []) or "<none>"
                return localized(
                    language,
                    f"链 `{chain}` 的默认 RPC workload：\n- single: {single}\n- mixed: {mixed}\n- methods: {methods}",
                    f"Default RPC workload for `{chain}`:\n- single: {single}\n- mixed: {mixed}\n- methods: {methods}",
                )
        resolution = research_chain_identity(state, subject)
        canonical_name = str(resolution.get("canonical_chain_name") or subject).strip()
        family = str(resolution.get("adapter_family") or "unknown").strip()
        protocol = str(resolution.get("protocol_or_api") or "").strip()
        search_result = resolution.get("search_result")
        grounded = isinstance(search_result, dict) and search_result.get("available") is True
        if resolution.get("chain_exists") is True:
            source = localized(
                language,
                "google_search 官方资料核实" if grounded else "当前模型知识判断（未进行互联网核实）",
                "official google_search grounding" if grounded else "current model knowledge (not web-grounded)",
            )
            return localized(
                language,
                f"`{canonical_name}` 不在当前已配置链模板中。{source}：协议/API 为 `{protocol or '<unknown>'}`，协议族判断为 `{family}`。如要测试，需要先由你确认链名和协议，再进入 endpoint/RPC 验证流程。",
                f"`{canonical_name}` is not a configured chain template. Based on {source}, its protocol/API is `{protocol or '<unknown>'}` and the proposed adapter family is `{family}`. To test it, confirm the chain identity and protocol before endpoint/RPC validation.",
            )
        return localized(
            language,
            f"`{subject}` 不在当前已配置链模板中，当前模型也无法可靠确认它的链身份或协议。若要继续，请确认它是真实链并提供官方协议/RPC 资料；之后会进入 Case 2 或 Case 3。",
            f"`{subject}` is not a configured chain template, and the current model could not reliably confirm its identity or protocol. To continue, confirm that it is a real chain and provide official protocol/RPC evidence; the flow will then enter Case 2 or Case 3.",
        )
    return localized(language, "已配置链：", "Configured chains: ") + ", ".join(chains)


def _requirements(state: AgentGraphState, language: str) -> str:
    requirements = localized(
        language,
        "测试前会逐组确认：1. 测试类型和链；2. 云和机器（`CLOUD_REGION`、`CLOUD_ZONE`、`MACHINE_TYPE`）；3. Ledger/data、可选 accounts/state 磁盘和网络；4. endpoint 和节点进程；5. RPC workload、自定义 method、权重和 fixtures；6. QPS profile；7. 可观测性和高级配置；8. preflight/smoke、job 和报告。用户可以随时跳转或回退，完成后从状态重新计算下一个缺失项。real-node 需要 `LOCAL_RPC_URL`；sync-observe 需要真实节点来源，不配置 RPC workload/QPS，也不走 Vegeta。",
        "Before testing, the Agent confirms: 1. test type and chain; 2. cloud and machine (`CLOUD_REGION`, `CLOUD_ZONE`, `MACHINE_TYPE`); 3. Ledger/data disk, optional accounts/state disk, and network; 4. endpoint and node process; 5. RPC workload, custom methods, weights, and fixtures; 6. QPS profile; 7. observability and advanced settings; 8. preflight/smoke, jobs, and reports. Users may jump or go back at any time; the next missing group is recomputed from state. real-node requires `LOCAL_RPC_URL`; sync-observe requires a real-node source and does not configure RPC workload/QPS or run Vegeta.",
    )
    if not state.get("target_mode"):
        requirements += localized(
            language,
            "\n如果目标只是低成本确认整个工具链能否闭环，先选择 fake-node smoke；它不需要真实 endpoint，但不能代表真实节点性能。",
            "\nFor the lowest-cost end-to-end toolchain check, start with fake-node smoke. It needs no real endpoint but does not represent real-node performance.",
        )
    return requirements


def _mode_comparison(language: str, *, performance_goal: bool = False) -> str:
    comparison = localized(
        language,
        "fake-node 的作用是使用 fixtures 验证框架闭环，不代表真实节点性能，也不能回答真实节点支持多少 QPS；real-node benchmark 使用 `LOCAL_RPC_URL` 对真实 endpoint 执行 RPC 压测，用于 QPS、延迟和瓶颈分析；sync-observe 观察真实节点追块、进程 CPU/内存、磁盘、网络及客户端可用的 MGas/s，不走 Vegeta。",
        "fake-node only validates the framework loop with fixtures; it does not represent real-node performance or answer real-node QPS capacity. real-node benchmark uses `LOCAL_RPC_URL` to load a real endpoint for QPS, latency, and bottleneck analysis. sync-observe watches real-node catch-up, process CPU/memory, disk, network, and client-native MGas/s when available, without Vegeta.",
    )
    if performance_goal:
        return comparison + localized(
            language,
            "\n因此，如果目标是测试可支持的 QPS 和瓶颈，应选择 real-node benchmark。",
            "\nTherefore, choose real-node benchmark when the goal is QPS capacity and bottleneck discovery.",
        )
    return comparison


def _job_status(state: AgentGraphState, language: str) -> str:
    job = state.get("job") or {}
    job_id = str(job.get("job_id") or "").strip()
    if not job_id:
        try:
            jobs = list_jobs(limit=1)
            job = dict(jobs[0]) if jobs else {}
            job_id = str(job.get("job_id") or "").strip()
        except Exception:
            job = {}
    if not job_id:
        return localized(language, "当前没有历史 job。", "No historical job is available.")
    try:
        persisted = get_job(job_id)
    except (FileNotFoundError, OSError, ValueError):
        persisted = {}
    status = str(persisted.get("status") or job.get("status") or "unknown")
    return localized(language, f"当前 job：`{job_id}`，状态：`{status}`。", f"Current job: `{job_id}`, status: `{status}`.")

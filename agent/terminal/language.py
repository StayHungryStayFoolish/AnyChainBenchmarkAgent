"""Small language policy helpers for terminal UX."""

from __future__ import annotations

import re


_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_ASCII_ALPHA_RE = re.compile(r"[A-Za-z]")
_SINGLE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.:/+-]+[,，;；、]?$")
_TECHNICAL_SCALAR_RE = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_.:/-]*\s*=\s*[^=,\s]+)(\s*[,，]\s*[A-Za-z_][A-Za-z0-9_.:/-]*\s*=\s*[^=,\s]+)*[,，;；]?$"
)
_CONFIG_FACT_RE = re.compile(
    r"\b(region|zone|machine|ledger|accounts|network|bandwidth|iops|throughput|endpoint|qps|rpc|grafana|prometheus)\b|[A-Za-z_][A-Za-z0-9_.:/-]*\s*=",
    re.IGNORECASE,
)
_TECHNICAL_DOC_RE = re.compile(
    r"\b(path parameters?|query parameters?|request|response|method|jsonrpc|rest api|endpoint|curl|headers?)\b",
    re.IGNORECASE,
)
_TECHNICAL_PASTE_RE = re.compile(
    r"(^|\n)\s*(curl\b|--data\b|--header\b|response\s*:|request\s*:|traceback\b|\{|\[)",
    re.IGNORECASE,
)


def detect_language(text: str, default: str = "en") -> str:
    """Return the preferred response language for a user turn."""
    raw = text or ""
    stripped = raw.strip().lower()
    if stripped in {"y", "n", "yes", "no", "back", "previous", "undo"}:
        return default if default in {"zh", "en"} else "en"
    if stripped in {"hi", "hello", "hey"}:
        return "en"
    if _CJK_RE.search(raw):
        return "zh"
    if default == "zh" and ("\n" in raw or _TECHNICAL_PASTE_RE.search(raw)):
        return "zh"
    if default == "zh" and _SINGLE_TOKEN_RE.fullmatch(raw.strip()):
        return "zh"
    if default == "zh" and _TECHNICAL_SCALAR_RE.fullmatch(raw.strip()):
        return "zh"
    if default == "zh" and (_CONFIG_FACT_RE.search(raw) or _TECHNICAL_DOC_RE.search(raw)):
        return "zh"
    if _ASCII_ALPHA_RE.search(raw):
        return "en"
    if default in {"zh", "en"}:
        return default
    return "en"


def t(language: str, key: str, **values: object) -> str:
    """Translate fixed terminal-shell messages."""
    table = _ZH if language == "zh" else _EN
    template = table.get(key) or _EN.get(key) or key
    return template.format(**values)


_ZH = {
    "welcome": "AnyChain Benchmark Agent 已启动。",
    "mode": "当前模型配置：provider={provider}, model={model}, auth={auth_mode}",
    "web_research": "Web research：{status}",
    "adk": "ADK runtime：{status}",
    "job_found": "检测到最近 job：{job_id}，状态：{status}",
    "job_next_actions": "可选下一步：{actions}",
    "job_none": "没有检测到历史 job。",
    "prompt": "User> ",
    "agent": "Agent> {message}",
    "bye": "已退出 AnyChain Benchmark Agent。",
    "help": "你可以说：测试 solana、使用 fake-node、doctor、jobs、status、logs <job_id>、follow <job_id>、help、exit。",
    "resume_session_offer": "检测到之前的 Agent 配置会话：\n{summary}\n请选择下一步：\n1. 继续之前的配置\n2. 保留已确认配置，但告诉我你要修改哪一项\n3. 清空之前的配置，重新开始",
    "resume_session_continue": "已继续之前的配置。{next_step}",
    "resume_session_modify": "已保留已确认配置，并退出旧的待回答问题。请直接告诉我要修改哪一项，例如链、模式、磁盘、QPS、RPC 或可观测性。",
    "resume_session_clear": "已清空之前的 Agent 配置会话。请告诉我这次要测试什么，或先选择 fake-node / real-node / sync-observe。",
    "resume_session_invalid": "请回复 1、2 或 3：1 继续，2 修改已有配置，3 清空重新开始。",
    "resume_pending_next": "当前待确认：{prompt}",
    "resume_no_pending_next": "当前没有待确认问题，你可以继续描述目标或要修改的配置。",
    "startup_doctor_start": "正在启动时执行只读环境和依赖检查。",
    "startup_doctor_summary": "启动检查完成：status={status}，cloud={cloud}，deployment={deployment}，缺失依赖={missing}，能力={chains} chains / {methods} RPC methods。",
    "environment_inference_summary": "环境推断草案：\n{summary}",
    "doctor_start": "正在执行只读环境检查。",
    "doctor_summary": "检查完成：status={status}，缺失依赖={missing}，能力={chains} chains / {methods} RPC methods。",
    "dependency_offer": "检测到缺失依赖：{missing}。我可以在你确认后运行 scripts/install_deps.sh --yes。是否允许？[Y/n]",
    "dependency_required_for_benchmark": "执行 smoke 或 benchmark 前需要先处理缺失依赖：{missing}。是否允许我现在运行 scripts/install_deps.sh --yes？[Y/n]",
    "dependency_declined": "已跳过依赖安装。后续真实 benchmark 可能仍会被 preflight 阻止。",
    "dependency_install_start": "开始安装 benchmark 依赖。这一步可能需要一些时间。",
    "dependency_install_done": "依赖安装命令完成，exit_code={exit_code}。",
    "agent_runtime_offer": "检测到 Agent runtime 依赖缺失：google-adk。是否允许我运行 scripts/install_agent_deps.sh --yes 安装到隔离环境？[Y/n]",
    "agent_runtime_declined": "已跳过 Agent runtime 安装。底层 LLM/ADK 能力仍不可用。",
    "agent_runtime_install_start": "开始安装 Agent runtime 依赖到隔离环境。",
    "agent_runtime_install_done": "Agent runtime 安装命令完成，exit_code={exit_code}。",
    "llm_config_warning": "LLM 配置还不完整：{errors}",
    "jobs_empty": "没有找到 job。",
    "jobs_header": "最近 job：",
    "job_not_found": "没有找到 job：{job_id}",
    "log_path": "job={job_id} 的日志路径：{path}",
    "log_missing": "日志文件尚未生成。job 可能刚启动，稍后可再次输入 logs 或 follow。",
    "log_empty": "日志文件当前为空。",
    "follow_start": "开始跟踪 job={job_id} 日志：{path}\n按 Ctrl+C 只会退出日志跟踪，不会停止 benchmark，也不会退出 Agent。",
    "follow_stopped": "已退出日志跟踪。benchmark 如果仍在运行会继续执行。你可以把日志片段复制到 User>，Agent 会按 evidence 分析。日志路径：{path}",
    "follow_done": "日志跟踪结束，job 状态：{status}",
    "unknown": "ADK 没有返回可显示内容。你可以继续描述测试目标，或输入 doctor/status/jobs 查看确定性状态。",
    "adk_runtime_error": "底层模型调用暂时失败，我不会展示内部错误。这个自然语言请求尚未完成；请重试，或输入 doctor/status/jobs 查看确定性状态。",
    "llm_billing_error": "底层模型服务返回余额或配额不足。自然语言 Agent 能力暂时不可用；请补充模型账户余额/配额后重试。doctor/status/jobs 这些确定性命令仍可使用。",
    "thinking": "[thinking] 正在处理当前请求。如果超过 15 秒，你可以按 Ctrl+C 取消本轮，不会退出 Agent。",
    "turn_cancelled": "已取消当前这一轮。Agent 会话仍在，你可以继续输入。",
    "pasted_evidence_detected": "检测到你粘贴的是日志、旧对话或错误信息。我会把它当作 evidence 分析，不会直接写入 benchmark 配置。",
    "pasted_evidence_buffered": "已记录粘贴的日志/旧对话片段。你可以继续粘贴，或直接输入要分析的问题。",
    "pending_answer_blocked": "这条回复没有通过当前问题的校验：{blockers}",
    "pending_answer_recorded": "已记录该回复。你可以继续描述下一步目标。",
    "unbound_structural_answer": "当前没有待确认的问题，这个短回复无法绑定到任何决策。请直接说明你的目标，例如：测试 solana fake-node quick、查看 job 状态、或分析最近报告。",
    "framework_context_loaded": "已加载框架事实：{chains} chains，{families} adapter families，{methods} RPC methods，fake-node fixtures={fixtures}。",
    "ctrl_c_exit": "收到 Ctrl+C，正在退出 AnyChain Benchmark Agent。",
    "adk_missing_hint": "注意：google-adk 当前不可用。请允许 Agent 安装隔离运行时，或先运行 bash scripts/install_agent_deps.sh --yes。",
}


_EN = {
    "welcome": "AnyChain Benchmark Agent started.",
    "mode": "Model config: provider={provider}, model={model}, auth={auth_mode}",
    "web_research": "Web research: {status}",
    "adk": "ADK runtime: {status}",
    "job_found": "Found latest job: {job_id}, status: {status}",
    "job_next_actions": "Available next actions: {actions}",
    "job_none": "No previous job was found.",
    "prompt": "User> ",
    "agent": "Agent> {message}",
    "bye": "Exited AnyChain Benchmark Agent.",
    "help": "Try: benchmark solana, use fake-node, doctor, jobs, status, logs <job_id>, follow <job_id>, help, exit.",
    "resume_session_offer": "Found an existing Agent configuration session:\n{summary}\nChoose next step:\n1. Continue the previous configuration\n2. Keep confirmed values, but tell me what to change\n3. Clear the previous configuration and start over",
    "resume_session_continue": "Continuing the previous configuration. {next_step}",
    "resume_session_modify": "Kept confirmed values and cleared the old pending question. Tell me what to change, such as chain, mode, disk, QPS, RPC, or observability.",
    "resume_session_clear": "Cleared the previous Agent configuration session. Tell me what to test, or choose fake-node / real-node / sync-observe first.",
    "resume_session_invalid": "Reply with 1, 2, or 3: 1 continue, 2 modify existing config, 3 clear and start over.",
    "resume_pending_next": "Current pending question: {prompt}",
    "resume_no_pending_next": "There is no pending question. Continue describing the goal or the configuration to change.",
    "startup_doctor_start": "Running startup read-only environment and dependency diagnostics.",
    "startup_doctor_summary": "Startup diagnostics complete: status={status}, cloud={cloud}, deployment={deployment}, missing dependencies={missing}, capabilities={chains} chains / {methods} RPC methods.",
    "environment_inference_summary": "Environment inference draft:\n{summary}",
    "doctor_start": "Running read-only environment diagnostics.",
    "doctor_summary": "Doctor complete: status={status}, missing dependencies={missing}, capabilities={chains} chains / {methods} RPC methods.",
    "dependency_offer": "Missing dependencies detected: {missing}. I can run scripts/install_deps.sh --yes after your confirmation. Allow this? [Y/n]",
    "dependency_required_for_benchmark": "Smoke or benchmark execution needs missing dependencies first: {missing}. Allow me to run scripts/install_deps.sh --yes now? [Y/n]",
    "dependency_declined": "Skipped dependency installation. A real benchmark may still be blocked by preflight.",
    "dependency_install_start": "Starting benchmark dependency installation. This may take a while.",
    "dependency_install_done": "Dependency installation command completed, exit_code={exit_code}.",
    "agent_runtime_offer": "Agent runtime dependency is missing: google-adk. Allow me to run scripts/install_agent_deps.sh --yes and install it into an isolated environment? [Y/n]",
    "agent_runtime_declined": "Skipped Agent runtime installation. Underlying LLM/ADK capabilities remain unavailable.",
    "agent_runtime_install_start": "Starting Agent runtime dependency installation into the isolated environment.",
    "agent_runtime_install_done": "Agent runtime installation command completed, exit_code={exit_code}.",
    "llm_config_warning": "LLM configuration is incomplete: {errors}",
    "jobs_empty": "No jobs found.",
    "jobs_header": "Recent jobs:",
    "job_not_found": "Job not found: {job_id}",
    "log_path": "Log path for job={job_id}: {path}",
    "log_missing": "The log file has not been created yet. The job may have just started; try logs or follow again later.",
    "log_empty": "The log file is currently empty.",
    "follow_start": "Following logs for job={job_id}: {path}\nPress Ctrl+C to leave log-follow mode only; it will not stop the benchmark or exit the Agent.",
    "follow_stopped": "Stopped log-follow mode. The benchmark continues if it is still running. Paste any log snippet at User> for analysis. Log path: {path}",
    "follow_done": "Log follow finished; job status: {status}",
    "unknown": "ADK did not return displayable text. You can continue describing the benchmark goal, or type doctor/status/jobs for deterministic state.",
    "adk_runtime_error": "The underlying model call failed temporarily. Internal errors are hidden. This natural-language request was not completed; retry, or type doctor/status/jobs for deterministic state.",
    "llm_billing_error": "The underlying model provider reported insufficient balance or quota. Natural-language Agent capability is unavailable until the model account is funded or quota is restored. Deterministic commands such as doctor/status/jobs still work.",
    "thinking": "[thinking] Processing the current request. If this takes more than 15 seconds, press Ctrl+C to cancel this turn without exiting the Agent.",
    "turn_cancelled": "Cancelled the current turn. The Agent session is still active; you can continue.",
    "pasted_evidence_detected": "Detected pasted logs, transcript, or error evidence. Treating it as evidence; it will not be written directly into benchmark configuration.",
    "pasted_evidence_buffered": "Buffered pasted log/transcript evidence. Continue pasting, or type the question you want me to analyze.",
    "pending_answer_blocked": "This reply did not pass validation for the current question: {blockers}",
    "pending_answer_recorded": "Recorded that reply. You can continue with the next goal.",
    "unbound_structural_answer": "There is no active question to confirm, so this short reply cannot be bound to a decision. State the goal directly, for example: benchmark solana fake-node quick, check job status, or analyze the latest report.",
    "framework_context_loaded": "Loaded framework facts: {chains} chains, {families} adapter families, {methods} RPC methods, fake-node fixtures={fixtures}.",
    "ctrl_c_exit": "Received Ctrl+C; exiting AnyChain Benchmark Agent.",
    "adk_missing_hint": "Note: google-adk is not available. Allow the Agent to install the isolated runtime, or run bash scripts/install_agent_deps.sh --yes first.",
}

"""Small language policy helpers for terminal UX."""

from __future__ import annotations

import re


_CJK_RE = re.compile(r"[\u3400-\u9fff]")


def detect_language(text: str, default: str = "en") -> str:
    """Return the preferred response language for a user turn."""
    if _CJK_RE.search(text or ""):
        return "zh"
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
    "follow_stopped": "已退出日志跟踪。benchmark 如果仍在运行会继续执行。你可以复制日志片段到 User> 让我分析。日志路径：{path}",
    "follow_done": "日志跟踪结束，job 状态：{status}",
    "unknown": "ADK 没有返回可显示内容。你可以继续描述测试目标，或输入 doctor/status/jobs 查看确定性状态。",
    "adk_runtime_error": "底层模型调用暂时失败，我不会展示内部错误。这个自然语言请求尚未完成；请重试，或输入 doctor/status/jobs 查看确定性状态。",
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
    "adk_runtime_error": "The underlying model call failed temporarily. I will not show internal errors. This natural-language request was not completed; retry, or type doctor/status/jobs for deterministic state.",
    "framework_context_loaded": "Loaded framework facts: {chains} chains, {families} adapter families, {methods} RPC methods, fake-node fixtures={fixtures}.",
    "ctrl_c_exit": "Received Ctrl+C; exiting AnyChain Benchmark Agent.",
    "adk_missing_hint": "Note: google-adk is not available. Allow the Agent to install the isolated runtime, or run bash scripts/install_agent_deps.sh --yes first.",
}

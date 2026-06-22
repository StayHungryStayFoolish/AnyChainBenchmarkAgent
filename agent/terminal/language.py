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
    """Translate a small fixed terminal message."""
    table = _ZH if language == "zh" else _EN
    template = table.get(key) or _EN.get(key) or key
    return template.format(**values)


_ZH = {
    "welcome": "AnyChain Benchmark Agent 已启动。",
    "mode": "当前模型配置：provider={provider}, model={model}, auth={auth_mode}",
    "adk": "ADK runtime：{status}",
    "job_found": "检测到最近 job：{job_id}，状态：{status}",
    "job_next_actions": "可选下一步：{actions}",
    "job_none": "没有检测到历史 job。",
    "prompt": "User> ",
    "agent": "Agent> {message}",
    "bye": "已退出 AnyChain Benchmark Agent。",
    "help": "你可以说：测试 solana、使用 fake-node、doctor、jobs、status、help、exit。",
    "doctor_start": "正在执行只读环境检查。",
    "doctor_summary": "检查完成：status={status}，缺失依赖={missing}，能力={chains} chains / {methods} RPC methods。",
    "dependency_offer": "检测到缺失依赖：{missing}。我可以在你确认后运行 scripts/install_deps.sh --yes。是否允许？[Y/n]",
    "dependency_declined": "已跳过依赖安装。后续真实 benchmark 可能仍会被 preflight 阻止。",
    "dependency_install_start": "开始安装 benchmark 依赖。这一步可能需要一些时间。",
    "dependency_install_done": "依赖安装命令完成，exit_code={exit_code}。",
    "llm_config_warning": "LLM 配置还不完整：{errors}",
    "jobs_empty": "没有找到 job。",
    "jobs_header": "最近 job：",
    "state_reset": "已重置当前 Agent workflow。你可以重新描述要测试的链、节点和目标。",
    "unknown": "我已经记录你的输入。下一步会由 benchmark workflow 判断需要确认的配置；当前可先输入 doctor 或“测试 solana”。",
    "benchmark_chain": "我理解你要测试 {chain}。是否确认这个链？[Y/n]",
    "select_target": "使用 fake-node 闭环测试，还是真实节点？[fake-node/real-node]",
    "fake_node_selected": "已选择 fake-node。这个模式不需要真实 LOCAL_RPC_URL，但仍需要确认 RPC mode、method、weight 和必要的 TARGET_* 样本。",
    "ask_rpc_mode": "请选择 RPC 模式：[single/mixed]",
    "confirm_single_method": "默认先使用 single workload。下一步需要确认具体 RPC method 和必要的 TARGET_* 样本。",
    "confirm_mixed_weights": "mixed workload 需要确认每个 RPC method 的权重，且总和必须等于 100。",
    "confirm_param_samples": "还需要确认所选 RPC method 的 TARGET_* 参数样本。使用链模板默认样本继续吗？[Y/n]",
    "gate_blocked": "现在还不能执行 smoke，仍缺少：{missing}",
    "gate_ready": "配置门禁已通过，可以进入 smoke 前确认。",
    "prepare_smoke_offer": "我将根据已确认配置生成 benchmark plan、执行 preflight，并写入本次 job 的 runtime.env 预览。是否继续？[Y/n]",
    "prepare_start": "正在生成 plan 并执行 preflight。",
    "prepare_ok": "preflight 通过。plan={plan_file}，runbook={runbook_file}。是否执行 mock smoke 生命周期验证？[Y/n]",
    "prepare_blocked": "preflight 被阻止：{blockers}。plan={plan_file}",
    "mock_smoke_start": "正在执行 mock smoke 生命周期验证，不会向节点产生真实流量。",
    "mock_smoke_done": "mock smoke 完成：job={job_id}，status={status}，runtime_env={runtime_env_file}，artifact_index={artifact_index}",
    "smoke_declined": "已暂停 smoke。你可以继续调整配置，或输入 status/jobs 查看状态。",
    "real_node_selected": "已选择真实节点。后续需要确认 LOCAL_RPC_URL、进程名、磁盘、网络和 workload。",
    "ask_required_value": "请提供或确认 {name}。",
    "recorded_value": "已记录 {name}。",
    "real_node_required_done": "真实节点必填运行变量已记录完毕。",
    "not_manual_export": "这个问题应该由 Agent 写入 runtime env 或子进程 env 处理，不应该要求你手动 export PATH。",
    "interrupted": "已中断当前输入。输入 exit 退出，或继续告诉我你的测试目标。",
    "adk_missing_hint": "注意：google-adk 当前不可用。请先运行 bash scripts/install_agent_deps.sh --yes；在此之前，Agent 只能使用有限的终端 workflow 和可用的直连 LLM provider。",
}


_EN = {
    "welcome": "AnyChain Benchmark Agent started.",
    "mode": "Model config: provider={provider}, model={model}, auth={auth_mode}",
    "adk": "ADK runtime: {status}",
    "job_found": "Found latest job: {job_id}, status: {status}",
    "job_next_actions": "Available next actions: {actions}",
    "job_none": "No previous job was found.",
    "prompt": "User> ",
    "agent": "Agent> {message}",
    "bye": "Exited AnyChain Benchmark Agent.",
    "help": "Try: benchmark solana, use fake-node, doctor, jobs, status, help, exit.",
    "doctor_start": "Running read-only environment diagnostics.",
    "doctor_summary": "Doctor complete: status={status}, missing dependencies={missing}, capabilities={chains} chains / {methods} RPC methods.",
    "dependency_offer": "Missing dependencies detected: {missing}. I can run scripts/install_deps.sh --yes after your confirmation. Allow this? [Y/n]",
    "dependency_declined": "Skipped dependency installation. A real benchmark may still be blocked by preflight.",
    "dependency_install_start": "Starting benchmark dependency installation. This may take a while.",
    "dependency_install_done": "Dependency installation command completed, exit_code={exit_code}.",
    "llm_config_warning": "LLM configuration is incomplete: {errors}",
    "jobs_empty": "No jobs found.",
    "jobs_header": "Recent jobs:",
    "state_reset": "Reset the current Agent workflow. You can describe the chain, node, and benchmark goal again.",
    "unknown": "I recorded your input. The benchmark workflow will decide the next required configuration; you can start with doctor or benchmark solana.",
    "benchmark_chain": "I understand you want to benchmark {chain}. Confirm this chain? [Y/n]",
    "select_target": "Use fake-node closed-loop testing or a real node? [fake-node/real-node]",
    "fake_node_selected": "Selected fake-node. This mode does not need a real LOCAL_RPC_URL, but still needs RPC mode, method, weight, and required TARGET_* samples.",
    "ask_rpc_mode": "Choose RPC mode: [single/mixed]",
    "confirm_single_method": "Defaulting to a single workload. Next we need to confirm the RPC method and required TARGET_* samples.",
    "confirm_mixed_weights": "A mixed workload requires per-method weights, and the total must equal 100.",
    "confirm_param_samples": "We still need to confirm TARGET_* parameter samples for the selected RPC method. Continue with chain-template defaults? [Y/n]",
    "gate_blocked": "Smoke cannot run yet. Missing: {missing}",
    "gate_ready": "Configuration gates passed; ready for smoke confirmation.",
    "prepare_smoke_offer": "I will generate a benchmark plan, run preflight, and write the job-local runtime.env preview from the confirmed configuration. Continue? [Y/n]",
    "prepare_start": "Generating plan and running preflight.",
    "prepare_ok": "Preflight passed. plan={plan_file}, runbook={runbook_file}. Run a mock smoke lifecycle verification now? [Y/n]",
    "prepare_blocked": "Preflight is blocked: {blockers}. plan={plan_file}",
    "mock_smoke_start": "Running mock smoke lifecycle verification. This does not send real node traffic.",
    "mock_smoke_done": "Mock smoke completed: job={job_id}, status={status}, runtime_env={runtime_env_file}, artifact_index={artifact_index}",
    "smoke_declined": "Smoke is paused. You can continue adjusting configuration, or type status/jobs.",
    "real_node_selected": "Selected real node. Next we need LOCAL_RPC_URL, process names, disks, network, and workload.",
    "ask_required_value": "Please provide or confirm {name}.",
    "recorded_value": "Recorded {name}.",
    "real_node_required_done": "Required real-node runtime variables have been recorded.",
    "not_manual_export": "The Agent should fix this through runtime env or child-process env, not ask you to manually export PATH.",
    "interrupted": "Interrupted the current input. Type exit to quit, or continue with your benchmark goal.",
    "adk_missing_hint": "Note: google-adk is not available. Run bash scripts/install_agent_deps.sh --yes first; until then, the Agent can only use limited terminal workflow and any available direct LLM provider.",
}

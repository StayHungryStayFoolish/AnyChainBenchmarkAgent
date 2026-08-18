"""Evidence/report analysis response definitions."""

MESSAGES = {
    "analysis.model_document": {
        "en": "",
        "zh": "",
        "arguments": {},
        "kinds": {"message", "evidence"},
        "payload_kind": "analysis_document",
    },
    "analysis.response.collection_started": {
        "en": "Started collecting multi-line RPC evidence. Continue pasting request/response/docs; type `END` when done. I will parse only complete evidence and will not change protocol from a partial line.",
        "zh": "已开始接收多行 RPC 证据。请继续粘贴 request/response/docs；完成后输入 `END`。我会在完整证据后再解析，不会用半截内容改变协议。",
        "arguments": {},
        "kinds": {"message"},
    },
    "analysis.response.line_recorded": {
        "en": "Recorded evidence line {count}. Continue pasting, or type `END` when done.",
        "zh": "已记录第 {count} 行证据。请继续粘贴，完成后输入 `END`。",
        "arguments": {"count": "integer"},
        "kinds": {"status"},
    },
    "analysis.response.collection_empty": {
        "en": "No parseable multi-line evidence was collected. Provide request/response/docs again, or enter params JSON directly.",
        "zh": "没有可解析的多行证据。请重新提供 request/response/docs，或直接输入 params JSON。",
        "arguments": {},
        "kinds": {"warning"},
    },
    "analysis.response.logs_saved": {
        "en": "Saved {count} lines of error/log evidence. You can ask me to analyze the cause or generate a fix/retry plan from it.",
        "zh": "已保存 {count} 行错误/日志证据。你可以继续问我分析原因，或让我基于这些证据生成修复/重试建议。",
        "arguments": {"count": "integer"},
        "kinds": {"evidence"},
    },
    "analysis.response.analysis_hint": {
        "en": "I will analyze this as error/log evidence. If you want to continue benchmark configuration instead, name the configuration area.",
        "zh": "这类内容会作为错误/日志证据分析；如果你希望继续配置 benchmark，也可以直接说明要回到哪个配置项。",
        "arguments": {},
        "kinds": {"message"},
    },
    "analysis.response.collection_resumed": {
        "en": "Resumed evidence collection with {count} saved line(s). Continue pasting, or type `END` when done.",
        "zh": "已恢复证据收集，当前保留 {count} 行。请继续粘贴，完成后输入 `END`。",
        "arguments": {"count": "integer"},
        "kinds": {"status"},
    },
    "analysis.response.evidence_help": {
        "en": "Paste real logs, a stack trace, or command output; multiline content is collected as one evidence block and ends with `END`. You may also name a `job_id` so I can read existing logs and artifacts.",
        "zh": "请粘贴真实日志、错误栈或命令输出；多行内容会作为一个证据块接收，完成后输入 `END`。也可以指定 `job_id`，让我读取已有日志和产物。",
        "arguments": {},
        "kinds": {"message"},
    },
    "analysis.response.no_job": {
        "en": "No historical job was found. Run fake-node smoke, real-node benchmark, or sync-observe before analyzing a report.",
        "zh": "没有找到历史 job。请先运行 fake-node smoke、real-node benchmark 或 sync-observe，再分析报告。",
        "arguments": {},
        "kinds": {"warning"},
    },
    "analysis.response.job_read_failed": {
        "en": "Could not read job `{job_id}`: {error_type}. Use `jobs` to list available jobs, or provide the correct job_id.",
        "zh": "无法读取 job `{job_id}`：{error_type}。请用 `jobs` 查看可用任务，或提供正确的 job_id。",
        "arguments": {"job_id": "string", "error_type": "string"},
        "kinds": {"warning"},
    },
    "analysis.response.waiting_logs": {
        "en": "No new log content was received. Paste the actual log/stack trace and type `END` when done; to leave log analysis, name the configuration area to return to.",
        "zh": "还没有收到新的日志内容。请继续粘贴真实日志/错误栈，完成后输入 `END`；如果要退出日志分析，可以直接说明要回到哪个配置项。",
        "arguments": {},
        "kinds": {"message"},
    },
    "analysis.response.waiting_rpc": {
        "en": "No new RPC evidence was received. Continue pasting request/response/docs and type `END` when done; to leave this flow, name the configuration area to return to.",
        "zh": "还没有收到新的 RPC 证据。请继续粘贴 request/response/docs，完成后输入 `END`；如果要退出，请直接说明要回到哪个配置项。",
        "arguments": {},
        "kinds": {"message"},
    },
    "analysis.failure.collection_state_required": {
        "en": "Evidence operation `{operation}` requires collection state `{required_state}`.",
        "zh": "证据操作 `{operation}` 需要收集状态 `{required_state}`。",
        "arguments": {"operation": "string", "required_state": "string"},
        "kinds": {"error"},
    },
    "analysis.failure.unsupported_action": {
        "en": "The analysis owner does not support action `{action_type}`.",
        "zh": "分析 owner 不支持动作 `{action_type}`。",
        "arguments": {"action_type": "string"},
        "kinds": {"error"},
    },
}

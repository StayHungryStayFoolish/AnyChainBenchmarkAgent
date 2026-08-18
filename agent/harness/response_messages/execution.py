"""Execution response definitions."""

MESSAGES = {
    "execution.response.preflight_paused": {
        "en": "Preflight/smoke is paused. You can change chain, RPC, QPS, disk, or observability and approve execution when ready.",
        "zh": "已暂停 preflight/smoke。可以继续修改链、RPC、QPS、磁盘或可观测性；准备好后再批准执行。",
        "arguments": {},
        "kinds": {"message"},
    },
    "execution.response.final_paused": {
        "en": "The final benchmark is paused. The isolated-smoke result and current configuration are preserved; change the configuration or explicitly request final submission later.",
        "zh": "已暂停正式 benchmark。隔离 smoke 的结果和当前配置仍保留；你可以修改配置，或稍后明确要求提交正式 benchmark。",
        "arguments": {},
        "kinds": {"message"},
    },
    "execution.response.sync_observe_submitted": {
        "en": "Sync-observe job submitted: status={status}, job_id={job_id}. Use: {commands}",
        "zh": "sync-observe 任务已提交：status={status}，job_id={job_id}。可用命令：{commands}",
        "arguments": {
            "status": "string",
            "job_id": "string",
            "commands": "string",
        },
        "kinds": {"status"},
    },
    "execution.response.fake_node_smoke_submitted": {
        "en": "Fake-node smoke submitted: status={status}, job_id={job_id}. Use: {commands}",
        "zh": "fake-node smoke 已提交：status={status}，job_id={job_id}。可用命令：{commands}",
        "arguments": {"status": "string", "job_id": "string", "commands": "string"},
        "kinds": {"status"},
    },
    "execution.response.real_node_smoke_submitted": {
        "en": "Real-node isolated smoke submitted: status={status}, job_id={job_id}. Use: {commands}",
        "zh": "real-node 隔离 smoke 已提交：status={status}，job_id={job_id}。可用命令：{commands}",
        "arguments": {"status": "string", "job_id": "string", "commands": "string"},
        "kinds": {"status"},
    },
    "execution.response.final_benchmark_submitted": {
        "en": "Final real-node benchmark submitted: status={status}, job_id={job_id}. Use: {commands}",
        "zh": "正式 real-node benchmark 已提交：status={status}，job_id={job_id}。可用命令：{commands}",
        "arguments": {"status": "string", "job_id": "string", "commands": "string"},
        "kinds": {"status"},
    },
    "execution.response.final_not_submitted": {
        "en": "Final real-node benchmark was not submitted: {reason}",
        "zh": "未提交正式 real-node benchmark：{reason}",
        "arguments": {"reason": "string"},
        "kinds": {"warning"},
    },
    "execution.failure.unsupported_action": {
        "en": "The execution owner does not support action `{action_type}`.",
        "zh": "执行 owner 不支持动作 `{action_type}`。",
        "arguments": {"action_type": "string"},
        "kinds": {"error"},
    },
    "execution.failure.invalid_job_receipt": {
        "en": "The job-manager read receipt is invalid.",
        "zh": "job-manager 读取回执无效。",
        "arguments": {},
        "kinds": {"error"},
    },
    "execution.failure.real_smoke_required": {
        "en": "A completed isolated real-node smoke is required before the final benchmark.",
        "zh": "提交正式 benchmark 前必须先完成隔离的 real-node smoke。",
        "arguments": {},
        "kinds": {"error"},
    },
}

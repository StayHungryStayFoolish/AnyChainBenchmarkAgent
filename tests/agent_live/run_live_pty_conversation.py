#!/usr/bin/env python3
"""Run real AnyChain Agent terminal conversations through a PTY.

This is intentionally different from the ``--prompt`` live matrix. It starts
``./bin/anychain-agent`` as an interactive terminal program, waits for
``User>``, sends user-like messages, and captures the visible transcript.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import select
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SECRET_RE = re.compile(r"((?<![A-Za-z0-9])sk-[A-Za-z0-9_-]+|AIza[0-9A-Za-z_-]+)")
VISIBLE_RESPONSE_FORBIDDEN = [
    "底层模型调用暂时失败",
    "余额或配额不足",
    "insufficient balance",
    "insufficient quota",
    "Traceback (most recent call last):\n  File",
    "asyncio.exceptions",
    "CancelledError",
    "KeyboardInterrupt",
    "The user",
    "Let me",
    "I need",
    "I should",
    "I will",
    "调用工具",
    "子代理",
    "pending_question",
    "VISIBLE_RESPONSE",
    "job_status",
    "tail_job_log",
    "analyze_artifacts",
    "submit_benchmark_job",
    "prepare_benchmark_run",
]
VISIBLE_PROCESS_PATTERNS = [
    re.compile(r"Agent>\s*(我先|让我|让我先|我会先|现在让我|我来查|我来确认|我来检查)"),
    re.compile(r"Agent>\s*(Let me|I need|I should|I will|I'll first)", re.IGNORECASE),
]


def clean_agent_runtime_state() -> None:
    """Remove runtime state without deleting local task documents.

    Live PTY tests need clean sessions/jobs, but `.agent/task-docs` is the
    execution memory for this refactor and must survive test cleanup.
    """
    agent_root = REPO_ROOT / ".agent"
    if not agent_root.exists():
        return
    for name in ("jobs", "sessions", "terminal", "cache", "tmp"):
        shutil.rmtree(agent_root / name, ignore_errors=True)
    for path in agent_root.glob("*.json"):
        path.unlink(missing_ok=True)
    for path in agent_root.glob("*.log"):
        path.unlink(missing_ok=True)


def load_agent_config_env(env: dict[str, str]) -> dict[str, str]:
    """Load configured LLM env from config/agent_config.sh for live tests.

    The product CLI sources this file through its normal startup path. The live
    PTY harness must mirror that behavior so tests do not silently run with a
    different provider/key state than users.
    """
    config = REPO_ROOT / "config" / "agent_config.sh"
    if not config.is_file():
        return env
    keys = {
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_AUTH_MODE",
        "OPENAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
        "GOOGLE_GENAI_USE_VERTEXAI",
    }
    command = (
        "set -a; "
        f"source {str(config)!r} >/dev/null 2>&1; "
        "python3 - <<'PY'\n"
        "import os, json\n"
        f"keys = {sorted(keys)!r}\n"
        "print(json.dumps({k: os.environ.get(k, '') for k in keys}))\n"
        "PY"
    )
    try:
        completed = subprocess.run(
            ["bash", "-lc", command],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        loaded = json.loads(completed.stdout or "{}")
    except Exception:
        return env
    merged = dict(env)
    for key, value in loaded.items():
        if value and not merged.get(key):
            merged[key] = str(value)
    return merged


def smoke_llm_provider(env: dict[str, str], timeout: int) -> dict[str, object]:
    """Verify the configured live model can answer before PTY acceptance.

    A quota/auth/model failure means the live PTY matrix cannot exercise ADK
    intent recognition. Report that as provider-blocked instead of spending
    scenario time and mixing provider failures with Agent behavior failures.
    """
    command = (
        "PYTHONPATH=agent "
        f"{str(REPO_ROOT / '.venv-adk' / 'bin' / 'python')!r} - <<'PY'\n"
        "from llm.config import load_llm_config\n"
        "from llm.providers import provider_from_config\n"
        "from llm.types import LLMMessage, LLMRequest\n"
        "cfg = load_llm_config()\n"
        "print(f'provider={cfg.provider} model={cfg.model} auth={cfg.auth_mode}')\n"
        "try:\n"
        "    resp = provider_from_config(cfg).complete(LLMRequest(\n"
        "        messages=[LLMMessage(role='user', content='Reply with OK only.')],\n"
        "        temperature=0,\n"
        "        max_tokens=8,\n"
        "    ))\n"
        "    print('response=' + (resp.text or '').strip()[:40])\n"
        "except Exception as exc:\n"
        "    print(type(exc).__name__ + ': ' + str(exc)[:500])\n"
        "    raise SystemExit(3)\n"
        "PY"
    )
    try:
        completed = subprocess.run(
            ["bash", "-lc", command],
            cwd=REPO_ROOT,
            env=env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=max(20, min(timeout, 120)),
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "issue": "provider smoke timed out"}
    output = SECRET_RE.sub(r"\1***REDACTED***", completed.stdout or "")
    if completed.returncode != 0:
        return {"ok": False, "issue": "provider smoke failed", "output": output.strip()[-1000:]}
    return {"ok": True, "output": output.strip()[-1000:]}


def write_provider_blocked_summary(log_dir: Path, issue: str, output: str = "") -> None:
    result = {
        "name": "provider_smoke",
        "passed": False,
        "provider_blocked": True,
        "issues": [issue],
        "provider_output": output,
        "turns": [],
        "job_ids": [],
    }
    (log_dir / "summary.json").write_text(json.dumps([result], indent=2, sort_keys=True), encoding="utf-8")
    (log_dir / "transcripts.md").write_text(
        "# AnyChain Agent Live PTY Transcripts\n\n"
        "## provider_smoke\n\n"
        "Status: PROVIDER_BLOCKED\n\n"
        f"Issue: {issue}\n\n"
        "```text\n"
        f"{output}\n"
        "```\n",
        encoding="utf-8",
    )


SCENARIOS = [
    {
        "name": "zh_greeting_opening_no_benchmark_state",
        "turns": [
            "Hi",
            "1",
        ],
        "must_include_any": ["AnyChain", "Agent", "fake-node", "real-node", "历史", "帮助"],
        "must_include_all": ["fake-node", "Which chain do you want to benchmark?"],
        "must_not_include": [
            "你想怎么测试？",
            "target_mode",
            "pending_question",
            "Confirm CLOUD_REGION",
            "Confirm CLOUD_ZONE",
            "Review the solana",
            "solana single RPC workload",
        ],
    },
    {
        "name": "zh_language_switch_clarification_does_not_repeat_english_menu",
        "turns": [
            "Hi",
            "你好",
            "我没看懂你在说什么",
            "为什么是中英文混合呢",
        ],
        "must_include_all": ["AnyChain", "fake-node", "real-node"],
        "must_include_any": ["没关系", "抱歉", "简单解释", "中文"],
        "must_not_include": ["pending_question", "target_mode"],
        "agent_visible_max_count": {
            "Start a fake-node benchmark": 1,
            "No previous job is available": 1,
            "Reply `1`, `2`, or `3`, or type your goal directly.": 1,
        },
    },
    {
        "name": "zh_completed_job_then_new_fake_node_setup",
        "turns": [
            "Hi",
            "我需要查看上一次的日志和分析结果",
            "fake-node 和 real-node 有什么区别",
            "fake-node 测试",
            "2",
        ],
        "must_include_any": ["fake-node", "real-node", "链", "solana", "ethereum"],
        "must_not_include": ["当前没有待确认的问题", "这个短回复无法绑定", "target_mode", "pending_question"],
    },
    {
        "name": "en_partial_chain_token_does_not_enter_config_flow",
        "turns": [
            "Hi",
            "1",
            "sola",
        ],
        "must_include_any": ["chain identity", "not one of", "supported chain", "new chain", "adapter"],
        "must_not_include": [
            "Choose the ledger/data disk",
            "Confirm DATA_VOL_TYPE",
            "Detected DATA_VOL_SIZE",
            "pending_question",
            "target_mode",
        ],
        "state_must_equal": {
            "chain": "",
            "target_mode": "fake-node",
            "pending_question.id": "chain_identity_resolution",
            "chain_status": "identity_needs_confirmation",
            "chain_identity_candidate.normalized": "sola",
        },
        "state_must_not_equal": {
            "chain": "sola",
            "confirmed_config.BLOCKCHAIN_NODE": "sola",
        },
    },
    {
        "name": "en_alias_prefix_chain_name_does_not_collapse_to_existing_chain",
        "turns": [
            "Hi",
            "1",
            "bnb greenfield",
            "1",
            "1",
        ],
        "must_include_any": ["greenfield", "new", "unsupported", "not one of", "endpoint", "family", "BNB Greenfield", "needs_review"],
        "must_not_include": [
            "Choose the ledger/data disk",
            "Confirm DATA_VOL_TYPE",
            "Detected DATA_VOL_SIZE",
            "pending_question",
            "target_mode",
        ],
        "state_must_equal": {
            "chain": "bnb-greenfield",
            "target_mode": "fake-node",
        },
        "state_must_be_one_of": {
            "chain_status": ["unsupported_needs_endpoint_validation", "unsupported_needs_development_handoff"],
            "fixture_status.status": ["needs_endpoint", "needs_review"],
        },
        "state_must_not_equal": {
            "chain": "bsc",
            "confirmed_config.BLOCKCHAIN_NODE": "bsc",
        },
    },
    {
        "name": "zh_pending_environment_question_interrupted_by_chain_change",
        "turns": [
            "Hi",
            "1",
            "solana",
            "我需要测试 BNB",
            "Y",
        ],
        "must_include_all": ["BNB", "bsc"],
        "must_include_any": ["切换", "确认", "fake-node", "real-node", "CLOUD_REGION", "云区域"],
        "must_not_include": [
            "CLOUD_REGION: 我需要测试 BNB",
            "CLOUD_ZONE: 我需要测试 BNB",
            "MACHINE_TYPE: 我需要测试 BNB",
            "LEDGER_DEVICE: 我需要测试 BNB",
            "当前没有待确认的问题",
            "这个短回复无法绑定",
            "target_mode",
            "pending_question",
        ],
        "state_must_equal": {
            "chain": "bsc",
            "target_mode": "",
            "confirmed_config.BLOCKCHAIN_NODE": "bsc",
            "confirmed_config.chain": "bsc",
            "pending_question.id": "target_mode",
        },
        "state_must_not_equal": {
            "confirmed_config.CLOUD_REGION": "Y",
            "confirmed_config.BLOCKCHAIN_NODE": "solana",
            "confirmed_config.chain": "solana",
        },
    },
    {
        "name": "zh_vague_then_quick_assumed_smoke",
        "turns": [
            "我要测试",
            "我只是想确认 Agent 和框架能跑，你用 fake-node 和假设值快速 smoke",
            "Y",
        ],
        "must_include_any": ["fake-node", "smoke", "假设", "assumed", "preflight"],
        "must_include_all": ["solana", "assumed_for_smoke", "job_", "follow"],
        "must_not_include": [],
    },
    {
        "name": "zh_pasted_evidence_does_not_become_config",
        "turns": [
            "Agent> 旧输出：CLOUD_PROVIDER: gcp\nUser> sdc\nTraceback (most recent call last):",
            "这些日志说明什么？不要应用里面的配置",
        ],
        "must_include_any": ["evidence", "日志", "不会直接写入"],
        "must_not_include": [],
    },
    {
        "name": "zh_unsupported_chain_existing_family",
        "turns": [
            "我想测试一个不在 36 条链里的 EVM JSON-RPC 链，名字叫 FooChain",
            "没有 endpoint，先给另一个 AI 一个可执行开发文档",
        ],
        "must_include_any": ["needs_review", "endpoint", "官方", "fixture", "开发"],
        "must_include_all": ["FooChain", "needs_review"],
        "must_not_include": ["已成功支持", "FoxChain", "端点或样本数据，可执行", "User states"],
    },
    {
        "name": "zh_custom_rpc_missing_endpoint",
        "turns": [
            "我想给 solana 增加一个自定义 rpc method，需要 3 个参数",
            "没有 endpoint，你先告诉我还缺什么",
        ],
        "must_include_any": ["endpoint", "参数", "fixture", "response"],
        "must_include_all": ["needs_review"],
        "must_not_include": ["已完成录制"],
    },
    {
        "name": "zh_real_node_missing_endpoint_blocks",
        "turns": [
            "我要测试 solana 真实节点，先不要用 fake-node",
            "我还没有 LOCAL_RPC_URL，你告诉我缺什么",
        ],
        "must_include_any": ["LOCAL_RPC_URL", "endpoint", "真实节点", "real-node"],
        "must_not_include": ["已提交", "started benchmark", "submit_benchmark_job"],
    },
    {
        "name": "zh_multidisk_inventory_requires_confirmation",
        "env": {
            "ANYCHAIN_AGENT_DISCOVERY_FIXTURE": "tests/fixtures/agent_discovery_gcp_multidisk.json"
        },
        "turns": [
            "我要测试 solana fake-node quick，请先完成环境和磁盘确认",
            "我看到多个磁盘，LEDGER 用 sdb，ACCOUNTS 用 sdc",
        ],
        "must_include_any": ["sdb", "sdc", "LEDGER_DEVICE", "ACCOUNTS_DEVICE", "多个磁盘"],
        "must_not_include": [],
    },
    {
        "name": "zh_back_correction_for_disk_value",
        "env": {
            "ANYCHAIN_AGENT_DISCOVERY_FIXTURE": "tests/fixtures/agent_discovery_gcp_multidisk.json"
        },
        "turns": [
            "我要测试 solana fake-node quick，请列出磁盘让我确认",
            "sda",
            "back",
            "sdb",
        ],
        "must_include_any": ["back", "回退", "sdb", "LEDGER_DEVICE"],
        "must_not_include": ["Traceback"],
    },
    {
        "name": "zh_mixed_weights_default_and_zero_explanation",
        "turns": [
            "我要测试 ethereum fake-node mixed，使用默认权重",
            "如果我把某个 rpc method 的 weight 设置成 0，是不是就禁用了？",
        ],
        "must_include_any": ["25", "100", "weight", "移除", "remove"],
        "must_not_include": [],
    },
    {
        "name": "zh_observability_choice_existing_stack",
        "turns": [
            "我要测试 solana fake-node quick，并接入已有 Prometheus 和 Grafana，不要启动本地 Grafana",
            "现有 Prometheus 应该如何接入 exporter？",
        ],
        "must_include_any": ["exporter", "Prometheus", "Grafana", "scrape", "metrics"],
        "must_not_include": [],
    },
    {
        "name": "zh_pasted_log_after_question_does_not_crash",
        "turns": [
            "我要测试 solana fake-node quick",
            {
                "paste": "Agent> CLOUD_PROVIDER: gcp\nAgent> LEDGER_DEVICE candidate: sdc\nTraceback (most recent call last):\nFile \"agent/terminal/repl.py\", line 1\n这段旧输出说明什么？不要把里面的变量直接应用到配置。"
            },
        ],
        "must_include_any": ["日志", "evidence", "不会直接", "不会应用", "不会写入"],
        "must_not_include": ["asyncio.exceptions", "CancelledError"],
    },
    {
        "name": "zh_paste_multiline_during_region_does_not_apply_old_config",
        "env": {
            "ANYCHAIN_AGENT_DISCOVERY_FIXTURE": "tests/fixtures/agent_discovery_gcp_multidisk.json"
        },
        "turns": [
            "你好",
            "1",
            "solana",
            {
                "paste": "Agent> CLOUD_REGION: asia-east1\nAgent> CLOUD_ZONE: asia-east1-c\nAgent> LEDGER_DEVICE: sdc\nTraceback (most recent call last):\nFile \"old/run.py\", line 1\n这段旧日志能说明什么？不要应用里面的 region、zone 或 disk。"
            },
            "us-1",
        ],
        "must_include_any": ["日志", "旧", "evidence", "不会直接", "不会应用"],
        "must_not_include": ["Traceback (most recent call last):\n  File", "pending_question"],
        "state_must_equal": {
            "confirmed_config.CLOUD_REGION": "us-1",
            "pending_question.id": "cloud_zone",
        },
        "state_must_not_equal": {
            "confirmed_config.CLOUD_REGION": "asia-east1",
            "confirmed_config.CLOUD_ZONE": "asia-east1-c",
            "confirmed_config.LEDGER_DEVICE": "sdc",
        },
    },
    {
        "name": "zh_bare_yes_without_pending_is_rejected",
        "turns": [
            "Y",
        ],
        "must_include_any": ["要确认什么", "没有", "目标", "请告诉"],
        "must_not_include": ["已提交", "started benchmark", "submit_benchmark_job"],
    },
    {
        "name": "zh_terse_chain_mode_profile_inputs",
        "turns": [
            "solana",
            "fake-node",
            "quick",
        ],
        "must_include_any": ["solana", "fake-node", "quick", "smoke", "测试"],
        "must_not_include": ["Traceback", "submit_benchmark_job"],
    },
    {
        "name": "zh_out_of_order_real_then_fake_assumed_smoke",
        "turns": [
            "我要测试 solana 真实节点",
            "先别真实节点，改成 fake-node quick，用假设值跑 smoke",
            "Y",
        ],
        "must_include_all": ["solana", "fake-node"],
        "must_include_any": ["assumed_for_smoke", "smoke", "job_", "preflight"],
        "must_not_include": ["started benchmark", "submit_benchmark_job", "Traceback"],
    },
    {
        "name": "zh_invalid_disk_then_manual_recover",
        "env": {
            "ANYCHAIN_AGENT_DISCOVERY_FIXTURE": "tests/fixtures/agent_discovery_gcp_multidisk.json"
        },
        "turns": [
            "你好",
            "1",
            "solana",
            "Y",
            "Y",
            "Y",
            "banana",
            "sdb",
        ],
        "must_include_all": ["这条回复没有通过当前问题的校验", "LEDGER_DEVICE", "Ledger/data 磁盘类型"],
        "must_include_any": ["banana", "sdb", "磁盘", "设备"],
        "must_not_include": ["Traceback", "当前没有待处理的确认问题", "未识别 \"banana\""],
        "state_must_equal": {
            "confirmed_config.LEDGER_DEVICE": "sdb",
            "pending_question.id": "data_vol_type",
        },
    },
    {
        "name": "zh_detected_disk_size_is_confirmed_not_retyped",
        "env": {
            "ANYCHAIN_AGENT_DISCOVERY_FIXTURE": "tests/fixtures/agent_discovery_gcp_multidisk.json"
        },
        "turns": [
            "你好",
            "1",
            "solana",
            "Y",
            "Y",
            "Y",
            "sdb",
            "hyperdisk-extreme",
        ],
        "must_include_all": ["检测到 DATA_VOL_SIZE 为 `2048`"],
        "must_include_any": ["回复 `Y` 使用该值", "直接输入正确值"],
        "must_not_include": ["请输入 Ledger/data 磁盘容量"],
        "state_must_equal": {
            "confirmed_config.LEDGER_DEVICE": "sdb",
            "pending_question.id": "data_vol_size",
            "pending_question.current_value": "2048",
        },
    },
    {
        "name": "zh_detected_network_interface_accepts_yes_confirmation",
        "env": {
            "ANYCHAIN_AGENT_DISCOVERY_FIXTURE": "tests/fixtures/agent_discovery_gcp_multidisk.json"
        },
        "turns": [
            "你好",
            "1",
            "solana",
            "Y",
            "Y",
            "Y",
            "sdb",
            "hyperdisk-extreme",
            "Y",
            "30000",
            "1000",
            "N",
            "Y",
            "Y",
            "Y",
            "Y",
        ],
        "must_include_all": ["检测到 NETWORK_INTERFACE 为 `ens4`", "请输入网络带宽上限"],
        "must_not_include": ["yes requires pending_question.default_option", "Traceback"],
        "state_must_equal": {
            "confirmed_config.NETWORK_INTERFACE": "ens4",
            "pending_question.id": "network_max_bandwidth_gbps",
        },
    },
    {
        "name": "en_scalar_punctuation_and_qps_adjustment_flow",
        "env": {
            "ANYCHAIN_AGENT_DISCOVERY_FIXTURE": "tests/fixtures/agent_discovery_gcp_multidisk.json"
        },
        "turns": [
            "Hi",
            "1",
            "solana",
            "Y",
            "Y",
            "Y",
            "3",
            "hyperdisk-balanced,",
            "Y",
            "3000",
            "1000",
            "N",
            "Y",
            "100",
            "single",
            "1",
            "1",
            "N",
            "2",
            "25",
        ],
        "must_include_all": [
            "Use the selected mode's default QPS profile",
            "Choose which quick QPS parameter to adjust",
            "Enter the value for QUICK_MAX_QPS",
        ],
        "must_not_include": [
            "是否使用所选模式",
            "Provide required value",
            "This reply did not pass validation",
            "pending_question",
            "Traceback",
        ],
        "state_must_equal": {
            "confirmed_config.DATA_VOL_TYPE": "hyperdisk-balanced",
            "confirmed_config.QUICK_MAX_QPS": "25",
            "confirmed_config.qps_profile_confirmed": True,
        },
    },
    {
        "name": "en_accounts_prompts_are_specific_after_punctuation_input",
        "env": {
            "ANYCHAIN_AGENT_DISCOVERY_FIXTURE": "tests/fixtures/agent_discovery_gcp_multidisk.json"
        },
        "turns": [
            "Hi",
            "1",
            "solana",
            "Y",
            "Y",
            "Y",
            "sdb",
            "hyperdisk-balanced,",
            "Y",
            "3000",
            "1000",
            "Y",
            "sdc",
            "hyperdisk-balanced,",
            "Y",
        ],
        "must_include_all": [
            "Confirm ACCOUNTS_VOL_MAX_IOPS",
        ],
        "must_not_include": [
            "Provide required value: accounts_vol_max_iops",
            "This reply did not pass validation",
            "Traceback",
        ],
        "state_must_equal": {
            "confirmed_config.DATA_VOL_TYPE": "hyperdisk-balanced",
            "confirmed_config.ACCOUNTS_VOL_TYPE": "hyperdisk-balanced",
            "pending_question.id": "accounts_vol_max_iops",
        },
    },
    {
        "name": "en_numbered_choices_accept_trailing_punctuation",
        "env": {
            "ANYCHAIN_AGENT_DISCOVERY_FIXTURE": "tests/fixtures/agent_discovery_gcp_multidisk.json"
        },
        "turns": [
            "Hi",
            "1,",
            "solana",
            "Y",
            "Y",
            "Y",
            "1,",
            "hyperdisk-balanced",
            "Y",
            "3000",
            "1000",
            "N",
            "Y",
            "100",
            "single",
            "1,",
            "1,",
        ],
        "must_include_all": [
            "Use the selected mode's default QPS profile",
        ],
        "must_not_include": [
            "This reply did not pass validation",
            "answer does not match pending_question options",
            "pending_question",
            "Traceback",
        ],
        "state_must_equal": {
            "target_mode": "fake-node",
            "confirmed_config.LEDGER_DEVICE": "sda",
            "benchmark_profile.mode": "quick",
            "pending_question.id": "benchmark_profile_confirm",
        },
    },
    {
        "name": "zh_info_question_during_pending_config_returns_to_same_flow",
        "env": {
            "ANYCHAIN_AGENT_DISCOVERY_FIXTURE": "tests/fixtures/agent_discovery_gcp_multidisk.json"
        },
        "turns": [
            "你好",
            "1",
            "solana",
            "Y",
            "Y",
            "Y",
            "sdb",
            "hyperdisk-extreme",
            "Y",
            "fake-node 和 real-node 有什么区别",
            "30000",
        ],
        "must_include_any": ["fake-node", "real-node", "真实节点", "模拟节点"],
        "must_include_all": ["最大 IOPS"],
        "must_not_include": [
            "当前没有待确认的问题",
            "这个短回复无法绑定",
            "Traceback",
        ],
        "state_must_equal": {
            "confirmed_config.DATA_VOL_MAX_IOPS": "30000",
            "pending_question.id": "data_vol_max_throughput",
        },
    },
    {
        "name": "zh_chain_change_during_disk_choice_routes_to_chain_flow",
        "env": {
            "ANYCHAIN_AGENT_DISCOVERY_FIXTURE": "tests/fixtures/agent_discovery_gcp_multidisk.json"
        },
        "turns": [
            "Hi",
            "1",
            "solana",
            "我改成 BNB，继续 fake-node",
            "Y",
        ],
        "must_include_all": ["BNB", "bsc"],
        "must_include_any": ["切换", "确认", "fake-node", "RPC", "method", "workload", "Ledger/data"],
        "must_not_include": [
            "LEDGER_DEVICE: 我改成 BNB",
            "当前没有待确认的问题",
            "这个短回复无法绑定",
            "pending_question",
            "Traceback",
        ],
        "state_must_equal": {
            "chain": "bsc",
            "target_mode": "fake-node",
            "confirmed_config.BLOCKCHAIN_NODE": "bsc",
        },
        "state_must_not_equal": {
            "confirmed_config.BLOCKCHAIN_NODE": "solana",
            "confirmed_config.LEDGER_DEVICE": "我改成 BNB，继续 fake-node",
        },
    },
    {
        "name": "zh_chain_change_during_disk_type_routes_to_chain_flow",
        "env": {
            "ANYCHAIN_AGENT_DISCOVERY_FIXTURE": "tests/fixtures/agent_discovery_gcp_multidisk.json"
        },
        "turns": [
            "Hi",
            "1",
            "solana",
            "1",
            "我需要换成 eth，仍然 fake-node",
            "Y",
        ],
        "must_include_all": ["ethereum", "fake-node"],
        "must_include_any": ["切换", "确认", "RPC", "method", "workload", "Ledger/data"],
        "must_not_include": [
            "DATA_VOL_TYPE: 我需要换成 eth",
            "这个短回复无法绑定",
            "pending_question",
            "Traceback",
        ],
        "state_must_equal": {
            "chain": "ethereum",
            "target_mode": "fake-node",
            "confirmed_config.BLOCKCHAIN_NODE": "ethereum",
        },
        "state_must_not_equal": {
            "confirmed_config.BLOCKCHAIN_NODE": "solana",
            "confirmed_config.DATA_VOL_TYPE": "我需要换成 eth，仍然 fake-node",
        },
    },
    {
        "name": "zh_change_chain_during_qps_profile_revalidates_workload",
        "env": {
            "ANYCHAIN_AGENT_DISCOVERY_FIXTURE": "tests/fixtures/agent_discovery_gcp_multidisk.json"
        },
        "turns": [
            "Hi",
            "1",
            "solana",
            "1",
            "hyperdisk-balanced",
            "Y",
            "3000",
            "1000",
            "N",
            "Y",
            "Y",
            "Y",
            "Y",
            "100",
            "1",
            "我现在要换成 BNB，并重新确认 RPC method 和权重",
            "Y",
        ],
        "must_include_all": ["BNB", "bsc"],
        "must_include_any": ["RPC", "method", "权重", "workload", "mixed", "single"],
        "must_not_include": ["当前没有待确认的问题", "这个短回复无法绑定", "pending_question", "Traceback"],
        "state_must_equal": {
            "chain": "bsc",
            "confirmed_config.BLOCKCHAIN_NODE": "bsc",
        },
        "state_must_not_equal": {
            "confirmed_config.BLOCKCHAIN_NODE": "solana",
        },
    },
    {
        "name": "zh_unknown_chain_during_qps_enters_identity_then_protocol_gate",
        "env": {
            "ANYCHAIN_AGENT_DISCOVERY_FIXTURE": "tests/fixtures/agent_discovery_gcp_multidisk.json"
        },
        "turns": [
            "你好",
            "1",
            "solana",
            "Y",
            "Y",
            "Y",
            "3",
            "hyperdisk-balanced",
            "Y",
            "3000",
            "1000",
            "N",
            "Y",
            "100",
            "1",
            "我现在想换成 Flow 这个链，应该属于 EVM JSON-RPC",
            "1",
        ],
        "must_include_all": ["flow"],
        "must_include_any": ["协议", "family", "jsonrpc", "endpoint", "链身份"],
        "must_not_include": [
            "当前没有待确认的问题",
            "这个短回复无法绑定",
            "pending_question",
            "The user",
            "Traceback",
        ],
        "state_must_equal": {
            "chain_status": "protocol_needs_confirmation",
            "pending_question.id": "chain_protocol_resolution",
            "chain_protocol_candidate.chain": "flow",
        },
    },
    {
        "name": "en_partial_unknown_chain_during_region_uses_identity_gate_not_value",
        "env": {
            "ANYCHAIN_AGENT_DISCOVERY_FIXTURE": "tests/fixtures/agent_discovery_gcp_multidisk.json"
        },
        "turns": [
            "Hi",
            "1",
            "solana",
            "I want to switch to sola, check whether it is a real chain first",
        ],
        "must_include_all": ["sola"],
        "must_include_any": ["chain identity", "not one of", "Confirm", "new chain"],
        "must_not_include": [
            "CLOUD_REGION: I want to switch to sola",
            "Choose the ledger/data disk",
            "pending_question",
            "Traceback",
        ],
        "state_must_equal": {
            "chain": "",
            "chain_status": "identity_needs_confirmation",
            "pending_question.id": "chain_identity_resolution",
            "chain_identity_candidate.normalized": "sola",
        },
        "state_must_not_equal": {
            "confirmed_config.CLOUD_REGION": "I want to switch to sola, check whether it is a real chain first",
            "confirmed_config.BLOCKCHAIN_NODE": "solana",
        },
    },
    {
        "name": "zh_chain_change_during_chain_template_review_routes_to_chain_flow",
        "env": {
            "ANYCHAIN_AGENT_DISCOVERY_FIXTURE": "tests/fixtures/agent_discovery_gcp_multidisk.json"
        },
        "turns": [
            "Hi",
            "1",
            "solana",
            "1",
            "hyperdisk-balanced",
            "Y",
            "3000",
            "1000",
            "N",
            "Y",
            "Y",
            "Y",
            "Y",
            "100",
            "1",
            "Y",
            "我现在换成 BNB Greenfield，仍然用 fake-node",
            "后者，BNB Greenfield",
        ],
        "must_include_all": ["bnb-greenfield"],
        "must_include_any": ["endpoint", "RPC", "unsupported", "不在当前", "可访问", "needs_review", "开发交接"],
        "must_not_include": [
            "chain_template_reviewed: 我现在换成",
            "当前没有待确认的问题",
            "这个短回复无法绑定",
            "pending_question",
            "Traceback",
        ],
        "state_must_equal": {
            "chain": "bnb-greenfield",
            "target_mode": "fake-node",
        },
        "state_must_not_equal": {
            "chain": "bsc",
            "confirmed_config.BLOCKCHAIN_NODE": "solana",
        },
    },
    {
        "name": "zh_change_chain_and_target_mode_during_environment_config",
        "env": {
            "ANYCHAIN_AGENT_DISCOVERY_FIXTURE": "tests/fixtures/agent_discovery_gcp_multidisk.json"
        },
        "turns": [
            "Hi",
            "1",
            "solana",
            "我需要换成 eth，不使用 fake-node 模式",
            "Y",
        ],
        "must_include_all": ["ethereum", "real-node"],
        "must_include_any": ["LOCAL_RPC_URL", "真实节点", "RPC"],
        "must_not_include": [
            "Choose the target mode",
            "选择目标模式",
            "pending_question",
            "Traceback",
        ],
        "state_must_equal": {
            "chain": "ethereum",
            "target_mode": "real-node",
            "confirmed_config.BLOCKCHAIN_NODE": "ethereum",
        },
        "state_must_be_one_of": {
            "pending_question.id": ["local_rpc_url", "real_node_local_rpc_url"],
        },
    },
    {
        "name": "zh_chain_change_during_real_node_endpoint_routes_to_chain_flow",
        "env": {
            "ANYCHAIN_AGENT_DISCOVERY_FIXTURE": "tests/fixtures/agent_discovery_gcp_multidisk.json"
        },
        "turns": [
            "Hi",
            "2",
            "ethereum",
            "我改成 solana，并先用 fake-node 验证",
            "Y",
        ],
        "must_include_all": ["solana", "fake-node"],
        "must_include_any": ["切换", "确认", "Ledger/data", "RPC", "method"],
        "must_not_include": [
            "LOCAL_RPC_URL: 我改成 solana",
            "这个短回复无法绑定",
            "pending_question",
            "Traceback",
        ],
        "state_must_equal": {
            "chain": "solana",
            "target_mode": "fake-node",
            "confirmed_config.BLOCKCHAIN_NODE": "solana",
        },
        "state_must_not_equal": {
            "confirmed_config.BLOCKCHAIN_NODE": "ethereum",
        },
    },
    {
        "name": "zh_separate_accounts_disk_rejects_ledger_device",
        "env": {
            "ANYCHAIN_AGENT_DISCOVERY_FIXTURE": "tests/fixtures/agent_discovery_gcp_multidisk.json"
        },
        "turns": [
            "你好",
            "1",
            "solana",
            "Y",
            "Y",
            "Y",
            "3",
            "hyperdisk-extreme",
            "Y",
            "30000",
            "1000",
            "Y",
            "3",
        ],
        "must_include_all": ["独立 accounts/state 磁盘不能和 LEDGER_DEVICE 使用同一块设备"],
        "must_include_any": ["回退", "选择 N", "sdc"],
        "must_not_include": ["Traceback", "ACCOUNTS_DEVICE must be different"],
        "state_must_equal": {
            "confirmed_config.LEDGER_DEVICE": "sdb",
            "pending_question.id": "disk_accounts_choice",
        },
    },
    {
        "name": "zh_change_target_mode_after_environment_config_requires_endpoint",
        "turns": [
            "Hi",
            "1",
            "solana",
            "1",
            "hyperdisk-balanced",
            "Y",
            "3000",
            "1000",
            "N",
            "us-1",
            "us-1-z",
            "n2",
            "Y",
            "100",
            "我改主意了，不用 fake-node，切换到 real-node",
        ],
        "must_include_all": ["real-node"],
        "must_include_any": ["LOCAL_RPC_URL", "endpoint", "真实节点"],
        "must_not_include": ["当前没有待确认的问题", "这个短回复无法绑定", "pending_question", "Traceback"],
        "state_must_equal": {
            "target_mode": "real-node",
        },
        "state_must_not_equal": {
            "confirmed_config.BLOCKCHAIN_PROCESS_NAMES_STR": "fake-node",
            "confirmed_config.BLOCKCHAIN_PROCESS_NAMES": ["fake-node"],
            "confirmed_config.blockchain_process_names": ["fake-node"],
        },
        "state_must_be_one_of": {
            "pending_question.id": ["chain_selection", "local_rpc_url", "real_node_local_rpc_url"],
        },
    },
    {
        "name": "zh_compact_mixed_weight_request",
        "turns": [
            "ethereum mixed fake-node quick eth_blockNumber 70 eth_getBalance 30",
        ],
        "must_include_all": ["ethereum", "fake-node"],
        "must_include_any": ["70", "30", "100", "mixed", "权重"],
        "must_not_include": ["Traceback"],
    },
    {
        "name": "zh_compact_mixed_weight_invalid_total_blocks",
        "turns": [
            "ethereum fake-node mixed eth_blockNumber 70 eth_getBalance 20",
        ],
        "must_include_all": ["ethereum", "fake-node", "90", "100"],
        "must_include_any": ["权重", "weights", "mixed"],
        "must_not_include": [
            "请选择 Ledger/data 磁盘",
            "Choose the ledger/data disk",
            "Traceback",
        ],
        "state_must_equal": {
            "chain": "ethereum",
            "target_mode": "fake-node",
            "rpc_mode": "mixed",
            "mixed_weights.eth_blockNumber": 70,
            "mixed_weights.eth_getBalance": 20,
            "pending_question.id": "mixed_weights_confirm",
        },
    },
    {
        "name": "zh_compact_mixed_weight_invalid_then_corrects",
        "turns": [
            "ethereum fake-node mixed eth_blockNumber 70 eth_getBalance 20",
            "eth_blockNumber=70, eth_getBalance=30",
        ],
        "must_include_all": ["ethereum", "fake-node", "70", "30"],
        "must_include_any": ["CLOUD_REGION", "cloud region", "云区域"],
        "must_not_include": ["Traceback"],
        "state_must_equal": {
            "chain": "ethereum",
            "target_mode": "fake-node",
            "rpc_mode": "mixed",
            "mixed_weights.eth_blockNumber": 70,
            "mixed_weights.eth_getBalance": 30,
            "pending_question.id": "cloud_region",
        },
    },
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run real PTY AnyChain Agent conversations.")
    parser.add_argument("--log-dir", default="/tmp/anychain-agent-live-pty")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--scenario", action="append", help="Run only named scenario(s).")
    parser.add_argument(
        "--simulate-goal",
        choices=["fake-node-config-chaos"],
        help="Run a state-aware generated user simulation instead of fixed scenarios.",
    )
    parser.add_argument("--simulator-seed", type=int, default=7)
    parser.add_argument("--simulator-turns", type=int, default=36)
    parser.add_argument("--clean-state", action="store_true", help="Remove .agent state before running the matrix.")
    parser.add_argument("--clean-per-scenario", action="store_true", help="Remove .agent state before each selected scenario.")
    parser.add_argument(
        "--job-settle-timeout",
        type=int,
        default=180,
        help="Seconds to wait for jobs created by one scenario before reporting the scenario result.",
    )
    args = parser.parse_args(argv)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    if args.clean_state:
        clean_agent_runtime_state()
    env = os.environ.copy()
    env = load_agent_config_env(env)
    provider = args.provider or env.get("LLM_PROVIDER") or "deepseek"
    model = args.model or env.get("LLM_MODEL") or "deepseek-chat"
    env.update({
        "LLM_PROVIDER": provider,
        "LLM_MODEL": model,
        "LLM_AUTH_MODE": env.get("LLM_AUTH_MODE", "api_key"),
        "ANYCHAIN_AGENT_TURN_TIMEOUT_SECONDS": str(max(30, min(args.timeout, 120))),
        "LC_ALL": "C.UTF-8",
        "LANG": "C.UTF-8",
    })
    if provider == "deepseek" and not env.get("DEEPSEEK_API_KEY"):
        print("DEEPSEEK_API_KEY is not loaded in the environment/config for live PTY testing", file=sys.stderr)
        return 2
    smoke = smoke_llm_provider(env, args.timeout)
    if not smoke.get("ok"):
        issue = str(smoke.get("issue") or "provider smoke failed")
        output = str(smoke.get("output") or "")
        write_provider_blocked_summary(log_dir, issue, output)
        print(f"PROVIDER_BLOCKED provider={provider} model={model}: {issue}", file=sys.stderr)
        if output:
            print(output, file=sys.stderr)
        return 2

    if args.simulate_goal:
        if args.clean_state:
            clean_agent_runtime_state()
        result = run_generated_simulation(
            goal=args.simulate_goal,
            log_dir=log_dir,
            env=env,
            timeout=args.timeout,
            seed=args.simulator_seed,
            max_turns=args.simulator_turns,
        )
        summary_file = log_dir / "summary.json"
        summary_file.write_text(json.dumps([result], indent=2, sort_keys=True), encoding="utf-8")
        transcript_file = log_dir / "transcripts.md"
        transcript_file.write_text(_render_transcripts([result]), encoding="utf-8")
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{status} {result['name']} -> {result['log_file']}")
        for issue in result["issues"]:
            print(f"  - {issue}")
        print(f"summary -> {summary_file}")
        print(f"transcripts -> {transcript_file}")
        return 0 if result["passed"] else 1

    selected = [item for item in SCENARIOS if not args.scenario or item["name"] in set(args.scenario)]
    if not selected:
        print("No scenarios selected", file=sys.stderr)
        return 2

    results = []
    failures = 0
    for scenario in selected:
        if args.clean_per_scenario:
            wait_for_running_jobs_to_settle(timeout=args.job_settle_timeout)
            clean_agent_runtime_state()
        result = run_scenario(scenario, log_dir, env, args.timeout)
        job_issues = wait_for_scenario_jobs_to_settle(result["job_ids"], timeout=args.job_settle_timeout)
        if job_issues:
            result["issues"].extend(job_issues)
            result["passed"] = False
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{status} {scenario['name']} -> {result['log_file']}")
        for issue in result["issues"]:
            print(f"  - {issue}")
        failures += 0 if result["passed"] else 1
    summary_file = log_dir / "summary.json"
    summary_file.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    transcript_file = log_dir / "transcripts.md"
    transcript_file.write_text(_render_transcripts(results), encoding="utf-8")
    print(f"summary -> {summary_file}")
    print(f"transcripts -> {transcript_file}")
    return 0 if failures == 0 else 1


class GeneratedUserSimulator:
    """State-aware user simulator for real PTY conversations.

    This intentionally is not a fixed transcript. It reads the current Agent
    output and workflow state, then generates the next user turn from a goal,
    mutation flags, and the active pending question.
    """

    def __init__(self, goal: str, seed: int) -> None:
        self.goal = goal
        self.random = random.Random(seed)
        self.turn_count = 0
        self.language = "en"
        self.switched_chain = False
        self.changed_qps = False
        self.asked_explanation = False
        self.pasted_old_log = False
        self.completed = False
        self.requested_accounts_disk = False
        self.job_followup_index = 0

    def next_turn(self, latest_agent_text: str, workflow_state: dict[str, object]) -> str | dict | None:
        self.turn_count += 1
        question = workflow_state.get("pending_question") if isinstance(workflow_state, dict) else {}
        if not isinstance(question, dict):
            question = {}
        qid = str(question.get("id") or "")
        kind = str(question.get("kind") or "")
        field = str(question.get("field") or "")
        confirmed = workflow_state.get("confirmed_config") if isinstance(workflow_state, dict) else {}
        if not isinstance(confirmed, dict):
            confirmed = {}
        latest_job_id = str(workflow_state.get("latest_job_id") or "")
        active_workflow = str(workflow_state.get("active_workflow") or "")

        if self.turn_count == 1:
            return "Hi"
        if self.turn_count == 2:
            self.language = "zh"
            return "你好，我只是想先用 fake-node 快速验证一下，但过程中我可能会改配置"

        if not self.pasted_old_log and qid in {"cloud_region", "cloud_zone", "machine_type"}:
            self.pasted_old_log = True
            return {
                "paste": (
                    "Agent> CLOUD_REGION: asia-east1\n"
                    "Agent> CLOUD_ZONE: asia-east1-c\n"
                    "Agent> LEDGER_DEVICE: sdc\n"
                    "Traceback (most recent call last):\n"
                    "File \"old/agent.py\", line 1\n"
                    "这是一段旧日志，帮我解释一下，但不要把里面的变量直接写入本次配置。"
                )
            }

        if not self.switched_chain and qid in {"cloud_region", "data_vol_type", "disk_ledger_choice"}:
            self.switched_chain = True
            return "我改主意了，换成 BNB，还是用 fake-node"

        if qid in {"confirm_chain_change", "confirm_target_change", "confirm_chain_target_change"}:
            return self._yes()

        if qid in {"opening_help_choice", "target_mode"}:
            return self._choice(["1", "2"], prefer="1")
        if qid in {"chain", "chain_choice"}:
            return "solana"
        if qid == "disk_ledger_choice":
            return "1,"
        if qid == "data_vol_type":
            return "hyperdisk-balanced,"
        if qid == "data_vol_size":
            return self._yes()
        if qid == "data_vol_max_iops":
            return "20000"
        if qid == "data_vol_max_throughput":
            return "1000"
        if qid in {"disk_accounts_exists", "accounts_disk_exists", "accounts_device_exists", "has_accounts_device"}:
            self.requested_accounts_disk = True
            return self._yes()
        if qid == "disk_accounts_choice":
            return "2"
        if qid == "accounts_vol_type":
            return "hyperdisk-balanced,"
        if qid == "accounts_vol_size":
            return self._yes()
        if qid == "accounts_vol_max_iops":
            return "10000"
        if qid == "accounts_vol_max_throughput":
            return "500"
        if qid == "cloud_region":
            return "us-1"
        if qid == "cloud_zone":
            return "us-1-z"
        if qid == "machine_type":
            return "n2"
        if qid == "network_interface":
            return self._yes()
        if qid == "network_max_bandwidth_gbps":
            return "100"
        if qid in {"benchmark_profile_choice", "benchmark_profile"}:
            return "1"
        if qid == "benchmark_profile_confirm":
            if not self.changed_qps:
                self.changed_qps = True
                return "我需要重新调整 qps profile"
            return self._no()
        if qid == "benchmark_profile_adjust_item":
            return "1"
        if qid in {"quick_initial_qps", "QUICK_INITIAL_QPS"} or field == "QUICK_INITIAL_QPS":
            return "100"
        if qid in {"chain_template_confirm", "chain_template_samples_confirm", "workload_confirm"}:
            return self._yes()
        if qid == "rpc_mode":
            return "single"
        if qid in {"rpc_workload_confirm", "default_rpc_method_confirm", "rpc_method_confirm"}:
            return self._yes()
        if qid in {"observability_mode", "monitoring_mode"}:
            return "1"
        if qid in {"execution_approval", "smoke_approval", "preflight_smoke_confirm"}:
            self.completed = True
            return self._yes()

        if self.completed or active_workflow == "job_monitoring" or latest_job_id:
            return self._job_followup(latest_job_id, latest_agent_text)

        if _latest_prompt_contains(latest_agent_text, "Choose the ledger/data disk", "请选择 Ledger/data 磁盘"):
            return "1,"
        if _latest_prompt_contains(latest_agent_text, "Which chain", "哪条链"):
            return "solana"
        if _latest_prompt_contains(latest_agent_text, "benchmark mode", "benchmark profile", "benchmark 模式"):
            return "1"
        if _latest_prompt_contains(latest_agent_text, "NETWORK_INTERFACE", "网络接口"):
            return self._yes()

        if kind == "yes_no":
            return self._yes()
        if kind in {"numbered_choice", "multi_select"}:
            return "1"
        if kind == "manual_value":
            return "manual-value"

        if not self.asked_explanation and self.turn_count > 4:
            self.asked_explanation = True
            return "fake-node 和 real-node 有什么区别？解释完继续刚才的配置"
        if self.turn_count > 20:
            return "status"
        return "继续下一步"

    def _job_followup(self, latest_job_id: str, latest_agent_text: str) -> str:
        commands = []
        if latest_job_id:
            commands.extend([
                f"status {latest_job_id}",
                f"logs {latest_job_id}",
                "analyze latest job",
                f"status {latest_job_id}",
            ])
        else:
            commands.extend(["status", "jobs", "analyze latest job", "status"])
        if _latest_prompt_contains(latest_agent_text, "No previous job", "没有历史 job", "没有检测到历史 job"):
            return "测试 solana fake-node quick"
        command = commands[self.job_followup_index % len(commands)]
        self.job_followup_index += 1
        return command

    def _choice(self, values: list[str], *, prefer: str) -> str:
        if prefer in values and self.random.random() < 0.8:
            return prefer
        return self.random.choice(values)

    def _yes(self) -> str:
        return self.random.choice(["Y", "y", "yes"])

    def _no(self) -> str:
        return self.random.choice(["N", "n", "no"])


def run_generated_simulation(
    *,
    goal: str,
    log_dir: Path,
    env: dict[str, str],
    timeout: int,
    seed: int,
    max_turns: int,
) -> dict:
    import pty

    name = f"generated_{goal}_seed_{seed}"
    master_fd, slave_fd = pty.openpty()
    session_id = f"pty-{name}"
    state_file = log_dir / f"{name}.state.json"
    scenario_env = env.copy()
    scenario_env["ANYCHAIN_AGENT_DISCOVERY_FIXTURE"] = str(REPO_ROOT / "tests/fixtures/agent_discovery_gcp_multidisk.json")
    process = subprocess.Popen(
        [
            "./bin/anychain-agent",
            "--state-file",
            str(state_file),
            "--session-id",
            session_id,
            "--language",
            "zh",
        ],
        cwd=REPO_ROOT,
        env=scenario_env,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        text=False,
        preexec_fn=os.setsid,
    )
    os.close(slave_fd)
    transcript = bytearray()
    issues: list[str] = []
    turns: list[str] = []
    simulator = GeneratedUserSimulator(goal, seed)
    try:
        wait_for_prompt(master_fd, transcript, start_index=len(transcript), timeout=timeout)
        for _ in range(max_turns):
            workflow_state = load_scenario_workflow_state(session_id)
            transcript_text = transcript.decode("utf-8", errors="replace")
            invariant_issues = generated_invariant_issues(transcript_text, workflow_state, turns=turns)
            if invariant_issues:
                issues.extend(invariant_issues)
                break
            turn = simulator.next_turn(_latest_agent_visible_chunk(transcript_text), workflow_state)
            if turn is None:
                break
            turns.append(render_turn_for_summary(turn))
            start_index = len(transcript)
            write_turn(master_fd, turn)
            wait_for_turn_complete(master_fd, transcript, start_index=start_index, timeout=timeout)
            if simulator.completed:
                break
        os.write(master_fd, b"exit\n")
        wait_for_exit(process, master_fd, transcript, timeout=30)
    except TimeoutError as exc:
        issues.append(str(exc))
        terminate(process)
    finally:
        try:
            os.close(master_fd)
        except OSError:
            pass
        terminate(process)

    text = transcript.decode("utf-8", errors="replace")
    redacted = SECRET_RE.sub("<redacted>", text)
    workflow_state = load_scenario_workflow_state(session_id)
    issues.extend(generated_invariant_issues(redacted, workflow_state, turns=turns, final=True))
    log_file = log_dir / f"{name}.log"
    log_file.write_text(redacted, encoding="utf-8")
    state_file_out = log_dir / f"{name}.workflow_state.json"
    if workflow_state:
        state_file_out.write_text(json.dumps(workflow_state, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "name": name,
        "passed": not issues,
        "issues": sorted(set(issues)),
        "log_file": str(log_file),
        "workflow_state_file": str(state_file_out) if workflow_state else "",
        "turns": turns,
        "job_ids": sorted(set(re.findall(r"\bjob_\d{14}_[0-9a-f]{8}\b", redacted))),
    }


def generated_invariant_issues(text: str, workflow_state: dict, *, turns: list[str] | None = None, final: bool = False) -> list[str]:
    issues: list[str] = []
    agent_visible = _agent_visible_sections(SECRET_RE.sub("<redacted>", text))
    for term in VISIBLE_RESPONSE_FORBIDDEN:
        if term in agent_visible:
            issues.append(f"forbidden visible text: {term}")
    for pattern in VISIBLE_PROCESS_PATTERNS:
        if pattern.search(agent_visible):
            issues.append(f"forbidden process narration: {pattern.pattern}")
    if agent_visible.count("Start a fake-node benchmark") > 1:
        issues.append("stale English opening menu repeated after language/context changed")
    if agent_visible.count("开始一个 fake-node") > 2:
        issues.append("opening benchmark menu repeated too many times")
    if _latest_user_turn_contains_chinese(text) and _agent_chunk_looks_english_prompt(_latest_agent_visible_chunk(text)):
        issues.append("latest Agent response used English deterministic prompt in a Chinese conversation")
    validation_count = agent_visible.count("这条回复没有通过当前问题的校验") + agent_visible.count("This reply did not pass validation")
    if validation_count > 2:
        issues.append(f"repeated validation loop detected: {validation_count} validation messages")
    if turns:
        repeated = _repeated_user_turn_issues(turns)
        issues.extend(repeated)
    job_ids = re.findall(r"\bjob_\d{14}_[0-9a-f]{8}\b", text)
    if len(set(job_ids)) > 1:
        issues.append(f"multiple benchmark jobs were submitted in one generated conversation: {sorted(set(job_ids))}")
    if not workflow_state:
        return issues
    question = workflow_state.get("pending_question") or {}
    if question and not isinstance(question, dict):
        issues.append("pending_question is not an object")
    if isinstance(question, dict) and question:
        if not question.get("id"):
            issues.append("pending_question missing id")
        if not question.get("kind"):
            issues.append("pending_question missing kind")
        if question.get("kind") in {"numbered_choice", "multi_select", "device"} and not question.get("options"):
            issues.append(f"pending_question {question.get('id')} missing options")
    confirmed = workflow_state.get("confirmed_config") or {}
    if isinstance(confirmed, dict):
        wrong_values = {
            "CLOUD_REGION": "我需要测试 BNB",
            "CLOUD_ZONE": "我需要测试 BNB",
            "MACHINE_TYPE": "我需要测试 BNB",
            "LEDGER_DEVICE": "我需要测试 BNB",
        }
        for key, forbidden in wrong_values.items():
            if confirmed.get(key) == forbidden:
                issues.append(f"free-form interruption stored in wrong field: {key}")
        if confirmed.get("CLOUD_REGION") == "asia-east1" and "不要把里面的变量直接写入本次配置" in text:
            issues.append("pasted old CLOUD_REGION was applied as config")
        weights = workflow_state.get("mixed_weights") or {}
        if isinstance(weights, dict) and weights:
            try:
                total = sum(int(value) for value in weights.values())
            except Exception:
                total = -1
            if total != 100:
                issues.append(f"mixed weights total is not 100: {total}")
        issues.extend(_confirmed_config_shape_issues(confirmed))
        issues.extend(_accounts_branch_issues(text, confirmed))
    if final:
        history = workflow_state.get("history") or []
        if len(history) > 80:
            issues.append("generated simulation produced excessive workflow churn")
    return issues


def _latest_prompt_contains(text: str, *needles: str) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def _latest_agent_visible_chunk(text: str) -> str:
    visible = _agent_visible_sections(SECRET_RE.sub("<redacted>", text))
    markers = [match.start() for match in re.finditer(r"Agent>\s*", visible)]
    if not markers:
        return visible[-2000:]
    return visible[markers[-1]:]


def _contains_chinese_user_turn(text: str) -> bool:
    return bool(re.search(r"User>.*[\u4e00-\u9fff]", text))


def _latest_user_turn_contains_chinese(text: str) -> bool:
    matches = list(re.finditer(r"User>\s*(.*)", text))
    if not matches:
        return False
    return bool(re.search(r"[\u4e00-\u9fff]", matches[-1].group(1)))


def _agent_chunk_looks_english_prompt(text: str) -> bool:
    chunk = str(text or "")
    prefixes = (
        "Agent> Confirm ",
        "Agent> Detected ",
        "Agent> Choose ",
        "Agent> Does this node ",
        "Agent> Use the selected ",
        "Agent> Enter the value ",
        "Agent> Which chain ",
    )
    return chunk.startswith(prefixes)


def _repeated_user_turn_issues(turns: list[str]) -> list[str]:
    issues: list[str] = []
    streak_value = ""
    streak = 0
    for turn in turns:
        normalized = str(turn).strip()
        if normalized == streak_value:
            streak += 1
        else:
            streak_value = normalized
            streak = 1
        if normalized and streak > 3:
            issues.append(f"simulator repeated identical turn too many times: {normalized!r}")
            break
    return issues


def _confirmed_config_shape_issues(confirmed: dict) -> list[str]:
    issues: list[str] = []
    numeric_keys = {
        "DATA_VOL_SIZE",
        "DATA_VOL_MAX_IOPS",
        "DATA_VOL_MAX_THROUGHPUT",
        "ACCOUNTS_VOL_SIZE",
        "ACCOUNTS_VOL_MAX_IOPS",
        "ACCOUNTS_VOL_MAX_THROUGHPUT",
        "NETWORK_MAX_BANDWIDTH_GBPS",
        "QUICK_INITIAL_QPS",
        "QUICK_MAX_QPS",
        "QUICK_QPS_STEP",
        "QUICK_DURATION_SECONDS",
    }
    storage_type_keys = {"DATA_VOL_TYPE", "ACCOUNTS_VOL_TYPE"}
    numeric_re = re.compile(r"^[0-9]+(?:\.[0-9]+)?(?:\s*(?:gib|gb|tib|tb|mib/s|mb/s|gbps|iops))?$", re.IGNORECASE)
    storage_re = re.compile(r"^(?:hyperdisk-[a-z0-9-]+|pd-[a-z0-9-]+|local-ssd|nvme|ssd|hdd|standard|gp[0-9]+|io[0-9]+|st[0-9]+|sc[0-9]+)$", re.IGNORECASE)
    boolish_keys = {"has_accounts_device", "use_fake_node", "qps_profile_confirmed", "rpc_workload_confirmed", "rpc_param_samples_confirmed"}
    for key in numeric_keys:
        if key in confirmed and not numeric_re.fullmatch(str(confirmed.get(key) or "").strip()):
            issues.append(f"confirmed_config invalid numeric value: {key}={confirmed.get(key)!r}")
    for key in storage_type_keys:
        value = str(confirmed.get(key) or "").strip()
        if key in confirmed and (not storage_re.fullmatch(value) or value.isdigit()):
            issues.append(f"confirmed_config invalid storage type: {key}={confirmed.get(key)!r}")
    for key in boolish_keys:
        if key in confirmed and confirmed.get(key) not in {True, False, "true", "false", "yes", "no", "Y", "N", "y", "n"}:
            issues.append(f"confirmed_config invalid boolean-ish value: {key}={confirmed.get(key)!r}")
    return issues


def _accounts_branch_issues(text: str, confirmed: dict) -> list[str]:
    issues: list[str] = []
    requested = bool(re.search(r"(separate accounts/state disk|独立的 accounts/state 磁盘).*?\nUser>\s*(?:Y|y|yes)\b", text, re.DOTALL))
    if not requested:
        return issues
    has_accounts = confirmed.get("has_accounts_device")
    accounts_device = confirmed.get("ACCOUNTS_DEVICE") or confirmed.get("accounts_device")
    if has_accounts is not True:
        issues.append("accounts branch was requested but has_accounts_device is not true")
    if not accounts_device:
        issues.append("accounts branch was requested but ACCOUNTS_DEVICE is missing")
    return issues


def run_scenario(scenario: dict, log_dir: Path, env: dict[str, str], timeout: int) -> dict:
    import pty

    master_fd, slave_fd = pty.openpty()
    scenario_env = env.copy()
    for key, value in dict(scenario.get("env") or {}).items():
        path_value = str(value)
        if key.endswith("_FIXTURE") and path_value and not path_value.startswith("/"):
            path_value = str(REPO_ROOT / path_value)
        scenario_env[key] = path_value

    state_file = log_dir / f"{scenario['name']}.state.json"
    session_id = f"pty-{scenario['name']}"
    command = [
        "./bin/anychain-agent",
        "--state-file",
        str(state_file),
        "--session-id",
        session_id,
        "--language",
        "zh",
    ]
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=scenario_env,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        text=False,
        preexec_fn=os.setsid,
    )
    os.close(slave_fd)
    transcript = bytearray()
    issues: list[str] = []
    try:
        wait_for_prompt(master_fd, transcript, start_index=len(transcript), timeout=timeout)
        for turn in scenario["turns"]:
            start_index = len(transcript)
            write_turn(master_fd, turn)
            wait_for_turn_complete(master_fd, transcript, start_index=start_index, timeout=timeout)
        os.write(master_fd, b"exit\n")
        wait_for_exit(process, master_fd, transcript, timeout=30)
    except TimeoutError as exc:
        issues.append(str(exc))
        terminate(process)
    finally:
        try:
            os.close(master_fd)
        except OSError:
            pass
        terminate(process)

    text = transcript.decode("utf-8", errors="replace")
    redacted = SECRET_RE.sub("<redacted>", text)
    agent_visible_text = _agent_visible_sections(redacted)
    log_file = log_dir / f"{scenario['name']}.log"
    log_file.write_text(redacted, encoding="utf-8")
    workflow_state = load_scenario_workflow_state(session_id)
    state_file_out = log_dir / f"{scenario['name']}.workflow_state.json"
    if workflow_state:
        state_file_out.write_text(json.dumps(workflow_state, indent=2, sort_keys=True), encoding="utf-8")

    any_terms = scenario.get("must_include_any") or []
    if any_terms and not any(term in redacted for term in any_terms):
        issues.append(f"missing any required text: {', '.join(any_terms)}")
    for term in scenario.get("must_include_all", []):
        if term not in redacted:
            issues.append(f"missing required text: {term}")
    for term in VISIBLE_RESPONSE_FORBIDDEN + scenario.get("must_not_include", []):
        if term in agent_visible_text:
            issues.append(f"forbidden text present: {term}")
    for term, max_count in dict(scenario.get("agent_visible_max_count") or {}).items():
        count = agent_visible_text.count(term)
        if count > int(max_count):
            issues.append(f"visible text repeated too many times: {term!r} count={count} max={max_count}")
    for pattern in VISIBLE_PROCESS_PATTERNS:
        if pattern.search(agent_visible_text):
            issues.append(f"forbidden process narration matched: {pattern.pattern}")
    if workflow_state:
        for path, expected in dict(scenario.get("state_must_equal") or {}).items():
            actual = _nested_value(workflow_state, path)
            if actual != expected:
                issues.append(f"workflow state mismatch: {path} expected={expected!r} actual={actual!r}")
        for path, allowed in dict(scenario.get("state_must_be_one_of") or {}).items():
            actual = _nested_value(workflow_state, path)
            allowed_values = list(allowed or [])
            if actual not in allowed_values:
                issues.append(f"workflow state mismatch: {path} expected one of {allowed_values!r} actual={actual!r}")
        for path, forbidden in dict(scenario.get("state_must_not_equal") or {}).items():
            actual = _nested_value(workflow_state, path)
            if actual == forbidden:
                issues.append(f"workflow state forbidden value: {path}={forbidden!r}")
    guard_revisions = terminal_contract_guard_revisions(workflow_state)
    if guard_revisions:
        issues.append(
            "terminal_contract_guard used for product decision path: "
            + ", ".join(str(item) for item in guard_revisions)
        )
    return {
        "name": scenario["name"],
        "passed": not issues,
        "issues": issues,
        "log_file": str(log_file),
        "workflow_state_file": str(state_file_out) if workflow_state else "",
        "turns": [render_turn_for_summary(turn) for turn in scenario.get("turns", [])],
        "job_ids": sorted(set(re.findall(r"\bjob_\d{14}_[0-9a-f]{8}\b", redacted))),
    }


def load_scenario_workflow_state(session_id: str) -> dict:
    state_file = REPO_ROOT / ".agent" / "sessions" / session_id / "conversation_state.json"
    if not state_file.is_file():
        return {}
    try:
        return json.loads(state_file.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _nested_value(payload: dict, dotted_path: str):
    current = payload
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def terminal_contract_guard_revisions(workflow_state: dict) -> list[int | str]:
    revisions: list[int | str] = []
    if not workflow_state:
        return revisions
    for item in workflow_state.get("history") or []:
        question = item.get("pending_question") if isinstance(item, dict) else {}
        if isinstance(question, dict) and question.get("source") == "terminal_contract_guard":
            revisions.append(item.get("revision", "?"))
    question = workflow_state.get("pending_question") or {}
    if isinstance(question, dict) and question.get("source") == "terminal_contract_guard":
        revisions.append(workflow_state.get("revision", "?"))
    return revisions


def wait_for_scenario_jobs_to_settle(job_ids: list[str], timeout: int) -> list[str]:
    issues: list[str] = []
    for job_id in job_ids:
        status = wait_for_job_to_settle(job_id, timeout=timeout)
        if not status:
            issues.append(f"job did not settle before timeout: {job_id}")
            continue
        if status.get("status") != "completed":
            error = status.get("error", "")
            issues.append(f"job did not complete successfully: {job_id} status={status.get('status')} error={error}")
    return issues


def wait_for_running_jobs_to_settle(timeout: int) -> None:
    jobs_dir = REPO_ROOT / ".agent" / "jobs"
    if not jobs_dir.is_dir():
        return
    running = []
    for job_file in jobs_dir.glob("job_*/job.json"):
        try:
            job = json.loads(job_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if job.get("status") == "running":
            running.append(job.get("job_id") or job_file.parent.name)
    for job_id in running:
        wait_for_job_to_settle(str(job_id), timeout=timeout)


def wait_for_job_to_settle(job_id: str, timeout: int) -> dict | None:
    job_file = REPO_ROOT / ".agent" / "jobs" / job_id / "job.json"
    deadline = time.time() + max(1, timeout)
    last_status: dict | None = None
    while time.time() < deadline:
        if not job_file.is_file():
            return None
        try:
            last_status = json.loads(job_file.read_text(encoding="utf-8"))
        except Exception:
            time.sleep(1)
            continue
        if last_status.get("status") != "running":
            return last_status
        time.sleep(2)
    return last_status


def write_turn(master_fd: int, turn: str | dict) -> None:
    if isinstance(turn, dict) and "paste" in turn:
        payload = str(turn["paste"]).encode("utf-8")
        os.write(master_fd, b"\x1b[200~" + payload + b"\x1b[201~\n")
        return
    os.write(master_fd, str(turn).encode("utf-8") + b"\n")


def render_turn_for_summary(turn: str | dict) -> str:
    if isinstance(turn, dict) and "paste" in turn:
        return str(turn["paste"])
    return str(turn)


def _render_transcripts(results: list[dict]) -> str:
    sections = ["# AnyChain Agent Live PTY Transcripts", ""]
    for result in results:
        sections.append(f"## {result['name']}")
        sections.append("")
        sections.append(f"Status: {'PASS' if result['passed'] else 'FAIL'}")
        if result["issues"]:
            sections.append("")
            sections.append("Issues:")
            for issue in result["issues"]:
                sections.append(f"- {issue}")
        sections.append("")
        path = Path(result["log_file"])
        text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        sections.append("```text")
        sections.append(text.strip())
        sections.append("```")
        sections.append("")
    return "\n".join(sections)


def _agent_visible_sections(text: str) -> str:
    """Return terminal output excluding user-entered lines.

    The live PTY transcript intentionally includes both sides of the
    conversation. Forbidden-process checks should validate Agent-visible output
    only, otherwise natural user phrases such as "让我确认" create false
    positives.
    """
    kept: list[str] = []
    for line in text.splitlines():
        if "User>" in line:
            continue
        kept.append(line)
    return "\n".join(kept)


def wait_for(fd: int, transcript: bytearray, needle: bytes, timeout: int) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        readable, _, _ = select.select([fd], [], [], 0.5)
        if not readable:
            continue
        chunk = os.read(fd, 4096)
        if not chunk:
            return
        transcript.extend(chunk)
        if needle in transcript[-8192:]:
            return
    raise TimeoutError(f"timeout waiting for {needle.decode(errors='replace')!r}")


def wait_for_prompt(fd: int, transcript: bytearray, start_index: int, timeout: int) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        readable, _, _ = select.select([fd], [], [], 0.5)
        if not readable:
            continue
        chunk = os.read(fd, 4096)
        if not chunk:
            return
        transcript.extend(chunk)
        recent = transcript[start_index:]
        if b"User>" in recent:
            return
    raise TimeoutError("timeout waiting for terminal prompt 'User>'")


def wait_for_turn_complete(fd: int, transcript: bytearray, start_index: int, timeout: int) -> None:
    deadline = time.time() + timeout
    saw_agent_output = False
    while time.time() < deadline:
        readable, _, _ = select.select([fd], [], [], 0.5)
        if not readable:
            continue
        try:
            chunk = os.read(fd, 4096)
        except OSError:
            return
        if not chunk:
            return
        transcript.extend(chunk)
        recent = transcript[start_index:]
        if b"Agent>" in recent:
            saw_agent_output = True
        if saw_agent_output:
            tail_after_agent = recent.rsplit(b"Agent>", 1)[-1]
            if b"User>" in tail_after_agent:
                return
    raise TimeoutError("timeout waiting for one Agent response and next terminal prompt")


def wait_for_exit(process: subprocess.Popen, fd: int, transcript: bytearray, timeout: int) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            return
        readable, _, _ = select.select([fd], [], [], 0.5)
        if readable:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                return
            if chunk:
                transcript.extend(chunk)
    raise TimeoutError("timeout waiting for process exit")


def terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except Exception:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())

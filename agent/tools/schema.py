"""OpenAI-compatible tool schema for enterprise Agent platform integration."""

from __future__ import annotations

from typing import Any


def tool_schema() -> dict[str, list[dict[str, Any]]]:
    """Return tool definitions without binding the framework to one LLM vendor."""
    tools = [
        _tool(
            "discover_environment",
            "Run read-only discovery for cloud, VM/Kubernetes, CPU, memory, disks, network, and dependencies.",
            {},
        ),
        _tool(
            "audit_dependencies",
            "Run dependency checks in audit-only mode without changing the host.",
            {},
        ),
        _tool(
            "load_capabilities",
            "Return supported chain templates, RPC methods, adapter families, fake-node fixture status, and framework limits.",
            {},
        ),
        _tool(
            "load_framework_context",
            "Return compact AnyChain framework context for Agent grounding: runtime flow, config layers, current capability summary, and authoritative docs.",
            {"language": _string("Optional response/document language: en or zh.")},
        ),
        _tool(
            "load_execution_contract",
            "Return the hard benchmark execution contract: entrypoint phases, required variables, dependency expectations, and mandatory gates.",
            {"use_fake_node": {"type": "boolean", "description": "Optional target mode. true=fake-node, false=real-node, omitted=unknown target."}},
        ),
        _tool(
            "prepare_benchmark_run",
            "Run discovery, doctor, request drafting, plan generation, preflight, and runbook rendering without launching traffic.",
            {
                "source_prompt": _string("Optional original user request for evidence only."),
                "chain": _string("Optional chain name."),
                "goal": _string("Optional benchmark goal."),
                "rpc_mode": _string("Optional RPC mode: single or mixed."),
                "use_fake_node": _boolean("Whether to use fake-node."),
                "deployment_type": _string("Optional deployment type: vm, kubernetes, or unknown."),
                "cloud_provider": _string("Optional cloud provider such as gcp or aws."),
                "target_rpc_url": _string("Optional target node RPC URL."),
                "mainnet_rpc_url": _string("Optional mainnet/reference RPC URL."),
                "ledger_device": _string("Optional ledger device name."),
                "accounts_device": _string("Optional accounts/data device name."),
                "blockchain_process_names": {"type": "array", "items": {"type": "string"}, "description": "Node process names or command keywords."},
                "cloud_region": _string("Cloud region for report metadata."),
                "cloud_zone": _string("Cloud zone for report metadata."),
                "machine_type": _string("Machine or instance type."),
                "data_vol_type": _string("Ledger/data disk type."),
                "data_vol_size": _string("Ledger/data disk size in GiB."),
                "data_vol_max_iops": _string("Ledger/data disk provisioned IOPS."),
                "data_vol_max_throughput": _string("Ledger/data disk throughput in MiB/s."),
                "accounts_vol_type": _string("Optional accounts/state disk type."),
                "accounts_vol_size": _string("Optional accounts/state disk size in GiB."),
                "accounts_vol_max_iops": _string("Optional accounts/state disk provisioned IOPS."),
                "accounts_vol_max_throughput": _string("Optional accounts/state disk throughput in MiB/s."),
                "network_interface": _string("Network interface used by the node."),
                "network_max_bandwidth_gbps": _string("Instance or pod network bandwidth baseline in Gbps."),
                "qps_initial": {"type": "integer", "description": "Optional initial QPS."},
                "qps_max": {"type": "integer", "description": "Optional max QPS."},
                "qps_step": {"type": "integer", "description": "Optional QPS step."},
                "duration_seconds": {"type": "integer", "description": "Optional run duration in seconds."},
                "rpc_methods": {"type": "array", "items": {"type": "string"}, "description": "Optional RPC methods to test."},
                "mixed_weights": {"type": "object", "additionalProperties": {"type": "integer"}, "description": "Optional method->weight map for mixed workloads."},
                "output_dir": _string("Optional directory for prepared plan/runbook artifacts."),
            },
        ),
        _tool(
            "draft_request",
            "Draft a normalized benchmark request from explicit structured fields inferred by the calling Agent. prompt is optional evidence text only; this tool does not parse free-form text.",
            {
                "source_prompt": _string("Optional original user request for evidence only."),
                "chain": _string("Optional chain name."),
                "goal": _string("Optional benchmark goal such as smoke, baseline, stress, max_stable_qps, or bottleneck_confirmation."),
                "rpc_mode": _string("Optional RPC mode: single or mixed."),
                "use_fake_node": _boolean("Whether to use fake-node."),
                "deployment_type": _string("Optional deployment type: vm, kubernetes, or unknown."),
                "cloud_provider": _string("Optional cloud provider such as gcp or aws."),
                "target_rpc_url": _string("Optional target node RPC URL."),
                "mainnet_rpc_url": _string("Optional mainnet/reference RPC URL."),
                "ledger_device": _string("Optional ledger device name."),
                "accounts_device": _string("Optional accounts/data device name."),
                "blockchain_process_names": {"type": "array", "items": {"type": "string"}, "description": "Node process names or command keywords."},
                "cloud_region": _string("Cloud region for report metadata."),
                "cloud_zone": _string("Cloud zone for report metadata."),
                "machine_type": _string("Machine or instance type."),
                "data_vol_type": _string("Ledger/data disk type."),
                "data_vol_size": _string("Ledger/data disk size in GiB."),
                "data_vol_max_iops": _string("Ledger/data disk provisioned IOPS."),
                "data_vol_max_throughput": _string("Ledger/data disk throughput in MiB/s."),
                "accounts_vol_type": _string("Optional accounts/state disk type."),
                "accounts_vol_size": _string("Optional accounts/state disk size in GiB."),
                "accounts_vol_max_iops": _string("Optional accounts/state disk provisioned IOPS."),
                "accounts_vol_max_throughput": _string("Optional accounts/state disk throughput in MiB/s."),
                "network_interface": _string("Network interface used by the node."),
                "network_max_bandwidth_gbps": _string("Instance or pod network bandwidth baseline in Gbps."),
                "qps_initial": {"type": "integer", "description": "Optional initial QPS."},
                "qps_max": {"type": "integer", "description": "Optional max QPS."},
                "qps_step": {"type": "integer", "description": "Optional QPS step."},
                "duration_seconds": {"type": "integer", "description": "Optional run duration in seconds."},
                "rpc_methods": {"type": "array", "items": {"type": "string"}, "description": "Optional RPC methods to test."},
                "mixed_weights": {"type": "object", "additionalProperties": {"type": "integer"}, "description": "Optional method->weight map for mixed workloads."},
            },
        ),
        _tool(
            "generate_plan",
            "Generate a benchmark plan from a request JSON and optional discovery results.",
            {"request": _object("Normalized benchmark request.")},
            required=["request"],
        ),
        _tool(
            "run_preflight",
            "Validate a plan before execution and return blockers, warnings, and checklist questions.",
            {"plan": _object("Benchmark plan JSON.")},
            required=["plan"],
        ),
        _tool(
            "submit_job",
            "Submit a benchmark job. Real execution requires explicit approval; mock mode validates lifecycle only.",
            {
                "plan_file": _string("Path to a generated plan JSON."),
                "mock": _boolean("Run a lifecycle-only mock job."),
                "approved": _boolean("Explicit approval for real execution."),
                "jobs_dir": _string("Optional Agent jobs directory."),
            },
            required=["plan_file"],
        ),
        _tool(
            "run_fake_node_smoke_benchmark",
            "Run the real benchmark engine in quick fake-node mode with isolated job-local output. Requires explicit approval.",
            {
                "plan_file": _string("Path to a generated plan JSON."),
                "jobs_dir": _string("Optional Agent jobs directory."),
                "approved": _boolean("Explicit approval for execution."),
            },
            required=["plan_file"],
        ),
        _tool(
            "install_dependencies",
            "Install benchmark engine dependencies after explicit approval. Agent runtime/gcloud setup runs only when explicitly requested.",
            {
                "approved": _boolean("Explicit approval for installation."),
                "no_sudo": _boolean("Avoid sudo/system package changes. Defaults to true.", default=True),
                "include_vegeta": _boolean("Install vegeta when possible. Defaults to true.", default=True),
                "include_agent_runtime": _boolean("Reinstall/update Google ADK into an isolated venv. Defaults to false.", default=False),
                "include_gcloud": _boolean("Install Google Cloud CLI for ADC/impersonation workflows. Defaults to false."),
                "adk_venv": _string("ADK virtualenv path. Defaults to .venv-adk."),
                "allow_system_python": _boolean("Allow system Python package changes when required."),
            },
        ),
        _tool(
            "get_job_status",
            "Read current job metadata and status.",
            {"job_id": _string("Agent job id."), "jobs_dir": _string("Optional Agent jobs directory.")},
            required=["job_id"],
        ),
        _tool(
            "tail_job_log",
            "Read recent benchmark log lines for a job.",
            {
                "job_id": _string("Agent job id."),
                "jobs_dir": _string("Optional Agent jobs directory."),
                "lines": {"type": "integer", "description": "Maximum lines to return.", "default": 80},
            },
            required=["job_id"],
        ),
        _tool(
            "analyze_artifacts",
            "Analyze generated benchmark artifacts and summarize evidence paths, warnings, and failures.",
            {"job_id": _string("Agent job id."), "jobs_dir": _string("Optional Agent jobs directory.")},
            required=["job_id"],
        ),
        _tool(
            "answer_artifact_question",
            "Answer a question from generated CSV, HTML, runtime.env, job metadata, and archive artifacts.",
            {
                "question": _string("User question about benchmark artifacts."),
                "job_id": _string("Optional Agent job id."),
                "artifact_index": _string("Optional artifact index JSON path."),
            },
            required=["question"],
        ),
        _tool(
            "diagnose_artifacts",
            "Run deterministic bottleneck and chart diagnostics against benchmark artifacts.",
            {"job_id": _string("Optional Agent job id."), "artifact_index": _string("Optional artifact index JSON path.")},
        ),
        _tool(
            "draft_chain_template",
            "Generate a human-reviewed chain template draft for a new chain. The draft is not production support until validated.",
            {
                "chain": _string("New chain name."),
                "adapter_family": _string("Adapter family such as jsonrpc, rest, substrate, bitcoin, cosmos, or solana."),
                "methods": {"type": "array", "items": {"type": "string"}, "description": "RPC methods to include."},
            },
            required=["chain", "adapter_family"],
        ),
        _tool(
            "gap_analysis",
            "Explain whether a chain/RPC method is already supported and what onboarding work remains.",
            {
                "chain": _string("Chain name."),
                "methods": {"type": "array", "items": {"type": "string"}, "description": "RPC methods to check."},
            },
            required=["chain"],
        ),
        _tool(
            "knowledge_search",
            "Search the configured enterprise Knowledge Base adapter when enabled.",
            {"query": _string("Search query."), "chain": _string("Optional chain filter.")},
            required=["query"],
        ),
    ]
    return {"tools": tools}


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


def _string(description: str) -> dict[str, str]:
    return {"type": "string", "description": description}


def _boolean(description: str, default: bool = False) -> dict[str, Any]:
    return {"type": "boolean", "description": description, "default": default}


def _object(description: str) -> dict[str, str]:
    return {"type": "object", "description": description}

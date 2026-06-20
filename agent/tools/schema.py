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
            "load_capabilities",
            "Return supported chain templates, RPC methods, adapter families, fake-node fixture status, and framework limits.",
            {},
        ),
        _tool(
            "draft_request",
            "Convert a natural-language benchmark goal into a normalized request draft. Do not execute anything.",
            {"prompt": _string("Natural-language benchmark request.")},
            required=["prompt"],
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


def _boolean(description: str) -> dict[str, Any]:
    return {"type": "boolean", "description": description, "default": False}


def _object(description: str) -> dict[str, str]:
    return {"type": "object", "description": description}

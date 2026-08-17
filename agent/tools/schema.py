"""OpenAI-compatible tool schema for enterprise Agent platform integration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable

from agent.harness.sync_observe_contract import sync_observe_tool_properties


ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class ToolSpec:
    """One operation contract used for both model schema and dispatch."""

    name: str
    description: str
    properties: dict[str, Any]
    required: tuple[str, ...] = ()
    handler: ToolHandler | None = field(default=None, repr=False, compare=False)

    def bind(self, handler: ToolHandler) -> "ToolSpec":
        return replace(self, handler=handler)

    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.validate_arguments(arguments)
        if self.handler is None:
            raise RuntimeError(f"tool handler is not registered: {self.name}")
        return self.handler(arguments)

    def as_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.properties,
                    "required": list(self.required),
                    "additionalProperties": False,
                },
            },
        }

    def validate_arguments(self, arguments: dict[str, Any]) -> None:
        undeclared = sorted(str(key) for key in arguments if key not in self.properties)
        if undeclared:
            raise ValueError(f"undeclared arguments for {self.name}: {', '.join(undeclared)}")
        missing = [key for key in self.required if arguments.get(key) in (None, "")]
        if missing:
            raise ValueError(f"missing required arguments for {self.name}: {', '.join(missing)}")
        for name, value in arguments.items():
            _validate_json_value(value, self.properties[name], f"{self.name}.{name}")


def _validate_json_value(value: Any, schema: dict[str, Any], path: str) -> None:
    expected = schema.get("type")
    valid = {
        "string": isinstance(value, str),
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "array": isinstance(value, list),
        "object": isinstance(value, dict),
    }.get(str(expected), True)
    if not valid:
        raise ValueError(f"invalid type for {path}: expected {expected}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"invalid value for {path}: expected one of {schema['enum']}")
    if isinstance(value, list):
        if len(value) < int(schema.get("minItems") or 0):
            raise ValueError(f"invalid value for {path}: too few items")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate_json_value(item, item_schema, f"{path}[{index}]")
    if isinstance(value, dict):
        properties = schema.get("properties")
        if isinstance(properties, dict):
            unknown = sorted(set(value) - set(properties))
            if schema.get("additionalProperties") is False and unknown:
                raise ValueError(f"undeclared keys for {path}: {', '.join(unknown)}")
            for name, item in value.items():
                if name in properties:
                    _validate_json_value(item, properties[name], f"{path}.{name}")
        additional = schema.get("additionalProperties")
        if isinstance(additional, dict):
            for name, item in value.items():
                _validate_json_value(item, additional, f"{path}.{name}")


def tool_schema() -> dict[str, list[dict[str, Any]]]:
    """Generate model schemas from the callable operation registry."""

    from .executor import TOOL_OPERATIONS

    return {"tools": [spec.as_tool_schema() for spec in TOOL_OPERATIONS]}


def _build_tool_specs() -> tuple[ToolSpec, ...]:
    sync_observe_properties = sync_observe_tool_properties()
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
            "run_doctor",
            "Summarize readiness, missing dependencies, LLM auth, KB config, and capabilities.",
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
            "load_framework_index",
            "Return the local framework knowledge index: supported chains, RPC methods, authoritative docs, key code paths, extension boundaries, and validation commands.",
            {"index_path": _string("Optional framework index path. Defaults to .agent/knowledge/framework_index.json if present, otherwise builds from repo state.")},
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
                **sync_observe_properties,
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
                "blockchain_process_names": {"type": "array", "items": {"type": "string"}, "description": "Node process names or command-line fragments."},
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
                "observability_enabled": _boolean("Enable the observability stack."),
                "observability_mode": {"type": "string", "enum": ["local", "exporter"], "description": "Local Prometheus/Grafana or exporter-only mode."},
                "observability_auto_stop": _boolean("Stop locally managed observability with the benchmark."),
                "exporter_port": _string("Optional exporter port."),
                "prometheus_port": _string("Optional Prometheus port."),
                "grafana_port": _string("Optional Grafana port."),
                "confirmations": {"type": "array", "items": {"type": "string"}, "description": "Confirmed checklist ids, for example rpc_workload_confirmed and rpc_param_samples_confirmed."},
                "output_dir": _string("Optional directory for prepared plan/runbook artifacts."),
            },
        ),
        _tool(
            "draft_request",
            "Draft a normalized benchmark request from explicit structured fields inferred by the calling Agent. prompt is optional evidence text only; this tool does not parse free-form text.",
            {
                **sync_observe_properties,
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
                "blockchain_process_names": {"type": "array", "items": {"type": "string"}, "description": "Node process names or command-line fragments."},
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
                "observability_enabled": _boolean("Enable the observability stack."),
                "observability_mode": {"type": "string", "enum": ["local", "exporter"], "description": "Local Prometheus/Grafana or exporter-only mode."},
                "observability_auto_stop": _boolean("Stop locally managed observability with the benchmark."),
                "exporter_port": _string("Optional exporter port."),
                "prometheus_port": _string("Optional Prometheus port."),
                "grafana_port": _string("Optional Grafana port."),
                "confirmations": {"type": "array", "items": {"type": "string"}, "description": "Confirmed checklist ids, for example rpc_workload_confirmed and rpc_param_samples_confirmed."},
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
            "Submit a real benchmark job after explicit approval.",
            {
                "plan_file": _string("Path to a generated plan JSON."),
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
                "include_agent_runtime": _boolean("Install or update the core Agent runtime in an isolated venv. Defaults to false.", default=False),
                "include_gcloud": _boolean("Install Google Cloud CLI for ADC/impersonation workflows. Defaults to false."),
                "adk_venv": _string("Agent virtualenv path; adk_venv is a retained compatibility field. Defaults to .venv-adk."),
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
            "validate_rpc_workload",
            "Validate single/custom/mixed RPC workload choices and mixed weights.",
            {
                "chain": _string("Chain name."),
                "rpc_mode": _string("single or mixed."),
                "methods": {"type": "array", "items": {"type": "string"}, "description": "Selected RPC methods."},
                "mixed_weights": {"type": "object", "additionalProperties": {"type": "integer"}, "description": "method->weight map."},
            },
            required=["chain", "rpc_mode"],
        ),
        _tool(
            "load_default_workload",
            "Load chain-template default single method and mixed weights.",
            {"chain": _string("Chain name.")},
            required=["chain"],
        ),
        _tool(
            "validate_chain_template",
            "Validate selected chain template readiness.",
            {"chain": _string("Chain name.")},
            required=["chain"],
        ),
        _tool(
            "validate_execution_gate",
            "Validate preflight, smoke, and approval gates before execution.",
            {
                "plan": _object("Benchmark plan."),
                "preflight": _object("Preflight result."),
                "smoke": _object("Smoke result."),
                "approved": _boolean("Explicit user approval."),
                "real_execution": _boolean("Whether this is a real benchmark execution."),
            },
        ),
        _tool(
            "build_onboarding_handoff",
            "Build an evidence-aware chain/RPC onboarding handoff.",
            {
                "chain": _string("Chain name."),
                "family": _string("Adapter family."),
                "methods": {"type": "array", "items": {"type": "string"}, "description": "RPC methods."},
                "evidence": _object("Docs, KB, request samples, response samples, and sync-health evidence."),
            },
            required=["chain", "family"],
        ),
        _tool(
            "knowledge_search",
            "Search the configured enterprise Knowledge Base adapter when enabled.",
            {"query": _string("Search query."), "chain": _string("Optional chain filter.")},
            required=["query"],
        ),
        _tool(
            "validate_fake_node_fixture_coverage",
            "Validate fake-node fixture coverage through the canonical fake-node checker.",
            {
                "chains": _string("Chain filter, or 'all'. Defaults to all."),
                "modes": _string("Comma-separated modes to check. Defaults to single,mixed."),
                "strict": _boolean("Fail on placeholder fixtures. Defaults to true.", default=True),
            },
        ),
        _tool(
            "validate_fake_node_fixture_authenticity",
            "Validate that fixture files have matching recorded request/response evidence.",
            {
                "modes": _string("Comma-separated modes to check. Defaults to single,mixed."),
                "allow_incomplete": _boolean("Allow incomplete recorded evidence."),
            },
        ),
        _tool(
            "inspect_llm_auth",
            "Inspect configured LLM auth mode without reading or printing secrets.",
            {},
        ),
    ]
    return tuple(tools)


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str] | None = None) -> ToolSpec:
    return ToolSpec(name, description, properties, tuple(required or ()))


def _string(description: str) -> dict[str, str]:
    return {"type": "string", "description": description}


def _boolean(description: str, default: bool = False) -> dict[str, Any]:
    return {"type": "boolean", "description": description, "default": default}


def _object(description: str) -> dict[str, str]:
    return {"type": "object", "description": description}


TOOL_SPECS = _build_tool_specs()
if len({spec.name for spec in TOOL_SPECS}) != len(TOOL_SPECS):
    raise RuntimeError("duplicate Agent tool name")
for _spec in TOOL_SPECS:
    undeclared_required = set(_spec.required) - set(_spec.properties)
    if undeclared_required:
        raise RuntimeError(f"required tool arguments must be declared for {_spec.name}")

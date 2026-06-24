"""Quality gates for chain and RPC onboarding plans."""

from __future__ import annotations

from typing import Any

from onboarding.families import SUPPORTED_FAMILIES


REQUIRED_CHAIN_EVIDENCE = [
    "official protocol/RPC documentation URL or exported internal KB page",
    "node endpoint type: JSON-RPC, REST, Tendermint RPC/LCD, Substrate RPC/Sidecar, Bitcoin-like JSON-RPC, or dual protocol",
    "at least one real local-node request/response sample for each workload RPC method",
    "parameter contract for each method: order, type, encoding, path/query/body binding, and sample values",
    "sync-health or block-height method, including response shape and healthy/unhealthy interpretation",
    "rate-limit/auth requirements for public endpoints, indexers, sidecars, or mirror nodes",
]

REQUIRED_RPC_EVIDENCE = [
    "method name and transport route",
    "request body/path/query with all parameters",
    "real successful response body",
    "real error response body when params are missing or invalid",
    "fake-node fixture filename and mapping rule",
    "expected proxy method name for per-method attribution",
]


def onboarding_quality_gate(chain: str, adapter_family: str, methods: list[str] | None = None) -> dict[str, Any]:
    """Return the evidence checklist and support boundary for onboarding."""
    methods = [method.strip() for method in (methods or []) if method.strip()]
    family_known = adapter_family in SUPPORTED_FAMILIES
    return {
        "chain": chain,
        "adapter_family": adapter_family or "<unknown>",
        "family_known": family_known,
        "supported_families": SUPPORTED_FAMILIES,
        "decision": "existing_family_review" if family_known else "new_family_or_unknown_protocol",
        "llm_policy": [
            "Do not rely on the model's general blockchain knowledge as proof of support.",
            "Use LLM knowledge only to propose hypotheses; require official docs, KB evidence, or real node samples before code changes.",
            "If evidence is incomplete, ask the user for the missing documents/samples instead of drafting production support.",
        ],
        "required_chain_evidence": REQUIRED_CHAIN_EVIDENCE,
        "required_rpc_evidence": REQUIRED_RPC_EVIDENCE,
        "method_evidence": [
            {"method": method, "required": REQUIRED_RPC_EVIDENCE}
            for method in methods
        ] or [{"method": "<method_name>", "required": REQUIRED_RPC_EVIDENCE}],
        "quality_gates": [
            "family classification reviewed against supported_families",
            "chain template validates",
            "all workload methods have param_formats or param_spec",
            "mixed_weighted total equals 100",
            "fake-node fixtures exist and are not placeholders",
            "proxy attribution emits the expected method names",
            "sync-health behavior is configured or explicitly marked unsupported",
            "fake-node smoke passes before real-node benchmark",
        ],
    }


def coding_brief(package: dict[str, Any]) -> str:
    """Render a concise coding brief for a specialist implementation Agent."""
    gate = package.get("quality_gate", {})
    methods = package.get("workload_plugin", {}).get("mixed_weighted", [])
    method_names = [item.get("method", "") for item in methods if item.get("method")]
    lines = [
        f"Implementation brief: onboard {package.get('chain', '<chain>')}",
        "",
        "Objective:",
        "- Add or update support without bypassing AnyChain validation, fake-node, proxy attribution, or sync-health gates.",
        "",
        "Family decision:",
        f"- Proposed adapter_family: {package.get('adapter_family')}",
        f"- Known family: {gate.get('family_known')}",
        f"- Supported families: {', '.join(gate.get('supported_families', []))}",
        "- If the proposed family is unknown, stop and write a new-family design before coding.",
        "",
        "Evidence required before marking support complete:",
    ]
    lines.extend(f"- {item}" for item in gate.get("required_chain_evidence", []))
    lines.extend([
        "",
        "Files to edit:",
        "- config/chains/<chain>.json",
        "- tools/chain_adapters/<family>.py only if existing param formats cannot express the requests",
        "- tools/fake-node/configs/<family>.yaml and tools/fake-node/fixtures/<chain>/",
        "- docs/en/how-to-add-chain.md and docs/zh/how-to-add-chain.md only for user-visible behavior changes",
        "",
        "Existing-family chain path:",
        "- Choose one of the supported families only when the request envelope, parameter binding, proxy extraction, fake-node routing, and sync-health shape match repository evidence.",
        "- Prefer config-only changes: chain template, rpc_methods, param_formats/param_spec, proxy_extraction, _meta.sync_health, and fixture mappings.",
        "- Do not add adapter or fake-node handler code unless the existing family cannot express the method shape.",
        "",
        "New-family path:",
        "- Stop before coding unless the brief explains why all six current families are insufficient.",
        "- Add a design section covering transport, request envelope, parameter model, response envelope, sync-health semantics, proxy extraction, fake-node routing, and report impact.",
        "- Add new adapter/fake-node code only after the design and evidence are complete.",
        "",
        "New RPC method path:",
        "- For every method, define request parameters, sample values, expected response shape, error behavior, and method name as it must appear in proxy_method.csv.",
        "- Add the method to rpc_methods.single or rpc_methods.mixed_weighted, ensure mixed weights total 100, and add param_formats or param_spec.",
        "- Record or provide fake-node fixtures that exactly match method + params + chain response structure.",
        "",
        "RPC methods to implement:",
    ])
    if method_names:
        lines.extend(f"- {method}" for method in method_names)
    else:
        lines.append("- <method_name>")
    lines.extend([
        "",
        "Validation commands:",
    ])
    lines.extend(f"- {cmd}" for cmd in package.get("validation_commands", []))
    lines.extend([
        "",
        "Completion rule:",
        "- Do not call this supported until every quality gate passes and fake-node smoke proves request/response fixtures are usable.",
        "- If official docs, internal KB evidence, real request samples, or real response samples are missing, ask the user for them instead of inventing support.",
    ])
    return "\n".join(lines)

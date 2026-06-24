"""Local framework knowledge index for Agent grounding."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from knowledge.framework_capabilities import REPO_ROOT, load_framework_capabilities


DEFAULT_INDEX_PATH = REPO_ROOT / ".agent" / "knowledge" / "framework_index.json"


KEY_CODE_PATHS = [
    {
        "topic": "agent_terminal",
        "paths": ["bin/anychain-agent", "agent/terminal/repl.py", "agent/terminal/io.py"],
        "purpose": "Human-facing Agent terminal, dependency bootstrap, language routing, and workflow state.",
    },
    {
        "topic": "environment_discovery",
        "paths": ["agent/discovery/environment.py", "agent/diagnostics/doctor.py"],
        "purpose": "Read-only cloud, deployment, CPU, memory, disk, network, and dependency inference.",
    },
    {
        "topic": "benchmark_workflow",
        "paths": ["agent/workflows/benchmark_wizard.py", "agent/workflows/planning_bridge.py"],
        "purpose": "Interactive benchmark configuration, runtime.env preparation, preflight, and smoke gate.",
    },
    {
        "topic": "chain_templates",
        "paths": ["config/chains"],
        "purpose": "Supported chain templates, RPC methods, params, proxy extraction, and sync-health metadata.",
    },
    {
        "topic": "fake_node",
        "paths": ["tools/fake-node", "tools/fake-node/fixtures"],
        "purpose": "Local closed-loop node simulation and recorded RPC response fixtures.",
    },
    {
        "topic": "onboarding",
        "paths": ["agent/onboarding", "docs/en/how-to-add-chain.md", "docs/zh/how-to-add-chain.md"],
        "purpose": "New chain, new RPC method, new family, and secondary development handoff.",
    },
    {
        "topic": "reports",
        "paths": ["visualization", "analysis", "docs/en/framework-reference.md", "docs/zh/framework-reference.md"],
        "purpose": "HTML report, per-method attribution, bottleneck diagnostics, and framework reference docs.",
    },
]


VALIDATION_COMMANDS = [
    "python3 agent/cli.py framework-index --output /tmp/framework_index.json",
    "python3 agent/cli.py capabilities",
    "python3 tools/chain_adapters/cli.py validate-template --chain all",
    "python3 tools/fake-node/check_fixture_coverage.py --json",
    "python3 tools/fake-node/runtime_probe.py",
    "python3 -m unittest tests.test_agent_product_terminal tests.test_agent_runtime_contract",
]

DOC_INDEX = [
    {"topic": "quick_start_and_agent_usage", "en": "README.md", "zh": "README_ZH.md", "use_when": "Agent start, LLM config, ADC/API-key auth."},
    {"topic": "configuration_layer", "en": "config/README.md", "zh": "config/README.md", "use_when": "Required variables and config layering."},
    {"topic": "framework_flow", "en": "docs/en/framework-flow.md", "zh": "docs/zh/framework-flow.md", "use_when": "Benchmark lifecycle, reports, archive, observability."},
    {"topic": "module_guide", "en": "docs/en/module-guide.md", "zh": "docs/zh/module-guide.md", "use_when": "Module ownership and folder boundaries."},
    {"topic": "add_chain_or_rpc", "en": "docs/en/how-to-add-chain.md", "zh": "docs/zh/how-to-add-chain.md", "use_when": "New chain, RPC method, family, fixture, and param_spec."},
    {"topic": "closed_loop_fake_node", "en": "docs/en/local-closed-loop-testing.md", "zh": "docs/zh/local-closed-loop-testing.md", "use_when": "Local fake-node validation without real nodes."},
    {"topic": "secondary_development", "en": "docs/en/secondary-development-guide.md", "zh": "docs/zh/secondary-development-guide.md", "use_when": "KB, enterprise Agent platform, and extension handoff."},
    {"topic": "pr_quality_gates", "en": "docs/en/github-pr-gates.md", "zh": "docs/zh/github-pr-gates.md", "use_when": "PR, CI, quality gates, and review requirements."},
]


def build_framework_index(root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """Build a deterministic knowledge index from current repository files."""
    root = Path(root)
    capabilities = load_framework_capabilities(root)
    chains = []
    for row in capabilities.get("chains", []):
        chains.append({
            "chain": row.get("chain", ""),
            "adapter_family": row.get("family", ""),
            "single": row.get("single", ""),
            "mixed_methods": row.get("mixed_methods", []),
            "mixed_weighted": row.get("mixed_weighted", []),
            "methods": row.get("methods", []),
            "method_count": row.get("method_count", 0),
            "sync_health_mode": row.get("sync_health_mode", ""),
            "has_proxy_extraction": row.get("has_proxy_extraction", False),
            "template_path": f"config/chains/{row.get('chain', '')}.json",
        })

    fixture_by_chain = {
        item["chain"]: item["fixture_count"]
        for item in capabilities.get("fake_node", {}).get("chains", [])
    }
    for row in chains:
        row["fake_node_fixture_count"] = fixture_by_chain.get(row["chain"], 0)

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "agent.knowledge.framework_index",
        "summary": {
            "chain_count": capabilities.get("chain_count", 0),
            "family_count": capabilities.get("family_count", 0),
            "families": capabilities.get("families", {}),
            "unique_rpc_method_count": capabilities.get("unique_rpc_method_count", 0),
            "configured_rpc_method_entries": capabilities.get("configured_rpc_method_entries", 0),
            "fake_node_fixture_file_count": capabilities.get("fake_node", {}).get("fixture_file_count", 0),
        },
        "chains": chains,
        "authoritative_docs": _docs(root),
        "key_code_paths": _paths(root),
        "extension_boundaries": [
            "Existing-family chains should prefer config-only changes when param formats and request envelopes already fit.",
            "New protocol families require adapter, fake-node handler/config, sync-health, proxy extraction, and report-impact design.",
            "Custom RPC methods require param contract, fake-node fixture, proxy attribution, mixed-weight validation, and report validation.",
            "Docs must be updated with user-visible behavior because docs are part of the Agent knowledge source.",
        ],
        "validation_commands": VALIDATION_COMMANDS,
    }


def load_or_build_framework_index(root: str | Path = REPO_ROOT, index_path: str | Path | None = None) -> dict[str, Any]:
    """Load a local index when present, otherwise build from repository facts."""
    path = Path(index_path) if index_path else DEFAULT_INDEX_PATH
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return build_framework_index(root)


def write_framework_index(output: str | Path = DEFAULT_INDEX_PATH, root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """Build and write the framework index JSON."""
    path = Path(output)
    payload = build_framework_index(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _docs(root: Path) -> list[dict[str, Any]]:
    rows = []
    for item in DOC_INDEX:
        rows.append({
            "topic": item["topic"],
            "en": item.get("en", ""),
            "zh": item.get("zh", ""),
            "en_exists": bool((root / item.get("en", "")).exists()),
            "zh_exists": bool((root / item.get("zh", "")).exists()),
            "use_when": item.get("use_when", ""),
        })
    return rows


def _paths(root: Path) -> list[dict[str, Any]]:
    rows = []
    for item in KEY_CODE_PATHS:
        rows.append({
            "topic": item["topic"],
            "paths": item["paths"],
            "existing_paths": [path for path in item["paths"] if (root / path).exists()],
            "purpose": item["purpose"],
        })
    return rows

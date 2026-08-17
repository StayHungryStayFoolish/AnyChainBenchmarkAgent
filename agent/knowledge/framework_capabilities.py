"""Dynamic framework capability inventory from the local repository."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_framework_capabilities(root: str | Path = REPO_ROOT) -> dict[str, Any]:
    root = Path(root)
    chains_dir = root / "config" / "chains"
    fixture_dir = root / "tools" / "fake-node" / "fixtures"
    chains = []
    family_counts: Counter[str] = Counter()
    all_methods: set[str] = set()

    for path in sorted(chains_dir.glob("*.json")):
        data = _read_json(path)
        name = path.stem
        family = data.get("_meta", {}).get("adapter_family", "unknown")
        methods = _extract_rpc_methods(data.get("rpc_methods", {}))
        family_counts[family] += 1
        all_methods.update(methods)
        chains.append({
            "chain": name,
            "family": family,
            "single": _single_method(data.get("rpc_methods", {})),
            "mixed_methods": _mixed_methods(data.get("rpc_methods", {})),
            "mixed_weighted": _mixed_weighted(data.get("rpc_methods", {})),
            "method_count": len(methods),
            "methods": sorted(methods),
            "sync_health_mode": data.get("_meta", {}).get("sync_health", {}).get("mode", ""),
            "has_health_probe": bool(data.get("_meta", {}).get("health_probe")),
            "has_proxy_extraction": bool(data.get("proxy_extraction")),
        })

    fixture_chains = _fixture_summary(fixture_dir)
    client_metric_profiles = _client_metric_profiles(root / "config" / "client_metrics")
    return {
        "chain_count": len(chains),
        "family_count": len(family_counts),
        "families": dict(sorted(family_counts.items())),
        "unique_rpc_method_count": len(all_methods),
        "configured_rpc_method_entries": sum(chain["method_count"] for chain in chains),
        "chains": chains,
        "fake_node": {
            "fixture_chain_count": len(fixture_chains),
            "fixture_file_count": sum(item["fixture_count"] for item in fixture_chains),
            "chains": fixture_chains,
        },
        "client_metric_profiles": client_metric_profiles,
        "extension_points": [
            "config/chains/<chain>.json chain template",
            "rpc_methods.single and rpc_methods.mixed_weighted workload configuration",
            "param_formats and optional param_spec for method params",
            "_meta.adapter_family for protocol family routing",
            "proxy_extraction for per-method attribution",
            "tools/fake-node/fixtures/<chain>/ for local closed-loop responses",
        ],
        "run_modes": [
            {
                "id": "rpc_benchmark",
                "entrypoint": "./blockchain_node_benchmark.sh --quick|--standard|--intensive",
                "purpose": "Generate RPC workload, run Vegeta through the RPC proxy, monitor resources, analyze per-method latency/errors, and archive reports.",
                "requires": ["chain", "target mode", "RPC mode", "workload", "QPS profile", "resource metadata"],
            },
            {
                "id": "sync_observe",
                "entrypoint": "./blockchain_node_benchmark.sh --sync-observe",
                "purpose": "Observe node sync progress and runtime resources without RPC workload, proxy, Vegeta, or QPS ramp.",
                "stop_conditions": ["until stopped", "fixed duration", "until synced"],
                "requires": ["chain", "sync-health target/reference decision", "resource metadata", "node process identity"],
                "optional": ["NODE_PROMETHEUS_METRICS_URL for MGas/s or client-native execution metrics; BSC v1.7.x adds native import/finality/TPS report KPIs"],
            },
        ],
    }


def _client_metric_profiles(profile_dir: Path) -> list[dict[str, Any]]:
    """Project structured client-metric profiles into read-only product facts."""

    profiles: list[dict[str, Any]] = []
    if not profile_dir.is_dir():
        return profiles
    for path in sorted(profile_dir.glob("*.json")):
        data = _read_json(path)
        profile_id = str(data.get("profile_id") or "").strip()
        chains = [
            str(item).strip()
            for item in data.get("chains") or []
            if str(item).strip()
        ]
        fields = data.get("fields")
        upstream = data.get("upstream")
        capability = data.get("capability")
        if (
            not profile_id
            or not chains
            or not isinstance(fields, dict)
            or not isinstance(upstream, dict)
            or not isinstance(capability, dict)
        ):
            continue
        native_samples = sorted(
            {
                sample
                for binding in fields.values()
                if isinstance(binding, dict)
                for sample in (_prometheus_sample_identity(binding),)
                if sample
            }
        )
        profiles.append(
            {
                "profile_id": profile_id,
                "display_name": str(capability.get("display_name") or profile_id),
                "chains": chains,
                "metrics_path": str(upstream.get("metrics_path") or ""),
                "native_samples": native_samples,
                "report_kpis": [
                    str(item).strip()
                    for item in capability.get("report_kpis") or []
                    if str(item).strip()
                ],
                "requirements": _localized_capability_rows(
                    capability.get("requirements")
                ),
                "missing_endpoint": _localized_capability_policy(
                    capability.get("missing_endpoint")
                ),
            }
        )
    return profiles


def _localized_capability_policy(value: Any) -> dict[str, Any]:
    """Project one bilingual capability policy without inventing defaults."""

    if not isinstance(value, dict):
        return {}
    return {
        "workflow_continues": value.get("workflow_continues") is True,
        "shared_timeline_available": value.get("shared_timeline_available") is True,
        "en": str(value.get("en") or "").strip(),
        "zh": str(value.get("zh") or "").strip(),
    }


def _localized_capability_rows(value: Any) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in value or []:
        if not isinstance(item, dict):
            continue
        row = {
            key: str(item.get(key) or "").strip()
            for key in ("id", "en", "zh")
        }
        if all(row.values()):
            rows.append(row)
    return rows


def _prometheus_sample_identity(binding: dict[str, Any]) -> str:
    metric = str(binding.get("metric") or "").strip()
    if not metric:
        return ""
    labels = binding.get("labels")
    if not isinstance(labels, dict) or not labels:
        return metric
    rendered = ",".join(
        f"{key}={json.dumps(str(value), ensure_ascii=True)}"
        for key, value in sorted(labels.items())
    )
    return f"{metric}{{{rendered}}}"


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_rpc_methods(rpc_methods: dict[str, Any]) -> set[str]:
    methods = set()
    single = _single_method(rpc_methods)
    if single:
        methods.add(single)
    methods.update(_mixed_methods(rpc_methods))
    methods.update(item["method"] for item in _mixed_weighted(rpc_methods) if item.get("method"))
    return methods


def _single_method(rpc_methods: dict[str, Any]) -> str:
    value = rpc_methods.get("single", "")
    if isinstance(value, str):
        return value.strip()
    return ""


def _mixed_methods(rpc_methods: dict[str, Any]) -> list[str]:
    value = rpc_methods.get("mixed", [])
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _mixed_weighted(rpc_methods: dict[str, Any]) -> list[dict[str, Any]]:
    value = rpc_methods.get("mixed_weighted", [])
    if not isinstance(value, list):
        return []
    output = []
    for item in value:
        if not isinstance(item, dict):
            continue
        method = str(item.get("method", "")).strip()
        if not method:
            continue
        output.append({"method": method, "weight": item.get("weight", 0)})
    return output


def _fixture_summary(fixture_dir: Path) -> list[dict[str, Any]]:
    if not fixture_dir.is_dir():
        return []
    rows = []
    for child in sorted(fixture_dir.iterdir()):
        if child.is_dir():
            rows.append({"chain": child.name, "fixture_count": len(list(child.glob("*.json")))})
    rows.sort(key=lambda item: (-item["fixture_count"], item["chain"]))
    return rows

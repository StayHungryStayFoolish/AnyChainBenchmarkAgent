"""Benchmark configuration blocker matrices."""

from __future__ import annotations

from typing import Any

from knowledge.entry_contract import REAL_NODE_ENDPOINT_FIELDS, runtime_baseline_keys

COMMON_BLOCKERS = (
    "chain",
    "use_fake_node",
    "rpc_mode",
    "benchmark_mode_confirmed",
    "qps_profile_confirmed",
    "observability_choice_confirmed",
    "chain_template_reviewed",
    "rpc_workload_confirmed",
    "rpc_param_samples_confirmed",
)
ENVIRONMENT_BLOCKERS = runtime_baseline_keys()
FAKE_NODE_BLOCKERS: tuple[str, ...] = ()
REAL_NODE_ENDPOINT_BLOCKERS = tuple(field.key for field in REAL_NODE_ENDPOINT_FIELDS)
REAL_NODE_BLOCKERS = (*REAL_NODE_ENDPOINT_BLOCKERS, *ENVIRONMENT_BLOCKERS)


def missing_smoke_blockers(values: dict[str, Any]) -> list[str]:
    blockers = list(COMMON_BLOCKERS)
    if values.get("use_fake_node") is True:
        blockers.extend(FAKE_NODE_BLOCKERS)
        blockers.extend(ENVIRONMENT_BLOCKERS)
    elif values.get("use_fake_node") is False:
        blockers.extend(REAL_NODE_BLOCKERS)
    elif "use_fake_node" not in values or _is_missing(values.get("use_fake_node")):
        blockers.append("use_fake_node")
    if values.get("rpc_mode") == "mixed":
        blockers.append("mixed_weights_confirmed")
    return [key for key in blockers if key not in values or _is_missing(values.get(key))]


def _is_missing(value: Any) -> bool:
    if value is None or value is False:
        return True
    if isinstance(value, str):
        return value == ""
    if isinstance(value, (list, tuple, set, dict)):
        return not value
    return False

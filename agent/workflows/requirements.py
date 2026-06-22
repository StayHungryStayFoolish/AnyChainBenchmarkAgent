"""Benchmark configuration blocker matrices."""

from __future__ import annotations

from typing import Any


COMMON_BLOCKERS = ("chain", "rpc_mode", "rpc_workload_confirmed", "rpc_param_samples_confirmed")
FAKE_NODE_BLOCKERS = ("use_fake_node",)
REAL_NODE_BLOCKERS = (
    "local_rpc_url",
    "blockchain_process_names",
    "ledger_device",
    "data_vol_type",
    "data_vol_size",
    "data_vol_max_iops",
    "data_vol_max_throughput",
    "network_interface",
    "network_max_bandwidth_gbps",
)


def missing_smoke_blockers(values: dict[str, Any]) -> list[str]:
    blockers = list(COMMON_BLOCKERS)
    if values.get("use_fake_node") is True:
        blockers.extend(FAKE_NODE_BLOCKERS)
    elif values.get("use_fake_node") is False:
        blockers.extend(REAL_NODE_BLOCKERS)
    else:
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

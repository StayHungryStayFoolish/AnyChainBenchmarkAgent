#!/usr/bin/env python3
"""BSC client-native window analytics for sync-observe reports."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional

import pandas as pd


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").replace(
        [float("inf"), float("-inf")], float("nan")
    )


def _finite_mean(values: Iterable[object]) -> Optional[float]:
    series = pd.to_numeric(pd.Series(list(values), dtype="object"), errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.mean())


def _finite_median(values: Iterable[object]) -> Optional[float]:
    series = pd.to_numeric(pd.Series(list(values), dtype="object"), errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.median())


def _nearest_rank(values: Iterable[object], percentile: float) -> Optional[float]:
    series = sorted(
        float(value)
        for value in pd.to_numeric(pd.Series(list(values), dtype="object"), errors="coerce").dropna()
    )
    if not series:
        return None
    rank = max(1, math.ceil(percentile * len(series)))
    return series[min(rank - 1, len(series) - 1)]


def _base_result() -> Dict[str, Any]:
    return {
        "available": False,
        "profile": "none",
        "quality": "unavailable",
        "duration_seconds": None,
        "block_delta": None,
        "block_rate": None,
        "observed_imported_blocks": 0,
        "sample_coverage_pct": None,
        "transactions": None,
        "transactions_estimated": False,
        "tps": None,
        "avg_tx_per_block": None,
        "empty_block_rate_pct": None,
        "avg_block_gas_used_mgas": None,
        "block_gas_used_per_sec_mgas": None,
        "avg_gas_per_tx": None,
        "block_insert_ms_p50": None,
        "import_mgas_per_sec_p50": None,
        "justified_lag_p50": None,
        "justified_lag_p90": None,
        "justified_lag_p99": None,
        "finalized_lag_p50": None,
        "finalized_lag_p90": None,
        "finalized_lag_p99": None,
        "process_cpu_avg_pct": None,
        "process_memory_rss_avg_gib": None,
        "process_memory_avg_pct": None,
        "memory_used_avg_gib": None,
        "memory_usage_avg_pct": None,
    }


def calculate_bsc_sync_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate auditable BSC KPIs from one time-aligned observation window."""
    result = _base_result()
    if df is None or df.empty or "client_metric_profile" not in df.columns:
        return result

    profile_mask = df["client_metric_profile"].astype(str).str.strip().str.lower() == "bsc_v1_7"
    frame = df.loc[profile_mask].copy()
    if frame.empty or "timestamp" not in frame.columns:
        return result

    frame["_timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["_head"] = _numeric(frame, "client_head_block")
    frame = frame.dropna(subset=["_timestamp", "_head"]).sort_values("_timestamp")
    if frame.empty:
        return result

    result.update({"available": True, "profile": "bsc_v1_7", "quality": "partial"})
    duration = (frame["_timestamp"].iloc[-1] - frame["_timestamp"].iloc[0]).total_seconds()
    head_delta = max(float(frame["_head"].iloc[-1] - frame["_head"].iloc[0]), 0.0)
    if duration > 0:
        result["duration_seconds"] = float(duration)
    result["block_delta"] = head_delta
    if duration > 0:
        result["block_rate"] = head_delta / duration

    per_head = frame.drop_duplicates(subset=["_head"], keep="last")
    imported = per_head.iloc[1:].copy()
    observed_imported = len(imported)
    result["observed_imported_blocks"] = observed_imported
    if head_delta > 0:
        result["sample_coverage_pct"] = min(observed_imported / head_delta * 100.0, 100.0)

    tx_values = _numeric(imported, "client_block_tx_count").dropna()
    gas_values = _numeric(imported, "client_block_gas_used").dropna()
    result["avg_tx_per_block"] = _finite_mean(tx_values)
    if not tx_values.empty:
        result["empty_block_rate_pct"] = float(tx_values.eq(0).mean() * 100.0)
    avg_gas = _finite_mean(gas_values)
    result["avg_block_gas_used_mgas"] = avg_gas / 1_000_000.0 if avg_gas is not None else None

    complete_coverage = bool(head_delta > 0 and observed_imported >= head_delta)
    if head_delta > 0 and result["avg_tx_per_block"] is not None:
        if complete_coverage and len(tx_values) == observed_imported:
            transactions = float(tx_values.sum())
        else:
            transactions = float(result["avg_tx_per_block"] * head_delta)
            result["transactions_estimated"] = True
        result["transactions"] = transactions
        if duration > 0:
            result["tps"] = transactions / duration

    if avg_gas is not None and duration > 0 and head_delta > 0:
        total_gas = float(gas_values.sum()) if complete_coverage and len(gas_values) == observed_imported else avg_gas * head_delta
        result["block_gas_used_per_sec_mgas"] = total_gas / duration / 1_000_000.0

    paired = imported.assign(
        _tx=_numeric(imported, "client_block_tx_count"),
        _gas=_numeric(imported, "client_block_gas_used"),
    ).dropna(subset=["_tx", "_gas"])
    total_transactions = float(paired["_tx"].sum()) if not paired.empty else 0.0
    if total_transactions > 0:
        result["avg_gas_per_tx"] = float(paired["_gas"].sum() / total_transactions)

    result["block_insert_ms_p50"] = _finite_median(_numeric(imported, "client_block_insert_ms_p50"))
    result["import_mgas_per_sec_p50"] = _finite_median(_numeric(imported, "client_import_mgas_per_sec_p50"))

    justified = _numeric(per_head, "client_justified_block")
    finalized = _numeric(per_head, "client_finalized_block")
    justified_lag = (per_head["_head"] - justified).clip(lower=0).dropna()
    finalized_lag = (per_head["_head"] - finalized).clip(lower=0).dropna()
    for name, values in (("justified", justified_lag), ("finalized", finalized_lag)):
        result[f"{name}_lag_p50"] = _nearest_rank(values, 0.50)
        result[f"{name}_lag_p90"] = _nearest_rank(values, 0.90)
        result[f"{name}_lag_p99"] = _nearest_rank(values, 0.99)

    result["process_cpu_avg_pct"] = _finite_mean(_numeric(frame, "node_process_cpu_pct"))
    process_rss_mib = _finite_mean(_numeric(frame, "node_process_rss_mib"))
    result["process_memory_rss_avg_gib"] = process_rss_mib / 1024.0 if process_rss_mib is not None else None
    result["process_memory_avg_pct"] = _finite_mean(_numeric(frame, "node_process_memory_pct"))
    memory_mb = _finite_mean(_numeric(frame, "mem_used"))
    result["memory_used_avg_gib"] = memory_mb / 1024.0 if memory_mb is not None else None
    result["memory_usage_avg_pct"] = _finite_mean(_numeric(frame, "mem_usage"))

    required = (
        result["block_rate"],
        result["avg_tx_per_block"],
        result["avg_block_gas_used_mgas"],
        result["block_insert_ms_p50"],
        result["import_mgas_per_sec_p50"],
    )
    if all(value is not None for value in required):
        result["quality"] = "complete" if complete_coverage else "estimated"
    return result

"""Canonical benchmark-mode QPS profiles shared by planning and the Harness."""

from __future__ import annotations

from typing import Mapping


QPS_PROFILE_DEFAULTS: Mapping[str, Mapping[str, int]] = {
    "quick": {
        "INITIAL_QPS": 1000,
        "MAX_QPS": 1500,
        "QPS_STEP": 500,
        "DURATION": 60,
    },
    "standard": {
        "INITIAL_QPS": 2000,
        "MAX_QPS": 50000,
        "QPS_STEP": 500,
        "DURATION": 600,
    },
    "intensive": {
        "INITIAL_QPS": 50000,
        "MAX_QPS": 9999999,
        "QPS_STEP": 250,
        "DURATION": 600,
    },
}

STRATEGY_BENCHMARK_MODE: Mapping[str, str] = {
    "smoke": "quick",
    "baseline": "standard",
    "ramp": "standard",
    "stress": "intensive",
    "bottleneck-confirmation": "standard",
    "regression": "standard",
}


def qps_profile_defaults(mode: str) -> dict[str, str]:
    """Return the selected profile as runtime environment strings."""

    normalized = str(mode or "").strip().lower()
    selected = QPS_PROFILE_DEFAULTS.get(
        normalized,
        QPS_PROFILE_DEFAULTS["quick"],
    )
    return {key: str(value) for key, value in selected.items()}


def strategy_qps_defaults(strategy: str) -> dict[str, int]:
    """Return plan-generator keys derived from the same profile authority."""

    mode = STRATEGY_BENCHMARK_MODE[strategy]
    selected = QPS_PROFILE_DEFAULTS[mode]
    return {
        "initial": selected["INITIAL_QPS"],
        "max": selected["MAX_QPS"],
        "step": selected["QPS_STEP"],
        "duration_seconds": selected["DURATION"],
    }

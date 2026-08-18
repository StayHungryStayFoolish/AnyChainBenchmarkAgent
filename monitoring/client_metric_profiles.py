#!/usr/bin/env python3
"""Client-native Prometheus metric contracts for sync observation."""

from __future__ import annotations

import math
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional


@dataclass(frozen=True)
class PrometheusSample:
    name: str
    labels: Mapping[str, str]
    value: float


_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"\s*(?:\{(?P<labels>.*)\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?(?:Inf|NaN))"
    r"(?:\s+\d+)?\s*$"
)
_LABEL_RE = re.compile(
    r'(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"(?P<value>(?:\\.|[^"\\])*)"'
)


def _parse_labels(raw: str) -> Optional[Dict[str, str]]:
    if not raw.strip():
        return {}
    labels: Dict[str, str] = {}
    position = 0
    while position < len(raw):
        match = _LABEL_RE.match(raw, position)
        if not match:
            return None
        key = match.group("key")
        encoded_value = match.group("value")
        value_parts: List[str] = []
        index = 0
        while index < len(encoded_value):
            character = encoded_value[index]
            if character != "\\":
                value_parts.append(character)
                index += 1
                continue
            if index + 1 >= len(encoded_value):
                return None
            escaped = encoded_value[index + 1]
            replacements = {"\\": "\\", '"': '"', "n": "\n"}
            if escaped not in replacements:
                return None
            value_parts.append(replacements[escaped])
            index += 2
        value = "".join(value_parts)
        if key in labels:
            return None
        labels[key] = value
        position = match.end()
        while position < len(raw) and raw[position].isspace():
            position += 1
        if position == len(raw):
            break
        if raw[position] != ",":
            return None
        position += 1
        while position < len(raw) and raw[position].isspace():
            position += 1
    return labels


def parse_prometheus_samples(metrics_text: str) -> List[PrometheusSample]:
    """Parse finite Prometheus text samples without conflating labels or types."""
    samples: List[PrometheusSample] = []
    for raw_line in metrics_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE_RE.match(line)
        if not match:
            continue
        labels = _parse_labels(match.group("labels") or "")
        if labels is None:
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        samples.append(PrometheusSample(match.group("name"), labels, value))
    return samples


def select_sample(
    samples: Iterable[PrometheusSample],
    name: str,
    labels: Optional[Mapping[str, str]] = None,
) -> Optional[float]:
    """Return one exact sample; missing or duplicate samples are ambiguous."""
    expected_labels = dict(labels or {})
    matches = [
        sample.value
        for sample in samples
        if sample.name == name and dict(sample.labels) == expected_labels
    ]
    return matches[0] if len(matches) == 1 else None


CLIENT_METRIC_FIELDS = (
    "client_metric_profile",
    "client_block_insert_ms_p50",
    "client_import_mgas_per_sec_p50",
    "client_import_observation_count",
    "client_block_tx_count",
    "client_block_gas_used",
    "client_head_block",
    "client_justified_block",
    "client_finalized_block",
    "client_inserted_blocks_count",
    "client_metric_quality",
)

_PROFILE_DIR = Path(__file__).resolve().parents[1] / "config" / "client_metrics"


@lru_cache(maxsize=1)
def _load_profiles() -> List[Dict[str, object]]:
    profiles: List[Dict[str, object]] = []
    for path in sorted(_PROFILE_DIR.glob("*.json")):
        try:
            profile = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        fields = profile.get("fields")
        chains = profile.get("chains")
        if not isinstance(fields, dict) or not isinstance(chains, list):
            continue
        if set(fields) != set(CLIENT_METRIC_FIELDS[1:-1]):
            continue
        profiles.append(profile)
    return profiles


def _profile_for_chain(chain: str) -> Optional[Dict[str, object]]:
    canonical = str(chain or "").strip().lower()
    matches = [
        profile
        for profile in _load_profiles()
        if canonical in {str(item).strip().lower() for item in profile.get("chains", [])}
    ]
    return matches[0] if len(matches) == 1 else None


def _empty_result(profile: str, quality: str) -> Dict[str, object]:
    result: Dict[str, object] = {field: None for field in CLIENT_METRIC_FIELDS}
    result.update(
        {
            "client_metric_profile": profile,
            "client_metric_quality": quality,
            "execution_mgas_per_sec": None,
            "execution_gas_per_sec": None,
            "execution_metric_source": "no_supported_metric",
            "execution_metric_status": "unavailable",
        }
    )
    return result


def collect_client_metrics(chain: str, metrics_text: str) -> Dict[str, object]:
    """Collect profile-neutral fields from one client-native metrics scrape."""
    profile = _profile_for_chain(chain)
    if profile is None:
        return _empty_result("none", "unsupported")

    result = _empty_result(str(profile.get("profile_id") or "unknown"), "unavailable")
    samples = parse_prometheus_samples(metrics_text)
    present = 0
    fields = profile["fields"]
    for field, binding in fields.items():
        if not isinstance(binding, dict):
            continue
        name = str(binding.get("metric") or "")
        labels = binding.get("labels") if isinstance(binding.get("labels"), dict) else {}
        scale = float(binding.get("scale", 1.0))
        value = select_sample(samples, name, labels)
        if value is None:
            continue
        result[field] = value * scale
        present += 1

    if present == len(fields):
        result["client_metric_quality"] = "complete"
    elif present:
        result["client_metric_quality"] = "partial"

    execution_field = str(profile.get("execution_source") or "")
    mgas = result.get(execution_field)
    if isinstance(mgas, (int, float)):
        result.update(
            {
                "execution_mgas_per_sec": float(mgas),
                "execution_gas_per_sec": float(mgas) * 1_000_000.0,
                "execution_metric_source": str(profile.get("execution_source_label") or execution_field),
                "execution_metric_status": "available",
            }
        )
    return result

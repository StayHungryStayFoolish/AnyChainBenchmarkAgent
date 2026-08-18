"""Canonical sync-observe request and readiness contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


REAL_SYNC_SOURCES = frozenset({"existing_local_node", "endpoint_only"})
SYNC_STOP_CONDITIONS = frozenset({"until_stopped", "duration", "until_synced"})


def sync_observe_tool_properties() -> dict[str, dict[str, Any]]:
    """Return public tool fields generated from the canonical request."""

    return {
        "workflow_type": {
            "type": "string",
            "enum": ["rpc_benchmark", "sync_observe"],
            "description": "Select the RPC benchmark or sync-observe workflow.",
        },
        "sync_observe_source": {
            "type": "string",
            "enum": ["existing_local_node", "endpoint_only", "client_setup"],
            "description": "Real sync-observe source; client_setup is guidance only.",
        },
        "sync_observe_rpc_url": {
            "type": "string",
            "description": "Validated real-node RPC URL used for sync observation.",
        },
        "sync_observe_rpc_url_ready": {
            "type": "boolean",
            "description": "Whether endpoint validation evidence exists for the RPC URL.",
        },
        "node_process_identity": {
            "type": "string",
            "description": "Local node process name/PID used for process attribution.",
        },
        "node_prometheus_metrics_url": {
            "type": "string",
            "description": "Optional node Prometheus metrics endpoint.",
        },
        "mainnet_rpc_url_reviewed": {
            "type": "boolean",
            "description": "Whether sync-health or MAINNET_RPC_URL behavior was reviewed.",
        },
        "sync_observe_stop_condition": {
            "type": "string",
            "enum": sorted(SYNC_STOP_CONDITIONS),
            "description": "Stop on Ctrl+C, after a duration, or after sync completes.",
        },
        "sync_observe_duration_seconds": {
            "type": "integer",
            "minimum": 1,
            "description": "Positive duration required only for duration stop condition.",
        },
    }


@dataclass(frozen=True)
class SyncObserveBlocker:
    group: str
    reason: str
    question_id: str


@dataclass(frozen=True)
class SyncObserveRequest:
    source: str = ""
    rpc_url: str = ""
    rpc_url_ready: bool = False
    process_name: str = ""
    mainnet_rpc_reviewed: bool = False
    stop_condition: str = ""
    duration_seconds: int | None = None
    metrics_url: str = ""

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "SyncObserveRequest":
        sync = state.get("sync_observe") if isinstance(state.get("sync_observe"), Mapping) else {}
        confirmed = state.get("confirmed_config") if isinstance(state.get("confirmed_config"), Mapping) else {}
        evidence = state.get("endpoint_evidence") if isinstance(state.get("endpoint_evidence"), Mapping) else {}
        return cls(
            source=str(sync.get("source") or "").strip(),
            rpc_url=str(confirmed.get("SYNC_OBSERVE_RPC_URL") or "").strip(),
            rpc_url_ready=bool(evidence.get("sync_rpc_url_ready")),
            process_name=str(confirmed.get("BLOCKCHAIN_PROCESS_NAMES") or "").strip(),
            mainnet_rpc_reviewed=bool(confirmed.get("MAINNET_RPC_URL_REVIEWED")),
            stop_condition=str(sync.get("stop_condition") or "").strip(),
            duration_seconds=_positive_int_or_none(sync.get("duration_seconds")),
            metrics_url=str(confirmed.get("NODE_PROMETHEUS_METRICS_URL") or "").strip(),
        )

    @classmethod
    def from_request_values(cls, values: Mapping[str, Any]) -> "SyncObserveRequest":
        rpc_url = str(values.get("sync_observe_rpc_url") or values.get("SYNC_OBSERVE_RPC_URL") or "").strip()
        return cls(
            source=str(values.get("sync_observe_source") or "").strip(),
            rpc_url=rpc_url,
            rpc_url_ready=bool(values.get("sync_observe_rpc_url_ready", rpc_url)),
            process_name=str(values.get("node_process_identity") or values.get("BLOCKCHAIN_PROCESS_NAMES") or "").strip(),
            mainnet_rpc_reviewed=bool(values.get("mainnet_rpc_url_reviewed") or values.get("MAINNET_RPC_URL_REVIEWED")),
            stop_condition=str(values.get("sync_observe_stop_condition") or "").strip(),
            duration_seconds=_positive_int_or_none(values.get("sync_observe_duration_seconds")),
            metrics_url=str(values.get("node_prometheus_metrics_url") or values.get("NODE_PROMETHEUS_METRICS_URL") or "").strip(),
        )

    def blocker(self) -> SyncObserveBlocker | None:
        if not self.source:
            return SyncObserveBlocker("sync_observe", "choose sync-observe data source", "sync_observe_source")
        if self.source == "client_setup":
            return SyncObserveBlocker(
                "sync_observe",
                "choose a real sync-observe source after client setup guidance",
                "sync_observe_after_client_setup",
            )
        if self.source not in REAL_SYNC_SOURCES:
            return SyncObserveBlocker("sync_observe", "choose sync-observe data source", "sync_observe_source")
        if not self.rpc_url_ready or not self.rpc_url:
            return SyncObserveBlocker("endpoint_process", "validate real sync-observe RPC endpoint", "SYNC_OBSERVE_RPC_URL")
        if self.source == "existing_local_node" and not self.process_name:
            return SyncObserveBlocker(
                "endpoint_process",
                "confirm node process for sync-observe attribution",
                "BLOCKCHAIN_PROCESS_NAMES",
            )
        if not self.mainnet_rpc_reviewed:
            return SyncObserveBlocker(
                "endpoint_process",
                "confirm sync-health / MAINNET_RPC_URL behavior",
                "MAINNET_RPC_URL_REVIEWED",
            )
        if self.stop_condition not in SYNC_STOP_CONDITIONS:
            return SyncObserveBlocker(
                "sync_observe",
                "choose sync-observe stop condition",
                "sync_observe_stop_condition",
            )
        if self.stop_condition == "duration" and self.duration_seconds is None:
            return SyncObserveBlocker(
                "sync_observe",
                "confirm sync-observe duration",
                "sync_observe_duration_seconds",
            )
        return None

    def execution_values(self) -> dict[str, Any]:
        return {
            "sync_observe_rpc_url": self.rpc_url,
            "node_prometheus_metrics_url": self.metrics_url,
            "sync_observe_stop_condition": self.stop_condition,
            "sync_observe_duration_seconds": self.duration_seconds,
            "sync_observe_source": self.source,
            "mainnet_rpc_url_reviewed": self.mainnet_rpc_reviewed,
        }

    def request_values(self) -> dict[str, Any]:
        """Project the canonical request into planner/tool vocabulary."""

        return {
            **self.execution_values(),
            "sync_observe_rpc_url_ready": self.rpc_url_ready,
            "node_process_identity": self.process_name,
        }

    def checklist_presence(self) -> dict[str, bool]:
        return {
            "sync_observe_rpc_url": bool(self.rpc_url and self.rpc_url_ready),
            "sync_observe_stop_condition": self.stop_condition in SYNC_STOP_CONDITIONS,
            "sync_observe_duration_seconds": self.stop_condition != "duration" or self.duration_seconds is not None,
            "node_process_identity": self.source == "endpoint_only" or bool(self.process_name),
            "node_prometheus_metrics_url": bool(self.metrics_url),
            "mainnet_rpc_url_reviewed": self.mainnet_rpc_reviewed,
        }


def _positive_int_or_none(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None

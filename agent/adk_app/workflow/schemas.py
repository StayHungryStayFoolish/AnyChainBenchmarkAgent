"""Structured schemas for ADK workflow boundaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


IntentName = Literal[
    "START_BENCHMARK",
    "RESUME_JOB",
    "ANALYZE_ARTIFACTS",
    "ONBOARD_CHAIN_RPC",
    "CONFIG_HELP",
    "GENERAL_QA",
    "OUT_OF_SCOPE",
]

TargetType = Literal["fake-node", "real-node", "unknown"]


@dataclass(frozen=True)
class IntentEntities:
    chain: str = ""
    rpc_methods: list[str] = field(default_factory=list)
    rpc_mode: str = ""
    target: TargetType = "unknown"
    job_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IntentRoute:
    intent: IntentName
    confidence: float
    language: Literal["zh", "en"]
    entities: IntentEntities = field(default_factory=IntentEntities)
    missing_clarifications: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["confidence"] = round(max(0.0, min(1.0, float(self.confidence))), 3)
        return payload


@dataclass(frozen=True)
class WorkflowEvent:
    event_type: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    requires_user_input: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_intent_route(payload: dict[str, Any]) -> list[str]:
    """Return schema errors for an IntentRoute-like payload."""
    errors: list[str] = []
    if payload.get("intent") not in IntentName.__args__:  # type: ignore[attr-defined]
        errors.append("intent must be one of the supported route enum values")
    if payload.get("language") not in {"zh", "en"}:
        errors.append("language must be zh or en")
    try:
        confidence = float(payload.get("confidence"))
    except (TypeError, ValueError):
        errors.append("confidence must be numeric")
    else:
        if confidence < 0 or confidence > 1:
            errors.append("confidence must be between 0 and 1")
    entities = payload.get("entities", {})
    if not isinstance(entities, dict):
        errors.append("entities must be an object")
    else:
        if entities.get("target", "unknown") not in TargetType.__args__:  # type: ignore[attr-defined]
            errors.append("entities.target must be fake-node, real-node, or unknown")
        if not isinstance(entities.get("rpc_methods", []), list):
            errors.append("entities.rpc_methods must be an array")
    if not isinstance(payload.get("missing_clarifications", []), list):
        errors.append("missing_clarifications must be an array")
    return errors

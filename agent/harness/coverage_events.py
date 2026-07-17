"""Out-of-band observations for executable Harness coverage.

The product graph never depends on this module. Coverage runners may wrap the
same compiled-graph helper used by live tests and record what that invocation
actually received and returned. They cannot declare an edge successful.
"""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterator, Mapping


@dataclass(frozen=True)
class CoverageEvent:
    sequence: int
    event_type: str
    edge_key: str = ""
    component: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)
    monotonic_ns: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GraphTurnObservation:
    """Raw evidence captured around one compiled LangGraph invocation."""

    before: Mapping[str, Any]
    input_value: Any
    question: Mapping[str, Any]
    after: Mapping[str, Any]
    state_diff: Mapping[str, Any]
    response: tuple[Any, ...]
    next_question: Mapping[str, Any]


_ACTIVE_COLLECTOR: ContextVar[list[CoverageEvent] | None] = ContextVar(
    "anychain_harness_coverage_collector",
    default=None,
)


def emit_coverage_event(
    event_type: str,
    *,
    edge_key: str = "",
    component: str = "",
    details: Mapping[str, Any] | None = None,
) -> None:
    """Record an observation when a test collector is active."""

    collector = _ACTIVE_COLLECTOR.get()
    if collector is None:
        return
    collector.append(
        CoverageEvent(
            sequence=len(collector) + 1,
            event_type=str(event_type),
            edge_key=str(edge_key),
            component=str(component),
            details=deepcopy(dict(details or {})),
            monotonic_ns=time.monotonic_ns(),
        )
    )


@contextmanager
def capture_coverage_events() -> Iterator[list[CoverageEvent]]:
    """Capture observations for one isolated execution."""

    events: list[CoverageEvent] = []
    token = _ACTIVE_COLLECTOR.set(events)
    try:
        yield events
    finally:
        _ACTIVE_COLLECTOR.reset(token)


def observe_compiled_graph_turn(
    invoke: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    state: Mapping[str, Any],
    *,
    edge_key: str,
    input_value: Any,
    component: str = "tests.agent_live.graph_turn.invoke_product_graph_turn",
) -> GraphTurnObservation:
    """Invoke the real compiled-graph helper and observe its actual boundary."""

    callable_identity = f"{getattr(invoke, '__module__', '')}.{getattr(invoke, '__name__', '')}"
    if callable_identity != component:
        raise ValueError(
            f"coverage turn helper mismatch: expected {component}, got {callable_identity}"
        )
    before = deepcopy(dict(state))
    question = deepcopy(dict(before.get("pending_question") or {}))
    identity = {
        "before_hash": _content_hash(before),
        "input_hash": _content_hash(input_value),
        "question_hash": _content_hash(question),
    }
    emit_coverage_event(
        "compiled_graph_turn_started",
        edge_key=edge_key,
        component=component,
        details=identity,
    )
    try:
        after = deepcopy(dict(invoke(deepcopy(before))))
    except Exception as exc:
        emit_coverage_event(
            "compiled_graph_turn_raised",
            edge_key=edge_key,
            component=component,
            details={**identity, "error": f"{type(exc).__name__}: {exc}"},
        )
        raise

    response = tuple(deepcopy(list(after.get("visible_response") or [])))
    next_question = deepcopy(dict(after.get("pending_question") or {}))
    state_diff = state_diff_between(before, after)
    emit_coverage_event(
        "compiled_graph_turn_returned",
        edge_key=edge_key,
        component=component,
        details={
            **identity,
            "after_hash": _content_hash(after),
            "state_diff_hash": _content_hash(state_diff),
            "response_hash": _content_hash(list(response)),
            "next_question_hash": _content_hash(next_question),
        },
    )
    return GraphTurnObservation(
        before=before,
        input_value=deepcopy(input_value),
        question=question,
        after=after,
        state_diff=state_diff,
        response=response,
        next_question=next_question,
    )


def state_diff_between(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deterministic top-level diff with the actual values."""

    missing = {"missing": True}
    output: dict[str, Any] = {}
    for key in sorted(set(before) | set(after)):
        old = before[key] if key in before else missing
        new = after[key] if key in after else missing
        if old != new:
            output[str(key)] = {"before": deepcopy(old), "after": deepcopy(new)}
    return output


def _content_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

"""Versioned, non-secret terminal outcome projection protocol."""

from __future__ import annotations

import hashlib
import json
import os
import fcntl
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Mapping


TERMINAL_OUTCOME_PROJECTION_SCHEMA_VERSION = 3
TERMINAL_SESSION_EVENT_SCHEMA_VERSION = 3
TERMINAL_DETOUR_PROJECTION_SCHEMA_VERSION = 3
_OUTCOMES = frozenset({"committed", "aborted", "reconciliation_required"})
_DELIVERY_PHASES = frozenset({"live", "startup_replay"})
_STARTUP_STATUSES = frozenset({"ready", "blocked"})


class StartupFailureCategory(str, Enum):
    DEPENDENCY_CONSENT_PENDING = "dependency_consent_pending"
    PROVIDER_RUNTIME_UNAVAILABLE = "provider_runtime_unavailable"
    PROVIDER_READINESS_FAILED = "provider_readiness_failed"
    HARNESS_RUNTIME_UNAVAILABLE = "harness_runtime_unavailable"
    STARTUP_BLOCKED = "startup_blocked"


_STARTUP_FAILURE_CATEGORIES = frozenset(
    category.value
    for category in StartupFailureCategory
)
_DETOUR_EFFECT_CLASSES = frozenset({
    "read_only",
    "observation_refresh",
    "shell_state_mutation",
    "local_external_effect",
    "authority_resolution",
    "streaming_observation",
})
_PROJECTABLE_DETOUR_EFFECT_STATUSES = frozenset({
    "not_applicable",
    "succeeded",
    "failed",
})


class TerminalProtocolError(RuntimeError):
    """Raised when terminal evidence violates the shared protocol."""


@dataclass(frozen=True)
class TerminalOutcomeProjection:
    schema_version: int
    record_type: str
    projection_id: str
    event_id: str
    transaction_id: str
    product_authority_id: str
    logical_thread_id: str
    physical_thread_id: str
    outcome: str
    failure_category: str
    diagnostic_hash: str
    base_revision: int
    base_checkpoint_thread_id: str
    base_checkpoint_id: str
    base_fingerprint: str
    attempt_checkpoint_id: str
    attempt_fingerprint: str
    product_revision: int
    product_checkpoint_thread_id: str
    product_checkpoint_id: str
    product_fingerprint: str
    render_hash: str
    runtime_event_id: str
    runtime_event_sequence: int
    runtime_event_payload_hash: str
    presentation_hash: str
    origin_revision: Mapping[str, str]
    delivery_phase: Literal["live", "startup_replay"]
    projected_at: str
    record_hash: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TerminalSessionEvent:
    schema_version: int
    record_type: str
    session_event_id: str
    process_instance_id: str
    session_id: str
    session_purpose: str
    provider: str
    model: str
    auth_mode: str
    provider_ready: bool
    startup_status: Literal["ready", "blocked"]
    failure_category: str
    product_authority_id: str
    product_revision: int | None
    product_checkpoint_thread_id: str
    product_checkpoint_id: str
    product_fingerprint: str
    runtime_event_fence_sequence: int
    runtime_event_fence_terminal_event_id: str
    runtime_event_fence_id: str
    runtime_event_fence_hash: str
    presentation_hash: str
    replayed_projection_ids: tuple[str, ...]
    origin_revision: Mapping[str, str]
    projected_at: str
    record_hash: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TerminalDetourProjection:
    schema_version: int
    record_type: str
    projection_id: str
    detour_id: str
    product_authority_id: str
    logical_thread_id: str
    process_instance_id: str
    session_id: str
    session_purpose: str
    command_name: str
    input_hash: str
    result_kind: str
    interruption_kind: str
    termination_reason: str
    exit_code: int | None
    effect_class: str
    effect_status: str
    shell_state_before_hash: str
    shell_state_after_hash: str
    product_revision_before: int | None
    product_checkpoint_thread_id_before: str
    product_checkpoint_id_before: str
    product_fingerprint_before: str
    product_revision_after: int | None
    product_checkpoint_thread_id_after: str
    product_checkpoint_id_after: str
    product_fingerprint_after: str
    runtime_event_fence_sequence: int
    runtime_event_fence_terminal_event_id: str
    runtime_event_fence_id: str
    runtime_event_fence_hash: str
    response_hash: str
    stream_chunk_count: int
    stream_hash: str
    stream_stop_reason: str
    identity_trust: str
    presentation_hash: str
    origin_revision: Mapping[str, str]
    delivery_phase: Literal["live", "startup_replay"]
    projected_at: str
    record_hash: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def presentation_hash(rendered_frame: str) -> str:
    """Hash the normalized text actually emitted for one terminal frame."""

    normalized = str(rendered_frame).replace("\r\n", "\n").replace("\r", "\n").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_terminal_detour_projection(
    detour: Any,
    *,
    rendered_frame: str,
    delivery_phase: Literal["live", "startup_replay"],
) -> TerminalDetourProjection:
    if delivery_phase not in _DELIVERY_PHASES:
        raise TerminalProtocolError("terminal delivery phase is invalid")
    if str(detour.status) != "completed":
        raise TerminalProtocolError("only completed detours can be projected")
    frame_hash = presentation_hash(rendered_frame)
    projection_identity = _record_hash({
        "detour_id": str(detour.detour_id),
        "delivery_phase": delivery_phase,
        "presentation_hash": frame_hash,
    })
    payload: dict[str, Any] = {
        "schema_version": TERMINAL_DETOUR_PROJECTION_SCHEMA_VERSION,
        "record_type": "terminal_detour_projection",
        "projection_id": f"detour-projection:{projection_identity}",
        "detour_id": str(detour.detour_id),
        "product_authority_id": str(detour.logical_thread_id),
        "logical_thread_id": str(detour.logical_thread_id),
        "process_instance_id": str(detour.process_instance_id),
        "session_id": str(detour.session_id),
        "session_purpose": str(detour.session_purpose),
        "command_name": str(detour.command_name),
        "input_hash": str(detour.input_hash),
        "result_kind": str(detour.result_kind),
        "interruption_kind": str(detour.interruption_kind or ""),
        "termination_reason": str(detour.termination_reason or ""),
        "exit_code": detour.exit_code,
        "effect_class": str(detour.effect_class),
        "effect_status": str(detour.effect_status),
        "shell_state_before_hash": str(detour.shell_state_before_hash),
        "shell_state_after_hash": str(detour.shell_state_after_hash or ""),
        "product_revision_before": detour.product_revision_before,
        "product_checkpoint_thread_id_before": str(
            detour.product_checkpoint_thread_id_before or ""
        ),
        "product_checkpoint_id_before": str(
            detour.product_checkpoint_id_before or ""
        ),
        "product_fingerprint_before": str(
            detour.product_fingerprint_before or ""
        ),
        "product_revision_after": detour.product_revision_after,
        "product_checkpoint_thread_id_after": str(
            detour.product_checkpoint_thread_id_after or ""
        ),
        "product_checkpoint_id_after": str(
            detour.product_checkpoint_id_after or ""
        ),
        "product_fingerprint_after": str(
            detour.product_fingerprint_after or ""
        ),
        "runtime_event_fence_sequence": int(
            detour.runtime_event_fence_sequence
        ),
        "runtime_event_fence_terminal_event_id": str(
            detour.runtime_event_fence_terminal_event_id
        ),
        "runtime_event_fence_id": str(detour.runtime_event_fence_id),
        "runtime_event_fence_hash": str(detour.runtime_event_fence_hash),
        "response_hash": str(detour.response_hash or ""),
        "stream_chunk_count": int(detour.stream_chunk_count),
        "stream_hash": str(detour.stream_hash),
        "stream_stop_reason": str(detour.stream_stop_reason),
        "identity_trust": str(detour.identity_trust),
        "presentation_hash": frame_hash,
        "origin_revision": {
            "commit": str(detour.origin_revision_commit),
            "worktree_hash": str(detour.origin_revision_worktree_hash),
        },
        "delivery_phase": delivery_phase,
        "projected_at": datetime.now(timezone.utc).isoformat(
            timespec="microseconds"
        ),
    }
    payload["record_hash"] = _record_hash(payload)
    return validate_terminal_detour_projection(payload)


def validate_terminal_detour_projection(
    raw: Mapping[str, Any],
) -> TerminalDetourProjection:
    required = {
        "schema_version",
        "record_type",
        "projection_id",
        "detour_id",
        "product_authority_id",
        "logical_thread_id",
        "process_instance_id",
        "session_id",
        "session_purpose",
        "command_name",
        "input_hash",
        "result_kind",
        "interruption_kind",
        "termination_reason",
        "exit_code",
        "effect_class",
        "effect_status",
        "shell_state_before_hash",
        "shell_state_after_hash",
        "product_revision_before",
        "product_checkpoint_thread_id_before",
        "product_checkpoint_id_before",
        "product_fingerprint_before",
        "product_revision_after",
        "product_checkpoint_thread_id_after",
        "product_checkpoint_id_after",
        "product_fingerprint_after",
        "runtime_event_fence_sequence",
        "runtime_event_fence_terminal_event_id",
        "runtime_event_fence_id",
        "runtime_event_fence_hash",
        "response_hash",
        "stream_chunk_count",
        "stream_hash",
        "stream_stop_reason",
        "identity_trust",
        "presentation_hash",
        "origin_revision",
        "delivery_phase",
        "projected_at",
        "record_hash",
    }
    if set(raw) != required:
        raise TerminalProtocolError("terminal detour projection schema is invalid")
    if (
        raw.get("schema_version") != TERMINAL_DETOUR_PROJECTION_SCHEMA_VERSION
        or raw.get("record_type") != "terminal_detour_projection"
    ):
        raise TerminalProtocolError(
            "terminal detour projection version is unsupported"
        )
    phase = str(raw.get("delivery_phase") or "")
    if phase not in _DELIVERY_PHASES:
        raise TerminalProtocolError("terminal detour delivery phase is invalid")
    for name in (
        "projection_id",
        "detour_id",
        "product_authority_id",
        "logical_thread_id",
        "process_instance_id",
        "session_id",
        "session_purpose",
        "command_name",
        "input_hash",
        "result_kind",
        "effect_class",
        "effect_status",
        "shell_state_before_hash",
        "shell_state_after_hash",
        "response_hash",
        "stream_hash",
        "presentation_hash",
        "projected_at",
        "record_hash",
    ):
        if not str(raw.get(name) or "").strip():
            raise TerminalProtocolError(
                f"terminal detour projection is missing {name}"
            )
    result_kind = str(raw.get("result_kind") or "")
    interruption_kind = str(raw.get("interruption_kind") or "")
    termination_reason = str(raw.get("termination_reason") or "")
    exit_code = raw.get("exit_code")
    if result_kind == "session_termination":
        if termination_reason not in {"exit", "eof", "outer_ctrl_c"}:
            raise TerminalProtocolError(
                "terminal detour termination reason is invalid"
            )
        if exit_code not in {0, 130}:
            raise TerminalProtocolError(
                "terminal detour termination exit code is invalid"
            )
    elif result_kind == "command":
        if interruption_kind == "session_termination_interrupted":
            if (
                termination_reason not in {"exit", "eof", "outer_ctrl_c"}
                or exit_code not in {0, 130}
            ):
                raise TerminalProtocolError(
                    "interrupted termination fields are invalid"
                )
        elif termination_reason or exit_code is not None:
            raise TerminalProtocolError(
                "command detour carries termination fields"
            )
    else:
        raise TerminalProtocolError("terminal detour result kind is invalid")
    for name in (
        "input_hash",
        "shell_state_before_hash",
        "shell_state_after_hash",
        "response_hash",
        "stream_hash",
        "presentation_hash",
        "record_hash",
    ):
        _require_hash(name, str(raw[name]))
    stream_chunk_count = raw.get("stream_chunk_count")
    if (
        not isinstance(stream_chunk_count, int)
        or isinstance(stream_chunk_count, bool)
        or stream_chunk_count < 0
    ):
        raise TerminalProtocolError(
            "terminal detour stream chunk count is invalid"
        )
    if str(raw.get("stream_stop_reason") or "") not in {
        "not_streaming",
        "completed",
        "cancelled",
        "deadline",
        "limit_reached",
        "not_found",
        "interrupted",
        "error",
    }:
        raise TerminalProtocolError("terminal detour stream stop reason is invalid")
    stream_stop_reason = str(raw.get("stream_stop_reason") or "")
    effect_class = str(raw.get("effect_class") or "")
    effect_status = str(raw.get("effect_status") or "")
    if effect_class not in _DETOUR_EFFECT_CLASSES:
        raise TerminalProtocolError(
            "terminal detour effect class is invalid"
        )
    if effect_status not in _PROJECTABLE_DETOUR_EFFECT_STATUSES:
        raise TerminalProtocolError(
            "terminal detour effect status is invalid"
        )
    if effect_class == "streaming_observation":
        if effect_status == "failed" and stream_stop_reason not in {
            "cancelled",
            "not_found",
            "interrupted",
            "error",
        }:
            raise TerminalProtocolError(
                "failed stream has an invalid typed stop reason"
            )
        if effect_status == "succeeded" and stream_stop_reason not in {
            "completed",
            "deadline",
            "limit_reached",
        }:
            raise TerminalProtocolError(
                "successful stream has an invalid typed stop reason"
            )
    elif stream_stop_reason != "not_streaming":
        raise TerminalProtocolError(
            "non-streaming detour carries a stream stop reason"
        )
    if raw.get("identity_trust") != "trusted":
        raise TerminalProtocolError(
            "unverifiable legacy terminal detour cannot be projected"
        )
    if str(raw["product_authority_id"]) != str(raw["logical_thread_id"]):
        raise TerminalProtocolError(
            "terminal detour Product Head authority is inconsistent"
        )
    before = (
        raw.get("product_revision_before"),
        str(raw.get("product_checkpoint_thread_id_before") or ""),
        str(raw.get("product_checkpoint_id_before") or ""),
        str(raw.get("product_fingerprint_before") or ""),
    )
    after = (
        raw.get("product_revision_after"),
        str(raw.get("product_checkpoint_thread_id_after") or ""),
        str(raw.get("product_checkpoint_id_after") or ""),
        str(raw.get("product_fingerprint_after") or ""),
    )
    if before != after:
        raise TerminalProtocolError(
            "terminal detour projection advanced the Product Head"
        )
    if before[0] is None:
        if before[1] or before[2] or before[3]:
            raise TerminalProtocolError(
                "terminal detour absent Product Head identity is invalid"
            )
    else:
        if (
            not isinstance(before[0], int)
            or before[0] < 0
            or not before[1]
            or not before[2]
        ):
            raise TerminalProtocolError(
                "terminal detour Product Head identity is invalid"
            )
        _require_hash("product_fingerprint_before", before[3])
    fence_sequence = raw.get("runtime_event_fence_sequence")
    if (
        not isinstance(fence_sequence, int)
        or isinstance(fence_sequence, bool)
        or fence_sequence < 0
    ):
        raise TerminalProtocolError(
            "terminal detour runtime-event fence is invalid"
        )
    fence_terminal_event_id = str(
        raw.get("runtime_event_fence_terminal_event_id") or ""
    )
    fence_event_id = str(raw.get("runtime_event_fence_id") or "")
    fence_hash = str(raw.get("runtime_event_fence_hash") or "")
    if fence_sequence == 0:
        if fence_terminal_event_id or fence_event_id or fence_hash:
            raise TerminalProtocolError(
                "empty terminal detour runtime-event fence is invalid"
            )
    else:
        if not fence_terminal_event_id or not fence_event_id:
            raise TerminalProtocolError(
                "terminal detour runtime-event fence identity is missing"
            )
        _require_hash("runtime_event_fence_hash", fence_hash)
    revision = dict(raw.get("origin_revision") or {})
    if set(revision) != {"commit", "worktree_hash"} or not revision["commit"]:
        raise TerminalProtocolError("terminal detour revision is invalid")
    _require_hash(
        "origin_revision.worktree_hash",
        str(revision["worktree_hash"]),
    )
    expected_hash = _record_hash({
        key: value for key, value in raw.items() if key != "record_hash"
    })
    if str(raw["record_hash"]) != expected_hash:
        raise TerminalProtocolError(
            "terminal detour projection record hash mismatch"
        )
    return TerminalDetourProjection(
        schema_version=TERMINAL_DETOUR_PROJECTION_SCHEMA_VERSION,
        record_type="terminal_detour_projection",
        projection_id=str(raw["projection_id"]),
        detour_id=str(raw["detour_id"]),
        product_authority_id=str(raw["product_authority_id"]),
        logical_thread_id=str(raw["logical_thread_id"]),
        process_instance_id=str(raw["process_instance_id"]),
        session_id=str(raw["session_id"]),
        session_purpose=str(raw["session_purpose"]),
        command_name=str(raw["command_name"]),
        input_hash=str(raw["input_hash"]),
        result_kind=result_kind,
        interruption_kind=interruption_kind,
        termination_reason=termination_reason,
        exit_code=exit_code,
        effect_class=str(raw["effect_class"]),
        effect_status=str(raw["effect_status"]),
        shell_state_before_hash=str(raw["shell_state_before_hash"]),
        shell_state_after_hash=str(raw["shell_state_after_hash"]),
        product_revision_before=before[0],
        product_checkpoint_thread_id_before=before[1],
        product_checkpoint_id_before=before[2],
        product_fingerprint_before=before[3],
        product_revision_after=after[0],
        product_checkpoint_thread_id_after=after[1],
        product_checkpoint_id_after=after[2],
        product_fingerprint_after=after[3],
        runtime_event_fence_sequence=fence_sequence,
        runtime_event_fence_terminal_event_id=fence_terminal_event_id,
        runtime_event_fence_id=fence_event_id,
        runtime_event_fence_hash=fence_hash,
        response_hash=str(raw["response_hash"]),
        stream_chunk_count=stream_chunk_count,
        stream_hash=str(raw["stream_hash"]),
        stream_stop_reason=str(raw["stream_stop_reason"]),
        identity_trust=str(raw["identity_trust"]),
        presentation_hash=str(raw["presentation_hash"]),
        origin_revision={
            "commit": str(revision["commit"]),
            "worktree_hash": str(revision["worktree_hash"]),
        },
        delivery_phase=phase,  # type: ignore[arg-type]
        projected_at=str(raw["projected_at"]),
        record_hash=str(raw["record_hash"]),
    )


def build_terminal_outcome_projection(
    outcome: Any,
    *,
    rendered_frame: str,
    delivery_phase: Literal["live", "startup_replay"],
) -> TerminalOutcomeProjection:
    if delivery_phase not in _DELIVERY_PHASES:
        raise TerminalProtocolError("terminal delivery phase is invalid")
    if (
        str(outcome.outcome) == "committed"
        and str(outcome.runtime_event_status) != "published"
    ):
        raise TerminalProtocolError(
            "committed terminal outcome requires a published runtime event"
        )
    frame_hash = presentation_hash(rendered_frame)
    projection_identity = _record_hash({
        "event_id": str(outcome.event_id),
        "delivery_phase": delivery_phase,
        "presentation_hash": frame_hash,
    })
    payload: dict[str, Any] = {
        "schema_version": TERMINAL_OUTCOME_PROJECTION_SCHEMA_VERSION,
        "record_type": "terminal_outcome_projection",
        "projection_id": f"projection:{projection_identity}",
        "event_id": str(outcome.event_id),
        "transaction_id": str(outcome.transaction_id),
        "product_authority_id": str(outcome.logical_thread_id),
        "logical_thread_id": str(outcome.logical_thread_id),
        "physical_thread_id": str(outcome.physical_thread_id),
        "outcome": str(outcome.outcome),
        "failure_category": str(outcome.failure_category or ""),
        "diagnostic_hash": str(outcome.diagnostic_hash or ""),
        "base_revision": int(outcome.base_revision),
        "base_checkpoint_thread_id": str(outcome.base_checkpoint_thread_id),
        "base_checkpoint_id": str(outcome.base_checkpoint_id),
        "base_fingerprint": str(outcome.base_fingerprint),
        "attempt_checkpoint_id": str(outcome.attempt_checkpoint_id or ""),
        "attempt_fingerprint": str(outcome.attempt_fingerprint or ""),
        "product_revision": int(outcome.product_revision),
        "product_checkpoint_thread_id": str(
            outcome.product_checkpoint_thread_id
        ),
        "product_checkpoint_id": str(outcome.product_checkpoint_id),
        "product_fingerprint": str(outcome.product_fingerprint),
        "render_hash": str(outcome.render_hash or ""),
        "runtime_event_id": str(outcome.runtime_event_id or ""),
        "runtime_event_sequence": int(outcome.runtime_event_sequence),
        "runtime_event_payload_hash": str(
            outcome.runtime_event_payload_hash or ""
        ),
        "presentation_hash": frame_hash,
        "origin_revision": {
            "commit": str(outcome.origin_revision_commit),
            "worktree_hash": str(outcome.origin_revision_worktree_hash),
        },
        "delivery_phase": delivery_phase,
        "projected_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
    }
    payload["record_hash"] = _record_hash(payload)
    return validate_terminal_outcome_projection(payload)


def validate_terminal_outcome_projection(
    raw: Mapping[str, Any],
) -> TerminalOutcomeProjection:
    required = {
        "schema_version",
        "record_type",
        "projection_id",
        "event_id",
        "transaction_id",
        "product_authority_id",
        "logical_thread_id",
        "physical_thread_id",
        "outcome",
        "failure_category",
        "diagnostic_hash",
        "base_revision",
        "base_checkpoint_thread_id",
        "base_checkpoint_id",
        "base_fingerprint",
        "attempt_checkpoint_id",
        "attempt_fingerprint",
        "product_revision",
        "product_checkpoint_thread_id",
        "product_checkpoint_id",
        "product_fingerprint",
        "render_hash",
        "runtime_event_id",
        "runtime_event_sequence",
        "runtime_event_payload_hash",
        "presentation_hash",
        "origin_revision",
        "delivery_phase",
        "projected_at",
        "record_hash",
    }
    if set(raw) != required:
        raise TerminalProtocolError("terminal projection schema is invalid")
    if (
        raw.get("schema_version") != TERMINAL_OUTCOME_PROJECTION_SCHEMA_VERSION
        or raw.get("record_type") != "terminal_outcome_projection"
    ):
        raise TerminalProtocolError("terminal projection version is unsupported")
    outcome = str(raw.get("outcome") or "")
    phase = str(raw.get("delivery_phase") or "")
    if outcome not in _OUTCOMES or phase not in _DELIVERY_PHASES:
        raise TerminalProtocolError("terminal projection outcome or phase is invalid")
    for name in (
        "projection_id",
        "event_id",
        "transaction_id",
        "product_authority_id",
        "logical_thread_id",
        "physical_thread_id",
        "base_checkpoint_thread_id",
        "base_checkpoint_id",
        "base_fingerprint",
        "product_checkpoint_thread_id",
        "product_checkpoint_id",
        "product_fingerprint",
        "presentation_hash",
        "projected_at",
        "record_hash",
    ):
        if not str(raw.get(name) or "").strip():
            raise TerminalProtocolError(f"terminal projection is missing {name}")
    for name in (
        "base_fingerprint",
        "product_fingerprint",
        "presentation_hash",
        "record_hash",
    ):
        _require_hash(name, str(raw[name]))
    revision = dict(raw.get("origin_revision") or {})
    if set(revision) != {"commit", "worktree_hash"} or not revision["commit"]:
        raise TerminalProtocolError("terminal projection revision is invalid")
    _require_hash("origin_revision.worktree_hash", str(revision["worktree_hash"]))
    render_hash = str(raw.get("render_hash") or "")
    failure_category = str(raw.get("failure_category") or "")
    diagnostic_hash = str(raw.get("diagnostic_hash") or "")
    attempt_fingerprint = str(raw.get("attempt_fingerprint") or "")
    attempt_checkpoint_id = str(raw.get("attempt_checkpoint_id") or "")
    base_revision = raw.get("base_revision")
    product_revision = raw.get("product_revision")
    if (
        not isinstance(base_revision, int)
        or isinstance(base_revision, bool)
        or base_revision < 0
        or not isinstance(product_revision, int)
        or isinstance(product_revision, bool)
        or product_revision < 0
    ):
        raise TerminalProtocolError(
            "terminal projection Product Head revision is invalid"
        )
    if outcome == "committed":
        _require_hash("render_hash", render_hash)
        _require_hash("attempt_fingerprint", attempt_fingerprint)
        if failure_category or diagnostic_hash:
            raise TerminalProtocolError("committed projection has failure metadata")
        if (
            not str(raw.get("runtime_event_id") or "")
            or raw.get("runtime_event_sequence") != product_revision
        ):
            raise TerminalProtocolError(
                "committed projection runtime event identity is invalid"
            )
        _require_hash(
            "runtime_event_payload_hash",
            str(raw.get("runtime_event_payload_hash") or ""),
        )
        if product_revision != base_revision + 1:
            raise TerminalProtocolError(
                "committed projection did not advance Product Head revision"
            )
        if (
            not attempt_checkpoint_id
            or str(raw["product_checkpoint_thread_id"])
            != str(raw["physical_thread_id"])
            or str(raw["product_checkpoint_id"]) != attempt_checkpoint_id
            or str(raw["product_fingerprint"]) != attempt_fingerprint
        ):
            raise TerminalProtocolError(
                "committed projection checkpoint lineage is invalid"
            )
    elif outcome == "aborted":
        if (
            render_hash
            or attempt_checkpoint_id
            or attempt_fingerprint
            or not failure_category
        ):
            raise TerminalProtocolError("failed projection metadata is invalid")
        _require_hash("diagnostic_hash", diagnostic_hash)
        if product_revision != base_revision:
            raise TerminalProtocolError(
                "aborted projection advanced Product Head revision"
            )
        if (
            raw.get("runtime_event_id")
            or raw.get("runtime_event_sequence") != product_revision
            or raw.get("runtime_event_payload_hash")
        ):
            raise TerminalProtocolError(
                "aborted projection carries runtime event evidence"
            )
    else:
        if render_hash or not failure_category:
            raise TerminalProtocolError(
                "reconciliation projection metadata is invalid"
            )
        if bool(attempt_checkpoint_id) != bool(attempt_fingerprint):
            raise TerminalProtocolError(
                "reconciliation attempt checkpoint identity is incomplete"
            )
        if attempt_fingerprint:
            _require_hash("attempt_fingerprint", attempt_fingerprint)
        _require_hash("diagnostic_hash", diagnostic_hash)
        if product_revision != base_revision:
            raise TerminalProtocolError(
                "reconciliation projection advanced Product Head revision"
            )
        if (
            raw.get("runtime_event_id")
            or raw.get("runtime_event_sequence") != product_revision
            or raw.get("runtime_event_payload_hash")
        ):
            raise TerminalProtocolError(
                "reconciliation projection carries runtime event evidence"
            )
    if str(raw["product_authority_id"]) != str(raw["logical_thread_id"]):
        raise TerminalProtocolError(
            "terminal projection Product Head authority is inconsistent"
        )
    if outcome != "committed" and (
        str(raw["product_checkpoint_thread_id"])
        != str(raw["base_checkpoint_thread_id"])
        or str(raw["product_checkpoint_id"]) != str(raw["base_checkpoint_id"])
        or str(raw["product_fingerprint"]) != str(raw["base_fingerprint"])
    ):
        raise TerminalProtocolError(
            "non-committing projection changed Product Head identity"
        )
    expected_hash = _record_hash({
        key: value for key, value in raw.items() if key != "record_hash"
    })
    if str(raw["record_hash"]) != expected_hash:
        raise TerminalProtocolError("terminal projection record hash mismatch")
    return TerminalOutcomeProjection(
        schema_version=TERMINAL_OUTCOME_PROJECTION_SCHEMA_VERSION,
        record_type="terminal_outcome_projection",
        projection_id=str(raw["projection_id"]),
        event_id=str(raw["event_id"]),
        transaction_id=str(raw["transaction_id"]),
        product_authority_id=str(raw["product_authority_id"]),
        logical_thread_id=str(raw["logical_thread_id"]),
        physical_thread_id=str(raw["physical_thread_id"]),
        outcome=outcome,
        failure_category=failure_category,
        diagnostic_hash=diagnostic_hash,
        base_revision=base_revision,
        base_checkpoint_thread_id=str(raw["base_checkpoint_thread_id"]),
        base_checkpoint_id=str(raw["base_checkpoint_id"]),
        base_fingerprint=str(raw["base_fingerprint"]),
        attempt_checkpoint_id=attempt_checkpoint_id,
        attempt_fingerprint=attempt_fingerprint,
        product_revision=product_revision,
        product_checkpoint_thread_id=str(
            raw["product_checkpoint_thread_id"]
        ),
        product_checkpoint_id=str(raw["product_checkpoint_id"]),
        product_fingerprint=str(raw["product_fingerprint"]),
        render_hash=render_hash,
        runtime_event_id=str(raw["runtime_event_id"]),
        runtime_event_sequence=int(raw["runtime_event_sequence"]),
        runtime_event_payload_hash=str(raw["runtime_event_payload_hash"]),
        presentation_hash=str(raw["presentation_hash"]),
        origin_revision={
            "commit": str(revision["commit"]),
            "worktree_hash": str(revision["worktree_hash"]),
        },
        delivery_phase=phase,  # type: ignore[arg-type]
        projected_at=str(raw["projected_at"]),
        record_hash=str(raw["record_hash"]),
    )


def build_terminal_session_event(
    *,
    process_instance_id: str,
    session_id: str,
    session_purpose: str,
    provider: str,
    model: str,
    auth_mode: str,
    provider_ready: bool,
    startup_status: Literal["ready", "blocked"],
    failure_category: str,
    product_authority_id: str,
    product_revision: int | None,
    product_checkpoint_thread_id: str,
    product_checkpoint_id: str,
    product_fingerprint: str,
    runtime_event_fence_sequence: int,
    runtime_event_fence_terminal_event_id: str,
    runtime_event_fence_id: str,
    runtime_event_fence_hash: str,
    rendered_frame: str,
    origin_revision: Mapping[str, str],
    replayed_projection_ids: tuple[str, ...] = (),
) -> TerminalSessionEvent:
    if startup_status not in _STARTUP_STATUSES:
        raise TerminalProtocolError("terminal session startup status is invalid")
    normalized_failure_category = str(failure_category or "")
    if (
        startup_status == "blocked"
        and normalized_failure_category not in _STARTUP_FAILURE_CATEGORIES
    ):
        raise TerminalProtocolError(
            "terminal session startup failure category is invalid"
        )
    frame_hash = presentation_hash(rendered_frame)
    event_identity = _record_hash({
        "process_instance_id": process_instance_id,
        "session_id": session_id,
        "session_purpose": session_purpose,
    })
    payload: dict[str, Any] = {
        "schema_version": TERMINAL_SESSION_EVENT_SCHEMA_VERSION,
        "record_type": "terminal_session_event",
        "session_event_id": f"session:{event_identity}",
        "process_instance_id": process_instance_id,
        "session_id": session_id,
        "session_purpose": session_purpose,
        "provider": provider,
        "model": model,
        "auth_mode": auth_mode,
        "provider_ready": bool(provider_ready),
        "startup_status": startup_status,
        "failure_category": normalized_failure_category,
        "product_authority_id": product_authority_id,
        "product_revision": product_revision,
        "product_checkpoint_thread_id": product_checkpoint_thread_id,
        "product_checkpoint_id": product_checkpoint_id,
        "product_fingerprint": product_fingerprint,
        "runtime_event_fence_sequence": runtime_event_fence_sequence,
        "runtime_event_fence_terminal_event_id": (
            runtime_event_fence_terminal_event_id
        ),
        "runtime_event_fence_id": runtime_event_fence_id,
        "runtime_event_fence_hash": runtime_event_fence_hash,
        "presentation_hash": frame_hash,
        "replayed_projection_ids": list(replayed_projection_ids),
        "origin_revision": {
            "commit": str(origin_revision.get("commit") or ""),
            "worktree_hash": str(origin_revision.get("worktree_hash") or ""),
        },
        "projected_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
    }
    payload["record_hash"] = _record_hash(payload)
    return validate_terminal_session_event(payload)


def validate_terminal_session_event(
    raw: Mapping[str, Any],
) -> TerminalSessionEvent:
    required = {
        "schema_version",
        "record_type",
        "session_event_id",
        "process_instance_id",
        "session_id",
        "session_purpose",
        "provider",
        "model",
        "auth_mode",
        "provider_ready",
        "startup_status",
        "failure_category",
        "product_authority_id",
        "product_revision",
        "product_checkpoint_thread_id",
        "product_checkpoint_id",
        "product_fingerprint",
        "runtime_event_fence_sequence",
        "runtime_event_fence_terminal_event_id",
        "runtime_event_fence_id",
        "runtime_event_fence_hash",
        "presentation_hash",
        "replayed_projection_ids",
        "origin_revision",
        "projected_at",
        "record_hash",
    }
    if set(raw) != required:
        raise TerminalProtocolError("terminal session event schema is invalid")
    if (
        raw.get("schema_version") != TERMINAL_SESSION_EVENT_SCHEMA_VERSION
        or raw.get("record_type") != "terminal_session_event"
    ):
        raise TerminalProtocolError("terminal session event version is unsupported")
    status = str(raw.get("startup_status") or "")
    if status not in _STARTUP_STATUSES:
        raise TerminalProtocolError("terminal session startup status is invalid")
    for name in (
        "session_event_id",
        "process_instance_id",
        "session_id",
        "session_purpose",
        "provider",
        "model",
        "auth_mode",
        "presentation_hash",
        "projected_at",
        "record_hash",
    ):
        if not str(raw.get(name) or "").strip():
            raise TerminalProtocolError(f"terminal session event is missing {name}")
    if not isinstance(raw.get("provider_ready"), bool):
        raise TerminalProtocolError("terminal session provider readiness is invalid")
    failure_category = str(raw.get("failure_category") or "")
    if status == "ready":
        if not raw["provider_ready"] or failure_category:
            raise TerminalProtocolError("ready terminal session has failure metadata")
    else:
        if raw["provider_ready"] or not failure_category:
            raise TerminalProtocolError("blocked terminal session lacks failure metadata")
        if failure_category not in _STARTUP_FAILURE_CATEGORIES:
            raise TerminalProtocolError(
                "terminal session startup failure category is invalid"
            )
    product_authority_id = str(raw.get("product_authority_id") or "")
    product_revision = raw.get("product_revision")
    product_checkpoint_thread_id = str(
        raw.get("product_checkpoint_thread_id") or ""
    )
    product_checkpoint_id = str(raw.get("product_checkpoint_id") or "")
    product_fingerprint = str(raw.get("product_fingerprint") or "")
    fence_sequence = raw.get("runtime_event_fence_sequence")
    fence_terminal_event_id = str(
        raw.get("runtime_event_fence_terminal_event_id") or ""
    )
    fence_event_id = str(raw.get("runtime_event_fence_id") or "")
    fence_hash = str(raw.get("runtime_event_fence_hash") or "")
    if status == "ready":
        if (
            not product_authority_id
            or not isinstance(product_revision, int)
            or product_revision < 1
            or not product_checkpoint_thread_id
            or not product_checkpoint_id
        ):
            raise TerminalProtocolError(
                "ready terminal session Product Head identity is invalid"
            )
        _require_hash("product_fingerprint", product_fingerprint)
        if fence_sequence != product_revision:
            raise TerminalProtocolError(
                "startup runtime-event fence is behind the Product Head"
            )
        if not fence_terminal_event_id or not fence_event_id:
            raise TerminalProtocolError(
                "startup runtime-event fence identity is missing"
            )
        _require_hash("runtime_event_fence_hash", fence_hash)
    elif (
        product_authority_id
        or product_revision is not None
        or product_checkpoint_thread_id
        or product_checkpoint_id
        or product_fingerprint
        or fence_sequence != 0
        or fence_terminal_event_id
        or fence_event_id
        or fence_hash
    ):
        raise TerminalProtocolError(
            "blocked terminal session carries Product Head authority"
        )
    _require_hash("presentation_hash", str(raw["presentation_hash"]))
    replayed_projection_ids = tuple(
        str(item) for item in raw.get("replayed_projection_ids") or ()
    )
    if (
        len(set(replayed_projection_ids)) != len(replayed_projection_ids)
        or any(not item.strip() for item in replayed_projection_ids)
    ):
        raise TerminalProtocolError(
            "terminal session replay projection identities are invalid"
        )
    revision = dict(raw.get("origin_revision") or {})
    if set(revision) != {"commit", "worktree_hash"} or not revision["commit"]:
        raise TerminalProtocolError("terminal session revision is invalid")
    _require_hash("origin_revision.worktree_hash", str(revision["worktree_hash"]))
    expected_hash = _record_hash({
        key: value for key, value in raw.items() if key != "record_hash"
    })
    if str(raw["record_hash"]) != expected_hash:
        raise TerminalProtocolError("terminal session event record hash mismatch")
    return TerminalSessionEvent(
        schema_version=TERMINAL_SESSION_EVENT_SCHEMA_VERSION,
        record_type="terminal_session_event",
        session_event_id=str(raw["session_event_id"]),
        process_instance_id=str(raw["process_instance_id"]),
        session_id=str(raw["session_id"]),
        session_purpose=str(raw["session_purpose"]),
        provider=str(raw["provider"]),
        model=str(raw["model"]),
        auth_mode=str(raw["auth_mode"]),
        provider_ready=bool(raw["provider_ready"]),
        startup_status=status,  # type: ignore[arg-type]
        failure_category=failure_category,
        product_authority_id=product_authority_id,
        product_revision=product_revision,
        product_checkpoint_thread_id=product_checkpoint_thread_id,
        product_checkpoint_id=product_checkpoint_id,
        product_fingerprint=product_fingerprint,
        runtime_event_fence_sequence=int(fence_sequence),
        runtime_event_fence_terminal_event_id=fence_terminal_event_id,
        runtime_event_fence_id=fence_event_id,
        runtime_event_fence_hash=fence_hash,
        presentation_hash=str(raw["presentation_hash"]),
        replayed_projection_ids=replayed_projection_ids,
        origin_revision={
            "commit": str(revision["commit"]),
            "worktree_hash": str(revision["worktree_hash"]),
        },
        projected_at=str(raw["projected_at"]),
        record_hash=str(raw["record_hash"]),
    )


def append_terminal_session_event(
    path: str | Path,
    event: TerminalSessionEvent,
) -> None:
    """Durably publish one immutable startup event for a process instance."""

    destination = Path(path)
    with _projection_file_lock(destination):
        if destination.exists():
            for line in destination.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                raw = json.loads(line)
                if raw.get("record_type") != "terminal_session_event":
                    continue
                existing = validate_terminal_session_event(raw)
                if existing.session_event_id != event.session_event_id:
                    continue
                left = existing.as_dict()
                right = event.as_dict()
                for key in ("projected_at", "record_hash"):
                    left.pop(key, None)
                    right.pop(key, None)
                if left != right:
                    raise TerminalProtocolError(
                        "terminal session identity has conflicting immutable facts"
                    )
                return
        _append_jsonl_record_unlocked(destination, event.as_dict())


def append_terminal_outcome_projection(
    path: str | Path,
    projection: TerminalOutcomeProjection,
) -> None:
    """Append one durable JSONL projection before acknowledging delivery."""

    destination = Path(path)
    with _projection_file_lock(destination):
        if destination.exists():
            for line in destination.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                raw = json.loads(line)
                if raw.get("record_type") != "terminal_outcome_projection":
                    continue
                existing = validate_terminal_outcome_projection(raw)
                if existing.projection_id != projection.projection_id:
                    continue
                left = existing.as_dict()
                right = projection.as_dict()
                for key in ("projected_at", "record_hash"):
                    left.pop(key, None)
                    right.pop(key, None)
                if left != right:
                    raise TerminalProtocolError(
                        "terminal projection identity has conflicting immutable facts"
                    )
                return
        _append_jsonl_record_unlocked(destination, projection.as_dict())


def append_terminal_detour_projection(
    path: str | Path,
    projection: TerminalDetourProjection,
) -> None:
    """Append one durable detour projection before delivery acknowledgement."""

    destination = Path(path)
    with _projection_file_lock(destination):
        if destination.exists():
            for line in destination.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                raw = json.loads(line)
                if raw.get("record_type") != "terminal_detour_projection":
                    continue
                existing = validate_terminal_detour_projection(raw)
                if existing.projection_id != projection.projection_id:
                    continue
                left = existing.as_dict()
                right = projection.as_dict()
                for key in ("projected_at", "record_hash"):
                    left.pop(key, None)
                    right.pop(key, None)
                if left != right:
                    raise TerminalProtocolError(
                        "terminal detour projection has conflicting immutable facts"
                    )
                return
        _append_jsonl_record_unlocked(destination, projection.as_dict())


@contextmanager
def _projection_file_lock(destination: Path, *, shared: bool = False):
    """Serialize the read-check-append sequence across Linux CLI processes."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_name(f"{destination.name}.lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(
            descriptor,
            fcntl.LOCK_SH if shared else fcntl.LOCK_EX,
        )
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def append_jsonl_record(
    path: str | Path,
    payload: Mapping[str, Any],
) -> None:
    """Append one complete JSON record and durably publish its directory entry."""

    destination = Path(path)
    with _projection_file_lock(destination):
        _append_jsonl_record_unlocked(destination, payload)


def read_jsonl_records(path: str | Path) -> tuple[Mapping[str, Any], ...]:
    """Read one stable JSONL snapshot while excluding in-progress appends."""

    source = Path(path)
    with _projection_file_lock(source, shared=True):
        try:
            lines = source.read_text(encoding="utf-8").splitlines()
        except OSError:
            return ()
        return tuple(json.loads(line) for line in lines if line.strip())


def claim_jsonl_authority(
    path: str | Path,
    *,
    product_authority_id: str,
) -> Path:
    """Bind one projection path to exactly one Product Head authority."""

    source = Path(path)
    authority = product_authority_id.strip()
    if not authority:
        raise ValueError("JSONL Product Head authority is missing")
    marker = source.with_name(f"{source.name}.authority.json")
    with _projection_file_lock(source):
        if marker.exists():
            payload = json.loads(marker.read_text(encoding="utf-8"))
            unsigned = dict(payload)
            record_hash = str(unsigned.pop("record_hash", "") or "")
            if (
                set(payload)
                != {
                    "schema_version",
                    "record_type",
                    "product_authority_id",
                    "record_hash",
                }
                or payload.get("schema_version") != 1
                or payload.get("record_type") != "jsonl_authority"
                or str(payload.get("product_authority_id") or "") != authority
                or not record_hash
                or _record_hash(unsigned) != record_hash
            ):
                raise RuntimeError(
                    "JSONL projection path belongs to another or invalid authority"
                )
            return marker

        inferred_authorities: set[str] = set()
        if source.exists():
            for line in source.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                inferred = str(record.get("product_authority_id") or "")
                if not inferred:
                    thread_id = str(record.get("thread_id") or "")
                    session_purpose = str(
                        record.get("session_purpose") or ""
                    )
                    if thread_id and session_purpose:
                        inferred = f"{session_purpose}:{thread_id}"
                if not inferred:
                    raise RuntimeError(
                        "existing JSONL projection has no provable authority"
                    )
                inferred_authorities.add(inferred)
        if inferred_authorities and inferred_authorities != {authority}:
            raise RuntimeError(
                "JSONL projection path contains another Product Head authority"
            )

        unsigned = {
            "schema_version": 1,
            "record_type": "jsonl_authority",
            "product_authority_id": authority,
        }
        payload = {**unsigned, "record_hash": _record_hash(unsigned)}
        encoded = (
            json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
        temporary = marker.with_name(f"{marker.name}.tmp.{os.getpid()}")
        temporary.unlink(missing_ok=True)
        descriptor = os.open(
            temporary,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        try:
            try:
                remaining = memoryview(encoded)
                while remaining:
                    written = os.write(descriptor, remaining)
                    if written <= 0:
                        raise OSError(
                            "JSONL authority write made no progress"
                        )
                    remaining = remaining[written:]
                os.fsync(descriptor)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        finally:
            os.close(descriptor)
        os.replace(temporary, marker)
        directory = os.open(source.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return marker


def quarantine_jsonl_file(
    path: str | Path,
    *,
    reason: str,
) -> Path | None:
    """Atomically retain an incompatible projection log outside live authority."""

    source = Path(path)
    normalized_reason = reason.strip().lower().replace("_", "-")
    if (
        not normalized_reason
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
            for character in normalized_reason
        )
    ):
        raise ValueError("JSONL quarantine reason is invalid")
    with _projection_file_lock(source):
        try:
            content = source.read_bytes()
        except FileNotFoundError:
            return None
        digest = hashlib.sha256(content).hexdigest()
        quarantine = source.with_name(
            f"{source.name}.{normalized_reason}.{digest}.quarantine"
        )
        if quarantine.exists():
            if quarantine.read_bytes() != content:
                raise RuntimeError(
                    "JSONL quarantine identity collided with different content"
                )
            source.unlink()
        else:
            os.replace(source, quarantine)
        directory = os.open(source.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return quarantine


def _append_jsonl_record_unlocked(
    destination: Path,
    payload: Mapping[str, Any],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor = os.open(destination, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        remaining = memoryview(encoded)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("durable JSONL append made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _record_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_hash(name: str, value: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise TerminalProtocolError(f"{name} is not a sha256 digest")

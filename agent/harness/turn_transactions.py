"""SQLite authority for logical product heads and physical turn attempts."""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import uuid
import json
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, Sequence

import fcntl


TURN_TRANSACTION_SCHEMA_VERSION = 14

_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,254}\Z")
_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
_TERMINAL_OUTCOMES = frozenset({"committed", "aborted", "reconciliation_required"})
_RECONCILIATION_RESOLUTIONS = frozenset({
    "effect_confirmed",
    "effect_not_observed",
})
_EMPTY_RENDER_HASH = hashlib.sha256(b"[]").hexdigest()
_EMPTY_STREAM_HASH = hashlib.sha256(b"").hexdigest()
_LEGACY_REVISION_HASH = hashlib.sha256(b"legacy-unbound-revision").hexdigest()

AttemptStatus = Literal["active", "committed", "aborted", "reconciliation_required"]
TerminalOutcomeKind = Literal["committed", "aborted", "reconciliation_required"]
TerminalDetourEffectClass = Literal[
    "read_only",
    "observation_refresh",
    "shell_state_mutation",
    "local_external_effect",
    "authority_resolution",
    "streaming_observation",
]
TerminalDetourStatus = Literal[
    "prepared",
    "completed",
    "reconciliation_required",
]
TerminalDetourEffectStatus = Literal[
    "not_applicable",
    "succeeded",
    "failed",
    "uncertain",
]
TerminalDetourResultKind = Literal["command", "session_termination"]
TerminalTerminationReason = Literal["exit", "eof", "outer_ctrl_c"]
TerminalStreamStopReason = Literal[
    "not_streaming",
    "completed",
    "cancelled",
    "deadline",
    "limit_reached",
    "not_found",
    "interrupted",
    "error",
]

_DETOUR_EFFECT_CLASSES = frozenset({
    "read_only",
    "observation_refresh",
    "shell_state_mutation",
    "local_external_effect",
    "authority_resolution",
    "streaming_observation",
})
_DETOUR_STATUSES = frozenset({
    "prepared",
    "completed",
    "reconciliation_required",
})
_DETOUR_EFFECT_STATUSES = frozenset({
    "not_applicable",
    "succeeded",
    "failed",
    "uncertain",
})
_DETOUR_RESULT_KINDS = frozenset({"command", "session_termination"})
_TERMINATION_REASONS = frozenset({"exit", "eof", "outer_ctrl_c"})
_STREAM_STOP_REASONS = frozenset({
    "not_streaming",
    "completed",
    "cancelled",
    "deadline",
    "limit_reached",
    "not_found",
    "interrupted",
    "error",
})
_LEGACY_V10_DETOUR_SNAPSHOT_FIELDS = frozenset({
    "detour_id",
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
    "status",
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
    "response_json",
    "response_hash",
    "stream_chunk_count",
    "stream_hash",
    "stream_messages_json",
    "stream_stop_reason",
    "identity_trust",
    "effect_status",
    "diagnostic_hash",
    "resolution",
    "evidence_hash",
    "origin_revision_commit",
    "origin_revision_worktree_hash",
    "created_at",
    "completed_at",
    "delivered_at",
})
_LEGACY_V12_DETOUR_SNAPSHOT_FIELDS = (
    _LEGACY_V10_DETOUR_SNAPSHOT_FIELDS
    | frozenset({
        "runtime_event_fence_sequence",
        "runtime_event_fence_terminal_event_id",
        "runtime_event_fence_id",
        "runtime_event_fence_hash",
    })
)


class TurnTransactionError(RuntimeError):
    """Base error for turn transaction authority failures."""


class TurnTransactionValidationError(TurnTransactionError, ValueError):
    """Raised when an identifier or fingerprint violates the ledger contract."""


class TurnTransactionConflictError(TurnTransactionError):
    """Raised when an operation conflicts with immutable ledger state."""


class ReconciliationRequiredError(TurnTransactionConflictError):
    """Raised while an uncertain external operation still blocks the head."""

    def __init__(self, transaction_id: str) -> None:
        self.transaction_id = transaction_id
        super().__init__(
            "logical Product Head requires reconciliation for transaction "
            f"{transaction_id}"
        )


class TurnTransactionSchemaError(TurnTransactionError):
    """Raised when the database uses an unsupported ledger schema."""


class ProductAuthorityLease:
    """Process-held exclusive lease for one logical Product Head."""

    def __init__(self, checkpoint_path: str | Path, logical_thread_id: str) -> None:
        authority = _identifier("logical_thread_id", logical_thread_id)
        digest = hashlib.sha256(authority.encode("utf-8")).hexdigest()[:24]
        self.path = Path(f"{Path(checkpoint_path)}.{digest}.authority.lock")
        self._descriptor: int | None = None

    def acquire(self) -> None:
        if self._descriptor is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(descriptor)
            raise TurnTransactionConflictError(
                "logical Product Head is already owned by another live runtime"
            ) from exc
        self._descriptor = descriptor

    def close(self) -> None:
        if self._descriptor is None:
            return
        descriptor, self._descriptor = self._descriptor, None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def __enter__(self) -> "ProductAuthorityLease":
        self.acquire()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


@dataclass(frozen=True)
class ProductHead:
    logical_thread_id: str
    checkpoint_thread_id: str
    checkpoint_id: str
    state_fingerprint: str
    revision: int
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class TurnAttempt:
    transaction_id: str
    logical_thread_id: str
    physical_thread_id: str
    base_revision: int
    base_checkpoint_thread_id: str
    base_checkpoint_id: str
    base_fingerprint: str
    origin_revision_commit: str
    origin_revision_worktree_hash: str
    status: AttemptStatus
    attempt_checkpoint_id: str | None
    attempt_fingerprint: str | None
    diagnostic_hash: str | None
    created_at: str
    completed_at: str | None


@dataclass(frozen=True)
class TerminalOutcome:
    event_id: str
    transaction_id: str
    logical_thread_id: str
    physical_thread_id: str
    outcome: TerminalOutcomeKind
    base_revision: int
    base_checkpoint_thread_id: str
    base_checkpoint_id: str
    base_fingerprint: str
    origin_revision_commit: str
    origin_revision_worktree_hash: str
    attempt_checkpoint_id: str | None
    attempt_fingerprint: str | None
    product_revision: int
    product_checkpoint_thread_id: str
    product_checkpoint_id: str
    product_fingerprint: str
    diagnostic_hash: str | None
    render_hash: str
    failure_category: str
    runtime_event_id: str
    runtime_event_sequence: int
    runtime_event_status: Literal["pending", "published", "not_applicable"]
    runtime_event_payload_hash: str | None
    runtime_event_published_at: str | None
    created_at: str
    delivered_at: str | None


@dataclass(frozen=True)
class ReconciliationResolution:
    transaction_id: str
    logical_thread_id: str
    resolution: Literal["effect_confirmed", "effect_not_observed"]
    evidence_hash: str
    created_at: str


@dataclass(frozen=True)
class TerminalDetour:
    detour_id: str
    logical_thread_id: str
    process_instance_id: str
    session_id: str
    session_purpose: str
    command_name: str
    input_hash: str
    result_kind: TerminalDetourResultKind
    interruption_kind: str | None
    termination_reason: TerminalTerminationReason | None
    exit_code: int | None
    effect_class: TerminalDetourEffectClass
    status: TerminalDetourStatus
    shell_state_before_hash: str
    shell_state_after_hash: str | None
    product_revision_before: int | None
    product_checkpoint_thread_id_before: str | None
    product_checkpoint_id_before: str | None
    product_fingerprint_before: str | None
    product_revision_after: int | None
    product_checkpoint_thread_id_after: str | None
    product_checkpoint_id_after: str | None
    product_fingerprint_after: str | None
    runtime_event_fence_sequence: int
    runtime_event_fence_terminal_event_id: str
    runtime_event_fence_id: str
    runtime_event_fence_hash: str
    response_messages: tuple[str, ...]
    response_hash: str | None
    stream_chunk_count: int
    stream_hash: str
    stream_messages: tuple[str, ...]
    stream_stop_reason: TerminalStreamStopReason
    identity_trust: Literal["trusted", "legacy_unbound"]
    effect_status: TerminalDetourEffectStatus
    diagnostic_hash: str | None
    resolution: str | None
    evidence_hash: str | None
    origin_revision_commit: str
    origin_revision_worktree_hash: str
    created_at: str
    completed_at: str | None
    delivered_at: str | None


class TurnTransactionStore:
    """Own the committed product head independently from LangGraph checkpoints."""

    def __init__(self, checkpoint_path: str | Path, *, timeout_seconds: float = 30.0) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if timeout_seconds <= 0:
            raise TurnTransactionValidationError("timeout_seconds must be positive")
        self.timeout_seconds = float(timeout_seconds)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def bootstrap_product_head(
        self,
        *,
        logical_thread_id: str,
        checkpoint_thread_id: str,
        checkpoint_id: str,
        state_fingerprint: str,
    ) -> ProductHead:
        logical_thread_id = _identifier("logical_thread_id", logical_thread_id)
        checkpoint_thread_id = _identifier("checkpoint_thread_id", checkpoint_thread_id)
        checkpoint_id = _identifier("checkpoint_id", checkpoint_id)
        state_fingerprint = _fingerprint("state_fingerprint", state_fingerprint)
        now = _utc_now()
        with self._write_transaction() as connection:
            existing = self._select_head(connection, logical_thread_id)
            if existing is not None:
                expected = (checkpoint_thread_id, checkpoint_id, state_fingerprint)
                actual = (
                    existing.checkpoint_thread_id,
                    existing.checkpoint_id,
                    existing.state_fingerprint,
                )
                if actual != expected:
                    raise TurnTransactionConflictError(
                        "logical product head is already bootstrapped with a different checkpoint"
                    )
                return existing
            connection.execute(
                """
                INSERT INTO anychain_product_heads (
                    logical_thread_id,
                    checkpoint_thread_id,
                    checkpoint_id,
                    state_fingerprint,
                    revision,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    logical_thread_id,
                    checkpoint_thread_id,
                    checkpoint_id,
                    state_fingerprint,
                    now,
                    now,
                ),
            )
            return self._require_head(connection, logical_thread_id)

    def begin_attempt(
        self,
        *,
        logical_thread_id: str,
        transaction_id: str | None = None,
        origin_revision_commit: str = "legacy-unbound",
        origin_revision_worktree_hash: str = _LEGACY_REVISION_HASH,
    ) -> TurnAttempt:
        logical_thread_id = _identifier("logical_thread_id", logical_thread_id)
        transaction_id = _transaction_id(transaction_id or str(uuid.uuid4()))
        origin_revision_commit = _identifier(
            "origin_revision_commit",
            origin_revision_commit,
        )
        origin_revision_worktree_hash = _fingerprint(
            "origin_revision_worktree_hash",
            origin_revision_worktree_hash,
        )
        physical_thread_id = f"attempt:{transaction_id}"
        now = _utc_now()
        with self._write_transaction() as connection:
            head = self._require_head(connection, logical_thread_id)
            existing = self._select_attempt(connection, transaction_id)
            if existing is not None:
                expected = (
                    logical_thread_id,
                    physical_thread_id,
                    head.revision,
                    head.checkpoint_thread_id,
                    head.checkpoint_id,
                    head.state_fingerprint,
                    origin_revision_commit,
                    origin_revision_worktree_hash,
                )
                actual = (
                    existing.logical_thread_id,
                    existing.physical_thread_id,
                    existing.base_revision,
                    existing.base_checkpoint_thread_id,
                    existing.base_checkpoint_id,
                    existing.base_fingerprint,
                    existing.origin_revision_commit,
                    existing.origin_revision_worktree_hash,
                )
                if actual != expected:
                    raise TurnTransactionConflictError(
                        "transaction_id is already bound to a different immutable base"
                    )
                return existing
            unresolved = self._select_unresolved_reconciliation(
                connection,
                logical_thread_id,
            )
            if unresolved is not None:
                raise ReconciliationRequiredError(unresolved.transaction_id)
            open_detour = self._select_open_detour(
                connection,
                logical_thread_id,
            )
            if open_detour is not None:
                raise ReconciliationRequiredError(open_detour.detour_id)
            active = connection.execute(
                """
                SELECT transaction_id
                FROM anychain_turn_attempts
                WHERE logical_thread_id = ? AND status = 'active'
                """,
                (logical_thread_id,),
            ).fetchone()
            if active is not None:
                raise TurnTransactionConflictError(
                    f"logical thread already has active transaction {active['transaction_id']}"
                )
            connection.execute(
                """
                INSERT INTO anychain_turn_attempts (
                    transaction_id,
                    logical_thread_id,
                    physical_thread_id,
                    base_revision,
                    base_checkpoint_thread_id,
                    base_checkpoint_id,
                    base_fingerprint,
                    origin_revision_commit,
                    origin_revision_worktree_hash,
                    status,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?)
                """,
                (
                    transaction_id,
                    logical_thread_id,
                    physical_thread_id,
                    head.revision,
                    head.checkpoint_thread_id,
                    head.checkpoint_id,
                    head.state_fingerprint,
                    origin_revision_commit,
                    origin_revision_worktree_hash,
                    now,
                ),
            )
            return self._require_attempt(connection, transaction_id)

    def commit_attempt(
        self,
        *,
        logical_thread_id: str,
        transaction_id: str,
        physical_thread_id: str,
        attempt_checkpoint_id: str,
        attempt_fingerprint: str,
        render_hash: str = _EMPTY_RENDER_HASH,
    ) -> TerminalOutcome:
        return self._finish_attempt(
            logical_thread_id=logical_thread_id,
            transaction_id=transaction_id,
            physical_thread_id=physical_thread_id,
            outcome="committed",
            attempt_checkpoint_id=attempt_checkpoint_id,
            attempt_fingerprint=attempt_fingerprint,
            diagnostic_hash=None,
            render_hash=render_hash,
            failure_category="",
        )

    def abort_attempt(
        self,
        *,
        logical_thread_id: str,
        transaction_id: str,
        physical_thread_id: str,
        diagnostic_hash: str,
        failure_category: str = "unexpected_failure",
    ) -> TerminalOutcome:
        return self._finish_attempt(
            logical_thread_id=logical_thread_id,
            transaction_id=transaction_id,
            physical_thread_id=physical_thread_id,
            outcome="aborted",
            attempt_checkpoint_id=None,
            attempt_fingerprint=None,
            diagnostic_hash=diagnostic_hash,
            render_hash="",
            failure_category=failure_category,
        )

    def require_reconciliation(
        self,
        *,
        logical_thread_id: str,
        transaction_id: str,
        physical_thread_id: str,
        diagnostic_hash: str,
        attempt_checkpoint_id: str | None = None,
        attempt_fingerprint: str | None = None,
        failure_category: str = "external_effect_uncertain",
    ) -> TerminalOutcome:
        if (attempt_checkpoint_id is None) != (attempt_fingerprint is None):
            raise TurnTransactionValidationError(
                "attempt checkpoint identity and fingerprint must be provided together"
            )
        return self._finish_attempt(
            logical_thread_id=logical_thread_id,
            transaction_id=transaction_id,
            physical_thread_id=physical_thread_id,
            outcome="reconciliation_required",
            attempt_checkpoint_id=attempt_checkpoint_id,
            attempt_fingerprint=attempt_fingerprint,
            diagnostic_hash=diagnostic_hash,
            render_hash="",
            failure_category=failure_category,
        )

    def resolve_reconciliation(
        self,
        *,
        logical_thread_id: str,
        transaction_id: str,
        resolution: str,
        evidence_hash: str,
    ) -> ReconciliationResolution:
        """Record one immutable operator resolution for an uncertain effect."""

        logical_thread_id = _identifier("logical_thread_id", logical_thread_id)
        transaction_id = _transaction_id(transaction_id)
        if resolution not in _RECONCILIATION_RESOLUTIONS:
            raise TurnTransactionValidationError(
                f"unsupported reconciliation resolution: {resolution}"
            )
        evidence_hash = _fingerprint("evidence_hash", evidence_hash)
        now = _utc_now()
        with self._write_transaction() as connection:
            outcome = self._require_outcome(connection, transaction_id)
            if (
                outcome.logical_thread_id != logical_thread_id
                or outcome.outcome != "reconciliation_required"
            ):
                raise TurnTransactionConflictError(
                    "transaction is not an unresolved reconciliation for this head"
                )
            existing = self._select_resolution(connection, transaction_id)
            if existing is not None:
                expected = (resolution, evidence_hash)
                actual = (existing.resolution, existing.evidence_hash)
                if actual != expected:
                    raise TurnTransactionConflictError(
                        "reconciliation already has a different immutable resolution"
                    )
                return existing
            connection.execute(
                """
                INSERT INTO anychain_reconciliation_resolutions (
                    transaction_id,
                    logical_thread_id,
                    resolution,
                    evidence_hash,
                    created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    transaction_id,
                    logical_thread_id,
                    resolution,
                    evidence_hash,
                    now,
                ),
            )
            return self._require_resolution(connection, transaction_id)

    def get_product_head(self, logical_thread_id: str) -> ProductHead | None:
        logical_thread_id = _identifier("logical_thread_id", logical_thread_id)
        with self._read_connection() as connection:
            return self._select_head(connection, logical_thread_id)

    def runtime_event_fence(
        self,
        logical_thread_id: str,
    ) -> tuple[int, str, str, str]:
        """Return the latest durably published runtime-event identity."""

        logical_thread_id = _identifier("logical_thread_id", logical_thread_id)
        with self._read_connection() as connection:
            pending = connection.execute(
                """
                SELECT event_id
                FROM anychain_terminal_outbox
                WHERE logical_thread_id = ?
                  AND runtime_event_status = 'pending'
                LIMIT 1
                """,
                (logical_thread_id,),
            ).fetchone()
            if pending is not None:
                raise TurnTransactionConflictError(
                    "runtime event publication is incomplete"
                )
            return self._published_runtime_event_fence(
                connection,
                logical_thread_id,
            )

    def get_attempt(self, transaction_id: str) -> TurnAttempt | None:
        transaction_id = _transaction_id(transaction_id)
        with self._read_connection() as connection:
            return self._select_attempt(connection, transaction_id)

    def get_terminal_outcome(self, transaction_id: str) -> TerminalOutcome | None:
        transaction_id = _transaction_id(transaction_id)
        with self._read_connection() as connection:
            return self._select_outcome(connection, transaction_id)

    def get_reconciliation_resolution(
        self,
        transaction_id: str,
    ) -> ReconciliationResolution | None:
        transaction_id = _transaction_id(transaction_id)
        with self._read_connection() as connection:
            return self._select_resolution(connection, transaction_id)

    def get_unresolved_reconciliation(
        self,
        logical_thread_id: str,
    ) -> TerminalOutcome | None:
        logical_thread_id = _identifier("logical_thread_id", logical_thread_id)
        with self._read_connection() as connection:
            return self._select_unresolved_reconciliation(
                connection,
                logical_thread_id,
            )

    def list_attempts(self, logical_thread_id: str) -> tuple[TurnAttempt, ...]:
        logical_thread_id = _identifier("logical_thread_id", logical_thread_id)
        with self._read_connection() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM anychain_turn_attempts
                WHERE logical_thread_id = ?
                ORDER BY created_at, transaction_id
                """,
                (logical_thread_id,),
            ).fetchall()
            return tuple(_attempt_from_row(row) for row in rows)

    def list_terminal_outcomes(self, logical_thread_id: str) -> tuple[TerminalOutcome, ...]:
        logical_thread_id = _identifier("logical_thread_id", logical_thread_id)
        with self._read_connection() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM anychain_terminal_outbox
                WHERE logical_thread_id = ?
                ORDER BY created_at, transaction_id
                """,
                (logical_thread_id,),
            ).fetchall()
            return tuple(_outcome_from_row(row) for row in rows)

    def list_undelivered_outcomes(
        self,
        logical_thread_id: str,
    ) -> tuple[TerminalOutcome, ...]:
        logical_thread_id = _identifier("logical_thread_id", logical_thread_id)
        with self._read_connection() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM anychain_terminal_outbox
                WHERE logical_thread_id = ? AND delivered_at IS NULL
                ORDER BY created_at, transaction_id
                """,
                (logical_thread_id,),
            ).fetchall()
            return tuple(_outcome_from_row(row) for row in rows)

    def mark_terminal_delivered(
        self,
        *,
        logical_thread_id: str,
        event_id: str,
    ) -> TerminalOutcome:
        logical_thread_id = _identifier("logical_thread_id", logical_thread_id)
        event_id = _identifier("event_id", event_id)
        with self._write_transaction() as connection:
            row = connection.execute(
                """
                SELECT transaction_id
                FROM anychain_terminal_outbox
                WHERE event_id = ? AND logical_thread_id = ?
                """,
                (event_id, logical_thread_id),
            ).fetchone()
            if row is None:
                raise TurnTransactionConflictError(
                    "terminal outcome does not belong to this Product Head"
                )
            connection.execute(
                """
                UPDATE anychain_terminal_outbox
                SET delivered_at = COALESCE(delivered_at, ?)
                WHERE event_id = ? AND logical_thread_id = ?
                """,
                (_utc_now(), event_id, logical_thread_id),
            )
            return self._require_outcome(connection, str(row["transaction_id"]))

    def mark_runtime_event_published(
        self,
        *,
        logical_thread_id: str,
        event_id: str,
        runtime_event_id: str,
        payload_hash: str,
    ) -> TerminalOutcome:
        """Acknowledge a durably appended runtime-event projection."""

        logical_thread_id = _identifier("logical_thread_id", logical_thread_id)
        event_id = _identifier("event_id", event_id)
        runtime_event_id = _identifier("runtime_event_id", runtime_event_id)
        payload_hash = _fingerprint("payload_hash", payload_hash)
        with self._write_transaction() as connection:
            row = connection.execute(
                """
                SELECT transaction_id, outcome, runtime_event_id,
                       runtime_event_status, runtime_event_payload_hash
                FROM anychain_terminal_outbox
                WHERE event_id = ? AND logical_thread_id = ?
                """,
                (event_id, logical_thread_id),
            ).fetchone()
            if row is None or str(row["outcome"]) != "committed":
                raise TurnTransactionConflictError(
                    "runtime event does not belong to a committed Product Head"
                )
            existing_id = str(row["runtime_event_id"] or "")
            if existing_id and existing_id != runtime_event_id:
                raise TurnTransactionConflictError(
                    "committed outcome is bound to a different runtime event"
                )
            existing_hash = str(row["runtime_event_payload_hash"] or "")
            if (
                str(row["runtime_event_status"]) == "published"
                and existing_hash != payload_hash
            ):
                raise TurnTransactionConflictError(
                    "published runtime event payload is immutable"
                )
            connection.execute(
                """
                UPDATE anychain_terminal_outbox
                SET runtime_event_id = ?,
                    runtime_event_status = 'published',
                    runtime_event_payload_hash = ?,
                    runtime_event_published_at = COALESCE(
                        runtime_event_published_at,
                        ?
                    )
                WHERE event_id = ? AND logical_thread_id = ?
                """,
                (
                    runtime_event_id,
                    payload_hash,
                    _utc_now(),
                    event_id,
                    logical_thread_id,
                ),
            )
            return self._require_outcome(
                connection,
                str(row["transaction_id"]),
            )

    def requeue_runtime_events_after_legacy_log_quarantine(
        self,
        logical_thread_id: str,
    ) -> int:
        """Invalidate publication receipts whose projection log was quarantined."""

        logical_thread_id = _identifier(
            "logical_thread_id",
            logical_thread_id,
        )
        with self._write_transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE anychain_terminal_outbox
                SET runtime_event_status = 'pending',
                    runtime_event_payload_hash = NULL,
                    runtime_event_published_at = NULL
                WHERE logical_thread_id = ?
                  AND outcome = 'committed'
                """,
                (logical_thread_id,),
            )
            return int(cursor.rowcount)

    def prepare_terminal_detour(
        self,
        *,
        logical_thread_id: str,
        process_instance_id: str,
        session_id: str,
        session_purpose: str,
        command_name: str,
        input_hash: str,
        effect_class: TerminalDetourEffectClass,
        shell_state_before_hash: str,
        origin_revision_commit: str,
        origin_revision_worktree_hash: str,
        result_kind: TerminalDetourResultKind = "command",
        termination_reason: TerminalTerminationReason | None = None,
        exit_code: int | None = None,
        detour_id: str | None = None,
    ) -> TerminalDetour:
        """Persist command intent before terminal-owned work starts."""

        logical_thread_id = _identifier("logical_thread_id", logical_thread_id)
        process_instance_id = _identifier(
            "process_instance_id",
            process_instance_id,
        )
        session_id = _identifier("session_id", session_id)
        session_purpose = _identifier("session_purpose", session_purpose)
        command_name = _identifier("command_name", command_name)
        input_hash = _fingerprint("input_hash", input_hash)
        if result_kind not in _DETOUR_RESULT_KINDS:
            raise TurnTransactionValidationError(
                f"unsupported terminal detour result kind: {result_kind}"
            )
        if result_kind == "session_termination":
            if termination_reason not in _TERMINATION_REASONS:
                raise TurnTransactionValidationError(
                    "session termination requires a typed reason"
                )
            if exit_code not in {0, 130}:
                raise TurnTransactionValidationError(
                    "session termination requires exit code 0 or 130"
                )
        elif termination_reason is not None or exit_code is not None:
            raise TurnTransactionValidationError(
                "command detour cannot carry termination fields"
            )
        if effect_class not in _DETOUR_EFFECT_CLASSES:
            raise TurnTransactionValidationError(
                f"unsupported terminal detour effect class: {effect_class}"
            )
        shell_state_before_hash = _fingerprint(
            "shell_state_before_hash",
            shell_state_before_hash,
        )
        origin_revision_commit = _identifier(
            "origin_revision_commit",
            origin_revision_commit,
        )
        origin_revision_worktree_hash = _fingerprint(
            "origin_revision_worktree_hash",
            origin_revision_worktree_hash,
        )
        detour_id = _transaction_id(detour_id or str(uuid.uuid4()))
        now = _utc_now()
        with self._write_transaction() as connection:
            existing = self._select_detour(connection, detour_id)
            head = self._select_head(connection, logical_thread_id)
            head_identity = _head_identity(head)
            pending_runtime_event = connection.execute(
                """
                SELECT event_id
                FROM anychain_terminal_outbox
                WHERE logical_thread_id = ?
                  AND runtime_event_status = 'pending'
                LIMIT 1
                """,
                (logical_thread_id,),
            ).fetchone()
            if pending_runtime_event is not None:
                raise TurnTransactionConflictError(
                    "terminal detour requires all committed runtime events "
                    "to be durably published"
                )
            runtime_event_fence = self._published_runtime_event_fence(
                connection,
                logical_thread_id,
            )
            if existing is not None:
                expected = (
                    logical_thread_id,
                    process_instance_id,
                    session_id,
                    session_purpose,
                    command_name,
                    input_hash,
                    result_kind,
                    termination_reason,
                    exit_code,
                    effect_class,
                    shell_state_before_hash,
                    *head_identity,
                    *runtime_event_fence,
                    origin_revision_commit,
                    origin_revision_worktree_hash,
                )
                actual = (
                    existing.logical_thread_id,
                    existing.process_instance_id,
                    existing.session_id,
                    existing.session_purpose,
                    existing.command_name,
                    existing.input_hash,
                    existing.result_kind,
                    existing.termination_reason,
                    existing.exit_code,
                    existing.effect_class,
                    existing.shell_state_before_hash,
                    existing.product_revision_before,
                    existing.product_checkpoint_thread_id_before,
                    existing.product_checkpoint_id_before,
                    existing.product_fingerprint_before,
                    existing.runtime_event_fence_sequence,
                    existing.runtime_event_fence_terminal_event_id,
                    existing.runtime_event_fence_id,
                    existing.runtime_event_fence_hash,
                    existing.origin_revision_commit,
                    existing.origin_revision_worktree_hash,
                )
                if actual != expected:
                    raise TurnTransactionConflictError(
                        "detour_id is already bound to different immutable facts"
                    )
                return existing
            unresolved = self._select_prepared_detour(
                connection,
                logical_thread_id,
            )
            if unresolved is not None:
                raise ReconciliationRequiredError(unresolved.detour_id)
            active = connection.execute(
                """
                SELECT transaction_id
                FROM anychain_turn_attempts
                WHERE logical_thread_id = ? AND status = 'active'
                """,
                (logical_thread_id,),
            ).fetchone()
            if active is not None:
                raise TurnTransactionConflictError(
                    "terminal detour cannot start while a workflow attempt is active"
                )
            connection.execute(
                """
                INSERT INTO anychain_terminal_detours (
                    detour_id,
                    logical_thread_id,
                    process_instance_id,
                    session_id,
                    session_purpose,
                    command_name,
                    input_hash,
                    result_kind,
                    termination_reason,
                    exit_code,
                    effect_class,
                    status,
                    shell_state_before_hash,
                    product_revision_before,
                    product_checkpoint_thread_id_before,
                    product_checkpoint_id_before,
                    product_fingerprint_before,
                    runtime_event_fence_sequence,
                    runtime_event_fence_terminal_event_id,
                    runtime_event_fence_id,
                    runtime_event_fence_hash,
                    response_json,
                    effect_status,
                    origin_revision_commit,
                    origin_revision_worktree_hash,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'prepared', ?, ?,
                          ?, ?, ?, ?, ?, ?, ?, '[]', 'not_applicable', ?, ?, ?)
                """,
                (
                    detour_id,
                    logical_thread_id,
                    process_instance_id,
                    session_id,
                    session_purpose,
                    command_name,
                    input_hash,
                    result_kind,
                    termination_reason,
                    exit_code,
                    effect_class,
                    shell_state_before_hash,
                    *head_identity,
                    *runtime_event_fence,
                    origin_revision_commit,
                    origin_revision_worktree_hash,
                    now,
                ),
            )
            return self._require_detour(connection, detour_id)

    def record_terminal_detour_stream_chunk(
        self,
        *,
        detour_id: str,
        chunk_message: str,
    ) -> TerminalDetour:
        """Advance the durable rolling hash before one stream chunk is shown."""

        detour_id = _transaction_id(detour_id)
        chunk_message = str(chunk_message)
        if not chunk_message.strip():
            raise TurnTransactionValidationError(
                "stream chunk message cannot be empty"
            )
        chunk_hash = state_fingerprint(chunk_message.encode("utf-8"))
        with self._write_transaction() as connection:
            detour = self._require_detour(connection, detour_id)
            if detour.status != "prepared":
                raise TurnTransactionConflictError(
                    "only a prepared detour can accept stream chunks"
                )
            if detour.effect_class != "streaming_observation":
                raise TurnTransactionConflictError(
                    "stream chunks require a streaming-observation detour"
                )
            next_hash = state_fingerprint(
                f"{detour.stream_hash}:{chunk_hash}".encode("ascii")
            )
            next_messages = json.dumps(
                [*detour.stream_messages, chunk_message],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            connection.execute(
                """
                UPDATE anychain_terminal_detours
                SET stream_chunk_count = stream_chunk_count + 1,
                    stream_hash = ?,
                    stream_messages_json = ?
                WHERE detour_id = ? AND status = 'prepared'
                """,
                (next_hash, next_messages, detour_id),
            )
            return self._require_detour(connection, detour_id)

    def complete_terminal_detour(
        self,
        *,
        detour_id: str,
        shell_state_after_hash: str,
        response_messages: Sequence[str],
        effect_status: TerminalDetourEffectStatus = "not_applicable",
        diagnostic_hash: str | None = None,
        stream_stop_reason: TerminalStreamStopReason = "not_streaming",
    ) -> TerminalDetour:
        """Complete a detour only while the Product Head is unchanged."""

        detour_id = _transaction_id(detour_id)
        shell_state_after_hash = _fingerprint(
            "shell_state_after_hash",
            shell_state_after_hash,
        )
        if effect_status not in _DETOUR_EFFECT_STATUSES - {"uncertain"}:
            raise TurnTransactionValidationError(
                f"unsupported completed detour effect status: {effect_status}"
            )
        messages = tuple(str(message) for message in response_messages)
        if not messages or any(not message.strip() for message in messages):
            raise TurnTransactionValidationError(
                "completed terminal detour requires non-empty response messages"
            )
        response_json = json.dumps(
            list(messages),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        response_hash = state_fingerprint(response_json.encode("utf-8"))
        if stream_stop_reason not in _STREAM_STOP_REASONS:
            raise TurnTransactionValidationError(
                f"unsupported stream stop reason: {stream_stop_reason}"
            )
        if diagnostic_hash is not None:
            diagnostic_hash = _fingerprint("diagnostic_hash", diagnostic_hash)
        if effect_status == "failed" and diagnostic_hash is None:
            raise TurnTransactionValidationError(
                "failed terminal detour requires diagnostic_hash"
            )
        if effect_status != "failed" and diagnostic_hash is not None:
            raise TurnTransactionValidationError(
                "successful terminal detour cannot carry diagnostic_hash"
            )
        now = _utc_now()
        with self._write_transaction() as connection:
            detour = self._require_detour(connection, detour_id)
            if detour.status == "completed":
                expected = (
                    shell_state_after_hash,
                    messages,
                    response_hash,
                    effect_status,
                    diagnostic_hash,
                    stream_stop_reason,
                )
                actual = (
                    detour.shell_state_after_hash,
                    detour.response_messages,
                    detour.response_hash,
                    detour.effect_status,
                    detour.diagnostic_hash,
                    detour.stream_stop_reason,
                )
                if actual != expected:
                    raise TurnTransactionConflictError(
                        "terminal detour already has a different completion"
                    )
                return detour
            if detour.status != "prepared":
                raise TurnTransactionConflictError(
                    "terminal detour requires reconciliation before completion"
                )
            if detour.effect_class == "streaming_observation":
                if stream_stop_reason == "not_streaming":
                    raise TurnTransactionValidationError(
                        "streaming detour requires a typed stop reason"
                    )
                if len(messages) < detour.stream_chunk_count:
                    raise TurnTransactionValidationError(
                        "streaming response omits persisted stream chunks"
                    )
                observed_stream_hash = _rolling_stream_hash(
                    messages[:detour.stream_chunk_count]
                )
                if observed_stream_hash != detour.stream_hash:
                    raise TurnTransactionConflictError(
                        "streaming response does not match persisted chunk evidence"
                    )
            elif stream_stop_reason != "not_streaming":
                raise TurnTransactionValidationError(
                    "non-streaming detour cannot carry a stream stop reason"
                )
            head = self._select_head(connection, detour.logical_thread_id)
            head_identity = _head_identity(head)
            pending_runtime_event = connection.execute(
                """
                SELECT event_id
                FROM anychain_terminal_outbox
                WHERE logical_thread_id = ?
                  AND runtime_event_status = 'pending'
                LIMIT 1
                """,
                (detour.logical_thread_id,),
            ).fetchone()
            runtime_event_fence = self._published_runtime_event_fence(
                connection,
                detour.logical_thread_id,
            )
            expected_head = (
                detour.product_revision_before,
                detour.product_checkpoint_thread_id_before,
                detour.product_checkpoint_id_before,
                detour.product_fingerprint_before,
            )
            if head_identity != expected_head:
                raise TurnTransactionConflictError(
                    "terminal detour observed a Product Head mutation"
                )
            if (
                pending_runtime_event is not None
                or runtime_event_fence
                != (
                    detour.runtime_event_fence_sequence,
                    detour.runtime_event_fence_terminal_event_id,
                    detour.runtime_event_fence_id,
                    detour.runtime_event_fence_hash,
                )
            ):
                raise TurnTransactionConflictError(
                    "terminal detour crossed a committed runtime-event fence"
                )
            connection.execute(
                """
                UPDATE anychain_terminal_detours
                SET status = 'completed',
                    shell_state_after_hash = ?,
                    product_revision_after = ?,
                    product_checkpoint_thread_id_after = ?,
                    product_checkpoint_id_after = ?,
                    product_fingerprint_after = ?,
                    response_json = ?,
                    response_hash = ?,
                    effect_status = ?,
                    diagnostic_hash = ?,
                    stream_stop_reason = ?,
                    completed_at = ?
                WHERE detour_id = ? AND status = 'prepared'
                """,
                (
                    shell_state_after_hash,
                    *head_identity,
                    response_json,
                    response_hash,
                    effect_status,
                    diagnostic_hash,
                    stream_stop_reason,
                    now,
                    detour_id,
                ),
            )
            return self._require_detour(connection, detour_id)

    def require_terminal_detour_reconciliation(
        self,
        *,
        detour_id: str,
        diagnostic_hash: str,
    ) -> TerminalDetour:
        """Fail closed when an external/streaming detour may have taken effect."""

        detour_id = _transaction_id(detour_id)
        diagnostic_hash = _fingerprint("diagnostic_hash", diagnostic_hash)
        with self._write_transaction() as connection:
            detour = self._require_detour(connection, detour_id)
            if detour.status == "reconciliation_required":
                if detour.diagnostic_hash != diagnostic_hash:
                    raise TurnTransactionConflictError(
                        "terminal detour reconciliation evidence changed"
                    )
                return detour
            if detour.status != "prepared":
                raise TurnTransactionConflictError(
                    "only a prepared terminal detour can require reconciliation"
                )
            if detour.effect_class != "local_external_effect":
                raise TurnTransactionConflictError(
                    "non-effect terminal detour cannot require reconciliation"
                )
            connection.execute(
                """
                UPDATE anychain_terminal_detours
                SET status = 'reconciliation_required',
                    effect_status = 'uncertain',
                    diagnostic_hash = ?
                WHERE detour_id = ? AND status = 'prepared'
                """,
                (diagnostic_hash, detour_id),
            )
            return self._require_detour(connection, detour_id)

    def list_undelivered_terminal_detours(
        self,
        logical_thread_id: str,
    ) -> tuple[TerminalDetour, ...]:
        logical_thread_id = _identifier("logical_thread_id", logical_thread_id)
        with self._read_connection() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM anychain_terminal_detours
                WHERE logical_thread_id = ?
                  AND status = 'completed'
                  AND delivered_at IS NULL
                  AND identity_trust = 'trusted'
                ORDER BY created_at, detour_id
                """,
                (logical_thread_id,),
            ).fetchall()
            return tuple(_detour_from_row(row) for row in rows)

    def list_terminal_detours(
        self,
        logical_thread_id: str,
    ) -> tuple[TerminalDetour, ...]:
        logical_thread_id = _identifier("logical_thread_id", logical_thread_id)
        with self._read_connection() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM anychain_terminal_detours
                WHERE logical_thread_id = ?
                ORDER BY created_at, detour_id
                """,
                (logical_thread_id,),
            ).fetchall()
            return tuple(_detour_from_row(row) for row in rows)

    def get_unresolved_terminal_detour(
        self,
        logical_thread_id: str,
    ) -> TerminalDetour | None:
        logical_thread_id = _identifier("logical_thread_id", logical_thread_id)
        with self._read_connection() as connection:
            return self._select_unresolved_detour(connection, logical_thread_id)

    def recover_prepared_terminal_detours(
        self,
        *,
        logical_thread_id: str,
        interrupted_message: str,
    ) -> tuple[TerminalDetour, ...]:
        """Recover command intents left open by a process interruption."""

        logical_thread_id = _identifier("logical_thread_id", logical_thread_id)
        with self._read_connection() as connection:
            rows = connection.execute(
                """
                SELECT detour_id
                FROM anychain_terminal_detours
                WHERE logical_thread_id = ? AND status = 'prepared'
                ORDER BY created_at, detour_id
                """,
                (logical_thread_id,),
            ).fetchall()
        recovered: list[TerminalDetour] = []
        diagnostic_hash = state_fingerprint(
            b"terminal-detour-process-interrupted"
        )
        for row in rows:
            detour_id = str(row["detour_id"])
            with self._read_connection() as connection:
                detour = self._require_detour(connection, detour_id)
            if detour.identity_trust != "trusted":
                raise TurnTransactionSchemaError(
                    "legacy terminal detour has unverifiable identity and is quarantined"
                )
            if detour.effect_class == "local_external_effect":
                recovered.append(
                    self.require_terminal_detour_reconciliation(
                        detour_id=detour_id,
                        diagnostic_hash=diagnostic_hash,
                    )
                )
            else:
                result_kind: TerminalDetourResultKind = detour.result_kind
                stream_stop_reason: TerminalStreamStopReason = "not_streaming"
                if detour.result_kind == "session_termination":
                    result_kind = "command"
                if detour.effect_class == "streaming_observation":
                    stream_stop_reason = "interrupted"
                if result_kind != detour.result_kind:
                    with self._write_transaction() as connection:
                        connection.execute(
                            """
                            UPDATE anychain_terminal_detours
                            SET result_kind = ?,
                                interruption_kind =
                                    'session_termination_interrupted'
                            WHERE detour_id = ? AND status = 'prepared'
                            """,
                            (result_kind, detour_id),
                        )
                recovered.append(
                    self.complete_terminal_detour(
                        detour_id=detour_id,
                        shell_state_after_hash=detour.shell_state_before_hash,
                        response_messages=(
                            *detour.stream_messages,
                            interrupted_message,
                        ),
                        effect_status="failed",
                        diagnostic_hash=diagnostic_hash,
                        stream_stop_reason=stream_stop_reason,
                    )
                )
        return tuple(recovered)

    def resolve_with_terminal_detour(
        self,
        *,
        logical_thread_id: str,
        process_instance_id: str,
        session_id: str,
        session_purpose: str,
        target_id: str,
        input_hash: str,
        resolution: str,
        evidence_hash: str,
        shell_state_hash: str,
        response_messages: Sequence[str],
        origin_revision_commit: str,
        origin_revision_worktree_hash: str,
        detour_id: str | None = None,
    ) -> TerminalDetour:
        """Atomically resolve uncertain authority and persist its terminal reply."""

        logical_thread_id = _identifier("logical_thread_id", logical_thread_id)
        process_instance_id = _identifier(
            "process_instance_id",
            process_instance_id,
        )
        session_id = _identifier("session_id", session_id)
        session_purpose = _identifier("session_purpose", session_purpose)
        target_id = _transaction_id(target_id)
        input_hash = _fingerprint("input_hash", input_hash)
        if resolution not in _RECONCILIATION_RESOLUTIONS:
            raise TurnTransactionValidationError(
                f"unsupported reconciliation resolution: {resolution}"
            )
        evidence_hash = _fingerprint("evidence_hash", evidence_hash)
        shell_state_hash = _fingerprint(
            "shell_state_hash",
            shell_state_hash,
        )
        origin_revision_commit = _identifier(
            "origin_revision_commit",
            origin_revision_commit,
        )
        origin_revision_worktree_hash = _fingerprint(
            "origin_revision_worktree_hash",
            origin_revision_worktree_hash,
        )
        messages = tuple(str(message) for message in response_messages)
        if not messages or any(not message.strip() for message in messages):
            raise TurnTransactionValidationError(
                "reconciliation detour requires non-empty response messages"
            )
        response_json = json.dumps(
            list(messages),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        response_hash = state_fingerprint(response_json.encode("utf-8"))
        detour_id = _transaction_id(detour_id or str(uuid.uuid4()))
        now = _utc_now()
        with self._write_transaction() as connection:
            if self._select_detour(connection, detour_id) is not None:
                raise TurnTransactionConflictError(
                    "reconciliation detour identity already exists"
                )
            head = self._select_head(connection, logical_thread_id)
            head_identity = _head_identity(head)
            target_outcome = self._select_outcome(connection, target_id)
            target_detour = self._select_detour(connection, target_id)
            if target_outcome is not None:
                if (
                    target_outcome.logical_thread_id != logical_thread_id
                    or target_outcome.outcome != "reconciliation_required"
                ):
                    raise TurnTransactionConflictError(
                        "target is not an unresolved workflow reconciliation"
                    )
                existing_resolution = self._select_resolution(
                    connection,
                    target_id,
                )
                if existing_resolution is not None:
                    raise TurnTransactionConflictError(
                        "workflow reconciliation is already resolved"
                    )
                connection.execute(
                    """
                    INSERT INTO anychain_reconciliation_resolutions (
                        transaction_id,
                        logical_thread_id,
                        resolution,
                        evidence_hash,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        target_id,
                        logical_thread_id,
                        resolution,
                        evidence_hash,
                        now,
                    ),
                )
            elif target_detour is not None:
                if (
                    target_detour.logical_thread_id != logical_thread_id
                    or target_detour.status != "reconciliation_required"
                ):
                    raise TurnTransactionConflictError(
                        "target is not an unresolved terminal reconciliation"
                    )
                connection.execute(
                    """
                    UPDATE anychain_terminal_detours
                    SET status = 'completed',
                        shell_state_after_hash = shell_state_before_hash,
                        product_revision_after = product_revision_before,
                        product_checkpoint_thread_id_after =
                            product_checkpoint_thread_id_before,
                        product_checkpoint_id_after =
                            product_checkpoint_id_before,
                        product_fingerprint_after =
                            product_fingerprint_before,
                        response_json = '[]',
                        response_hash = ?,
                        effect_status = ?,
                        resolution = ?,
                        evidence_hash = ?,
                        completed_at = ?,
                        delivered_at = ?
                    WHERE detour_id = ?
                      AND status = 'reconciliation_required'
                    """,
                    (
                        state_fingerprint(b"[]"),
                        (
                            "succeeded"
                            if resolution == "effect_confirmed"
                            else "failed"
                        ),
                        resolution,
                        evidence_hash,
                        now,
                        now,
                        target_id,
                    ),
                )
            else:
                raise TurnTransactionConflictError(
                    "unknown reconciliation target"
                )
            if _head_identity(
                self._select_head(connection, logical_thread_id)
            ) != head_identity:
                raise TurnTransactionConflictError(
                    "reconciliation detour observed a Product Head mutation"
                )
            connection.execute(
                """
                INSERT INTO anychain_terminal_detours (
                    detour_id,
                    logical_thread_id,
                    process_instance_id,
                    session_id,
                    session_purpose,
                    command_name,
                    input_hash,
                    result_kind,
                    effect_class,
                    status,
                    shell_state_before_hash,
                    shell_state_after_hash,
                    product_revision_before,
                    product_checkpoint_thread_id_before,
                    product_checkpoint_id_before,
                    product_fingerprint_before,
                    product_revision_after,
                    product_checkpoint_thread_id_after,
                    product_checkpoint_id_after,
                    product_fingerprint_after,
                    response_json,
                    response_hash,
                    effect_status,
                    resolution,
                    evidence_hash,
                    origin_revision_commit,
                    origin_revision_worktree_hash,
                    created_at,
                    completed_at
                ) VALUES (?, ?, ?, ?, ?, 'reconcile', ?, 'command',
                          'authority_resolution',
                          'completed', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                          'not_applicable', ?, ?, ?, ?, ?, ?)
                """,
                (
                    detour_id,
                    logical_thread_id,
                    process_instance_id,
                    session_id,
                    session_purpose,
                    input_hash,
                    shell_state_hash,
                    shell_state_hash,
                    *head_identity,
                    *head_identity,
                    response_json,
                    response_hash,
                    resolution,
                    evidence_hash,
                    origin_revision_commit,
                    origin_revision_worktree_hash,
                    now,
                    now,
                ),
            )
            return self._require_detour(connection, detour_id)

    def mark_terminal_detour_delivered(
        self,
        *,
        logical_thread_id: str,
        detour_id: str,
    ) -> TerminalDetour:
        logical_thread_id = _identifier("logical_thread_id", logical_thread_id)
        detour_id = _transaction_id(detour_id)
        with self._write_transaction() as connection:
            detour = self._require_detour(connection, detour_id)
            if (
                detour.logical_thread_id != logical_thread_id
                or detour.status != "completed"
            ):
                raise TurnTransactionConflictError(
                    "terminal detour is not deliverable for this Product Head"
                )
            connection.execute(
                """
                UPDATE anychain_terminal_detours
                SET delivered_at = COALESCE(delivered_at, ?)
                WHERE detour_id = ?
                """,
                (_utc_now(), detour_id),
            )
            return self._require_detour(connection, detour_id)

    def _finish_attempt(
        self,
        *,
        logical_thread_id: str,
        transaction_id: str,
        physical_thread_id: str,
        outcome: TerminalOutcomeKind,
        attempt_checkpoint_id: str | None,
        attempt_fingerprint: str | None,
        diagnostic_hash: str | None,
        render_hash: str,
        failure_category: str,
    ) -> TerminalOutcome:
        logical_thread_id = _identifier("logical_thread_id", logical_thread_id)
        transaction_id = _transaction_id(transaction_id)
        physical_thread_id = _identifier("physical_thread_id", physical_thread_id)
        if outcome not in _TERMINAL_OUTCOMES:
            raise TurnTransactionValidationError(f"unsupported terminal outcome: {outcome}")
        if attempt_checkpoint_id is not None:
            attempt_checkpoint_id = _identifier("attempt_checkpoint_id", attempt_checkpoint_id)
        if attempt_fingerprint is not None:
            attempt_fingerprint = _fingerprint("attempt_fingerprint", attempt_fingerprint)
        if diagnostic_hash is not None:
            diagnostic_hash = _fingerprint("diagnostic_hash", diagnostic_hash)
        if render_hash:
            render_hash = _fingerprint("render_hash", render_hash)
        failure_category = str(failure_category or "").strip()
        if failure_category:
            failure_category = _identifier(
                "failure_category",
                failure_category,
            )
        if outcome == "committed" and (
            attempt_checkpoint_id is None
            or attempt_fingerprint is None
            or diagnostic_hash is not None
            or not render_hash
            or failure_category
        ):
            raise TurnTransactionValidationError(
                "committed outcome requires checkpoint/render hashes and no failure metadata"
            )
        if outcome != "committed" and (
            diagnostic_hash is None or render_hash or not failure_category
        ):
            raise TurnTransactionValidationError(
                f"{outcome} outcome requires diagnostic/failure metadata only"
            )

        now = _utc_now()
        with self._write_transaction() as connection:
            attempt = self._require_attempt(connection, transaction_id)
            self._validate_attempt_binding(
                attempt,
                logical_thread_id=logical_thread_id,
                physical_thread_id=physical_thread_id,
            )
            existing = self._select_outcome(connection, transaction_id)
            if existing is not None:
                self._validate_idempotent_outcome(
                    existing,
                    outcome=outcome,
                    attempt_checkpoint_id=attempt_checkpoint_id,
                    attempt_fingerprint=attempt_fingerprint,
                    diagnostic_hash=diagnostic_hash,
                    render_hash=render_hash,
                    failure_category=failure_category,
                )
                return existing
            if attempt.status != "active":
                raise TurnTransactionConflictError(
                    "terminal attempt is missing its immutable outbox outcome"
                )
            head = self._require_head(connection, logical_thread_id)
            self._validate_base_head(attempt, head)

            if outcome == "committed":
                product_revision = head.revision + 1
                product_checkpoint_thread_id = attempt.physical_thread_id
                product_checkpoint_id = attempt_checkpoint_id
                product_fingerprint = attempt_fingerprint
                connection.execute(
                    """
                    UPDATE anychain_product_heads
                    SET checkpoint_thread_id = ?,
                        checkpoint_id = ?,
                        state_fingerprint = ?,
                        revision = revision + 1,
                        updated_at = ?
                    WHERE logical_thread_id = ?
                    """,
                    (
                        product_checkpoint_thread_id,
                        product_checkpoint_id,
                        product_fingerprint,
                        now,
                        logical_thread_id,
                    ),
                )
            else:
                product_revision = head.revision
                product_checkpoint_thread_id = head.checkpoint_thread_id
                product_checkpoint_id = head.checkpoint_id
                product_fingerprint = head.state_fingerprint

            connection.execute(
                """
                UPDATE anychain_turn_attempts
                SET status = ?,
                    attempt_checkpoint_id = ?,
                    attempt_fingerprint = ?,
                    diagnostic_hash = ?,
                    completed_at = ?
                WHERE transaction_id = ? AND status = 'active'
                """,
                (
                    outcome,
                    attempt_checkpoint_id,
                    attempt_fingerprint,
                    diagnostic_hash,
                    now,
                    transaction_id,
                ),
            )
            event_id = str(uuid.uuid4())
            runtime_event_id = (
                str(uuid.uuid4()) if outcome == "committed" else ""
            )
            runtime_event_status = (
                "pending" if outcome == "committed" else "not_applicable"
            )
            connection.execute(
                """
                INSERT INTO anychain_terminal_outbox (
                    event_id,
                    transaction_id,
                    logical_thread_id,
                    physical_thread_id,
                    outcome,
                    base_revision,
                    base_checkpoint_thread_id,
                    base_checkpoint_id,
                    base_fingerprint,
                    origin_revision_commit,
                    origin_revision_worktree_hash,
                    attempt_checkpoint_id,
                    attempt_fingerprint,
                    product_revision,
                    product_checkpoint_thread_id,
                    product_checkpoint_id,
                    product_fingerprint,
                    diagnostic_hash,
                    render_hash,
                    failure_category,
                    runtime_event_id,
                    runtime_event_sequence,
                    runtime_event_status,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    transaction_id,
                    logical_thread_id,
                    physical_thread_id,
                    outcome,
                    attempt.base_revision,
                    attempt.base_checkpoint_thread_id,
                    attempt.base_checkpoint_id,
                    attempt.base_fingerprint,
                    attempt.origin_revision_commit,
                    attempt.origin_revision_worktree_hash,
                    attempt_checkpoint_id,
                    attempt_fingerprint,
                    product_revision,
                    product_checkpoint_thread_id,
                    product_checkpoint_id,
                    product_fingerprint,
                    diagnostic_hash,
                    render_hash,
                    failure_category,
                    runtime_event_id,
                    product_revision,
                    runtime_event_status,
                    now,
                ),
            )
            return self._require_outcome(connection, transaction_id)

    def _ensure_schema(self) -> None:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                schema_sql = """
                    CREATE TABLE IF NOT EXISTS anychain_turn_transaction_meta (
                        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                        schema_version INTEGER NOT NULL,
                        created_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS anychain_product_heads (
                        logical_thread_id TEXT PRIMARY KEY,
                        checkpoint_thread_id TEXT NOT NULL,
                        checkpoint_id TEXT NOT NULL,
                        state_fingerprint TEXT NOT NULL,
                        revision INTEGER NOT NULL CHECK (revision >= 0),
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS anychain_turn_attempts (
                        transaction_id TEXT PRIMARY KEY,
                        logical_thread_id TEXT NOT NULL,
                        physical_thread_id TEXT NOT NULL UNIQUE,
                        base_revision INTEGER NOT NULL CHECK (base_revision >= 0),
                        base_checkpoint_thread_id TEXT NOT NULL,
                        base_checkpoint_id TEXT NOT NULL,
                        base_fingerprint TEXT NOT NULL,
                        origin_revision_commit TEXT NOT NULL,
                        origin_revision_worktree_hash TEXT NOT NULL,
                        status TEXT NOT NULL CHECK (
                            status IN (
                                'active',
                                'committed',
                                'aborted',
                                'reconciliation_required'
                            )
                        ),
                        attempt_checkpoint_id TEXT,
                        attempt_fingerprint TEXT,
                        diagnostic_hash TEXT,
                        created_at TEXT NOT NULL,
                        completed_at TEXT,
                        FOREIGN KEY (logical_thread_id)
                            REFERENCES anychain_product_heads(logical_thread_id)
                    );

                    CREATE UNIQUE INDEX IF NOT EXISTS anychain_one_active_attempt_per_thread
                    ON anychain_turn_attempts(logical_thread_id)
                    WHERE status = 'active';

                    CREATE TABLE IF NOT EXISTS anychain_terminal_outbox (
                        event_id TEXT PRIMARY KEY,
                        transaction_id TEXT NOT NULL UNIQUE,
                        logical_thread_id TEXT NOT NULL,
                        physical_thread_id TEXT NOT NULL,
                        outcome TEXT NOT NULL CHECK (
                            outcome IN ('committed', 'aborted', 'reconciliation_required')
                        ),
                        base_revision INTEGER NOT NULL CHECK (base_revision >= 0),
                        base_checkpoint_thread_id TEXT NOT NULL,
                        base_checkpoint_id TEXT NOT NULL,
                        base_fingerprint TEXT NOT NULL,
                        origin_revision_commit TEXT NOT NULL,
                        origin_revision_worktree_hash TEXT NOT NULL,
                        attempt_checkpoint_id TEXT,
                        attempt_fingerprint TEXT,
                        product_revision INTEGER NOT NULL CHECK (product_revision >= 0),
                        product_checkpoint_thread_id TEXT NOT NULL,
                        product_checkpoint_id TEXT NOT NULL,
                        product_fingerprint TEXT NOT NULL,
                        diagnostic_hash TEXT,
                        render_hash TEXT NOT NULL DEFAULT '',
                        failure_category TEXT NOT NULL DEFAULT '',
                        runtime_event_id TEXT NOT NULL DEFAULT '',
                        runtime_event_sequence INTEGER NOT NULL DEFAULT 0
                            CHECK (runtime_event_sequence >= 0),
                        runtime_event_status TEXT NOT NULL DEFAULT 'not_applicable'
                            CHECK (
                                runtime_event_status IN (
                                    'pending',
                                    'published',
                                    'not_applicable'
                                )
                            ),
                        runtime_event_payload_hash TEXT,
                        runtime_event_published_at TEXT,
                        created_at TEXT NOT NULL,
                        delivered_at TEXT,
                        FOREIGN KEY (transaction_id)
                            REFERENCES anychain_turn_attempts(transaction_id),
                        FOREIGN KEY (logical_thread_id)
                            REFERENCES anychain_product_heads(logical_thread_id)
                    );

                    CREATE TABLE IF NOT EXISTS anychain_reconciliation_resolutions (
                        transaction_id TEXT PRIMARY KEY,
                        logical_thread_id TEXT NOT NULL,
                        resolution TEXT NOT NULL CHECK (
                            resolution IN ('effect_confirmed', 'effect_not_observed')
                        ),
                        evidence_hash TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (transaction_id)
                            REFERENCES anychain_turn_attempts(transaction_id),
                        FOREIGN KEY (logical_thread_id)
                            REFERENCES anychain_product_heads(logical_thread_id)
                    );

                    CREATE TABLE IF NOT EXISTS anychain_terminal_detours (
                        detour_id TEXT PRIMARY KEY,
                        logical_thread_id TEXT NOT NULL,
                        process_instance_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        session_purpose TEXT NOT NULL,
                        command_name TEXT NOT NULL,
                        input_hash TEXT NOT NULL,
                        result_kind TEXT NOT NULL CHECK (
                            result_kind IN ('command', 'session_termination')
                        ),
                        interruption_kind TEXT,
                        termination_reason TEXT CHECK (
                            termination_reason IN ('exit', 'eof', 'outer_ctrl_c')
                        ),
                        exit_code INTEGER,
                        effect_class TEXT NOT NULL CHECK (
                            effect_class IN (
                                'read_only',
                                'observation_refresh',
                                'shell_state_mutation',
                                'local_external_effect',
                                'authority_resolution',
                                'streaming_observation'
                            )
                        ),
                        status TEXT NOT NULL CHECK (
                            status IN (
                                'prepared',
                                'completed',
                                'reconciliation_required'
                            )
                        ),
                        shell_state_before_hash TEXT NOT NULL,
                        shell_state_after_hash TEXT,
                        product_revision_before INTEGER,
                        product_checkpoint_thread_id_before TEXT,
                        product_checkpoint_id_before TEXT,
                        product_fingerprint_before TEXT,
                        product_revision_after INTEGER,
                        product_checkpoint_thread_id_after TEXT,
                        product_checkpoint_id_after TEXT,
                        product_fingerprint_after TEXT,
                        runtime_event_fence_sequence INTEGER NOT NULL DEFAULT 0
                            CHECK (runtime_event_fence_sequence >= 0),
                        runtime_event_fence_terminal_event_id TEXT NOT NULL
                            DEFAULT '',
                        runtime_event_fence_id TEXT NOT NULL DEFAULT '',
                        runtime_event_fence_hash TEXT NOT NULL DEFAULT '',
                        response_json TEXT NOT NULL DEFAULT '[]',
                        response_hash TEXT,
                        stream_chunk_count INTEGER NOT NULL DEFAULT 0,
                        stream_hash TEXT NOT NULL DEFAULT
                            'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
                        stream_messages_json TEXT NOT NULL DEFAULT '[]',
                        stream_stop_reason TEXT NOT NULL DEFAULT 'not_streaming',
                        identity_trust TEXT NOT NULL DEFAULT 'trusted',
                        effect_status TEXT NOT NULL CHECK (
                            effect_status IN (
                                'not_applicable',
                                'succeeded',
                                'failed',
                                'uncertain'
                            )
                        ),
                        diagnostic_hash TEXT,
                        resolution TEXT,
                        evidence_hash TEXT,
                        origin_revision_commit TEXT NOT NULL,
                        origin_revision_worktree_hash TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        completed_at TEXT,
                        delivered_at TEXT
                    );

                    CREATE UNIQUE INDEX IF NOT EXISTS anychain_one_active_detour_per_thread
                    ON anychain_terminal_detours(logical_thread_id)
                    WHERE status = 'prepared';

                    CREATE TABLE IF NOT EXISTS anychain_terminal_detour_quarantine (
                        detour_id TEXT PRIMARY KEY,
                        logical_thread_id TEXT NOT NULL,
                        command_name TEXT NOT NULL,
                        quarantine_reason TEXT NOT NULL,
                        record_json TEXT NOT NULL,
                        record_hash TEXT NOT NULL,
                        quarantined_at TEXT NOT NULL
                    );
                    """
                for statement in schema_sql.split(";"):
                    if statement.strip():
                        connection.execute(statement)
                row = connection.execute(
                    """
                    SELECT schema_version
                    FROM anychain_turn_transaction_meta
                    WHERE singleton = 1
                    """
                ).fetchone()
                if row is None:
                    connection.execute(
                        """
                        INSERT INTO anychain_turn_transaction_meta (
                            singleton,
                            schema_version,
                            created_at
                        ) VALUES (1, ?, ?)
                        """,
                        (TURN_TRANSACTION_SCHEMA_VERSION, _utc_now()),
                    )
                elif int(row["schema_version"]) == 1:
                    self._migrate_v1_to_v2(connection)
                    self._migrate_v2_to_v3(connection)
                    self._migrate_v3_to_v4(connection)
                    self._migrate_v4_to_v5(connection)
                    self._migrate_v5_to_v6(connection)
                    self._migrate_v6_to_v7(connection)
                    self._migrate_v7_to_v8(connection)
                    self._migrate_v8_to_v9(connection)
                elif int(row["schema_version"]) == 2:
                    self._migrate_v2_to_v3(connection)
                    self._migrate_v3_to_v4(connection)
                    self._migrate_v4_to_v5(connection)
                    self._migrate_v5_to_v6(connection)
                    self._migrate_v6_to_v7(connection)
                    self._migrate_v7_to_v8(connection)
                    self._migrate_v8_to_v9(connection)
                elif int(row["schema_version"]) == 3:
                    self._migrate_v3_to_v4(connection)
                    self._migrate_v4_to_v5(connection)
                    self._migrate_v5_to_v6(connection)
                    self._migrate_v6_to_v7(connection)
                    self._migrate_v7_to_v8(connection)
                    self._migrate_v8_to_v9(connection)
                elif int(row["schema_version"]) == 4:
                    self._migrate_v4_to_v5(connection)
                    self._migrate_v5_to_v6(connection)
                    self._migrate_v6_to_v7(connection)
                    self._migrate_v7_to_v8(connection)
                    self._migrate_v8_to_v9(connection)
                elif int(row["schema_version"]) == 5:
                    self._migrate_v5_to_v6(connection)
                    self._migrate_v6_to_v7(connection)
                    self._migrate_v7_to_v8(connection)
                    self._migrate_v8_to_v9(connection)
                elif int(row["schema_version"]) == 6:
                    self._migrate_v6_to_v7(connection)
                    self._migrate_v7_to_v8(connection)
                    self._migrate_v8_to_v9(connection)
                elif int(row["schema_version"]) == 7:
                    self._migrate_v7_to_v8(connection)
                    self._migrate_v8_to_v9(connection)
                elif int(row["schema_version"]) == 8:
                    self._migrate_v8_to_v9(connection)
                elif int(row["schema_version"]) == 9:
                    pass
                elif int(row["schema_version"]) == 10:
                    pass
                elif int(row["schema_version"]) == 11:
                    pass
                elif int(row["schema_version"]) == 12:
                    pass
                elif int(row["schema_version"]) == 13:
                    pass
                elif int(row["schema_version"]) != TURN_TRANSACTION_SCHEMA_VERSION:
                    raise TurnTransactionSchemaError(
                        "turn transaction schema version "
                        f"{row['schema_version']} is unsupported; "
                        f"expected {TURN_TRANSACTION_SCHEMA_VERSION}"
                    )
                migrated = connection.execute(
                    """
                    SELECT schema_version
                    FROM anychain_turn_transaction_meta
                    WHERE singleton = 1
                    """
                ).fetchone()
                if migrated is not None and int(migrated["schema_version"]) == 9:
                    self._migrate_v9_to_v10(connection)
                    migrated = connection.execute(
                        """
                        SELECT schema_version
                        FROM anychain_turn_transaction_meta
                        WHERE singleton = 1
                        """
                    ).fetchone()
                if migrated is not None and int(migrated["schema_version"]) == 10:
                    self._migrate_v10_to_v11(connection)
                    migrated = connection.execute(
                        """
                        SELECT schema_version
                        FROM anychain_turn_transaction_meta
                        WHERE singleton = 1
                        """
                    ).fetchone()
                if migrated is not None and int(migrated["schema_version"]) == 11:
                    self._migrate_v11_to_v12(connection)
                    migrated = connection.execute(
                        """
                        SELECT schema_version
                        FROM anychain_turn_transaction_meta
                        WHERE singleton = 1
                        """
                    ).fetchone()
                if migrated is not None and int(migrated["schema_version"]) == 12:
                    self._migrate_v12_to_v13(connection)
                    migrated = connection.execute(
                        """
                        SELECT schema_version
                        FROM anychain_turn_transaction_meta
                        WHERE singleton = 1
                        """
                    ).fetchone()
                if migrated is not None and int(migrated["schema_version"]) == 13:
                    self._migrate_v13_to_v14(connection)
                connection.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS
                        anychain_runtime_event_sequence_per_thread
                    ON anychain_terminal_outbox(
                        logical_thread_id,
                        runtime_event_sequence
                    )
                    WHERE outcome = 'committed'
                    """
                )
                connection.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS anychain_runtime_event_id
                    ON anychain_terminal_outbox(runtime_event_id)
                    WHERE runtime_event_id <> ''
                    """
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    @staticmethod
    def _migrate_v1_to_v2(connection: sqlite3.Connection) -> None:
        """Add delivery and reconciliation authority to the unpublished v1 ledger."""

        columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(anychain_terminal_outbox)"
            ).fetchall()
        }
        additions = (
            ("render_hash", "TEXT NOT NULL DEFAULT ''"),
            ("failure_category", "TEXT NOT NULL DEFAULT ''"),
            ("delivered_at", "TEXT"),
        )
        for name, declaration in additions:
            if name not in columns:
                connection.execute(
                    f"ALTER TABLE anychain_terminal_outbox "
                    f"ADD COLUMN {name} {declaration}"
                )
        # Version 1 had no replay hash or delivery acknowledgement. Those
        # outcomes predate the replay contract and may already have been shown
        # by the terminal, so they must not be presented as fresh messages.
        connection.execute(
            """
            UPDATE anychain_terminal_outbox
            SET delivered_at = COALESCE(delivered_at, created_at),
                failure_category = CASE
                    WHEN outcome = 'committed' THEN ''
                    ELSE 'legacy_unclassified'
                END
            """
        )
        connection.execute(
            """
            UPDATE anychain_turn_transaction_meta
            SET schema_version = ?
            WHERE singleton = 1
            """,
            (2,),
        )

    @staticmethod
    def _migrate_v2_to_v3(connection: sqlite3.Connection) -> None:
        """Bind every attempt and terminal result to its originating revision."""

        for table in ("anychain_turn_attempts", "anychain_terminal_outbox"):
            columns = {
                str(row["name"])
                for row in connection.execute(
                    f"PRAGMA table_info({table})"
                ).fetchall()
            }
            additions = (
                ("origin_revision_commit", "TEXT NOT NULL DEFAULT 'legacy-unbound'"),
                (
                    "origin_revision_worktree_hash",
                    f"TEXT NOT NULL DEFAULT '{_LEGACY_REVISION_HASH}'",
                ),
            )
            for name, declaration in additions:
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE {table} ADD COLUMN {name} {declaration}"
                    )
        connection.execute(
            """
            UPDATE anychain_turn_transaction_meta
            SET schema_version = ?
            WHERE singleton = 1
            """,
            (3,),
        )

    @staticmethod
    def _migrate_v3_to_v4(connection: sqlite3.Connection) -> None:
        """Add typed non-workflow terminal completion authority."""

        schema = """
            CREATE TABLE IF NOT EXISTS anychain_terminal_detours (
                detour_id TEXT PRIMARY KEY,
                logical_thread_id TEXT NOT NULL,
                process_instance_id TEXT NOT NULL,
                command_name TEXT NOT NULL,
                effect_class TEXT NOT NULL CHECK (
                    effect_class IN (
                        'read_only', 'observation_refresh',
                        'shell_state_mutation', 'local_external_effect',
                        'authority_resolution', 'streaming_observation'
                    )
                ),
                status TEXT NOT NULL CHECK (
                    status IN ('prepared', 'completed', 'reconciliation_required')
                ),
                shell_state_before_hash TEXT NOT NULL,
                shell_state_after_hash TEXT,
                product_revision_before INTEGER,
                product_checkpoint_thread_id_before TEXT,
                product_checkpoint_id_before TEXT,
                product_fingerprint_before TEXT,
                product_revision_after INTEGER,
                product_checkpoint_thread_id_after TEXT,
                product_checkpoint_id_after TEXT,
                product_fingerprint_after TEXT,
                response_json TEXT NOT NULL DEFAULT '[]',
                response_hash TEXT,
                effect_status TEXT NOT NULL CHECK (
                    effect_status IN (
                        'not_applicable', 'succeeded', 'failed', 'uncertain'
                    )
                ),
                diagnostic_hash TEXT,
                resolution TEXT,
                evidence_hash TEXT,
                origin_revision_commit TEXT NOT NULL,
                origin_revision_worktree_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                delivered_at TEXT
            )
        """
        connection.execute(schema)
        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS anychain_one_open_detour_per_thread
            ON anychain_terminal_detours(logical_thread_id)
            WHERE status IN ('prepared', 'reconciliation_required')
            """
        )
        connection.execute(
            """
            UPDATE anychain_turn_transaction_meta
            SET schema_version = ?
            WHERE singleton = 1
            """,
            (4,),
        )

    @staticmethod
    def _migrate_v4_to_v5(connection: sqlite3.Connection) -> None:
        """Allow diagnostic detours while an external effect awaits resolution."""

        connection.execute(
            "DROP INDEX IF EXISTS anychain_one_open_detour_per_thread"
        )
        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS anychain_one_active_detour_per_thread
            ON anychain_terminal_detours(logical_thread_id)
            WHERE status = 'prepared'
            """
        )
        connection.execute(
            """
            UPDATE anychain_turn_transaction_meta
            SET schema_version = ?
            WHERE singleton = 1
            """,
            (5,),
        )

    @staticmethod
    def _migrate_v5_to_v6(connection: sqlite3.Connection) -> None:
        """Bind terminal detours to the exact submitted input."""

        columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(anychain_terminal_detours)"
            ).fetchall()
        }
        if "input_hash" not in columns:
            connection.execute(
                "ALTER TABLE anychain_terminal_detours "
                f"ADD COLUMN input_hash TEXT NOT NULL DEFAULT '{_LEGACY_REVISION_HASH}'"
            )
        connection.execute(
            """
            UPDATE anychain_turn_transaction_meta
            SET schema_version = ?
            WHERE singleton = 1
            """,
            (6,),
        )

    @staticmethod
    def _migrate_v6_to_v7(connection: sqlite3.Connection) -> None:
        """Bind detours to session identity and durable stream progress."""

        columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(anychain_terminal_detours)"
            ).fetchall()
        }
        additions = (
            ("session_id", "TEXT NOT NULL DEFAULT 'legacy-session'"),
            ("session_purpose", "TEXT NOT NULL DEFAULT 'legacy'"),
            ("stream_chunk_count", "INTEGER NOT NULL DEFAULT 0"),
            (
                "stream_hash",
                f"TEXT NOT NULL DEFAULT '{_EMPTY_STREAM_HASH}'",
            ),
        )
        for name, declaration in additions:
            if name not in columns:
                connection.execute(
                    "ALTER TABLE anychain_terminal_detours "
                    f"ADD COLUMN {name} {declaration}"
                )
        connection.execute(
            """
            UPDATE anychain_turn_transaction_meta
            SET schema_version = ?
            WHERE singleton = 1
            """,
            (7,),
        )

    @staticmethod
    def _migrate_v7_to_v8(connection: sqlite3.Connection) -> None:
        """Add a typed terminal-session termination result."""

        columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(anychain_terminal_detours)"
            ).fetchall()
        }
        additions = (
            ("result_kind", "TEXT NOT NULL DEFAULT 'command'"),
            ("termination_reason", "TEXT"),
            ("exit_code", "INTEGER"),
        )
        for name, declaration in additions:
            if name not in columns:
                connection.execute(
                    "ALTER TABLE anychain_terminal_detours "
                    f"ADD COLUMN {name} {declaration}"
                )
        connection.execute(
            """
            UPDATE anychain_turn_transaction_meta
            SET schema_version = ?
            WHERE singleton = 1
            """,
            (8,),
        )

    @staticmethod
    def _migrate_v8_to_v9(connection: sqlite3.Connection) -> None:
        """Type stream termination and quarantine unverifiable legacy detours."""

        columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(anychain_terminal_detours)"
            ).fetchall()
        }
        additions = (
            ("stream_stop_reason", "TEXT NOT NULL DEFAULT 'not_streaming'"),
            ("identity_trust", "TEXT NOT NULL DEFAULT 'legacy_unbound'"),
            ("stream_messages_json", "TEXT NOT NULL DEFAULT '[]'"),
        )
        for name, declaration in additions:
            if name not in columns:
                connection.execute(
                    "ALTER TABLE anychain_terminal_detours "
                    f"ADD COLUMN {name} {declaration}"
                )
        connection.execute(
            """
            UPDATE anychain_turn_transaction_meta
            SET schema_version = ?
            WHERE singleton = 1
            """,
            (9,),
        )

    @staticmethod
    def _migrate_v9_to_v10(connection: sqlite3.Connection) -> None:
        """Quarantine unverifiable open history and type interrupted intent."""

        columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(anychain_terminal_detours)"
            ).fetchall()
        }
        if "interruption_kind" not in columns:
            connection.execute(
                "ALTER TABLE anychain_terminal_detours "
                "ADD COLUMN interruption_kind TEXT"
            )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS anychain_terminal_detour_quarantine (
                detour_id TEXT PRIMARY KEY,
                logical_thread_id TEXT NOT NULL,
                command_name TEXT NOT NULL,
                quarantine_reason TEXT NOT NULL,
                record_json TEXT NOT NULL DEFAULT '{}',
                record_hash TEXT NOT NULL,
                quarantined_at TEXT NOT NULL
            )
            """
        )
        quarantine_columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(anychain_terminal_detour_quarantine)"
            ).fetchall()
        }
        if "record_json" not in quarantine_columns:
            connection.execute(
                "ALTER TABLE anychain_terminal_detour_quarantine "
                "ADD COLUMN record_json TEXT NOT NULL DEFAULT '{}'"
            )
        rows = connection.execute(
            """
            SELECT *
            FROM anychain_terminal_detours
            WHERE status = 'prepared' AND identity_trust = 'legacy_unbound'
            """
        ).fetchall()
        for row in rows:
            record_json = json.dumps(
                dict(row),
                sort_keys=True,
                separators=(",", ":"),
            )
            record_hash = state_fingerprint(record_json.encode("utf-8"))
            connection.execute(
                """
                INSERT OR IGNORE INTO anychain_terminal_detour_quarantine (
                    detour_id, logical_thread_id, command_name,
                    quarantine_reason, record_json, record_hash, quarantined_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["detour_id"],
                    row["logical_thread_id"],
                    row["command_name"],
                    "legacy_unbound_identity",
                    record_json,
                    record_hash,
                    _utc_now(),
                ),
            )
            connection.execute(
                "DELETE FROM anychain_terminal_detours WHERE detour_id = ?",
                (row["detour_id"],),
            )
        connection.execute(
            """
            UPDATE anychain_turn_transaction_meta
            SET schema_version = ?
            WHERE singleton = 1
            """,
            (10,),
        )

    @staticmethod
    def _migrate_v10_to_v11(connection: sqlite3.Connection) -> None:
        """Persist complete before/after Product Head identity."""

        committed_rows = connection.execute(
            """
            SELECT COUNT(*)
            FROM anychain_terminal_outbox
            WHERE outcome = 'committed'
            """
        ).fetchone()[0]
        if int(committed_rows) > 0:
            raise TurnTransactionSchemaError(
                "v10 turn history has no authoritative revision ordering; "
                "archive it and start a fresh Product Head"
            )
        for table, additions in (
            (
                "anychain_turn_attempts",
                (("base_revision", "INTEGER NOT NULL DEFAULT 0"),),
            ),
            (
                "anychain_terminal_outbox",
                (
                    ("base_revision", "INTEGER NOT NULL DEFAULT 0"),
                    ("product_revision", "INTEGER NOT NULL DEFAULT 0"),
                ),
            ),
        ):
            columns = {
                str(row["name"])
                for row in connection.execute(
                    f"PRAGMA table_info({table})"
                ).fetchall()
            }
            for name, declaration in additions:
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE {table} ADD COLUMN {name} {declaration}"
                    )

        connection.execute(
            """
            UPDATE anychain_turn_attempts AS attempt
            SET base_revision = (
                SELECT COUNT(*)
                FROM anychain_terminal_outbox AS prior
                WHERE prior.logical_thread_id = attempt.logical_thread_id
                  AND prior.outcome = 'committed'
                  AND (
                      prior.created_at < attempt.created_at
                      OR (
                          prior.created_at = attempt.created_at
                          AND prior.transaction_id < attempt.transaction_id
                      )
                  )
            )
            """
        )
        connection.execute(
            """
            UPDATE anychain_terminal_outbox AS outcome
            SET base_revision = (
                    SELECT COUNT(*)
                    FROM anychain_terminal_outbox AS prior
                    WHERE prior.logical_thread_id = outcome.logical_thread_id
                      AND prior.outcome = 'committed'
                      AND (
                          prior.created_at < outcome.created_at
                          OR (
                              prior.created_at = outcome.created_at
                              AND prior.transaction_id < outcome.transaction_id
                          )
                      )
                ),
                product_revision = (
                    SELECT COUNT(*)
                    FROM anychain_terminal_outbox AS prior
                    WHERE prior.logical_thread_id = outcome.logical_thread_id
                      AND prior.outcome = 'committed'
                      AND (
                          prior.created_at < outcome.created_at
                          OR (
                              prior.created_at = outcome.created_at
                              AND prior.transaction_id <= outcome.transaction_id
                          )
                      )
                )
            """
        )
        connection.execute(
            """
            UPDATE anychain_turn_transaction_meta
            SET schema_version = ?
            WHERE singleton = 1
            """,
            (11,),
        )

    @staticmethod
    def _migrate_v11_to_v12(connection: sqlite3.Connection) -> None:
        """Add a durable publication lifecycle for committed runtime events."""

        columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(anychain_terminal_outbox)"
            ).fetchall()
        }
        additions = (
            ("runtime_event_id", "TEXT NOT NULL DEFAULT ''"),
            ("runtime_event_sequence", "INTEGER NOT NULL DEFAULT 0"),
            (
                "runtime_event_status",
                "TEXT NOT NULL DEFAULT 'not_applicable'",
            ),
            ("runtime_event_payload_hash", "TEXT"),
            ("runtime_event_published_at", "TEXT"),
        )
        for name, declaration in additions:
            if name not in columns:
                connection.execute(
                    "ALTER TABLE anychain_terminal_outbox "
                    f"ADD COLUMN {name} {declaration}"
                )
        detour_columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(anychain_terminal_detours)"
            ).fetchall()
        }
        detour_additions = (
            (
                "runtime_event_fence_sequence",
                "INTEGER NOT NULL DEFAULT 0",
            ),
            (
                "runtime_event_fence_terminal_event_id",
                "TEXT NOT NULL DEFAULT ''",
            ),
            ("runtime_event_fence_id", "TEXT NOT NULL DEFAULT ''"),
            ("runtime_event_fence_hash", "TEXT NOT NULL DEFAULT ''"),
        )
        for name, declaration in detour_additions:
            if name not in detour_columns:
                connection.execute(
                    "ALTER TABLE anychain_terminal_detours "
                    f"ADD COLUMN {name} {declaration}"
                )
        quarantine_columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(anychain_terminal_detour_quarantine)"
            ).fetchall()
        }
        if "record_json" not in quarantine_columns:
            connection.execute(
                "ALTER TABLE anychain_terminal_detour_quarantine "
                "ADD COLUMN record_json TEXT NOT NULL DEFAULT '{}'"
            )
        connection.execute(
            """
            UPDATE anychain_terminal_outbox
            SET runtime_event_sequence = product_revision,
                runtime_event_status = CASE
                    WHEN outcome = 'committed' THEN 'pending'
                    ELSE 'not_applicable'
                END
            """
        )
        connection.execute(
            """
            UPDATE anychain_terminal_detours
            SET identity_trust = 'legacy_unbound'
            """
        )
        rows = connection.execute(
            """
            SELECT *
            FROM anychain_terminal_detours
            WHERE identity_trust = 'legacy_unbound'
            """
        ).fetchall()
        for row in rows:
            record_json = json.dumps(
                dict(row),
                sort_keys=True,
                separators=(",", ":"),
            )
            record_hash = state_fingerprint(record_json.encode("utf-8"))
            connection.execute(
                """
                INSERT OR IGNORE INTO anychain_terminal_detour_quarantine (
                    detour_id, logical_thread_id, command_name,
                    quarantine_reason, record_json, record_hash, quarantined_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["detour_id"],
                    row["logical_thread_id"],
                    row["command_name"],
                    "legacy_runtime_fence_unbound",
                    record_json,
                    record_hash,
                    _utc_now(),
                ),
            )
            connection.execute(
                "DELETE FROM anychain_terminal_detours WHERE detour_id = ?",
                (row["detour_id"],),
            )
        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS
                anychain_runtime_event_sequence_per_thread
            ON anychain_terminal_outbox(
                logical_thread_id,
                runtime_event_sequence
            )
            WHERE outcome = 'committed'
            """
        )
        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS anychain_runtime_event_id
            ON anychain_terminal_outbox(runtime_event_id)
            WHERE runtime_event_id <> ''
            """
        )
        connection.execute(
            """
            UPDATE anychain_turn_transaction_meta
            SET schema_version = ?
            WHERE singleton = 1
            """,
            (12,),
        )

    @staticmethod
    def _migrate_v12_to_v13(connection: sqlite3.Connection) -> None:
        """Bind migrated committed outcomes to stable runtime-event identities."""

        connection.execute(
            """
            UPDATE anychain_terminal_outbox
            SET runtime_event_id = 'migrated-runtime:' || event_id,
                runtime_event_sequence = product_revision,
                runtime_event_status = 'pending',
                runtime_event_payload_hash = NULL,
                runtime_event_published_at = NULL
            WHERE outcome = 'committed'
              AND runtime_event_id = ''
            """
        )
        connection.execute(
            """
            UPDATE anychain_turn_transaction_meta
            SET schema_version = ?
            WHERE singleton = 1
            """,
            (13,),
        )

    @staticmethod
    def _migrate_v13_to_v14(connection: sqlite3.Connection) -> None:
        """Retain complete canonical detour records in the audit quarantine."""

        columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(anychain_terminal_detour_quarantine)"
            ).fetchall()
        }
        if "record_json" not in columns:
            existing_rows = int(
                connection.execute(
                    "SELECT COUNT(*) "
                    "FROM anychain_terminal_detour_quarantine"
                ).fetchone()[0]
            )
            if existing_rows:
                raise TurnTransactionSchemaError(
                    "legacy detour quarantine rows do not contain their "
                    "canonical records; archive the ledger before migration"
                )
            connection.execute(
                "ALTER TABLE anychain_terminal_detour_quarantine "
                "ADD COLUMN record_json TEXT NOT NULL DEFAULT '{}'"
            )
        rows = connection.execute(
            """
            SELECT detour_id, logical_thread_id, command_name,
                   quarantine_reason, record_json, record_hash
            FROM anychain_terminal_detour_quarantine
            """
        ).fetchall()
        for row in rows:
            record_json = str(row["record_json"] or "")
            try:
                record = json.loads(record_json)
            except (TypeError, ValueError) as exc:
                raise TurnTransactionSchemaError(
                    "detour quarantine contains invalid canonical JSON"
                ) from exc
            if not isinstance(record, dict) or not record:
                raise TurnTransactionSchemaError(
                    "detour quarantine contains an incomplete canonical record"
                )
            canonical = json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
            )
            if canonical != record_json or state_fingerprint(
                record_json.encode("utf-8")
            ) != str(row["record_hash"] or ""):
                raise TurnTransactionSchemaError(
                    "detour quarantine canonical record hash is invalid"
                )
            _validate_quarantined_detour_snapshot(record, row)
        connection.execute(
            """
            UPDATE anychain_turn_transaction_meta
            SET schema_version = ?
            WHERE singleton = 1
            """,
            (TURN_TRANSACTION_SCHEMA_VERSION,),
        )

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self.checkpoint_path,
            timeout=self.timeout_seconds,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _read_connection(self) -> Iterator[sqlite3.Connection]:
        with self._connection() as connection:
            yield connection

    @contextmanager
    def _write_transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._require_schema(connection)
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    @staticmethod
    def _require_schema(connection: sqlite3.Connection) -> None:
        row = connection.execute(
            """
            SELECT schema_version
            FROM anychain_turn_transaction_meta
            WHERE singleton = 1
            """
        ).fetchone()
        if row is None or int(row["schema_version"]) != TURN_TRANSACTION_SCHEMA_VERSION:
            actual = "<missing>" if row is None else str(row["schema_version"])
            raise TurnTransactionSchemaError(
                f"turn transaction schema version {actual} is unsupported; "
                f"expected {TURN_TRANSACTION_SCHEMA_VERSION}"
            )

    @staticmethod
    def _select_head(
        connection: sqlite3.Connection,
        logical_thread_id: str,
    ) -> ProductHead | None:
        row = connection.execute(
            """
            SELECT *
            FROM anychain_product_heads
            WHERE logical_thread_id = ?
            """,
            (logical_thread_id,),
        ).fetchone()
        return None if row is None else _head_from_row(row)

    def _require_head(
        self,
        connection: sqlite3.Connection,
        logical_thread_id: str,
    ) -> ProductHead:
        head = self._select_head(connection, logical_thread_id)
        if head is None:
            raise TurnTransactionConflictError(
                f"logical product head is not bootstrapped: {logical_thread_id}"
            )
        return head

    @staticmethod
    def _select_attempt(
        connection: sqlite3.Connection,
        transaction_id: str,
    ) -> TurnAttempt | None:
        row = connection.execute(
            """
            SELECT *
            FROM anychain_turn_attempts
            WHERE transaction_id = ?
            """,
            (transaction_id,),
        ).fetchone()
        return None if row is None else _attempt_from_row(row)

    def _require_attempt(
        self,
        connection: sqlite3.Connection,
        transaction_id: str,
    ) -> TurnAttempt:
        attempt = self._select_attempt(connection, transaction_id)
        if attempt is None:
            raise TurnTransactionConflictError(f"unknown transaction_id: {transaction_id}")
        return attempt

    @staticmethod
    def _select_outcome(
        connection: sqlite3.Connection,
        transaction_id: str,
    ) -> TerminalOutcome | None:
        row = connection.execute(
            """
            SELECT *
            FROM anychain_terminal_outbox
            WHERE transaction_id = ?
            """,
            (transaction_id,),
        ).fetchone()
        return None if row is None else _outcome_from_row(row)

    def _require_outcome(
        self,
        connection: sqlite3.Connection,
        transaction_id: str,
    ) -> TerminalOutcome:
        outcome = self._select_outcome(connection, transaction_id)
        if outcome is None:
            raise TurnTransactionConflictError(
                f"terminal outcome was not persisted for transaction: {transaction_id}"
            )
        return outcome

    @staticmethod
    def _select_detour(
        connection: sqlite3.Connection,
        detour_id: str,
    ) -> TerminalDetour | None:
        row = connection.execute(
            """
            SELECT *
            FROM anychain_terminal_detours
            WHERE detour_id = ?
            """,
            (detour_id,),
        ).fetchone()
        return None if row is None else _detour_from_row(row)

    def _require_detour(
        self,
        connection: sqlite3.Connection,
        detour_id: str,
    ) -> TerminalDetour:
        detour = self._select_detour(connection, detour_id)
        if detour is None:
            raise TurnTransactionConflictError(
                f"unknown terminal detour: {detour_id}"
            )
        return detour

    @staticmethod
    def _select_unresolved_detour(
        connection: sqlite3.Connection,
        logical_thread_id: str,
    ) -> TerminalDetour | None:
        row = connection.execute(
            """
            SELECT *
            FROM anychain_terminal_detours
            WHERE logical_thread_id = ?
              AND status = 'reconciliation_required'
            ORDER BY created_at, detour_id
            LIMIT 1
            """,
            (logical_thread_id,),
        ).fetchone()
        return None if row is None else _detour_from_row(row)

    @staticmethod
    def _select_open_detour(
        connection: sqlite3.Connection,
        logical_thread_id: str,
    ) -> TerminalDetour | None:
        row = connection.execute(
            """
            SELECT *
            FROM anychain_terminal_detours
            WHERE logical_thread_id = ?
              AND status IN ('prepared', 'reconciliation_required')
            ORDER BY created_at, detour_id
            LIMIT 1
            """,
            (logical_thread_id,),
        ).fetchone()
        return None if row is None else _detour_from_row(row)

    @staticmethod
    def _select_prepared_detour(
        connection: sqlite3.Connection,
        logical_thread_id: str,
    ) -> TerminalDetour | None:
        row = connection.execute(
            """
            SELECT *
            FROM anychain_terminal_detours
            WHERE logical_thread_id = ? AND status = 'prepared'
            ORDER BY created_at, detour_id
            LIMIT 1
            """,
            (logical_thread_id,),
        ).fetchone()
        return None if row is None else _detour_from_row(row)

    @staticmethod
    def _select_resolution(
        connection: sqlite3.Connection,
        transaction_id: str,
    ) -> ReconciliationResolution | None:
        row = connection.execute(
            """
            SELECT *
            FROM anychain_reconciliation_resolutions
            WHERE transaction_id = ?
            """,
            (transaction_id,),
        ).fetchone()
        return None if row is None else _resolution_from_row(row)

    def _require_resolution(
        self,
        connection: sqlite3.Connection,
        transaction_id: str,
    ) -> ReconciliationResolution:
        resolution = self._select_resolution(connection, transaction_id)
        if resolution is None:
            raise TurnTransactionConflictError(
                f"reconciliation resolution was not persisted: {transaction_id}"
            )
        return resolution

    @staticmethod
    def _select_unresolved_reconciliation(
        connection: sqlite3.Connection,
        logical_thread_id: str,
    ) -> TerminalOutcome | None:
        row = connection.execute(
            """
            SELECT outcome.*
            FROM anychain_terminal_outbox AS outcome
            LEFT JOIN anychain_reconciliation_resolutions AS resolution
              ON resolution.transaction_id = outcome.transaction_id
            WHERE outcome.logical_thread_id = ?
              AND outcome.outcome = 'reconciliation_required'
              AND resolution.transaction_id IS NULL
            ORDER BY outcome.created_at, outcome.transaction_id
            LIMIT 1
            """,
            (logical_thread_id,),
        ).fetchone()
        return None if row is None else _outcome_from_row(row)

    @staticmethod
    def _published_runtime_event_fence(
        connection: sqlite3.Connection,
        logical_thread_id: str,
    ) -> tuple[int, str, str, str]:
        row = connection.execute(
            """
            SELECT runtime_event_sequence, event_id, runtime_event_id,
                   runtime_event_payload_hash
            FROM anychain_terminal_outbox
            WHERE logical_thread_id = ?
              AND runtime_event_status = 'published'
            ORDER BY runtime_event_sequence DESC
            LIMIT 1
            """,
            (logical_thread_id,),
        ).fetchone()
        if row is None:
            return (0, "", "", "")
        return (
            int(row["runtime_event_sequence"]),
            str(row["event_id"]),
            str(row["runtime_event_id"]),
            str(row["runtime_event_payload_hash"]),
        )

    @staticmethod
    def _validate_attempt_binding(
        attempt: TurnAttempt,
        *,
        logical_thread_id: str,
        physical_thread_id: str,
    ) -> None:
        if attempt.logical_thread_id != logical_thread_id:
            raise TurnTransactionConflictError(
                "transaction_id is bound to a different logical thread"
            )
        if attempt.physical_thread_id != physical_thread_id:
            raise TurnTransactionConflictError(
                "transaction_id is bound to a different physical thread"
            )

    @staticmethod
    def _validate_base_head(attempt: TurnAttempt, head: ProductHead) -> None:
        expected = (
            attempt.base_revision,
            attempt.base_checkpoint_thread_id,
            attempt.base_checkpoint_id,
            attempt.base_fingerprint,
        )
        actual = (
            head.revision,
            head.checkpoint_thread_id,
            head.checkpoint_id,
            head.state_fingerprint,
        )
        if actual != expected:
            raise TurnTransactionConflictError(
                "product head no longer matches the attempt's immutable base"
            )

    @staticmethod
    def _validate_idempotent_outcome(
        existing: TerminalOutcome,
        *,
        outcome: TerminalOutcomeKind,
        attempt_checkpoint_id: str | None,
        attempt_fingerprint: str | None,
        diagnostic_hash: str | None,
        render_hash: str,
        failure_category: str,
    ) -> None:
        expected = (
            outcome,
            attempt_checkpoint_id,
            attempt_fingerprint,
            diagnostic_hash,
            render_hash,
            failure_category,
        )
        actual = (
            existing.outcome,
            existing.attempt_checkpoint_id,
            existing.attempt_fingerprint,
            existing.diagnostic_hash,
            existing.render_hash,
            existing.failure_category,
        )
        if actual != expected:
            raise TurnTransactionConflictError(
                "transaction already has a different immutable terminal outcome"
            )


def _validate_quarantined_detour_snapshot(
    record: dict[str, Any],
    quarantine_row: sqlite3.Row,
) -> None:
    reason = str(quarantine_row["quarantine_reason"] or "")
    expected_fields = {
        "legacy_unbound_identity": _LEGACY_V10_DETOUR_SNAPSHOT_FIELDS,
        "legacy_runtime_fence_unbound": _LEGACY_V12_DETOUR_SNAPSHOT_FIELDS,
    }.get(reason)
    if expected_fields is None or set(record) != expected_fields:
        raise TurnTransactionSchemaError(
            "detour quarantine contains an incomplete canonical record"
        )
    for field in ("detour_id", "logical_thread_id", "command_name"):
        if str(record.get(field) or "") != str(quarantine_row[field] or ""):
            raise TurnTransactionSchemaError(
                "detour quarantine record identity does not match its audit row"
            )
    _validate_detour_semantics(
        record,
        has_runtime_fence=reason == "legacy_runtime_fence_unbound",
        required_identity_trust="legacy_unbound",
    )


def read_product_head(
    checkpoint_path: str | Path,
    logical_thread_id: str,
) -> ProductHead | None:
    """Read the logical Product Head without creating or migrating authority tables."""

    logical_thread_id = _identifier("logical_thread_id", logical_thread_id)
    path = Path(checkpoint_path)
    if not path.exists():
        return None
    uri = f"file:{path}?mode=ro"
    try:
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT *
                FROM anychain_product_heads
                WHERE logical_thread_id = ?
                """,
                (logical_thread_id,),
            ).fetchone()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).lower():
            return None
        raise
    return _head_from_row(row) if row is not None else None


def product_authority_id(logical_thread_id: str, session_purpose: str) -> str:
    """Return the durable transaction authority for one isolated product session."""

    logical_thread_id = _identifier("logical_thread_id", logical_thread_id)
    session_purpose = _identifier("session_purpose", session_purpose)
    return _identifier(
        "product_authority_id",
        f"{session_purpose}:{logical_thread_id}",
    )


def state_fingerprint(serialized_state: bytes) -> str:
    """Return the ledger's canonical SHA-256 fingerprint for serialized state."""

    if not isinstance(serialized_state, bytes):
        raise TurnTransactionValidationError("serialized_state must be bytes")
    return hashlib.sha256(serialized_state).hexdigest()


def _identifier(name: str, value: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise TurnTransactionValidationError(
            f"{name} must be a non-empty ASCII identifier of at most 255 characters"
        )
    return value


def _transaction_id(value: str) -> str:
    if not isinstance(value, str):
        raise TurnTransactionValidationError("transaction_id must be a UUID")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise TurnTransactionValidationError("transaction_id must be a UUID") from exc
    canonical = str(parsed)
    if value != canonical:
        raise TurnTransactionValidationError(
            "transaction_id must use canonical lowercase UUID form"
        )
    return canonical


def _fingerprint(name: str, value: str) -> str:
    if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
        raise TurnTransactionValidationError(
            f"{name} must be a lowercase SHA-256 hex digest"
        )
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _schema_identifier(name: str, value: Any) -> str:
    normalized = str(value)
    if not _IDENTIFIER_RE.fullmatch(normalized):
        raise TurnTransactionSchemaError(
            f"stored {name} is not a valid identifier"
        )
    return normalized


def _schema_hash(name: str, value: Any) -> str:
    normalized = str(value)
    if not _HASH_RE.fullmatch(normalized):
        raise TurnTransactionSchemaError(
            f"stored {name} is not a lowercase SHA-256 digest"
        )
    return normalized


def _head_from_row(row: sqlite3.Row) -> ProductHead:
    logical_thread_id = _schema_identifier(
        "logical_thread_id",
        row["logical_thread_id"],
    )
    checkpoint_thread_id = _schema_identifier(
        "checkpoint_thread_id",
        row["checkpoint_thread_id"],
    )
    checkpoint_id = _schema_identifier("checkpoint_id", row["checkpoint_id"])
    fingerprint = _schema_hash("state_fingerprint", row["state_fingerprint"])
    revision = int(row["revision"])
    if revision < 0:
        raise TurnTransactionSchemaError("Product Head revision is invalid")
    return ProductHead(
        logical_thread_id=logical_thread_id,
        checkpoint_thread_id=checkpoint_thread_id,
        checkpoint_id=checkpoint_id,
        state_fingerprint=fingerprint,
        revision=revision,
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _attempt_from_row(row: sqlite3.Row) -> TurnAttempt:
    transaction_id = str(row["transaction_id"])
    try:
        if str(uuid.UUID(transaction_id)) != transaction_id:
            raise ValueError
    except (ValueError, AttributeError) as exc:
        raise TurnTransactionSchemaError(
            "attempt transaction_id is invalid"
        ) from exc
    logical_thread_id = _schema_identifier(
        "logical_thread_id",
        row["logical_thread_id"],
    )
    physical_thread_id = _schema_identifier(
        "physical_thread_id",
        row["physical_thread_id"],
    )
    base_revision = int(row["base_revision"])
    if base_revision < 0:
        raise TurnTransactionSchemaError("attempt base revision is invalid")
    base_checkpoint_thread_id = _schema_identifier(
        "base_checkpoint_thread_id",
        row["base_checkpoint_thread_id"],
    )
    base_checkpoint_id = _schema_identifier(
        "base_checkpoint_id",
        row["base_checkpoint_id"],
    )
    base_fingerprint = _schema_hash(
        "base_fingerprint",
        row["base_fingerprint"],
    )
    origin_revision_commit = _schema_identifier(
        "origin_revision_commit",
        row["origin_revision_commit"],
    )
    origin_revision_worktree_hash = _schema_hash(
        "origin_revision_worktree_hash",
        row["origin_revision_worktree_hash"],
    )
    status = str(row["status"])
    if status not in {"active", *_TERMINAL_OUTCOMES}:
        raise TurnTransactionSchemaError("attempt status is invalid")
    attempt_checkpoint_id = row["attempt_checkpoint_id"]
    attempt_fingerprint = row["attempt_fingerprint"]
    diagnostic_hash = row["diagnostic_hash"]
    if attempt_checkpoint_id is not None:
        attempt_checkpoint_id = _schema_identifier(
            "attempt_checkpoint_id",
            attempt_checkpoint_id,
        )
    if attempt_fingerprint is not None:
        attempt_fingerprint = _schema_hash(
            "attempt_fingerprint",
            attempt_fingerprint,
        )
    if diagnostic_hash is not None:
        diagnostic_hash = _schema_hash("diagnostic_hash", diagnostic_hash)
    if (attempt_checkpoint_id is None) != (attempt_fingerprint is None):
        raise TurnTransactionSchemaError(
            "attempt checkpoint identity is incomplete"
        )
    return TurnAttempt(
        transaction_id=transaction_id,
        logical_thread_id=logical_thread_id,
        physical_thread_id=physical_thread_id,
        base_revision=base_revision,
        base_checkpoint_thread_id=base_checkpoint_thread_id,
        base_checkpoint_id=base_checkpoint_id,
        base_fingerprint=base_fingerprint,
        origin_revision_commit=origin_revision_commit,
        origin_revision_worktree_hash=origin_revision_worktree_hash,
        status=status,  # type: ignore[arg-type]
        attempt_checkpoint_id=attempt_checkpoint_id,
        attempt_fingerprint=attempt_fingerprint,
        diagnostic_hash=diagnostic_hash,
        created_at=str(row["created_at"]),
        completed_at=row["completed_at"],
    )


def _outcome_from_row(row: sqlite3.Row) -> TerminalOutcome:
    outcome = str(row["outcome"])
    if outcome not in _TERMINAL_OUTCOMES:
        raise TurnTransactionSchemaError("terminal outcome kind is invalid")
    event_id = _schema_identifier("event_id", row["event_id"])
    transaction_id = str(row["transaction_id"])
    try:
        if str(uuid.UUID(transaction_id)) != transaction_id:
            raise ValueError
    except (ValueError, AttributeError) as exc:
        raise TurnTransactionSchemaError(
            "terminal outcome transaction_id is invalid"
        ) from exc
    base_revision = int(row["base_revision"])
    product_revision = int(row["product_revision"])
    physical_thread_id = str(row["physical_thread_id"])
    base_checkpoint_thread_id = str(row["base_checkpoint_thread_id"])
    base_checkpoint_id = str(row["base_checkpoint_id"])
    base_fingerprint = str(row["base_fingerprint"])
    attempt_checkpoint_id = row["attempt_checkpoint_id"]
    attempt_fingerprint = row["attempt_fingerprint"]
    product_checkpoint_thread_id = str(row["product_checkpoint_thread_id"])
    product_checkpoint_id = str(row["product_checkpoint_id"])
    product_fingerprint = str(row["product_fingerprint"])
    diagnostic_hash = row["diagnostic_hash"]
    render_hash = str(row["render_hash"])
    failure_category = str(row["failure_category"])
    runtime_event_id = str(row["runtime_event_id"])
    runtime_event_sequence = int(row["runtime_event_sequence"])
    runtime_event_status = str(row["runtime_event_status"])
    runtime_event_payload_hash = row["runtime_event_payload_hash"]
    runtime_event_published_at = row["runtime_event_published_at"]
    if base_revision < 0 or product_revision < 0 or runtime_event_sequence < 0:
        raise TurnTransactionSchemaError(
            "terminal outcome revision or event sequence is invalid"
        )
    for name, value in (
        ("base_fingerprint", base_fingerprint),
        ("product_fingerprint", product_fingerprint),
        ("origin_revision_worktree_hash", row["origin_revision_worktree_hash"]),
    ):
        _schema_hash(name, value)
    for name, value in (
        ("logical_thread_id", row["logical_thread_id"]),
        ("physical_thread_id", physical_thread_id),
        ("base_checkpoint_thread_id", base_checkpoint_thread_id),
        ("base_checkpoint_id", base_checkpoint_id),
        ("product_checkpoint_thread_id", product_checkpoint_thread_id),
        ("product_checkpoint_id", product_checkpoint_id),
        ("origin_revision_commit", row["origin_revision_commit"]),
    ):
        _schema_identifier(name, value)
    if attempt_checkpoint_id is not None:
        _schema_identifier("attempt_checkpoint_id", attempt_checkpoint_id)
    if attempt_fingerprint is not None:
        _schema_hash("attempt_fingerprint", attempt_fingerprint)
    if diagnostic_hash is not None:
        _schema_hash("diagnostic_hash", diagnostic_hash)
    if render_hash:
        _schema_hash("render_hash", render_hash)
    if failure_category:
        _schema_identifier("failure_category", failure_category)
    if runtime_event_id:
        _schema_identifier("runtime_event_id", runtime_event_id)
    if outcome == "committed":
        if (
            product_revision != base_revision + 1
            or not attempt_checkpoint_id
            or not attempt_fingerprint
            or product_checkpoint_thread_id != physical_thread_id
            or product_checkpoint_id != str(attempt_checkpoint_id)
            or product_fingerprint != str(attempt_fingerprint)
            or diagnostic_hash is not None
            or not render_hash
            or failure_category
            or not runtime_event_id
            or runtime_event_status not in {"pending", "published"}
            or runtime_event_sequence != product_revision
            or (
                runtime_event_status == "pending"
                and (
                    runtime_event_payload_hash is not None
                    or runtime_event_published_at is not None
                )
            )
            or (
                runtime_event_status == "published"
                and (
                    runtime_event_payload_hash is None
                    or _HASH_RE.fullmatch(
                        str(runtime_event_payload_hash)
                    )
                    is None
                    or not runtime_event_published_at
                )
            )
        ):
            raise TurnTransactionSchemaError(
                "committed terminal outcome lineage is invalid"
            )
    else:
        if (
            product_revision != base_revision
            or product_checkpoint_thread_id != base_checkpoint_thread_id
            or product_checkpoint_id != base_checkpoint_id
            or product_fingerprint != base_fingerprint
            or diagnostic_hash is None
            or render_hash
            or not failure_category
            or runtime_event_status != "not_applicable"
            or runtime_event_id
            or runtime_event_sequence != product_revision
            or runtime_event_payload_hash is not None
        ):
            raise TurnTransactionSchemaError(
                "non-committing terminal outcome lineage is invalid"
            )
        if outcome == "aborted" and (
            attempt_checkpoint_id is not None
            or attempt_fingerprint is not None
        ):
            raise TurnTransactionSchemaError(
                "aborted terminal outcome carries attempt identity"
            )
        if outcome == "reconciliation_required" and (
            (attempt_checkpoint_id is None)
            != (attempt_fingerprint is None)
        ):
            raise TurnTransactionSchemaError(
                "reconciliation terminal outcome attempt identity is incomplete"
            )
    return TerminalOutcome(
        event_id=event_id,
        transaction_id=transaction_id,
        logical_thread_id=str(row["logical_thread_id"]),
        physical_thread_id=physical_thread_id,
        outcome=outcome,  # type: ignore[arg-type]
        base_revision=base_revision,
        base_checkpoint_thread_id=base_checkpoint_thread_id,
        base_checkpoint_id=base_checkpoint_id,
        base_fingerprint=base_fingerprint,
        origin_revision_commit=str(row["origin_revision_commit"]),
        origin_revision_worktree_hash=str(row["origin_revision_worktree_hash"]),
        attempt_checkpoint_id=attempt_checkpoint_id,
        attempt_fingerprint=attempt_fingerprint,
        product_revision=product_revision,
        product_checkpoint_thread_id=product_checkpoint_thread_id,
        product_checkpoint_id=product_checkpoint_id,
        product_fingerprint=product_fingerprint,
        diagnostic_hash=diagnostic_hash,
        render_hash=render_hash,
        failure_category=failure_category,
        runtime_event_id=runtime_event_id,
        runtime_event_sequence=runtime_event_sequence,
        runtime_event_status=runtime_event_status,  # type: ignore[arg-type]
        runtime_event_payload_hash=runtime_event_payload_hash,
        runtime_event_published_at=runtime_event_published_at,
        created_at=str(row["created_at"]),
        delivered_at=row["delivered_at"],
    )


def _resolution_from_row(row: sqlite3.Row) -> ReconciliationResolution:
    return ReconciliationResolution(
        transaction_id=str(row["transaction_id"]),
        logical_thread_id=str(row["logical_thread_id"]),
        resolution=str(row["resolution"]),  # type: ignore[arg-type]
        evidence_hash=str(row["evidence_hash"]),
        created_at=str(row["created_at"]),
    )


def _head_identity(
    head: ProductHead | None,
) -> tuple[int | None, str | None, str | None, str | None]:
    if head is None:
        return (None, None, None, None)
    return (
        head.revision,
        head.checkpoint_thread_id,
        head.checkpoint_id,
        head.state_fingerprint,
    )


def _validate_detour_semantics(
    raw: Mapping[str, Any],
    *,
    has_runtime_fence: bool,
    required_identity_trust: str | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate one detour record identically on read and migration."""

    def require_identifier(name: str) -> str:
        value = str(raw.get(name) or "")
        try:
            return _identifier(name, value)
        except TurnTransactionValidationError as exc:
            raise TurnTransactionSchemaError(
                "terminal detour identity is invalid"
            ) from exc

    def require_hash(name: str, *, optional: bool = False) -> str | None:
        value = raw.get(name)
        if value is None or value == "":
            if optional:
                return None
            raise TurnTransactionSchemaError(
                "terminal detour hash identity is invalid"
            )
        try:
            return _fingerprint(name, str(value))
        except TurnTransactionValidationError as exc:
            raise TurnTransactionSchemaError(
                "terminal detour hash identity is invalid"
            ) from exc

    for name in (
        "detour_id",
        "logical_thread_id",
        "process_instance_id",
        "session_id",
        "session_purpose",
        "command_name",
        "origin_revision_commit",
    ):
        require_identifier(name)
    for name in (
        "input_hash",
        "shell_state_before_hash",
        "origin_revision_worktree_hash",
    ):
        require_hash(name)

    try:
        response_payload = json.loads(str(raw.get("response_json") or "[]"))
        stream_payload = json.loads(
            str(raw.get("stream_messages_json") or "[]")
        )
    except (TypeError, ValueError) as exc:
        raise TurnTransactionSchemaError(
            "terminal detour message payload is invalid"
        ) from exc
    if (
        not isinstance(response_payload, list)
        or any(
            not isinstance(item, str) or not item.strip()
            for item in response_payload
        )
        or not isinstance(stream_payload, list)
        or any(
            not isinstance(item, str) or not item.strip()
            for item in stream_payload
        )
    ):
        raise TurnTransactionSchemaError(
            "terminal detour message payload is invalid"
        )
    response_messages = tuple(response_payload)
    stream_messages = tuple(stream_payload)

    result_kind = str(raw.get("result_kind") or "")
    interruption_kind = raw.get("interruption_kind")
    termination_reason = raw.get("termination_reason")
    exit_code = raw.get("exit_code")
    effect_class = str(raw.get("effect_class") or "")
    status = str(raw.get("status") or "")
    effect_status = str(raw.get("effect_status") or "")
    stream_stop_reason = str(raw.get("stream_stop_reason") or "")
    identity_trust = str(raw.get("identity_trust") or "")
    if (
        result_kind not in _DETOUR_RESULT_KINDS
        or effect_class not in _DETOUR_EFFECT_CLASSES
        or status not in _DETOUR_STATUSES
        or effect_status not in _DETOUR_EFFECT_STATUSES
        or stream_stop_reason not in _STREAM_STOP_REASONS
        or identity_trust not in {"trusted", "legacy_unbound"}
        or (
            required_identity_trust is not None
            and identity_trust != required_identity_trust
        )
    ):
        raise TurnTransactionSchemaError(
            "terminal detour typed fields are invalid"
        )
    if interruption_kind not in {None, "session_termination_interrupted"}:
        raise TurnTransactionSchemaError(
            "terminal detour interruption kind is invalid"
        )
    if isinstance(exit_code, bool):
        raise TurnTransactionSchemaError(
            "terminal detour termination fields are invalid"
        )
    if result_kind == "session_termination":
        if (
            interruption_kind is not None
            or termination_reason not in _TERMINATION_REASONS
            or exit_code not in {0, 130}
        ):
            raise TurnTransactionSchemaError(
                "terminal detour termination fields are invalid"
            )
    elif interruption_kind == "session_termination_interrupted":
        if termination_reason not in _TERMINATION_REASONS or exit_code not in {
            0,
            130,
        }:
            raise TurnTransactionSchemaError(
                "interrupted termination fields are invalid"
            )
    elif termination_reason is not None or exit_code is not None:
        raise TurnTransactionSchemaError(
            "command detour carries termination fields"
        )

    stream_chunk_count = raw.get("stream_chunk_count")
    if (
        not isinstance(stream_chunk_count, int)
        or isinstance(stream_chunk_count, bool)
        or stream_chunk_count < 0
        or stream_chunk_count != len(stream_messages)
    ):
        raise TurnTransactionSchemaError(
            "terminal detour stream count is invalid"
        )
    stream_hash = require_hash("stream_hash")
    if stream_hash != _rolling_stream_hash(stream_messages):
        raise TurnTransactionSchemaError(
            "terminal detour stream hash is invalid"
        )
    if effect_class != "streaming_observation":
        if stream_messages or stream_stop_reason != "not_streaming":
            raise TurnTransactionSchemaError(
                "non-streaming terminal detour carries stream evidence"
            )
    elif status == "completed":
        if effect_status == "succeeded" and stream_stop_reason not in {
            "completed",
            "deadline",
            "limit_reached",
        }:
            raise TurnTransactionSchemaError(
                "successful terminal stream stop reason is invalid"
            )
        if effect_status == "failed" and stream_stop_reason not in {
            "cancelled",
            "not_found",
            "interrupted",
            "error",
        }:
            raise TurnTransactionSchemaError(
                "failed terminal stream stop reason is invalid"
            )
        if response_messages[:stream_chunk_count] != stream_messages:
            raise TurnTransactionSchemaError(
                "terminal stream response omits persisted chunks"
            )

    def head_identity(suffix: str) -> tuple[Any, str, str, str]:
        revision = raw.get(f"product_revision_{suffix}")
        thread_id = str(
            raw.get(f"product_checkpoint_thread_id_{suffix}") or ""
        )
        checkpoint_id = str(raw.get(f"product_checkpoint_id_{suffix}") or "")
        fingerprint = str(raw.get(f"product_fingerprint_{suffix}") or "")
        if revision is None:
            if thread_id or checkpoint_id or fingerprint:
                raise TurnTransactionSchemaError(
                    "terminal detour absent Product Head identity is invalid"
                )
        elif (
            not isinstance(revision, int)
            or isinstance(revision, bool)
            or revision < 0
            or not thread_id
            or not checkpoint_id
            or require_hash(
                f"product_fingerprint_{suffix}",
            )
            != fingerprint
        ):
            raise TurnTransactionSchemaError(
                "terminal detour Product Head identity is invalid"
            )
        return revision, thread_id, checkpoint_id, fingerprint

    before = head_identity("before")
    after = head_identity("after")
    shell_state_after_hash = require_hash(
        "shell_state_after_hash",
        optional=True,
    )
    response_hash = require_hash("response_hash", optional=True)
    diagnostic_hash = require_hash("diagnostic_hash", optional=True)
    evidence_hash = require_hash("evidence_hash", optional=True)
    resolution = raw.get("resolution")
    completed_at = raw.get("completed_at")
    delivered_at = raw.get("delivered_at")

    if (resolution is None) != (evidence_hash is None) or (
        resolution is not None
        and str(resolution) not in _RECONCILIATION_RESOLUTIONS
    ):
        raise TurnTransactionSchemaError(
            "terminal detour reconciliation evidence is invalid"
        )
    if resolution is not None:
        expected_resolution_status = {
            "effect_confirmed": "succeeded",
            "effect_not_observed": "failed",
        }[str(resolution)]
        if effect_class == "local_external_effect":
            if (
                status != "completed"
                or effect_status != expected_resolution_status
                or diagnostic_hash is None
                or response_messages
            ):
                raise TurnTransactionSchemaError(
                    "resolved external-effect detour provenance is invalid"
                )
        elif effect_class == "authority_resolution":
            if (
                status != "completed"
                or effect_status != "not_applicable"
                or diagnostic_hash is not None
                or not response_messages
            ):
                raise TurnTransactionSchemaError(
                    "authority-resolution detour provenance is invalid"
                )
        else:
            raise TurnTransactionSchemaError(
                "terminal detour reconciliation provenance is invalid"
            )
    elif effect_class == "authority_resolution":
        raise TurnTransactionSchemaError(
            "authority-resolution detour lacks reconciliation evidence"
        )
    if status == "prepared":
        if (
            after != (None, "", "", "")
            or shell_state_after_hash is not None
            or response_messages
            or response_hash is not None
            or effect_status != "not_applicable"
            or diagnostic_hash is not None
            or resolution is not None
            or completed_at is not None
            or delivered_at is not None
            or stream_stop_reason != "not_streaming"
        ):
            raise TurnTransactionSchemaError(
                "prepared terminal detour carries completion evidence"
            )
    elif status == "reconciliation_required":
        if (
            effect_class != "local_external_effect"
            or effect_status != "uncertain"
            or diagnostic_hash is None
            or after != (None, "", "", "")
            or shell_state_after_hash is not None
            or response_messages
            or response_hash is not None
            or resolution is not None
            or completed_at is not None
            or delivered_at is not None
            or stream_stop_reason != "not_streaming"
        ):
            raise TurnTransactionSchemaError(
                "terminal detour reconciliation state is invalid"
            )
    else:
        expected_response_json = json.dumps(
            list(response_messages),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if (
            after != before
            or shell_state_after_hash is None
            or response_hash
            != state_fingerprint(expected_response_json.encode("utf-8"))
            or str(raw.get("response_json") or "") != expected_response_json
            or completed_at is None
            or effect_status == "uncertain"
            or (not response_messages and resolution is None)
            or (
                resolution is None
                and (
                    (effect_status == "failed") != (diagnostic_hash is not None)
                )
            )
        ):
            raise TurnTransactionSchemaError(
                "completed terminal detour evidence is invalid"
            )
    if delivered_at is not None and status != "completed":
        raise TurnTransactionSchemaError(
            "undeliverable terminal detour carries delivery evidence"
        )

    if has_runtime_fence:
        fence_sequence = raw.get("runtime_event_fence_sequence")
        if (
            not isinstance(fence_sequence, int)
            or isinstance(fence_sequence, bool)
            or fence_sequence < 0
        ):
            raise TurnTransactionSchemaError(
                "terminal detour runtime fence is invalid"
            )
        fence_terminal_event_id = str(
            raw.get("runtime_event_fence_terminal_event_id") or ""
        )
        fence_id = str(raw.get("runtime_event_fence_id") or "")
        fence_hash = str(raw.get("runtime_event_fence_hash") or "")
        if fence_sequence == 0:
            if fence_terminal_event_id or fence_id or fence_hash:
                raise TurnTransactionSchemaError(
                    "empty terminal detour runtime fence is invalid"
                )
        elif (
            not fence_terminal_event_id
            or not fence_id
            or require_hash("runtime_event_fence_hash") != fence_hash
        ):
            raise TurnTransactionSchemaError(
                "terminal detour runtime fence identity is invalid"
            )
    return response_messages, stream_messages


def _detour_from_row(row: sqlite3.Row) -> TerminalDetour:
    raw = dict(row)
    raw_messages, raw_stream_messages = _validate_detour_semantics(
        raw,
        has_runtime_fence=True,
        required_identity_trust="trusted",
    )
    result_kind = str(row["result_kind"])
    interruption_kind = row["interruption_kind"]
    termination_reason = row["termination_reason"]
    exit_code = row["exit_code"]
    effect_class = str(row["effect_class"])
    status = str(row["status"])
    effect_status = str(row["effect_status"])
    stream_chunk_count = int(row["stream_chunk_count"])
    stream_hash = str(row["stream_hash"])
    stream_stop_reason = str(row["stream_stop_reason"])
    identity_trust = str(row["identity_trust"])
    return TerminalDetour(
        detour_id=str(row["detour_id"]),
        logical_thread_id=str(row["logical_thread_id"]),
        process_instance_id=str(row["process_instance_id"]),
        session_id=str(row["session_id"]),
        session_purpose=str(row["session_purpose"]),
        command_name=str(row["command_name"]),
        input_hash=str(row["input_hash"]),
        result_kind=result_kind,  # type: ignore[arg-type]
        interruption_kind=interruption_kind,
        termination_reason=termination_reason,
        exit_code=exit_code,
        effect_class=effect_class,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        shell_state_before_hash=str(row["shell_state_before_hash"]),
        shell_state_after_hash=row["shell_state_after_hash"],
        product_revision_before=row["product_revision_before"],
        product_checkpoint_thread_id_before=(
            row["product_checkpoint_thread_id_before"]
        ),
        product_checkpoint_id_before=row["product_checkpoint_id_before"],
        product_fingerprint_before=row["product_fingerprint_before"],
        product_revision_after=row["product_revision_after"],
        product_checkpoint_thread_id_after=(
            row["product_checkpoint_thread_id_after"]
        ),
        product_checkpoint_id_after=row["product_checkpoint_id_after"],
        product_fingerprint_after=row["product_fingerprint_after"],
        runtime_event_fence_sequence=int(
            row["runtime_event_fence_sequence"]
        ),
        runtime_event_fence_terminal_event_id=str(
            row["runtime_event_fence_terminal_event_id"]
        ),
        runtime_event_fence_id=str(row["runtime_event_fence_id"]),
        runtime_event_fence_hash=str(row["runtime_event_fence_hash"]),
        response_messages=raw_messages,
        response_hash=row["response_hash"],
        stream_chunk_count=stream_chunk_count,
        stream_hash=stream_hash,
        stream_messages=raw_stream_messages,
        stream_stop_reason=stream_stop_reason,  # type: ignore[arg-type]
        identity_trust=identity_trust,  # type: ignore[arg-type]
        effect_status=effect_status,  # type: ignore[arg-type]
        diagnostic_hash=row["diagnostic_hash"],
        resolution=row["resolution"],
        evidence_hash=row["evidence_hash"],
        origin_revision_commit=str(row["origin_revision_commit"]),
        origin_revision_worktree_hash=str(
            row["origin_revision_worktree_hash"]
        ),
        created_at=str(row["created_at"]),
        completed_at=row["completed_at"],
        delivered_at=row["delivered_at"],
    )


def _rolling_stream_hash(messages: Sequence[str]) -> str:
    current = _EMPTY_STREAM_HASH
    for message in messages:
        chunk_hash = state_fingerprint(str(message).encode("utf-8"))
        current = state_fingerprint(f"{current}:{chunk_hash}".encode("ascii"))
    return current

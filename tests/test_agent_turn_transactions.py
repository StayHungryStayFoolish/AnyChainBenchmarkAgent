from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from agent.harness.turn_transactions import (
    ProductAuthorityLease,
    ReconciliationRequiredError,
    TURN_TRANSACTION_SCHEMA_VERSION,
    TurnTransactionConflictError,
    TurnTransactionSchemaError,
    TurnTransactionStore,
    TurnTransactionValidationError,
    state_fingerprint,
)


def _hash(label: str) -> str:
    return state_fingerprint(label.encode("utf-8"))


def _create_v9_ledger(path: Path) -> None:
    """Create the published v9 table shape without using current-schema DDL."""

    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE anychain_turn_transaction_meta (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            schema_version INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        INSERT INTO anychain_turn_transaction_meta
        VALUES (1, 9, '2026-07-27T00:00:00+00:00');

        CREATE TABLE anychain_product_heads (
            logical_thread_id TEXT PRIMARY KEY,
            checkpoint_thread_id TEXT NOT NULL,
            checkpoint_id TEXT NOT NULL,
            state_fingerprint TEXT NOT NULL,
            revision INTEGER NOT NULL CHECK (revision >= 0),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE anychain_turn_attempts (
            transaction_id TEXT PRIMARY KEY,
            logical_thread_id TEXT NOT NULL,
            physical_thread_id TEXT NOT NULL UNIQUE,
            base_checkpoint_thread_id TEXT NOT NULL,
            base_checkpoint_id TEXT NOT NULL,
            base_fingerprint TEXT NOT NULL,
            status TEXT NOT NULL,
            attempt_checkpoint_id TEXT,
            attempt_fingerprint TEXT,
            diagnostic_hash TEXT,
            created_at TEXT NOT NULL,
            completed_at TEXT,
            origin_revision_commit TEXT NOT NULL,
            origin_revision_worktree_hash TEXT NOT NULL
        );
        CREATE UNIQUE INDEX anychain_one_active_attempt_per_thread
        ON anychain_turn_attempts(logical_thread_id)
        WHERE status = 'active';

        CREATE TABLE anychain_terminal_outbox (
            event_id TEXT PRIMARY KEY,
            transaction_id TEXT NOT NULL UNIQUE,
            logical_thread_id TEXT NOT NULL,
            physical_thread_id TEXT NOT NULL,
            outcome TEXT NOT NULL,
            base_checkpoint_thread_id TEXT NOT NULL,
            base_checkpoint_id TEXT NOT NULL,
            base_fingerprint TEXT NOT NULL,
            attempt_checkpoint_id TEXT,
            attempt_fingerprint TEXT,
            product_checkpoint_thread_id TEXT NOT NULL,
            product_checkpoint_id TEXT NOT NULL,
            product_fingerprint TEXT NOT NULL,
            diagnostic_hash TEXT,
            created_at TEXT NOT NULL,
            render_hash TEXT NOT NULL DEFAULT '',
            failure_category TEXT NOT NULL DEFAULT '',
            delivered_at TEXT,
            origin_revision_commit TEXT NOT NULL,
            origin_revision_worktree_hash TEXT NOT NULL
        );

        CREATE TABLE anychain_reconciliation_resolutions (
            transaction_id TEXT PRIMARY KEY,
            logical_thread_id TEXT NOT NULL,
            resolution TEXT NOT NULL,
            evidence_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE anychain_terminal_detours (
            detour_id TEXT PRIMARY KEY,
            logical_thread_id TEXT NOT NULL,
            process_instance_id TEXT NOT NULL,
            command_name TEXT NOT NULL,
            effect_class TEXT NOT NULL,
            status TEXT NOT NULL,
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
            effect_status TEXT NOT NULL,
            diagnostic_hash TEXT,
            resolution TEXT,
            evidence_hash TEXT,
            origin_revision_commit TEXT NOT NULL,
            origin_revision_worktree_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            completed_at TEXT,
            delivered_at TEXT,
            input_hash TEXT NOT NULL,
            session_id TEXT NOT NULL,
            session_purpose TEXT NOT NULL,
            stream_chunk_count INTEGER NOT NULL DEFAULT 0,
            stream_hash TEXT NOT NULL,
            result_kind TEXT NOT NULL DEFAULT 'command',
            termination_reason TEXT,
            exit_code INTEGER,
            stream_stop_reason TEXT NOT NULL DEFAULT 'not_streaming',
            identity_trust TEXT NOT NULL DEFAULT 'legacy_unbound',
            stream_messages_json TEXT NOT NULL DEFAULT '[]'
        );
        CREATE UNIQUE INDEX anychain_one_active_detour_per_thread
        ON anychain_terminal_detours(logical_thread_id)
        WHERE status = 'prepared';
        """
    )
    connection.commit()
    connection.close()


class TurnTransactionStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.path = Path(self.tempdir.name) / "checkpoints.sqlite"
        self.store = TurnTransactionStore(self.path)
        self.base = self.store.bootstrap_product_head(
            logical_thread_id="logical-session",
            checkpoint_thread_id="logical-session",
            checkpoint_id="checkpoint-0",
            state_fingerprint=_hash("state-0"),
        )

    def _begin(self, transaction_id: str | None = None):
        return self.store.begin_attempt(
            logical_thread_id="logical-session",
            transaction_id=transaction_id,
        )

    def _assert_quarantined_detour_semantics_fail(
        self,
        mutate,
        expected_error: str,
    ) -> None:
        prepared = self.store.prepare_terminal_detour(
            logical_thread_id="logical-session",
            process_instance_id="legacy-process",
            session_id="legacy-session",
            session_purpose="chaos",
            command_name="help",
            input_hash=_hash("help"),
            effect_class="read_only",
            shell_state_before_hash=_hash("shell-before"),
            origin_revision_commit="test-revision",
            origin_revision_worktree_hash=_hash("revision"),
        )
        completed = self.store.complete_terminal_detour(
            detour_id=prepared.detour_id,
            shell_state_after_hash=_hash("shell-before"),
            response_messages=("help",),
        )
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        record = dict(
            connection.execute(
                "SELECT * FROM anychain_terminal_detours WHERE detour_id = ?",
                (completed.detour_id,),
            ).fetchone()
        )
        record["identity_trust"] = "legacy_unbound"
        mutate(record)
        record_json = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
        )
        connection.execute(
            "DELETE FROM anychain_terminal_detours WHERE detour_id = ?",
            (completed.detour_id,),
        )
        connection.execute(
            """
            INSERT INTO anychain_terminal_detour_quarantine (
                detour_id, logical_thread_id, command_name,
                quarantine_reason, record_json, record_hash, quarantined_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                completed.detour_id,
                "logical-session",
                "help",
                "legacy_runtime_fence_unbound",
                record_json,
                state_fingerprint(record_json.encode("utf-8")),
                "2026-07-27T00:00:00+00:00",
            ),
        )
        connection.execute(
            "UPDATE anychain_turn_transaction_meta "
            "SET schema_version = 13 WHERE singleton = 1"
        )
        connection.commit()
        connection.close()

        with self.assertRaisesRegex(
            TurnTransactionSchemaError,
            expected_error,
        ):
            TurnTransactionStore(self.path)

    def test_schema_coexists_with_existing_checkpoint_tables(self) -> None:
        other = Path(self.tempdir.name) / "shared.sqlite"
        connection = sqlite3.connect(other)
        connection.execute("CREATE TABLE checkpoints (thread_id TEXT PRIMARY KEY)")
        connection.execute("INSERT INTO checkpoints VALUES ('existing')")
        connection.commit()
        connection.close()

        store = TurnTransactionStore(other)
        store.bootstrap_product_head(
            logical_thread_id="shared-session",
            checkpoint_thread_id="shared-session",
            checkpoint_id="checkpoint-0",
            state_fingerprint=_hash("shared"),
        )

        reader = sqlite3.connect(other)
        self.addCleanup(reader.close)
        self.assertEqual(
            reader.execute("SELECT thread_id FROM checkpoints").fetchone()[0],
            "existing",
        )
        version = reader.execute(
            "SELECT schema_version FROM anychain_turn_transaction_meta WHERE singleton = 1"
        ).fetchone()[0]
        self.assertEqual(version, TURN_TRANSACTION_SCHEMA_VERSION)

    def test_v1_outbox_migration_does_not_replay_unverifiable_history(self) -> None:
        legacy = Path(self.tempdir.name) / "legacy.sqlite"
        connection = sqlite3.connect(legacy)
        connection.executescript(
            """
            CREATE TABLE anychain_turn_transaction_meta (
                singleton INTEGER PRIMARY KEY,
                schema_version INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            INSERT INTO anychain_turn_transaction_meta
            VALUES (1, 1, '2026-07-27T00:00:00+00:00');

            CREATE TABLE anychain_terminal_outbox (
                event_id TEXT PRIMARY KEY,
                transaction_id TEXT NOT NULL UNIQUE,
                logical_thread_id TEXT NOT NULL,
                physical_thread_id TEXT NOT NULL,
                outcome TEXT NOT NULL,
                base_checkpoint_thread_id TEXT NOT NULL,
                base_checkpoint_id TEXT NOT NULL,
                base_fingerprint TEXT NOT NULL,
                attempt_checkpoint_id TEXT,
                attempt_fingerprint TEXT,
                product_checkpoint_thread_id TEXT NOT NULL,
                product_checkpoint_id TEXT NOT NULL,
                product_fingerprint TEXT NOT NULL,
                diagnostic_hash TEXT,
                created_at TEXT NOT NULL
            );
            """
        )
        connection.execute(
            """
            INSERT INTO anychain_terminal_outbox
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-event",
                "00000000-0000-0000-0000-000000000001",
                "legacy-session",
                "attempt:00000000-0000-0000-0000-000000000001",
                "aborted",
                "legacy-session",
                "checkpoint-0",
                _hash("base"),
                None,
                None,
                "legacy-session",
                "checkpoint-0",
                _hash("base"),
                _hash("failure"),
                "2026-07-27T00:00:00+00:00",
            ),
        )
        connection.commit()
        connection.close()

        migrated = TurnTransactionStore(legacy)
        outcomes = migrated.list_terminal_outcomes("legacy-session")

        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0].failure_category, "legacy_unclassified")
        self.assertIsNotNone(outcomes[0].delivered_at)
        self.assertEqual(
            migrated.list_undelivered_outcomes("legacy-session"),
            (),
        )

    def test_failed_multi_step_migration_rolls_back_schema_and_version(self) -> None:
        legacy = Path(self.tempdir.name) / "migration-failure.sqlite"
        connection = sqlite3.connect(legacy)
        connection.executescript(
            """
            CREATE TABLE anychain_turn_transaction_meta (
                singleton INTEGER PRIMARY KEY,
                schema_version INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            INSERT INTO anychain_turn_transaction_meta
            VALUES (1, 1, '2026-07-27T00:00:00+00:00');

            CREATE TABLE anychain_terminal_outbox (
                event_id TEXT PRIMARY KEY,
                transaction_id TEXT NOT NULL UNIQUE,
                logical_thread_id TEXT NOT NULL,
                physical_thread_id TEXT NOT NULL,
                outcome TEXT NOT NULL,
                base_checkpoint_thread_id TEXT NOT NULL,
                base_checkpoint_id TEXT NOT NULL,
                base_fingerprint TEXT NOT NULL,
                attempt_checkpoint_id TEXT,
                attempt_fingerprint TEXT,
                product_checkpoint_thread_id TEXT NOT NULL,
                product_checkpoint_id TEXT NOT NULL,
                product_fingerprint TEXT NOT NULL,
                diagnostic_hash TEXT,
                created_at TEXT NOT NULL
            );
            """
        )
        connection.commit()
        connection.close()

        with patch.object(
            TurnTransactionStore,
            "_migrate_v2_to_v3",
            side_effect=RuntimeError("injected second migration failure"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "injected second migration failure",
            ):
                TurnTransactionStore(legacy)

        reader = sqlite3.connect(legacy)
        self.addCleanup(reader.close)
        version = reader.execute(
            "SELECT schema_version FROM anychain_turn_transaction_meta"
        ).fetchone()[0]
        columns = {
            row[1]
            for row in reader.execute(
                "PRAGMA table_info(anychain_terminal_outbox)"
            ).fetchall()
        }
        self.assertEqual(version, 1)
        self.assertNotIn("render_hash", columns)
        self.assertNotIn("failure_category", columns)
        self.assertNotIn("delivered_at", columns)

    def test_product_authority_lease_prevents_concurrent_runtime_ownership(
        self,
    ) -> None:
        first = ProductAuthorityLease(self.path, "logical-session")
        second = ProductAuthorityLease(self.path, "logical-session")
        first.acquire()
        self.addCleanup(first.close)

        with self.assertRaises(TurnTransactionConflictError):
            second.acquire()

        first.close()
        second.acquire()
        second.close()

    def test_unsupported_schema_fails_closed(self) -> None:
        connection = sqlite3.connect(self.path)
        connection.execute(
            "UPDATE anychain_turn_transaction_meta SET schema_version = 999 WHERE singleton = 1"
        )
        connection.commit()
        connection.close()

        with self.assertRaises(TurnTransactionSchemaError):
            TurnTransactionStore(self.path)

    def test_bootstrap_is_idempotent_but_identity_is_immutable(self) -> None:
        repeated = self.store.bootstrap_product_head(
            logical_thread_id="logical-session",
            checkpoint_thread_id="logical-session",
            checkpoint_id="checkpoint-0",
            state_fingerprint=_hash("state-0"),
        )
        self.assertEqual(repeated, self.base)

        with self.assertRaises(TurnTransactionConflictError):
            self.store.bootstrap_product_head(
                logical_thread_id="logical-session",
                checkpoint_thread_id="logical-session",
                checkpoint_id="checkpoint-other",
                state_fingerprint=_hash("other"),
            )

    def test_begin_attempt_uses_unique_physical_thread_and_immutable_base(self) -> None:
        transaction_id = str(uuid.uuid4())
        attempt = self._begin(transaction_id)
        repeated = self._begin(transaction_id)

        self.assertEqual(attempt, repeated)
        self.assertEqual(attempt.physical_thread_id, f"attempt:{transaction_id}")
        self.assertEqual(attempt.base_checkpoint_thread_id, self.base.checkpoint_thread_id)
        self.assertEqual(attempt.base_checkpoint_id, self.base.checkpoint_id)
        self.assertEqual(attempt.base_fingerprint, self.base.state_fingerprint)
        with self.assertRaises(TurnTransactionConflictError):
            self._begin()

    def test_terminal_detour_completes_without_advancing_product_head(self) -> None:
        before = self.store.get_product_head("logical-session")
        detour = self.store.prepare_terminal_detour(
            logical_thread_id="logical-session",
            process_instance_id="process-1",
            session_id="session-1",
            session_purpose="chaos",
            command_name="help",
            input_hash=_hash("help"),
            effect_class="read_only",
            shell_state_before_hash=_hash("shell-before"),
            origin_revision_commit="test-revision",
            origin_revision_worktree_hash=_hash("revision"),
        )
        completed = self.store.complete_terminal_detour(
            detour_id=detour.detour_id,
            shell_state_after_hash=_hash("shell-before"),
            response_messages=("help text",),
        )
        after = self.store.get_product_head("logical-session")

        self.assertEqual(completed.status, "completed")
        self.assertEqual(completed.response_messages, ("help text",))
        self.assertEqual(before, after)
        self.assertEqual(
            completed.product_fingerprint_before,
            completed.product_fingerprint_after,
        )
        self.assertEqual(
            self.store.list_undelivered_terminal_detours("logical-session"),
            (completed,),
        )
        delivered = self.store.mark_terminal_detour_delivered(
            logical_thread_id="logical-session",
            detour_id=completed.detour_id,
        )
        self.assertIsNotNone(delivered.delivered_at)

    def test_open_detour_and_workflow_attempt_are_mutually_exclusive(self) -> None:
        detour = self.store.prepare_terminal_detour(
            logical_thread_id="logical-session",
            process_instance_id="process-1",
            session_id="session-1",
            session_purpose="chaos",
            command_name="doctor",
            input_hash=_hash("doctor"),
            effect_class="observation_refresh",
            shell_state_before_hash=_hash("shell-before"),
            origin_revision_commit="test-revision",
            origin_revision_worktree_hash=_hash("revision"),
        )
        with self.assertRaises(ReconciliationRequiredError):
            self._begin()
        self.store.complete_terminal_detour(
            detour_id=detour.detour_id,
            shell_state_after_hash=_hash("shell-after"),
            response_messages=("doctor complete",),
        )

        attempt = self._begin()
        with self.assertRaises(TurnTransactionConflictError):
            self.store.prepare_terminal_detour(
                logical_thread_id="logical-session",
                process_instance_id="process-1",
                session_id="session-1",
                session_purpose="chaos",
                command_name="help",
                input_hash=_hash("help"),
                effect_class="read_only",
                shell_state_before_hash=_hash("shell-after"),
                origin_revision_commit="test-revision",
                origin_revision_worktree_hash=_hash("revision"),
            )
        self.store.abort_attempt(
            logical_thread_id="logical-session",
            transaction_id=attempt.transaction_id,
            physical_thread_id=attempt.physical_thread_id,
            diagnostic_hash=_hash("cancelled"),
            failure_category="cancelled",
        )

    def test_interrupted_external_detour_fails_closed_for_reconciliation(self) -> None:
        detour = self.store.prepare_terminal_detour(
            logical_thread_id="logical-session",
            process_instance_id="process-1",
            session_id="session-1",
            session_purpose="chaos",
            command_name="install_dependencies",
            input_hash=_hash("install"),
            effect_class="local_external_effect",
            shell_state_before_hash=_hash("shell-before"),
            origin_revision_commit="test-revision",
            origin_revision_worktree_hash=_hash("revision"),
        )
        recovered = self.store.recover_prepared_terminal_detours(
            logical_thread_id="logical-session",
            interrupted_message="retry command",
        )

        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0].detour_id, detour.detour_id)
        self.assertEqual(recovered[0].status, "reconciliation_required")
        self.assertEqual(recovered[0].effect_status, "uncertain")
        with self.assertRaises(ReconciliationRequiredError):
            self._begin()

    def test_interrupted_read_only_detour_becomes_replayable_failure(self) -> None:
        detour = self.store.prepare_terminal_detour(
            logical_thread_id="logical-session",
            process_instance_id="process-1",
            session_id="session-1",
            session_purpose="chaos",
            command_name="jobs",
            input_hash=_hash("jobs"),
            effect_class="read_only",
            shell_state_before_hash=_hash("shell-before"),
            origin_revision_commit="test-revision",
            origin_revision_worktree_hash=_hash("revision"),
        )
        recovered = self.store.recover_prepared_terminal_detours(
            logical_thread_id="logical-session",
            interrupted_message="retry command",
        )

        self.assertEqual(recovered[0].detour_id, detour.detour_id)
        self.assertEqual(recovered[0].status, "completed")
        self.assertEqual(recovered[0].effect_status, "failed")
        self.assertEqual(recovered[0].response_messages, ("retry command",))

    def test_interrupted_termination_intent_does_not_terminate_new_process(
        self,
    ) -> None:
        detour = self.store.prepare_terminal_detour(
            logical_thread_id="logical-session",
            process_instance_id="process-1",
            session_id="session-1",
            session_purpose="chaos",
            command_name="exit",
            input_hash=_hash("exit"),
            result_kind="session_termination",
            termination_reason="exit",
            exit_code=0,
            effect_class="read_only",
            shell_state_before_hash=_hash("shell-before"),
            origin_revision_commit="test-revision",
            origin_revision_worktree_hash=_hash("revision"),
        )

        recovered = self.store.recover_prepared_terminal_detours(
            logical_thread_id="logical-session",
            interrupted_message="termination interrupted",
        )

        self.assertEqual(recovered[0].detour_id, detour.detour_id)
        self.assertEqual(
            recovered[0].result_kind,
            "command",
        )
        self.assertEqual(recovered[0].effect_status, "failed")
        self.assertEqual(
            recovered[0].interruption_kind,
            "session_termination_interrupted",
        )
        self.assertEqual(recovered[0].termination_reason, "exit")
        self.assertEqual(recovered[0].exit_code, 0)

    def test_v9_unbound_open_detour_is_quarantined_without_blocking_session(
        self,
    ) -> None:
        legacy = Path(self.tempdir.name) / "real-v9.sqlite"
        _create_v9_ledger(legacy)
        detour_id = "legacy-detour"
        connection = sqlite3.connect(legacy)
        connection.execute(
            """
            INSERT INTO anychain_terminal_detours (
                detour_id, logical_thread_id, process_instance_id,
                command_name, effect_class, status,
                shell_state_before_hash, effect_status,
                origin_revision_commit, origin_revision_worktree_hash,
                created_at, input_hash, session_id, session_purpose,
                stream_hash, result_kind, stream_stop_reason,
                identity_trust, stream_messages_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                detour_id,
                "logical-session",
                "legacy-process",
                "help",
                "read_only",
                "prepared",
                _hash("shell-before"),
                "not_applicable",
                "test-revision",
                _hash("revision"),
                "2026-07-27T00:00:00+00:00",
                _hash("help"),
                "legacy-session",
                "chaos",
                state_fingerprint(b""),
                "command",
                "not_streaming",
                "legacy_unbound",
                "[]",
            ),
        )
        connection.commit()
        connection.close()

        migrated = TurnTransactionStore(legacy)

        self.assertIsNone(
            migrated.get_unresolved_terminal_detour("logical-session")
        )
        connection = sqlite3.connect(legacy)
        self.addCleanup(connection.close)
        quarantined = connection.execute(
            """
            SELECT quarantine_reason
            FROM anychain_terminal_detour_quarantine
            WHERE detour_id = ?
            """,
            (detour_id,),
        ).fetchone()
        self.assertEqual(quarantined[0], "legacy_unbound_identity")
        self.assertEqual(
            connection.execute(
                """
                SELECT schema_version
                FROM anychain_turn_transaction_meta
                WHERE singleton = 1
                """
            ).fetchone()[0],
            TURN_TRANSACTION_SCHEMA_VERSION,
        )

    def test_v12_migration_binds_pending_committed_runtime_event_identity(
        self,
    ) -> None:
        attempt = self._begin()
        outcome = self.store.commit_attempt(
            logical_thread_id=attempt.logical_thread_id,
            transaction_id=attempt.transaction_id,
            physical_thread_id=attempt.physical_thread_id,
            attempt_checkpoint_id="checkpoint-1",
            attempt_fingerprint=_hash("state-1"),
        )
        connection = sqlite3.connect(self.path)
        connection.execute(
            """
            UPDATE anychain_terminal_outbox
            SET runtime_event_id = '',
                runtime_event_status = 'pending',
                runtime_event_payload_hash = NULL,
                runtime_event_published_at = NULL
            WHERE event_id = ?
            """,
            (outcome.event_id,),
        )
        connection.execute(
            """
            UPDATE anychain_turn_transaction_meta
            SET schema_version = 12
            WHERE singleton = 1
            """
        )
        connection.commit()
        connection.close()

        migrated = TurnTransactionStore(self.path)
        recovered = migrated.get_terminal_outcome(attempt.transaction_id)

        self.assertIsNotNone(recovered)
        self.assertEqual(recovered.runtime_event_status, "pending")
        self.assertEqual(
            recovered.runtime_event_id,
            f"migrated-runtime:{outcome.event_id}",
        )

    def test_real_v11_shape_migrates_to_v13(self) -> None:
        attempt = self._begin()
        outcome = self.store.commit_attempt(
            logical_thread_id=attempt.logical_thread_id,
            transaction_id=attempt.transaction_id,
            physical_thread_id=attempt.physical_thread_id,
            attempt_checkpoint_id="checkpoint-1",
            attempt_fingerprint=_hash("state-1"),
        )
        connection = sqlite3.connect(self.path)
        connection.execute("DROP INDEX anychain_runtime_event_id")
        connection.execute(
            "DROP INDEX anychain_runtime_event_sequence_per_thread"
        )
        for column in (
            "runtime_event_id",
            "runtime_event_sequence",
            "runtime_event_status",
            "runtime_event_payload_hash",
            "runtime_event_published_at",
        ):
            connection.execute(
                f"ALTER TABLE anychain_terminal_outbox DROP COLUMN {column}"
            )
        for column in (
            "runtime_event_fence_sequence",
            "runtime_event_fence_terminal_event_id",
            "runtime_event_fence_id",
            "runtime_event_fence_hash",
        ):
            connection.execute(
                f"ALTER TABLE anychain_terminal_detours DROP COLUMN {column}"
            )
        connection.execute(
            """
            UPDATE anychain_turn_transaction_meta
            SET schema_version = 11
            WHERE singleton = 1
            """
        )
        connection.commit()
        connection.close()

        migrated = TurnTransactionStore(self.path)
        recovered = migrated.get_terminal_outcome(attempt.transaction_id)

        self.assertEqual(recovered.base_revision, 0)
        self.assertEqual(recovered.product_revision, 1)
        self.assertEqual(recovered.runtime_event_status, "pending")
        self.assertEqual(
            recovered.runtime_event_id,
            f"migrated-runtime:{outcome.event_id}",
        )

    def test_v11_completed_detour_is_audited_in_quarantine(self) -> None:
        prepared = self.store.prepare_terminal_detour(
            logical_thread_id="logical-session",
            process_instance_id="legacy-process",
            session_id="legacy-session",
            session_purpose="chaos",
            command_name="help",
            input_hash=_hash("help"),
            effect_class="read_only",
            shell_state_before_hash=_hash("shell"),
            origin_revision_commit="test-revision",
            origin_revision_worktree_hash=_hash("revision"),
        )
        completed = self.store.complete_terminal_detour(
            detour_id=prepared.detour_id,
            shell_state_after_hash=_hash("shell"),
            response_messages=("help",),
        )
        connection = sqlite3.connect(self.path)
        connection.execute("DROP INDEX anychain_runtime_event_id")
        connection.execute(
            "DROP INDEX anychain_runtime_event_sequence_per_thread"
        )
        for column in (
            "runtime_event_id",
            "runtime_event_sequence",
            "runtime_event_status",
            "runtime_event_payload_hash",
            "runtime_event_published_at",
        ):
            connection.execute(
                f"ALTER TABLE anychain_terminal_outbox DROP COLUMN {column}"
            )
        for column in (
            "runtime_event_fence_sequence",
            "runtime_event_fence_terminal_event_id",
            "runtime_event_fence_id",
            "runtime_event_fence_hash",
        ):
            connection.execute(
                f"ALTER TABLE anychain_terminal_detours DROP COLUMN {column}"
            )
        connection.execute(
            """
            UPDATE anychain_turn_transaction_meta
            SET schema_version = 11
            WHERE singleton = 1
            """
        )
        connection.commit()
        connection.close()

        migrated = TurnTransactionStore(self.path)
        self.assertEqual(
            migrated.list_undelivered_terminal_detours("logical-session"),
            (),
        )
        connection = sqlite3.connect(self.path)
        self.addCleanup(connection.close)
        quarantined = connection.execute(
            """
            SELECT quarantine_reason, record_json, record_hash
            FROM anychain_terminal_detour_quarantine
            WHERE detour_id = ?
            """,
            (completed.detour_id,),
        ).fetchone()
        self.assertEqual(
            quarantined[0],
            "legacy_runtime_fence_unbound",
        )
        record_json = str(quarantined[1])
        record = json.loads(record_json)
        self.assertEqual(record["status"], "completed")
        self.assertEqual(
            json.loads(record["response_json"]),
            ["help"],
        )
        self.assertEqual(
            quarantined[2],
            state_fingerprint(record_json.encode("utf-8")),
        )

    def test_v13_quarantine_table_migrates_to_complete_record_schema(
        self,
    ) -> None:
        connection = sqlite3.connect(self.path)
        connection.execute(
            "ALTER TABLE anychain_terminal_detour_quarantine "
            "DROP COLUMN record_json"
        )
        connection.execute(
            """
            UPDATE anychain_turn_transaction_meta
            SET schema_version = 13
            WHERE singleton = 1
            """
        )
        connection.commit()
        connection.close()

        TurnTransactionStore(self.path)
        connection = sqlite3.connect(self.path)
        self.addCleanup(connection.close)
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(anychain_terminal_detour_quarantine)"
            )
        }
        version = connection.execute(
            """
            SELECT schema_version
            FROM anychain_turn_transaction_meta
            WHERE singleton = 1
            """
        ).fetchone()[0]

        self.assertIn("record_json", columns)
        self.assertEqual(version, TURN_TRANSACTION_SCHEMA_VERSION)

    def test_v13_incomplete_quarantine_history_fails_closed(self) -> None:
        connection = sqlite3.connect(self.path)
        connection.execute(
            """
            INSERT INTO anychain_terminal_detour_quarantine (
                detour_id, logical_thread_id, command_name,
                quarantine_reason, record_json, record_hash, quarantined_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-incomplete",
                "logical-session",
                "help",
                "legacy",
                "{}",
                _hash("{}"),
                "2026-07-27T00:00:00+00:00",
            ),
        )
        connection.execute(
            "ALTER TABLE anychain_terminal_detour_quarantine "
            "DROP COLUMN record_json"
        )
        connection.execute(
            """
            UPDATE anychain_turn_transaction_meta
            SET schema_version = 13
            WHERE singleton = 1
            """
        )
        connection.commit()
        connection.close()

        with self.assertRaisesRegex(
            TurnTransactionSchemaError,
            "do not contain their canonical records",
        ):
            TurnTransactionStore(self.path)

        connection = sqlite3.connect(self.path)
        self.addCleanup(connection.close)
        version = connection.execute(
            """
            SELECT schema_version
            FROM anychain_turn_transaction_meta
            WHERE singleton = 1
            """
        ).fetchone()[0]
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(anychain_terminal_detour_quarantine)"
            )
        }
        self.assertEqual(version, 13)
        self.assertNotIn("record_json", columns)

    def test_v13_tampered_quarantine_record_fails_closed(self) -> None:
        record_json = json.dumps(
            {"detour_id": "legacy-record", "status": "completed"},
            sort_keys=True,
            separators=(",", ":"),
        )
        connection = sqlite3.connect(self.path)
        connection.execute(
            """
            INSERT INTO anychain_terminal_detour_quarantine (
                detour_id, logical_thread_id, command_name,
                quarantine_reason, record_json, record_hash, quarantined_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-record",
                "logical-session",
                "help",
                "legacy",
                record_json,
                _hash("wrong-record"),
                "2026-07-27T00:00:00+00:00",
            ),
        )
        connection.execute(
            """
            UPDATE anychain_turn_transaction_meta
            SET schema_version = 13
            WHERE singleton = 1
            """
        )
        connection.commit()
        connection.close()

        with self.assertRaisesRegex(
            TurnTransactionSchemaError,
            "canonical record hash is invalid",
        ):
            TurnTransactionStore(self.path)

    def test_v13_correctly_hashed_partial_quarantine_record_fails_closed(
        self,
    ) -> None:
        record_json = json.dumps(
            {"detour_id": "partial-record", "status": "completed"},
            sort_keys=True,
            separators=(",", ":"),
        )
        connection = sqlite3.connect(self.path)
        connection.execute(
            """
            INSERT INTO anychain_terminal_detour_quarantine (
                detour_id, logical_thread_id, command_name,
                quarantine_reason, record_json, record_hash, quarantined_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "partial-record",
                "logical-session",
                "help",
                "legacy_runtime_fence_unbound",
                record_json,
                state_fingerprint(record_json.encode("utf-8")),
                "2026-07-27T00:00:00+00:00",
            ),
        )
        connection.execute(
            "UPDATE anychain_turn_transaction_meta "
            "SET schema_version = 13 WHERE singleton = 1"
        )
        connection.commit()
        connection.close()

        with self.assertRaisesRegex(
            TurnTransactionSchemaError,
            "incomplete canonical record",
        ):
            TurnTransactionStore(self.path)

    def test_v13_quarantine_record_identity_mismatch_fails_closed(
        self,
    ) -> None:
        prepared = self.store.prepare_terminal_detour(
            logical_thread_id="logical-session",
            process_instance_id="legacy-process",
            session_id="legacy-session",
            session_purpose="chaos",
            command_name="help",
            input_hash=_hash("help"),
            effect_class="read_only",
            shell_state_before_hash=_hash("shell-before"),
            origin_revision_commit="test-revision",
            origin_revision_worktree_hash=_hash("revision"),
        )
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM anychain_terminal_detours WHERE detour_id = ?",
            (prepared.detour_id,),
        ).fetchone()
        record = dict(row)
        record["identity_trust"] = "legacy_unbound"
        record["logical_thread_id"] = "different-session"
        record_json = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
        )
        connection.execute(
            "DELETE FROM anychain_terminal_detours WHERE detour_id = ?",
            (prepared.detour_id,),
        )
        connection.execute(
            """
            INSERT INTO anychain_terminal_detour_quarantine (
                detour_id, logical_thread_id, command_name,
                quarantine_reason, record_json, record_hash, quarantined_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prepared.detour_id,
                "logical-session",
                "help",
                "legacy_runtime_fence_unbound",
                record_json,
                state_fingerprint(record_json.encode("utf-8")),
                "2026-07-27T00:00:00+00:00",
            ),
        )
        connection.execute(
            "UPDATE anychain_turn_transaction_meta "
            "SET schema_version = 13 WHERE singleton = 1"
        )
        connection.commit()
        connection.close()

        with self.assertRaisesRegex(
            TurnTransactionSchemaError,
            "identity does not match",
        ):
            TurnTransactionStore(self.path)

    def test_v13_quarantine_stream_hash_mismatch_fails_closed(self) -> None:
        def corrupt(record: dict[str, object]) -> None:
            record["stream_chunk_count"] = 1
            record["stream_messages_json"] = '["chunk"]'
            record["stream_hash"] = _hash("wrong-stream")

        self._assert_quarantined_detour_semantics_fail(
            corrupt,
            "stream hash is invalid",
        )

    def test_v13_quarantine_termination_combination_fails_closed(self) -> None:
        def corrupt(record: dict[str, object]) -> None:
            record["result_kind"] = "session_termination"
            record["termination_reason"] = None
            record["exit_code"] = None

        self._assert_quarantined_detour_semantics_fail(
            corrupt,
            "termination fields are invalid",
        )

    def test_v13_quarantine_runtime_fence_combination_fails_closed(self) -> None:
        def corrupt(record: dict[str, object]) -> None:
            record["runtime_event_fence_sequence"] = 1
            record["runtime_event_fence_terminal_event_id"] = ""
            record["runtime_event_fence_id"] = ""
            record["runtime_event_fence_hash"] = ""

        self._assert_quarantined_detour_semantics_fail(
            corrupt,
            "runtime fence identity is invalid",
        )

    def test_v13_quarantine_response_hash_mismatch_fails_closed(self) -> None:
        def corrupt(record: dict[str, object]) -> None:
            record["response_hash"] = _hash("wrong-response")

        self._assert_quarantined_detour_semantics_fail(
            corrupt,
            "completed terminal detour evidence is invalid",
        )

    def test_v13_quarantine_reconciliation_provenance_fails_closed(self) -> None:
        def corrupt(record: dict[str, object]) -> None:
            record["resolution"] = "effect_confirmed"
            record["evidence_hash"] = _hash("operator-evidence")

        self._assert_quarantined_detour_semantics_fail(
            corrupt,
            "reconciliation provenance is invalid",
        )

    def test_current_detour_response_hash_mismatch_fails_closed(self) -> None:
        prepared = self.store.prepare_terminal_detour(
            logical_thread_id="logical-session",
            process_instance_id="process-1",
            session_id="session-1",
            session_purpose="chaos",
            command_name="help",
            input_hash=_hash("help"),
            effect_class="read_only",
            shell_state_before_hash=_hash("shell-before"),
            origin_revision_commit="test-revision",
            origin_revision_worktree_hash=_hash("revision"),
        )
        self.store.complete_terminal_detour(
            detour_id=prepared.detour_id,
            shell_state_after_hash=_hash("shell-before"),
            response_messages=("help",),
        )
        connection = sqlite3.connect(self.path)
        connection.execute(
            "UPDATE anychain_terminal_detours SET response_hash = ? "
            "WHERE detour_id = ?",
            (_hash("wrong-response"), prepared.detour_id),
        )
        connection.commit()
        connection.close()

        with self.assertRaisesRegex(
            TurnTransactionSchemaError,
            "completed terminal detour evidence is invalid",
        ):
            self.store.list_terminal_detours("logical-session")

    def test_current_detour_reconciliation_provenance_fails_closed(self) -> None:
        prepared = self.store.prepare_terminal_detour(
            logical_thread_id="logical-session",
            process_instance_id="process-1",
            session_id="session-1",
            session_purpose="chaos",
            command_name="help",
            input_hash=_hash("help"),
            effect_class="read_only",
            shell_state_before_hash=_hash("shell-before"),
            origin_revision_commit="test-revision",
            origin_revision_worktree_hash=_hash("revision"),
        )
        self.store.complete_terminal_detour(
            detour_id=prepared.detour_id,
            shell_state_after_hash=_hash("shell-before"),
            response_messages=("help",),
        )
        connection = sqlite3.connect(self.path)
        connection.execute(
            """
            UPDATE anychain_terminal_detours
            SET resolution = 'effect_confirmed', evidence_hash = ?
            WHERE detour_id = ?
            """,
            (_hash("operator-evidence"), prepared.detour_id),
        )
        connection.commit()
        connection.close()

        with self.assertRaisesRegex(
            TurnTransactionSchemaError,
            "reconciliation provenance is invalid",
        ):
            self.store.list_terminal_detours("logical-session")

    def test_runtime_event_requeue_is_scoped_to_one_product_authority(
        self,
    ) -> None:
        first_attempt = self._begin()
        first = self.store.commit_attempt(
            logical_thread_id="logical-session",
            transaction_id=first_attempt.transaction_id,
            physical_thread_id=first_attempt.physical_thread_id,
            attempt_checkpoint_id="checkpoint-a1",
            attempt_fingerprint=_hash("state-a1"),
        )
        self.store.mark_runtime_event_published(
            logical_thread_id="logical-session",
            event_id=first.event_id,
            runtime_event_id=first.runtime_event_id,
            payload_hash=_hash("payload-a"),
        )

        self.store.bootstrap_product_head(
            logical_thread_id="other-session",
            checkpoint_thread_id="other-session",
            checkpoint_id="checkpoint-b0",
            state_fingerprint=_hash("state-b0"),
        )
        second_attempt = self.store.begin_attempt(
            logical_thread_id="other-session"
        )
        second = self.store.commit_attempt(
            logical_thread_id="other-session",
            transaction_id=second_attempt.transaction_id,
            physical_thread_id=second_attempt.physical_thread_id,
            attempt_checkpoint_id="checkpoint-b1",
            attempt_fingerprint=_hash("state-b1"),
        )
        self.store.mark_runtime_event_published(
            logical_thread_id="other-session",
            event_id=second.event_id,
            runtime_event_id=second.runtime_event_id,
            payload_hash=_hash("payload-b"),
        )

        changed = (
            self.store.requeue_runtime_events_after_legacy_log_quarantine(
                "logical-session"
            )
        )

        self.assertEqual(changed, 1)
        self.assertEqual(
            self.store.get_terminal_outcome(
                first_attempt.transaction_id
            ).runtime_event_status,
            "pending",
        )
        unaffected = self.store.get_terminal_outcome(
            second_attempt.transaction_id
        )
        self.assertEqual(unaffected.runtime_event_status, "published")
        self.assertEqual(
            unaffected.runtime_event_payload_hash,
            _hash("payload-b"),
        )

    def test_real_v10_history_fails_closed_instead_of_guessing_revisions(
        self,
    ) -> None:
        attempt = self._begin()
        outcome = self.store.commit_attempt(
            logical_thread_id=attempt.logical_thread_id,
            transaction_id=attempt.transaction_id,
            physical_thread_id=attempt.physical_thread_id,
            attempt_checkpoint_id="checkpoint-1",
            attempt_fingerprint=_hash("state-1"),
        )
        connection = sqlite3.connect(self.path)
        connection.execute("DROP INDEX anychain_runtime_event_id")
        connection.execute(
            "DROP INDEX anychain_runtime_event_sequence_per_thread"
        )
        for column in (
            "runtime_event_id",
            "runtime_event_sequence",
            "runtime_event_status",
            "runtime_event_payload_hash",
            "runtime_event_published_at",
            "base_revision",
            "product_revision",
        ):
            connection.execute(
                f"ALTER TABLE anychain_terminal_outbox DROP COLUMN {column}"
            )
        connection.execute(
            "ALTER TABLE anychain_turn_attempts DROP COLUMN base_revision"
        )
        for column in (
            "runtime_event_fence_sequence",
            "runtime_event_fence_terminal_event_id",
            "runtime_event_fence_id",
            "runtime_event_fence_hash",
        ):
            connection.execute(
                f"ALTER TABLE anychain_terminal_detours DROP COLUMN {column}"
            )
        connection.execute(
            """
            UPDATE anychain_turn_transaction_meta
            SET schema_version = 10
            WHERE singleton = 1
            """
        )
        connection.commit()
        connection.close()

        with self.assertRaisesRegex(
            TurnTransactionSchemaError,
            "no authoritative revision ordering",
        ):
            TurnTransactionStore(self.path)

    def test_streaming_detour_records_chunks_before_completion(self) -> None:
        detour = self.store.prepare_terminal_detour(
            logical_thread_id="logical-session",
            process_instance_id="process-1",
            session_id="session-1",
            session_purpose="chaos",
            command_name="follow",
            input_hash=_hash("follow job-1"),
            effect_class="streaming_observation",
            shell_state_before_hash=_hash("shell-before"),
            origin_revision_commit="test-revision",
            origin_revision_worktree_hash=_hash("revision"),
        )
        first = self.store.record_terminal_detour_stream_chunk(
            detour_id=detour.detour_id,
            chunk_message="chunk-1",
        )
        second = self.store.record_terminal_detour_stream_chunk(
            detour_id=detour.detour_id,
            chunk_message="chunk-2",
        )
        completed = self.store.complete_terminal_detour(
            detour_id=detour.detour_id,
            shell_state_after_hash=_hash("shell-before"),
            response_messages=("chunk-1", "chunk-2"),
            effect_status="succeeded",
            stream_stop_reason="completed",
        )

        self.assertEqual(first.stream_chunk_count, 1)
        self.assertEqual(second.stream_chunk_count, 2)
        self.assertNotEqual(first.stream_hash, second.stream_hash)
        self.assertEqual(completed.stream_chunk_count, 2)
        self.assertEqual(completed.stream_hash, second.stream_hash)

    def test_interrupted_stream_is_replayable_failure_not_effect_uncertainty(
        self,
    ) -> None:
        detour = self.store.prepare_terminal_detour(
            logical_thread_id="logical-session",
            process_instance_id="process-1",
            session_id="session-1",
            session_purpose="chaos",
            command_name="follow",
            input_hash=_hash("follow job-1"),
            effect_class="streaming_observation",
            shell_state_before_hash=_hash("shell-before"),
            origin_revision_commit="test-revision",
            origin_revision_worktree_hash=_hash("revision"),
        )
        self.store.record_terminal_detour_stream_chunk(
            detour_id=detour.detour_id,
            chunk_message="chunk-1",
        )
        recovered = self.store.recover_prepared_terminal_detours(
            logical_thread_id="logical-session",
            interrupted_message="stream interrupted",
        )

        self.assertEqual(recovered[0].status, "completed")
        self.assertEqual(recovered[0].effect_status, "failed")
        self.assertEqual(recovered[0].stream_chunk_count, 1)
        self.assertEqual(
            recovered[0].response_messages,
            ("chunk-1", "stream interrupted"),
        )
        self.assertEqual(recovered[0].stream_stop_reason, "interrupted")

    def test_reconciliation_resolution_and_detour_commit_atomically(self) -> None:
        attempt = self._begin()
        outcome = self.store.require_reconciliation(
            logical_thread_id="logical-session",
            transaction_id=attempt.transaction_id,
            physical_thread_id=attempt.physical_thread_id,
            diagnostic_hash=_hash("uncertain"),
        )
        before = self.store.get_product_head("logical-session")

        detour = self.store.resolve_with_terminal_detour(
            logical_thread_id="logical-session",
            process_instance_id="process-1",
            session_id="session-1",
            session_purpose="chaos",
            target_id=outcome.transaction_id,
            input_hash=_hash("reconcile"),
            resolution="effect_not_observed",
            evidence_hash=_hash("operator-evidence"),
            shell_state_hash=_hash("shell"),
            response_messages=("reconciliation recorded",),
            origin_revision_commit="test-revision",
            origin_revision_worktree_hash=_hash("revision"),
        )

        resolution = self.store.get_reconciliation_resolution(
            outcome.transaction_id
        )
        after = self.store.get_product_head("logical-session")
        self.assertIsNotNone(resolution)
        self.assertEqual(detour.effect_class, "authority_resolution")
        self.assertEqual(detour.status, "completed")
        self.assertEqual(before, after)

    def test_external_detour_reconciliation_provenance_is_preserved(
        self,
    ) -> None:
        for index, resolution in enumerate(
            ("effect_confirmed", "effect_not_observed"),
            start=1,
        ):
            target = self.store.prepare_terminal_detour(
                logical_thread_id="logical-session",
                process_instance_id=f"process-{index}",
                session_id=f"session-{index}",
                session_purpose="chaos",
                command_name="install",
                input_hash=_hash(f"install-{index}"),
                effect_class="local_external_effect",
                shell_state_before_hash=_hash("shell-before"),
                origin_revision_commit="test-revision",
                origin_revision_worktree_hash=_hash("revision"),
            )
            self.store.require_terminal_detour_reconciliation(
                detour_id=target.detour_id,
                diagnostic_hash=_hash(f"uncertain-{index}"),
            )
            authority = self.store.resolve_with_terminal_detour(
                logical_thread_id="logical-session",
                process_instance_id=f"resolver-{index}",
                session_id=f"resolver-session-{index}",
                session_purpose="chaos",
                target_id=target.detour_id,
                input_hash=_hash(f"resolve-{index}"),
                resolution=resolution,
                evidence_hash=_hash(f"evidence-{index}"),
                shell_state_hash=_hash("shell-before"),
                response_messages=(f"resolved {index}",),
                origin_revision_commit="test-revision",
                origin_revision_worktree_hash=_hash("revision"),
            )

            resolved = next(
                item
                for item in self.store.list_terminal_detours(
                    "logical-session"
                )
                if item.detour_id == target.detour_id
            )
            self.assertEqual(resolved.effect_class, "local_external_effect")
            self.assertEqual(resolved.resolution, resolution)
            self.assertEqual(
                resolved.effect_status,
                (
                    "succeeded"
                    if resolution == "effect_confirmed"
                    else "failed"
                ),
            )
            self.assertEqual(resolved.response_messages, ())
            self.assertEqual(authority.effect_class, "authority_resolution")
            self.assertEqual(authority.resolution, resolution)
            self.assertEqual(authority.effect_status, "not_applicable")

    def test_reconciliation_detour_failure_rolls_back_resolution(self) -> None:
        attempt = self._begin()
        outcome = self.store.require_reconciliation(
            logical_thread_id="logical-session",
            transaction_id=attempt.transaction_id,
            physical_thread_id=attempt.physical_thread_id,
            diagnostic_hash=_hash("uncertain"),
        )
        with patch.object(
            self.store,
            "_require_detour",
            side_effect=RuntimeError("injected detour outbox failure"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "injected detour outbox failure",
            ):
                self.store.resolve_with_terminal_detour(
                    logical_thread_id="logical-session",
                    process_instance_id="process-1",
                    session_id="session-1",
                    session_purpose="chaos",
                    target_id=outcome.transaction_id,
                    input_hash=_hash("reconcile"),
                    resolution="effect_confirmed",
                    evidence_hash=_hash("operator-evidence"),
                    shell_state_hash=_hash("shell"),
                    response_messages=("reconciliation recorded",),
                    origin_revision_commit="test-revision",
                    origin_revision_worktree_hash=_hash("revision"),
                )

        self.assertIsNone(
            self.store.get_reconciliation_resolution(outcome.transaction_id)
        )
        self.assertEqual(
            [
                item
                for item in self.store.list_terminal_detours(
                    "logical-session"
                )
                if item.command_name == "reconcile"
            ],
            [],
        )

    def test_commit_atomically_moves_head_and_writes_committed_outcome(self) -> None:
        attempt = self._begin()
        outcome = self.store.commit_attempt(
            logical_thread_id=attempt.logical_thread_id,
            transaction_id=attempt.transaction_id,
            physical_thread_id=attempt.physical_thread_id,
            attempt_checkpoint_id="checkpoint-1",
            attempt_fingerprint=_hash("state-1"),
        )

        head = self.store.get_product_head("logical-session")
        self.assertIsNotNone(head)
        assert head is not None
        self.assertEqual(head.checkpoint_thread_id, attempt.physical_thread_id)
        self.assertEqual(head.checkpoint_id, "checkpoint-1")
        self.assertEqual(head.state_fingerprint, _hash("state-1"))
        self.assertEqual(head.revision, 1)
        self.assertEqual(outcome.outcome, "committed")
        self.assertEqual(outcome.product_checkpoint_id, head.checkpoint_id)
        self.assertEqual(self.store.get_attempt(attempt.transaction_id).status, "committed")

    def test_committed_outcome_recovery_rejects_corrupt_checkpoint_lineage(
        self,
    ) -> None:
        attempt = self._begin()
        self.store.commit_attempt(
            logical_thread_id=attempt.logical_thread_id,
            transaction_id=attempt.transaction_id,
            physical_thread_id=attempt.physical_thread_id,
            attempt_checkpoint_id="checkpoint-1",
            attempt_fingerprint=_hash("state-1"),
        )
        connection = sqlite3.connect(self.path)
        connection.execute(
            """
            UPDATE anychain_terminal_outbox
            SET product_checkpoint_id = ?
            WHERE transaction_id = ?
            """,
            ("different-checkpoint", attempt.transaction_id),
        )
        connection.commit()
        connection.close()

        with self.assertRaisesRegex(
            TurnTransactionSchemaError,
            "committed terminal outcome lineage is invalid",
        ):
            self.store.get_terminal_outcome(attempt.transaction_id)

    def test_authority_reads_reject_matching_but_invalid_hashes(self) -> None:
        attempt = self._begin()
        self.store.commit_attempt(
            logical_thread_id=attempt.logical_thread_id,
            transaction_id=attempt.transaction_id,
            physical_thread_id=attempt.physical_thread_id,
            attempt_checkpoint_id="checkpoint-1",
            attempt_fingerprint=_hash("state-1"),
        )
        connection = sqlite3.connect(self.path)
        connection.execute(
            """
            UPDATE anychain_terminal_outbox
            SET attempt_fingerprint = 'invalid',
                product_fingerprint = 'invalid'
            WHERE transaction_id = ?
            """,
            (attempt.transaction_id,),
        )
        connection.execute(
            """
            UPDATE anychain_product_heads
            SET state_fingerprint = 'invalid'
            WHERE logical_thread_id = ?
            """,
            (attempt.logical_thread_id,),
        )
        connection.commit()
        connection.close()

        with self.assertRaisesRegex(
            TurnTransactionSchemaError,
            "stored state_fingerprint",
        ):
            self.store.get_product_head(attempt.logical_thread_id)
        with self.assertRaisesRegex(
            TurnTransactionSchemaError,
            "stored product_fingerprint",
        ):
            self.store.get_terminal_outcome(attempt.transaction_id)

    def test_terminal_outbox_failure_rolls_back_head_and_attempt(self) -> None:
        attempt = self._begin()
        connection = sqlite3.connect(self.path)
        connection.execute(
            """
            CREATE TRIGGER reject_terminal_outbox
            BEFORE INSERT ON anychain_terminal_outbox
            BEGIN
                SELECT RAISE(ABORT, 'injected outbox failure');
            END
            """
        )
        connection.commit()
        connection.close()

        with self.assertRaises(sqlite3.IntegrityError):
            self.store.commit_attempt(
                logical_thread_id=attempt.logical_thread_id,
                transaction_id=attempt.transaction_id,
                physical_thread_id=attempt.physical_thread_id,
                attempt_checkpoint_id="checkpoint-1",
                attempt_fingerprint=_hash("state-1"),
            )

        self.assertEqual(self.store.get_product_head("logical-session"), self.base)
        self.assertEqual(self.store.get_attempt(attempt.transaction_id).status, "active")
        self.assertIsNone(self.store.get_terminal_outcome(attempt.transaction_id))

    def test_abort_is_atomic_idempotent_and_does_not_move_head(self) -> None:
        attempt = self._begin()
        diagnostic = _hash("provider-timeout")
        first = self.store.abort_attempt(
            logical_thread_id=attempt.logical_thread_id,
            transaction_id=attempt.transaction_id,
            physical_thread_id=attempt.physical_thread_id,
            diagnostic_hash=diagnostic,
        )
        second = self.store.abort_attempt(
            logical_thread_id=attempt.logical_thread_id,
            transaction_id=attempt.transaction_id,
            physical_thread_id=attempt.physical_thread_id,
            diagnostic_hash=diagnostic,
        )

        self.assertEqual(first, second)
        self.assertEqual(first.outcome, "aborted")
        self.assertEqual(self.store.get_product_head("logical-session"), self.base)
        self.assertEqual(len(self.store.list_terminal_outcomes("logical-session")), 1)
        with self.assertRaises(TurnTransactionConflictError):
            self.store.abort_attempt(
                logical_thread_id=attempt.logical_thread_id,
                transaction_id=attempt.transaction_id,
                physical_thread_id=attempt.physical_thread_id,
                diagnostic_hash=_hash("different-failure"),
            )

    def test_aborted_outcome_recovery_rejects_attempt_identity(self) -> None:
        attempt = self._begin()
        self.store.abort_attempt(
            logical_thread_id=attempt.logical_thread_id,
            transaction_id=attempt.transaction_id,
            physical_thread_id=attempt.physical_thread_id,
            diagnostic_hash=_hash("provider-timeout"),
        )
        connection = sqlite3.connect(self.path)
        connection.execute(
            """
            UPDATE anychain_terminal_outbox
            SET attempt_checkpoint_id = ?, attempt_fingerprint = ?
            WHERE transaction_id = ?
            """,
            ("unexpected-checkpoint", _hash("unexpected"), attempt.transaction_id),
        )
        connection.commit()
        connection.close()

        with self.assertRaisesRegex(
            TurnTransactionSchemaError,
            "aborted terminal outcome carries attempt identity",
        ):
            self.store.get_terminal_outcome(attempt.transaction_id)

    def test_reconciliation_required_keeps_head_and_preserves_attempt_identity(self) -> None:
        attempt = self._begin()
        outcome = self.store.require_reconciliation(
            logical_thread_id=attempt.logical_thread_id,
            transaction_id=attempt.transaction_id,
            physical_thread_id=attempt.physical_thread_id,
            attempt_checkpoint_id="effect-checkpoint",
            attempt_fingerprint=_hash("effect-state"),
            diagnostic_hash=_hash("effect-status-unknown"),
        )

        self.assertEqual(outcome.outcome, "reconciliation_required")
        self.assertEqual(outcome.attempt_checkpoint_id, "effect-checkpoint")
        self.assertEqual(outcome.product_checkpoint_id, self.base.checkpoint_id)
        self.assertEqual(self.store.get_product_head("logical-session"), self.base)
        self.assertEqual(
            self.store.get_attempt(attempt.transaction_id).status,
            "reconciliation_required",
        )

    def test_reconciliation_recovery_rejects_partial_attempt_identity(
        self,
    ) -> None:
        attempt = self._begin()
        self.store.require_reconciliation(
            logical_thread_id=attempt.logical_thread_id,
            transaction_id=attempt.transaction_id,
            physical_thread_id=attempt.physical_thread_id,
            attempt_checkpoint_id="effect-checkpoint",
            attempt_fingerprint=_hash("effect-state"),
            diagnostic_hash=_hash("effect-status-unknown"),
        )
        connection = sqlite3.connect(self.path)
        connection.execute(
            """
            UPDATE anychain_terminal_outbox
            SET attempt_fingerprint = NULL
            WHERE transaction_id = ?
            """,
            (attempt.transaction_id,),
        )
        connection.commit()
        connection.close()

        with self.assertRaisesRegex(
            TurnTransactionSchemaError,
            "attempt identity is incomplete",
        ):
            self.store.get_terminal_outcome(attempt.transaction_id)

    def test_unresolved_reconciliation_blocks_new_attempt_until_audited_resolution(
        self,
    ) -> None:
        attempt = self._begin()
        self.store.require_reconciliation(
            logical_thread_id=attempt.logical_thread_id,
            transaction_id=attempt.transaction_id,
            physical_thread_id=attempt.physical_thread_id,
            diagnostic_hash=_hash("uncertain"),
        )

        with self.assertRaises(ReconciliationRequiredError) as raised:
            self._begin()
        self.assertEqual(raised.exception.transaction_id, attempt.transaction_id)

        resolution = self.store.resolve_reconciliation(
            logical_thread_id=attempt.logical_thread_id,
            transaction_id=attempt.transaction_id,
            resolution="effect_not_observed",
            evidence_hash=_hash("operator-evidence"),
        )
        repeated = self.store.resolve_reconciliation(
            logical_thread_id=attempt.logical_thread_id,
            transaction_id=attempt.transaction_id,
            resolution="effect_not_observed",
            evidence_hash=_hash("operator-evidence"),
        )

        self.assertEqual(resolution, repeated)
        self.assertEqual(
            self.store.get_reconciliation_resolution(attempt.transaction_id),
            resolution,
        )
        self.assertIsNone(
            self.store.get_unresolved_reconciliation("logical-session")
        )
        self.assertEqual(self._begin().status, "active")

    def test_reconciliation_resolution_is_immutable(self) -> None:
        attempt = self._begin()
        self.store.require_reconciliation(
            logical_thread_id=attempt.logical_thread_id,
            transaction_id=attempt.transaction_id,
            physical_thread_id=attempt.physical_thread_id,
            diagnostic_hash=_hash("uncertain"),
        )
        self.store.resolve_reconciliation(
            logical_thread_id=attempt.logical_thread_id,
            transaction_id=attempt.transaction_id,
            resolution="effect_confirmed",
            evidence_hash=_hash("confirmed"),
        )

        for resolution, evidence_hash in (
            ("effect_not_observed", _hash("confirmed")),
            ("effect_confirmed", _hash("different")),
        ):
            with self.subTest(resolution=resolution, evidence_hash=evidence_hash):
                with self.assertRaises(TurnTransactionConflictError):
                    self.store.resolve_reconciliation(
                        logical_thread_id=attempt.logical_thread_id,
                        transaction_id=attempt.transaction_id,
                        resolution=resolution,
                        evidence_hash=evidence_hash,
                    )

    def test_terminal_delivery_acknowledgement_is_idempotent(self) -> None:
        attempt = self._begin()
        outcome = self.store.commit_attempt(
            logical_thread_id=attempt.logical_thread_id,
            transaction_id=attempt.transaction_id,
            physical_thread_id=attempt.physical_thread_id,
            attempt_checkpoint_id="checkpoint-1",
            attempt_fingerprint=_hash("state-1"),
            render_hash=_hash("rendered-response"),
        )

        self.assertEqual(
            self.store.list_undelivered_outcomes("logical-session"),
            (outcome,),
        )
        first = self.store.mark_terminal_delivered(
            logical_thread_id="logical-session",
            event_id=outcome.event_id,
        )
        second = self.store.mark_terminal_delivered(
            logical_thread_id="logical-session",
            event_id=outcome.event_id,
        )

        self.assertIsNotNone(first.delivered_at)
        self.assertEqual(first, second)
        self.assertEqual(
            self.store.list_undelivered_outcomes("logical-session"),
            (),
        )

    def test_conflicting_terminal_outcome_is_rejected(self) -> None:
        attempt = self._begin()
        self.store.abort_attempt(
            logical_thread_id=attempt.logical_thread_id,
            transaction_id=attempt.transaction_id,
            physical_thread_id=attempt.physical_thread_id,
            diagnostic_hash=_hash("cancelled"),
        )

        with self.assertRaises(TurnTransactionConflictError):
            self.store.commit_attempt(
                logical_thread_id=attempt.logical_thread_id,
                transaction_id=attempt.transaction_id,
                physical_thread_id=attempt.physical_thread_id,
                attempt_checkpoint_id="checkpoint-1",
                attempt_fingerprint=_hash("state-1"),
            )

    def test_thread_transaction_and_hash_tampering_fail_closed(self) -> None:
        attempt = self._begin()
        calls = (
            {
                "logical_thread_id": "other-session",
                "transaction_id": attempt.transaction_id,
                "physical_thread_id": attempt.physical_thread_id,
                "diagnostic_hash": _hash("failure"),
            },
            {
                "logical_thread_id": attempt.logical_thread_id,
                "transaction_id": attempt.transaction_id,
                "physical_thread_id": "attempt:00000000-0000-0000-0000-000000000000",
                "diagnostic_hash": _hash("failure"),
            },
        )
        for arguments in calls:
            with self.subTest(arguments=arguments):
                with self.assertRaises(TurnTransactionConflictError):
                    self.store.abort_attempt(**arguments)

        with self.assertRaises(TurnTransactionValidationError):
            self.store.abort_attempt(
                logical_thread_id=attempt.logical_thread_id,
                transaction_id=attempt.transaction_id,
                physical_thread_id=attempt.physical_thread_id,
                diagnostic_hash="not-a-sha256",
            )
        with self.assertRaises(TurnTransactionValidationError):
            self.store.get_attempt("not-a-uuid")

    def test_stale_base_rejects_commit_without_partial_writes(self) -> None:
        attempt = self._begin()
        connection = sqlite3.connect(self.path)
        connection.execute(
            """
            UPDATE anychain_product_heads
            SET checkpoint_id = ?, state_fingerprint = ?
            WHERE logical_thread_id = ?
            """,
            ("external-checkpoint", _hash("external"), "logical-session"),
        )
        connection.commit()
        connection.close()

        with self.assertRaises(TurnTransactionConflictError):
            self.store.commit_attempt(
                logical_thread_id=attempt.logical_thread_id,
                transaction_id=attempt.transaction_id,
                physical_thread_id=attempt.physical_thread_id,
                attempt_checkpoint_id="checkpoint-1",
                attempt_fingerprint=_hash("state-1"),
            )

        self.assertEqual(self.store.get_attempt(attempt.transaction_id).status, "active")
        self.assertIsNone(self.store.get_terminal_outcome(attempt.transaction_id))

    def test_read_apis_do_not_mutate_ledger(self) -> None:
        attempt = self._begin()
        connection = sqlite3.connect(self.path)
        before = connection.total_changes
        connection.close()

        self.assertEqual(self.store.get_product_head("logical-session"), self.base)
        self.assertEqual(self.store.get_attempt(attempt.transaction_id), attempt)
        self.assertIsNone(self.store.get_terminal_outcome(attempt.transaction_id))
        self.assertEqual(self.store.list_attempts("logical-session"), (attempt,))
        self.assertEqual(self.store.list_terminal_outcomes("logical-session"), ())

        reader = sqlite3.connect(self.path)
        counts = {
            table: reader.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "anychain_product_heads",
                "anychain_turn_attempts",
                "anychain_terminal_outbox",
            )
        }
        reader.close()
        self.assertEqual(before, 0)
        self.assertEqual(
            counts,
            {
                "anychain_product_heads": 1,
                "anychain_turn_attempts": 1,
                "anychain_terminal_outbox": 0,
            },
        )

    def test_ledger_schema_has_no_free_text_or_secret_payload_columns(self) -> None:
        connection = sqlite3.connect(self.path)
        columns = {
            row[1]
            for table in (
                "anychain_product_heads",
                "anychain_turn_attempts",
                "anychain_terminal_outbox",
            )
            for row in connection.execute(f"PRAGMA table_info({table})")
        }
        connection.close()

        forbidden = {
            "user_text",
            "prompt",
            "response",
            "payload",
            "content",
            "secret",
            "api_key",
            "endpoint",
        }
        self.assertTrue(columns.isdisjoint(forbidden))


if __name__ == "__main__":
    unittest.main()

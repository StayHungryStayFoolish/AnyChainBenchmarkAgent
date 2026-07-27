"""Product contracts for typed terminal delivery and runtime correlation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


class TerminalProjectionContractTest(unittest.TestCase):
    @staticmethod
    def _reseal(payload: dict[str, object]) -> dict[str, object]:
        body = {key: value for key, value in payload.items() if key != "record_hash"}
        payload["record_hash"] = hashlib.sha256(
            json.dumps(
                body,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return payload

    def test_runtime_projection_path_has_one_durable_product_authority(
        self,
    ) -> None:
        from agent.harness.terminal_protocol import claim_jsonl_authority

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "runtime.jsonl"
            first = claim_jsonl_authority(
                path,
                product_authority_id="chaos:session-a",
            )
            second = claim_jsonl_authority(
                path,
                product_authority_id="chaos:session-a",
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "another or invalid authority",
            ):
                claim_jsonl_authority(
                    path,
                    product_authority_id="chaos:session-b",
                )

        self.assertEqual(first, second)

    def test_existing_runtime_projection_cannot_be_claimed_cross_authority(
        self,
    ) -> None:
        from agent.harness.terminal_protocol import claim_jsonl_authority

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "runtime.jsonl"
            path.write_text(
                json.dumps({
                    "schema_version": 5,
                    "thread_id": "session-a",
                    "session_purpose": "chaos",
                })
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "another Product Head authority",
            ):
                claim_jsonl_authority(
                    path,
                    product_authority_id="chaos:session-b",
                )

    @staticmethod
    def _make_outcome():
        from agent.harness.turn_transactions import TerminalOutcome

        return TerminalOutcome(
            event_id="terminal-event",
            transaction_id="00000000-0000-0000-0000-000000000001",
            logical_thread_id="terminal-protocol:user",
            physical_thread_id=(
                "attempt:00000000-0000-0000-0000-000000000001"
            ),
            outcome="committed",
            base_revision=0,
            base_checkpoint_thread_id="head",
            base_checkpoint_id="checkpoint-0",
            base_fingerprint="0" * 64,
            origin_revision_commit="test-revision",
            origin_revision_worktree_hash="1" * 64,
            attempt_checkpoint_id="checkpoint-1",
            attempt_fingerprint="2" * 64,
            product_revision=1,
            product_checkpoint_thread_id=(
                "attempt:00000000-0000-0000-0000-000000000001"
            ),
            product_checkpoint_id="checkpoint-1",
            product_fingerprint="2" * 64,
            diagnostic_hash=None,
            render_hash="3" * 64,
            failure_category="",
            runtime_event_id="runtime-event-1",
            runtime_event_sequence=1,
            runtime_event_status="published",
            runtime_event_payload_hash="5" * 64,
            runtime_event_published_at="2026-07-27T00:00:00+00:00",
            created_at="2026-07-27T00:00:00+00:00",
            delivered_at=None,
        )

    @staticmethod
    def _make_detour():
        from agent.harness.turn_transactions import TerminalDetour

        return TerminalDetour(
            detour_id="00000000-0000-0000-0000-000000000002",
            logical_thread_id="terminal-protocol:user",
            process_instance_id="process-1",
            session_id="terminal-protocol",
            session_purpose="user",
            command_name="help",
            input_hash="8" * 64,
            result_kind="command",
            interruption_kind=None,
            termination_reason=None,
            exit_code=None,
            effect_class="read_only",
            status="completed",
            shell_state_before_hash="4" * 64,
            shell_state_after_hash="4" * 64,
            product_revision_before=2,
            product_checkpoint_thread_id_before="head",
            product_checkpoint_id_before="checkpoint-2",
            product_fingerprint_before="5" * 64,
            product_revision_after=2,
            product_checkpoint_thread_id_after="head",
            product_checkpoint_id_after="checkpoint-2",
            product_fingerprint_after="5" * 64,
            runtime_event_fence_sequence=0,
            runtime_event_fence_terminal_event_id="",
            runtime_event_fence_id="",
            runtime_event_fence_hash="",
            response_messages=("secret-bearing help response",),
            response_hash="6" * 64,
            stream_chunk_count=0,
            stream_hash=hashlib.sha256(b"").hexdigest(),
            stream_messages=(),
            stream_stop_reason="not_streaming",
            identity_trust="trusted",
            effect_status="not_applicable",
            diagnostic_hash=None,
            resolution=None,
            evidence_hash=None,
            origin_revision_commit="test-revision",
            origin_revision_worktree_hash="1" * 64,
            created_at="2026-07-27T00:00:00+00:00",
            completed_at="2026-07-27T00:00:01+00:00",
            delivered_at=None,
        )

    def test_detour_projection_is_non_secret_and_keeps_head_unchanged(self) -> None:
        from agent.harness.terminal_protocol import (
            build_terminal_detour_projection,
            validate_terminal_detour_projection,
        )

        projection = build_terminal_detour_projection(
            self._make_detour(),
            rendered_frame="Agent> redacted help",
            delivery_phase="live",
        )
        validated = validate_terminal_detour_projection(projection.as_dict())
        encoded = json.dumps(validated.as_dict(), sort_keys=True)

        self.assertEqual(validated.command_name, "help")
        self.assertEqual(
            validated.product_fingerprint_before,
            validated.product_fingerprint_after,
        )
        self.assertNotIn("secret-bearing", encoded)
        self.assertNotIn("redacted help", encoded)

    def test_session_termination_projection_is_typed(self) -> None:
        from agent.harness.terminal_protocol import (
            build_terminal_detour_projection,
        )

        projection = build_terminal_detour_projection(
            replace(
                self._make_detour(),
                command_name="exit",
                result_kind="session_termination",
                termination_reason="exit",
                exit_code=0,
            ),
            rendered_frame="Agent> bye",
            delivery_phase="live",
        )

        self.assertEqual(projection.result_kind, "session_termination")
        self.assertEqual(projection.termination_reason, "exit")
        self.assertEqual(projection.exit_code, 0)

    def test_detour_projection_rejects_product_head_advance(self) -> None:
        from agent.harness.terminal_protocol import (
            TerminalProtocolError,
            build_terminal_detour_projection,
            validate_terminal_detour_projection,
        )

        projection = build_terminal_detour_projection(
            self._make_detour(),
            rendered_frame="Agent> help",
            delivery_phase="live",
        )
        payload = projection.as_dict()
        payload["product_revision_after"] = 3
        payload["record_hash"] = "0" * 64

        with self.assertRaisesRegex(
            TerminalProtocolError,
            "advanced the Product Head",
        ):
            validate_terminal_detour_projection(payload)

    def test_detour_projection_rejects_product_authority_mismatch(self) -> None:
        from agent.harness.terminal_protocol import (
            TerminalProtocolError,
            build_terminal_detour_projection,
            validate_terminal_detour_projection,
        )

        payload = build_terminal_detour_projection(
            self._make_detour(),
            rendered_frame="Agent> help",
            delivery_phase="live",
        ).as_dict()
        payload["product_authority_id"] = "different:user"

        with self.assertRaisesRegex(
            TerminalProtocolError,
            "authority is inconsistent",
        ):
            validate_terminal_detour_projection(payload)

    def test_detour_projection_rejects_unknown_effect_class(self) -> None:
        from agent.harness.terminal_protocol import (
            TerminalProtocolError,
            build_terminal_detour_projection,
            validate_terminal_detour_projection,
        )

        payload = build_terminal_detour_projection(
            self._make_detour(),
            rendered_frame="Agent> help",
            delivery_phase="live",
        ).as_dict()
        payload["effect_class"] = "unknown_effect"
        self._reseal(payload)

        with self.assertRaisesRegex(
            TerminalProtocolError,
            "effect class is invalid",
        ):
            validate_terminal_detour_projection(payload)

    def test_detour_projection_rejects_unprojectable_effect_status(self) -> None:
        from agent.harness.terminal_protocol import (
            TerminalProtocolError,
            build_terminal_detour_projection,
            validate_terminal_detour_projection,
        )

        payload = build_terminal_detour_projection(
            self._make_detour(),
            rendered_frame="Agent> help",
            delivery_phase="live",
        ).as_dict()
        payload["effect_status"] = "uncertain"
        self._reseal(payload)

        with self.assertRaisesRegex(
            TerminalProtocolError,
            "effect status is invalid",
        ):
            validate_terminal_detour_projection(payload)

    def test_projection_contains_only_typed_non_secret_evidence(self) -> None:
        from agent.harness.terminal_protocol import (
            build_terminal_outcome_projection,
            validate_terminal_outcome_projection,
        )

        projection = build_terminal_outcome_projection(
            self._make_outcome(),
            rendered_frame="Agent> complete",
            delivery_phase="live",
        )
        validated = validate_terminal_outcome_projection(projection.as_dict())
        encoded = json.dumps(validated.as_dict(), sort_keys=True)

        self.assertEqual(
            validated.transaction_id,
            self._make_outcome().transaction_id,
        )
        self.assertEqual(validated.origin_revision["commit"], "test-revision")
        self.assertNotIn("endpoint", encoded.lower())
        self.assertNotIn("Agent> complete", encoded)

    def test_projection_rejects_corrupt_product_authority(self) -> None:
        from agent.harness.terminal_protocol import (
            TerminalProtocolError,
            build_terminal_outcome_projection,
            validate_terminal_outcome_projection,
        )

        payload = build_terminal_outcome_projection(
            self._make_outcome(),
            rendered_frame="Agent> complete",
            delivery_phase="live",
        ).as_dict()
        payload["product_authority_id"] = "different:user"

        with self.assertRaisesRegex(
            TerminalProtocolError,
            "authority is inconsistent",
        ):
            validate_terminal_outcome_projection(payload)

    def test_projection_rejects_committed_checkpoint_lineage_mismatch(
        self,
    ) -> None:
        from agent.harness.terminal_protocol import (
            TerminalProtocolError,
            build_terminal_outcome_projection,
            validate_terminal_outcome_projection,
        )

        payload = build_terminal_outcome_projection(
            self._make_outcome(),
            rendered_frame="Agent> complete",
            delivery_phase="live",
        ).as_dict()
        payload["attempt_checkpoint_id"] = "different-checkpoint"

        with self.assertRaisesRegex(
            TerminalProtocolError,
            "checkpoint lineage is invalid",
        ):
            validate_terminal_outcome_projection(payload)

    def test_projection_rejects_noncommitting_product_identity_change(
        self,
    ) -> None:
        from agent.harness.terminal_protocol import (
            TerminalProtocolError,
            build_terminal_outcome_projection,
            validate_terminal_outcome_projection,
        )

        outcome = replace(
            self._make_outcome(),
            outcome="aborted",
            attempt_checkpoint_id=None,
            attempt_fingerprint=None,
            diagnostic_hash="4" * 64,
            render_hash="",
            failure_category="cancelled",
            product_revision=0,
            product_checkpoint_thread_id="head",
            product_checkpoint_id="checkpoint-0",
            product_fingerprint="0" * 64,
            runtime_event_id="",
            runtime_event_sequence=0,
            runtime_event_status="not_applicable",
            runtime_event_payload_hash=None,
            runtime_event_published_at=None,
        )
        payload = build_terminal_outcome_projection(
            outcome,
            rendered_frame="Agent> cancelled",
            delivery_phase="live",
        ).as_dict()
        payload["product_checkpoint_id"] = "different-checkpoint"

        with self.assertRaisesRegex(
            TerminalProtocolError,
            "changed Product Head identity",
        ):
            validate_terminal_outcome_projection(payload)

    def test_reconciliation_projection_preserves_optional_attempt_evidence(
        self,
    ) -> None:
        from agent.harness.terminal_protocol import (
            build_terminal_outcome_projection,
        )

        outcome = replace(
            self._make_outcome(),
            outcome="reconciliation_required",
            diagnostic_hash="4" * 64,
            render_hash="",
            failure_category="external_effect_uncertain",
            product_revision=0,
            product_checkpoint_thread_id="head",
            product_checkpoint_id="checkpoint-0",
            product_fingerprint="0" * 64,
            runtime_event_id="",
            runtime_event_sequence=0,
            runtime_event_status="not_applicable",
            runtime_event_payload_hash=None,
            runtime_event_published_at=None,
        )
        projection = build_terminal_outcome_projection(
            outcome,
            rendered_frame="Agent> reconciliation required",
            delivery_phase="live",
        )

        self.assertEqual(projection.outcome, "reconciliation_required")
        self.assertEqual(
            projection.attempt_fingerprint,
            outcome.attempt_fingerprint,
        )

    def test_projection_failure_does_not_acknowledge_outbox_delivery(self) -> None:
        from agent.terminal.repl import AnyChainTerminal, TerminalSession

        outcome = self._make_outcome()
        calls: list[str] = []

        class IO:
            def begin_frame(self) -> None:
                calls.append("begin")

            def agent(self, _language: str, message: str) -> str:
                calls.append("render")
                return f"Agent> {message}"

            def rendered_frame(self) -> str:
                return "Agent> complete"

        class Harness:
            def ensure_runtime_observation(self, _outcome):
                calls.append("observe")

            def terminal_messages(self, _outcome):
                return ("complete",)

            def mark_terminal_delivered(self, _outcome):
                calls.append("mark")
                return _outcome

        app = AnyChainTerminal(state=TerminalSession(language="en"), io=IO())
        app._harness = Harness()
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {
                "ANYCHAIN_AGENT_TERMINAL_OUTCOME_FILE": str(
                    Path(tmpdir) / "outcomes.jsonl"
                )
            },
        ), patch(
            "agent.terminal.repl.append_terminal_outcome_projection",
            side_effect=OSError("projection storage failed"),
        ):
            with self.assertRaisesRegex(OSError, "projection storage failed"):
                app._deliver_terminal_outcome(outcome)

        self.assertEqual(calls, ["observe", "render"])

    def test_runtime_observation_failure_prevents_terminal_presentation(self) -> None:
        from agent.terminal.repl import AnyChainTerminal, TerminalSession

        calls: list[str] = []

        class IO:
            def agent(self, _language: str, _message: str) -> str:
                calls.append("render")
                return "Agent> complete"

        class Harness:
            def ensure_runtime_observation(self, _outcome):
                calls.append("observe")
                raise OSError("runtime event unavailable")

            def mark_terminal_delivered(self, _outcome):
                calls.append("mark")

        app = AnyChainTerminal(state=TerminalSession(language="en"), io=IO())
        app._harness = Harness()
        with self.assertRaisesRegex(OSError, "runtime event unavailable"):
            app._deliver_terminal_outcome(self._make_outcome())

        self.assertEqual(calls, ["observe"])

    def test_projection_is_durable_before_delivery_ack(self) -> None:
        from agent.terminal.repl import AnyChainTerminal, TerminalSession

        outcome = self._make_outcome()
        calls: list[str] = []

        class IO:
            def agent(self, _language: str, message: str) -> str:
                calls.append("render")
                return f"Agent> {message}"

            def rendered_frame(self) -> str:
                return "Agent> complete"

        class Harness:
            def ensure_runtime_observation(self, _outcome):
                calls.append("observe")

            def terminal_messages(self, _outcome):
                return ("complete",)

            def mark_terminal_delivered(self, _outcome):
                calls.append("mark")
                return _outcome

        app = AnyChainTerminal(state=TerminalSession(language="en"), io=IO())
        app._harness = Harness()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "outcomes.jsonl"
            with patch.dict(
                os.environ,
                {"ANYCHAIN_AGENT_TERMINAL_OUTCOME_FILE": str(path)},
            ):
                app._deliver_terminal_outcome(outcome)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(calls, ["observe", "render", "mark"])
        self.assertEqual(payload["event_id"], outcome.event_id)
        self.assertEqual(payload["delivery_phase"], "live")

    def test_durable_append_completes_after_short_writes(self) -> None:
        from agent.harness import terminal_protocol

        real_write = os.write

        def short_write(descriptor: int, payload) -> int:
            return real_write(descriptor, bytes(payload[:7]))

        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            terminal_protocol.os,
            "write",
            side_effect=short_write,
        ):
            path = Path(tmpdir) / "record.jsonl"
            terminal_protocol.append_jsonl_record(
                path,
                {"schema_version": 1, "value": "x" * 100},
            )
            record = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(record["value"], "x" * 100)

    def test_same_delivery_occurrence_is_projected_idempotently(self) -> None:
        from agent.harness.terminal_protocol import (
            append_terminal_outcome_projection,
            build_terminal_outcome_projection,
        )

        first = build_terminal_outcome_projection(
            self._make_outcome(),
            rendered_frame="Agent> complete",
            delivery_phase="live",
        )
        second = build_terminal_outcome_projection(
            self._make_outcome(),
            rendered_frame="Agent> complete",
            delivery_phase="live",
        )
        self.assertEqual(first.projection_id, second.projection_id)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "outcomes.jsonl"
            append_terminal_outcome_projection(path, first)
            append_terminal_outcome_projection(path, second)
            records = path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(records), 1)

    def test_startup_identity_is_typed_and_can_share_the_projection_stream(
        self,
    ) -> None:
        from agent.harness.terminal_protocol import (
            append_terminal_outcome_projection,
            append_terminal_session_event,
            build_terminal_outcome_projection,
            build_terminal_session_event,
            validate_terminal_session_event,
        )

        session = build_terminal_session_event(
            process_instance_id="process-1",
            session_id="session-1",
            session_purpose="dynamic-dual-ai-chaos",
            provider="deepseek",
            model="deepseek-v4-pro",
            auth_mode="api_key",
            provider_ready=True,
            startup_status="ready",
            failure_category="",
            product_authority_id="dynamic-dual-ai-chaos:session-1",
            product_revision=1,
            product_checkpoint_thread_id=(
                "attempt:00000000-0000-0000-0000-000000000001"
            ),
            product_checkpoint_id="checkpoint-1",
            product_fingerprint="2" * 64,
            runtime_event_fence_sequence=1,
            runtime_event_fence_terminal_event_id="terminal-event",
            runtime_event_fence_id="runtime-event-1",
            runtime_event_fence_hash="5" * 64,
            rendered_frame="Agent> arbitrary localized startup",
            origin_revision={
                "commit": "test-revision",
                "worktree_hash": "1" * 64,
            },
        )
        outcome = build_terminal_outcome_projection(
            self._make_outcome(),
            rendered_frame="Agent> complete",
            delivery_phase="live",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "terminal.jsonl"
            append_terminal_session_event(path, session)
            append_terminal_outcome_projection(path, outcome)
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]

        validated = validate_terminal_session_event(records[0])
        self.assertEqual(validated.provider, "deepseek")
        self.assertEqual(validated.model, "deepseek-v4-pro")
        self.assertEqual(
            [record["record_type"] for record in records],
            ["terminal_session_event", "terminal_outcome_projection"],
        )

    def test_blocked_startup_requires_typed_failure_category(self) -> None:
        from agent.harness.terminal_protocol import (
            TerminalProtocolError,
            build_terminal_session_event,
        )

        with self.assertRaises(TerminalProtocolError):
            build_terminal_session_event(
                process_instance_id="process-1",
                session_id="session-1",
                session_purpose="user",
                provider="deepseek",
                model="deepseek-v4-pro",
                auth_mode="api_key",
                provider_ready=False,
                startup_status="blocked",
                failure_category="",
                product_authority_id="",
                product_revision=None,
                product_checkpoint_thread_id="",
                product_checkpoint_id="",
                product_fingerprint="",
                runtime_event_fence_sequence=0,
                runtime_event_fence_terminal_event_id="",
                runtime_event_fence_id="",
                runtime_event_fence_hash="",
                rendered_frame="Agent> blocked",
                origin_revision={
                    "commit": "test-revision",
                    "worktree_hash": "1" * 64,
                },
            )

    def test_blocked_startup_rejects_unknown_failure_category(self) -> None:
        from agent.harness.terminal_protocol import (
            TerminalProtocolError,
            build_terminal_session_event,
        )

        with self.assertRaisesRegex(
            TerminalProtocolError,
            "startup failure category is invalid",
        ):
            build_terminal_session_event(
                process_instance_id="process-1",
                session_id="session-1",
                session_purpose="user",
                provider="deepseek",
                model="deepseek-v4-pro",
                auth_mode="api_key",
                provider_ready=False,
                startup_status="blocked",
                failure_category="provider_readdiness_failed",
                product_authority_id="",
                product_revision=None,
                product_checkpoint_thread_id="",
                product_checkpoint_id="",
                product_fingerprint="",
                runtime_event_fence_sequence=0,
                runtime_event_fence_terminal_event_id="",
                runtime_event_fence_id="",
                runtime_event_fence_hash="",
                rendered_frame="Agent> blocked",
                origin_revision={
                    "commit": "test-revision",
                    "worktree_hash": "1" * 64,
                },
            )

    def test_ready_startup_rejects_revision_zero_without_publication_fence(
        self,
    ) -> None:
        from agent.harness.terminal_protocol import (
            TerminalProtocolError,
            build_terminal_session_event,
        )

        with self.assertRaisesRegex(
            TerminalProtocolError,
            "Product Head identity is invalid",
        ):
            build_terminal_session_event(
                process_instance_id="process-1",
                session_id="session-1",
                session_purpose="user",
                provider="deepseek",
                model="deepseek-v4-pro",
                auth_mode="api_key",
                provider_ready=True,
                startup_status="ready",
                failure_category="",
                product_authority_id="user:session-1",
                product_revision=0,
                product_checkpoint_thread_id="session-1",
                product_checkpoint_id="checkpoint-0",
                product_fingerprint="2" * 64,
                runtime_event_fence_sequence=0,
                runtime_event_fence_terminal_event_id="",
                runtime_event_fence_id="",
                runtime_event_fence_hash="",
                rendered_frame="Agent> ready",
                origin_revision={
                    "commit": "test-revision",
                    "worktree_hash": "1" * 64,
                },
            )


class InvariantRecoveryTransactionTest(unittest.TestCase):
    def test_invariant_recovery_has_one_committed_terminal_result(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AnyChainGraphRuntime(
                thread_id="single-recovery-outcome",
                checkpoint_path=Path(tmpdir) / "checkpoint.sqlite",
            )
            before = runtime.snapshot()

            def invalid_result(state, **_kwargs):
                result = dict(state)
                result["turn_index"] = int(state.get("turn_index") or 0) + 1
                result["pending_question"] = {
                    "id": "invalid-pending",
                    "prompt": "missing typed owner",
                }
                return result

            with patch.object(runtime.graph, "invoke", side_effect=invalid_result):
                recovered = runtime.invoke("anything", language="en")
            attempts = runtime.turn_transactions.list_attempts(
                runtime.transaction_authority_id
            )
            outcomes = runtime.turn_transactions.list_terminal_outcomes(
                runtime.transaction_authority_id
            )
            runtime.close()

        self.assertEqual(recovered["active_group"], "failure_recovery")
        self.assertEqual(recovered["turn_index"], int(before["turn_index"]) + 1)
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0].status, "committed")
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0].outcome, "committed")


if __name__ == "__main__":
    unittest.main()

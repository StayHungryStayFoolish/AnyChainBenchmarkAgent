from __future__ import annotations

import threading
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.harness.secret_refs import (
    SecretRegistryConflict,
    clear_secret_references,
    materialize_state_secret_references,
    project_state_secret_values,
    raw_secret_paths_in_state,
    redact_secret_references,
    resolve_secret_reference,
    secret_references_in_value,
    secret_registry_transaction,
    store_secret_reference,
)


class SecretReferenceRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_secret_references()

    def tearDown(self) -> None:
        clear_secret_references()

    def test_unrelated_transactions_commit_without_whole_turn_serialization(
        self,
    ) -> None:
        first_staged = threading.Event()
        second_committed = threading.Event()
        allow_first_commit = threading.Event()
        results: dict[str, tuple[str, str]] = {}

        def first_worker() -> None:
            with secret_registry_transaction() as transaction:
                results["first"] = store_secret_reference(
                    "first-secret",
                    draft_id="first-draft",
                    atom_id="first-atom",
                )
                first_staged.set()
                if not allow_first_commit.wait(timeout=2):
                    raise AssertionError("test did not release first transaction")
                transaction.prepare()
                transaction.commit()

        def second_worker() -> None:
            with secret_registry_transaction() as transaction:
                results["second"] = store_secret_reference(
                    "second-secret",
                    draft_id="second-draft",
                    atom_id="second-atom",
                )
                transaction.prepare()
                transaction.commit()
            second_committed.set()

        first_thread = threading.Thread(target=first_worker)
        first_thread.start()
        self.assertTrue(first_staged.wait(timeout=2))

        second_thread = threading.Thread(target=second_worker)
        second_thread.start()
        self.assertTrue(second_committed.wait(timeout=2))
        self.assertTrue(first_thread.is_alive())

        allow_first_commit.set()
        first_thread.join(timeout=2)
        second_thread.join(timeout=2)
        self.assertFalse(first_thread.is_alive())
        self.assertFalse(second_thread.is_alive())
        for name in ("first", "second"):
            reference, value_hash = results[name]
            self.assertEqual(
                resolve_secret_reference(
                    reference,
                    draft_id=f"{name}-draft",
                    atom_id=f"{name}-atom",
                    expected_hash=value_hash,
                ),
                f"{name}-secret",
            )

    def test_transaction_rollback_never_clobbers_concurrent_commit(self) -> None:
        reference, _ = store_secret_reference(
            "initial-secret",
            draft_id="initial-draft",
            atom_id="initial-atom",
        )
        staged = threading.Event()
        allow_rollback = threading.Event()

        def rollback_worker() -> None:
            with secret_registry_transaction():
                store_secret_reference(
                    "transaction-secret",
                    draft_id="transaction-draft",
                    atom_id="transaction-atom",
                    reference=reference,
                )
                staged.set()
                if not allow_rollback.wait(timeout=2):
                    raise AssertionError("test did not release rollback")

        thread = threading.Thread(target=rollback_worker)
        thread.start()
        self.assertTrue(staged.wait(timeout=2))
        _, committed_hash = store_secret_reference(
            "committed-secret",
            draft_id="committed-draft",
            atom_id="committed-atom",
            reference=reference,
        )
        allow_rollback.set()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(
            resolve_secret_reference(
                reference,
                draft_id="committed-draft",
                atom_id="committed-atom",
                expected_hash=committed_hash,
            ),
            "committed-secret",
        )

    def test_same_reference_version_conflict_fails_closed(self) -> None:
        reference, _ = store_secret_reference(
            "initial-secret",
            draft_id="initial-draft",
            atom_id="initial-atom",
        )
        both_staged = threading.Barrier(2)
        first_committed = threading.Event()
        results: dict[str, object] = {}

        def first_worker() -> None:
            with secret_registry_transaction() as transaction:
                results["first"] = store_secret_reference(
                    "first-winner",
                    draft_id="first-draft",
                    atom_id="first-atom",
                    reference=reference,
                )
                both_staged.wait(timeout=2)
                transaction.prepare()
                transaction.commit()
            first_committed.set()

        def second_worker() -> None:
            try:
                with secret_registry_transaction() as transaction:
                    store_secret_reference(
                        "second-loser",
                        draft_id="second-draft",
                        atom_id="second-atom",
                        reference=reference,
                    )
                    both_staged.wait(timeout=2)
                    if not first_committed.wait(timeout=2):
                        raise AssertionError("first transaction did not commit")
                    transaction.prepare()
            except SecretRegistryConflict as exc:
                results["conflict"] = exc

        first_thread = threading.Thread(target=first_worker)
        second_thread = threading.Thread(target=second_worker)
        first_thread.start()
        second_thread.start()
        first_thread.join(timeout=2)
        second_thread.join(timeout=2)
        self.assertFalse(first_thread.is_alive())
        self.assertFalse(second_thread.is_alive())
        self.assertIsInstance(results.get("conflict"), SecretRegistryConflict)
        _, value_hash = results["first"]  # type: ignore[misc]
        self.assertEqual(
            resolve_secret_reference(
                reference,
                draft_id="first-draft",
                atom_id="first-atom",
                expected_hash=value_hash,
            ),
            "first-winner",
        )

    def test_failed_nested_enter_releases_its_rlock_acquisition(self) -> None:
        with secret_registry_transaction() as transaction:
            with self.assertRaisesRegex(RuntimeError, "cannot be nested"):
                with secret_registry_transaction():
                    pass
            transaction.prepare()
            transaction.commit()

        completed = threading.Event()

        def worker() -> None:
            store_secret_reference(
                "after-failed-enter",
                draft_id="worker-draft",
                atom_id="worker-atom",
            )
            completed.set()

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertTrue(completed.is_set())

    def test_clear_is_rolled_back_with_the_surrounding_transaction(self) -> None:
        reference, value_hash = store_secret_reference(
            "retained-after-rollback",
            draft_id="retained-draft",
            atom_id="retained-atom",
        )

        with secret_registry_transaction():
            clear_secret_references()

        self.assertEqual(
            resolve_secret_reference(
                reference,
                draft_id="retained-draft",
                atom_id="retained-atom",
                expected_hash=value_hash,
            ),
            "retained-after-rollback",
        )

    def test_clear_epoch_conflict_fails_closed(self) -> None:
        reference, value_hash = store_secret_reference(
            "clear-epoch-secret",
            draft_id="clear-epoch-draft",
            atom_id="clear-epoch-atom",
        )
        read_complete = threading.Event()
        allow_prepare = threading.Event()
        observed: dict[str, object] = {}

        def worker() -> None:
            try:
                with secret_registry_transaction() as transaction:
                    observed["value"] = resolve_secret_reference(
                        reference,
                        draft_id="clear-epoch-draft",
                        atom_id="clear-epoch-atom",
                        expected_hash=value_hash,
                    )
                    read_complete.set()
                    if not allow_prepare.wait(timeout=2):
                        raise AssertionError("test did not release prepare")
                    transaction.prepare()
            except SecretRegistryConflict as exc:
                observed["conflict"] = exc

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(read_complete.wait(timeout=2))
        clear_secret_references()
        allow_prepare.set()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(observed["value"], "clear-epoch-secret")
        self.assertIsInstance(observed.get("conflict"), SecretRegistryConflict)

    def test_clear_transaction_conflicts_with_concurrent_registry_change(
        self,
    ) -> None:
        clear_staged = threading.Event()
        allow_prepare = threading.Event()
        observed: dict[str, object] = {}

        def worker() -> None:
            try:
                with secret_registry_transaction() as transaction:
                    clear_secret_references()
                    clear_staged.set()
                    if not allow_prepare.wait(timeout=2):
                        raise AssertionError("test did not release clear prepare")
                    transaction.prepare()
            except SecretRegistryConflict as exc:
                observed["conflict"] = exc

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(clear_staged.wait(timeout=2))
        reference, value_hash = store_secret_reference(
            "concurrent-secret",
            draft_id="concurrent-draft",
            atom_id="concurrent-atom",
        )
        allow_prepare.set()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertIsInstance(observed.get("conflict"), SecretRegistryConflict)
        self.assertEqual(
            resolve_secret_reference(
                reference,
                draft_id="concurrent-draft",
                atom_id="concurrent-atom",
                expected_hash=value_hash,
            ),
            "concurrent-secret",
        )

    def test_prepare_reservation_releases_after_product_head_failure(self) -> None:
        reference, _ = store_secret_reference(
            "initial-secret",
            draft_id="initial-draft",
            atom_id="initial-atom",
        )
        with secret_registry_transaction() as transaction:
            store_secret_reference(
                "staged-secret",
                draft_id="staged-draft",
                atom_id="staged-atom",
                reference=reference,
            )
            transaction.prepare()
            with self.assertRaises(SecretRegistryConflict):
                store_secret_reference(
                    "competing-secret",
                    draft_id="competing-draft",
                    atom_id="competing-atom",
                    reference=reference,
                )

        _, value_hash = store_secret_reference(
            "after-failure-secret",
            draft_id="after-failure-draft",
            atom_id="after-failure-atom",
            reference=reference,
        )
        self.assertEqual(
            resolve_secret_reference(
                reference,
                draft_id="after-failure-draft",
                atom_id="after-failure-atom",
                expected_hash=value_hash,
            ),
            "after-failure-secret",
        )

    def test_registry_mutation_after_prepare_fails_closed(self) -> None:
        with secret_registry_transaction() as transaction:
            store_secret_reference(
                "prepared-secret",
                draft_id="prepared-draft",
                atom_id="prepared-atom",
            )
            transaction.prepare()
            with self.assertRaisesRegex(
                SecretRegistryConflict,
                "after prepare",
            ):
                store_secret_reference(
                    "late-secret",
                    draft_id="late-draft",
                    atom_id="late-atom",
                )

    def test_graph_reserves_published_secret_until_product_head_commits(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime
        from agent.harness.state import new_state

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AnyChainGraphRuntime(
                thread_id="secret-prepare-order",
                checkpoint_path=Path(tmpdir) / "checkpoints.sqlite",
            )
            try:
                with secret_registry_transaction() as transaction:
                    reference, value_hash = store_secret_reference(
                        "publish-after-product-head",
                        draft_id="publish-draft",
                        atom_id="publish-atom",
                    )
                    original_commit = runtime._commit_turn_attempt
                    observed: dict[str, object] = {}

                    def commit_product_head(*args: object, **kwargs: object):
                        observed["prepared"] = transaction.prepared
                        visible: list[str] = []

                        def read_committed_registry() -> None:
                            try:
                                resolve_secret_reference(
                                    reference,
                                    draft_id="publish-draft",
                                    atom_id="publish-atom",
                                    expected_hash=value_hash,
                                )
                            except SecretRegistryConflict:
                                visible.append("reserved")

                        thread = threading.Thread(
                            target=read_committed_registry
                        )
                        thread.start()
                        thread.join(timeout=2)
                        self.assertFalse(thread.is_alive())
                        observed["visible_before_product_head"] = visible[0]
                        return original_commit(*args, **kwargs)

                    with patch.object(
                        runtime,
                        "_commit_turn_attempt",
                        side_effect=commit_product_head,
                    ):
                        runtime._persist_state(
                            new_state(
                                "secret-prepare-order",
                                language="en",
                            ),
                            registry_transaction=transaction,
                        )

                    self.assertTrue(transaction.committed)
                    self.assertTrue(observed["prepared"])
                    self.assertEqual(
                        observed["visible_before_product_head"],
                        "reserved",
                    )
                self.assertEqual(
                    resolve_secret_reference(
                        reference,
                        draft_id="publish-draft",
                        atom_id="publish-atom",
                        expected_hash=value_hash,
                    ),
                    "publish-after-product-head",
                )
            finally:
                runtime.close()

    def test_product_head_failure_aborts_prepared_registry_overlay(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime
        from agent.harness.state import new_state

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AnyChainGraphRuntime(
                thread_id="secret-product-head-failure",
                checkpoint_path=Path(tmpdir) / "checkpoints.sqlite",
            )
            try:
                reference = ""
                value_hash = ""
                with self.assertRaisesRegex(
                    RuntimeError,
                    "injected Product Head failure",
                ):
                    with secret_registry_transaction() as transaction:
                        reference, value_hash = store_secret_reference(
                            "must-not-publish",
                            draft_id="failure-draft",
                            atom_id="failure-atom",
                        )
                        with patch.object(
                            runtime,
                            "_commit_turn_attempt",
                            side_effect=RuntimeError(
                                "injected Product Head failure"
                            ),
                        ):
                            runtime._persist_state(
                                new_state(
                                    "secret-product-head-failure",
                                    language="en",
                                ),
                                registry_transaction=transaction,
                            )
                self.assertIsNone(
                    resolve_secret_reference(
                        reference,
                        draft_id="failure-draft",
                        atom_id="failure-atom",
                        expected_hash=value_hash,
                    )
                )
                store_secret_reference(
                    "reservation-released",
                    draft_id="released-draft",
                    atom_id="released-atom",
                    reference=reference,
                )
            finally:
                runtime.close()

    def test_commit_decision_gap_preserves_published_registry(
        self,
    ) -> None:
        from agent.harness.graph import AnyChainGraphRuntime
        from agent.harness.state import new_state

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AnyChainGraphRuntime(
                thread_id="secret-finalize-failure",
                checkpoint_path=Path(tmpdir) / "checkpoints.sqlite",
            )
            try:
                with self.assertRaisesRegex(RuntimeError, "finalize failure"):
                    with secret_registry_transaction() as transaction:
                        reference, value_hash = store_secret_reference(
                            "survives-finalize-failure",
                            draft_id="finalize-draft",
                            atom_id="finalize-atom",
                        )
                        with patch.object(
                            transaction,
                            "mark_product_head_committed",
                            side_effect=RuntimeError("finalize failure"),
                        ):
                            runtime._persist_state(
                                new_state(
                                    "secret-finalize-failure",
                                    language="en",
                                ),
                                registry_transaction=transaction,
                            )
                self.assertEqual(
                    resolve_secret_reference(
                        reference,
                        draft_id="finalize-draft",
                        atom_id="finalize-atom",
                        expected_hash=value_hash,
                    ),
                    "survives-finalize-failure",
                )
                outcomes = runtime.turn_transactions.list_terminal_outcomes(
                    runtime.transaction_authority_id
                )
                self.assertEqual(outcomes[-1].outcome, "committed")
            finally:
                runtime.close()

    def test_probe_failure_preserves_overlay_until_reconciliation(self) -> None:
        probe_calls = 0

        def durable_commit_probe() -> bool:
            nonlocal probe_calls
            probe_calls += 1
            if probe_calls == 1:
                raise RuntimeError("temporary Product Head read failure")
            return True

        reference = ""
        value_hash = ""
        with self.assertRaisesRegex(
            SecretRegistryConflict,
            "reconciliation is required",
        ):
            with secret_registry_transaction() as transaction:
                reference, value_hash = store_secret_reference(
                    "durable-secret",
                    draft_id="durable-draft",
                    atom_id="durable-atom",
                )
                transaction.bind_external_commit_probe(durable_commit_probe)
                transaction.prepare()
                transaction.publish_prepared()

        with self.assertRaisesRegex(
            SecretRegistryConflict,
            "prepared by another transaction",
        ):
            resolve_secret_reference(
                reference,
                draft_id="durable-draft",
                atom_id="durable-atom",
                expected_hash=value_hash,
            )

        with secret_registry_transaction() as transaction:
            transaction.commit()

        self.assertEqual(probe_calls, 2)
        self.assertEqual(
            resolve_secret_reference(
                reference,
                draft_id="durable-draft",
                atom_id="durable-atom",
                expected_hash=value_hash,
            ),
            "durable-secret",
        )

    def test_secret_reference_capabilities_are_found_in_mapping_keys(self) -> None:
        first = "semantic-secret:first-capability"
        second = "semantic-secret:second-capability"
        value = {
            f"prefix:{first}": {
                ("nested", second): "ordinary-value",
            },
        }

        self.assertEqual(
            secret_references_in_value(value),
            {first, second},
        )

    def test_redaction_rewrites_mapping_key_capabilities(self) -> None:
        reference = "semantic-secret:key-capability"
        projected = redact_secret_references({
            f"endpoint:{reference}": {
                ("nested", reference): reference,
            },
        })
        serialized = repr(projected)

        self.assertNotIn(reference, serialized)
        self.assertIn("secret-reference-hash:", serialized)

    def test_redaction_fails_closed_when_mapping_keys_collide(self) -> None:
        reference = "semantic-secret:collision-capability"
        redacted_reference = redact_secret_references(reference)

        with self.assertRaisesRegex(ValueError, "mapping key collision"):
            redact_secret_references({
                reference: "secret-key",
                redacted_reference: "preexisting-redacted-key",
            })

    def test_durable_projection_and_materialization_cover_mapping_keys(
        self,
    ) -> None:
        secret = "https://rpc.example/private-token-4821"
        reference, value_hash = store_secret_reference(
            secret,
            draft_id="turn-1",
            atom_id="endpoint",
        )
        state = {
            "secret_bindings": [{
                "reference": reference,
                "scope_id": "turn-1",
                "atom_id": "endpoint",
                "value_hash": value_hash,
            }],
        }
        projected = project_state_secret_values(
            {secret: {"endpoint": secret}},
            state,
        )
        self.assertNotIn(secret, repr(projected))
        self.assertIn(reference, projected)
        materialized = materialize_state_secret_references(projected, state)
        self.assertIn(secret, materialized)
        self.assertEqual(materialized[secret]["endpoint"], secret)

    def test_raw_secret_scanner_rejects_mapping_key_material(self) -> None:
        secret = "https://rpc.example/private-token-4821"
        state = {
            "confirmed_config": {
                secret: "value",
            },
        }
        self.assertIn(
            f"confirmed_config.{secret}",
            raw_secret_paths_in_state(state),
        )


if __name__ == "__main__":
    unittest.main()

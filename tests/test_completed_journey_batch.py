from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests.agent_live.completed_journey_batch import (
    G4_ARTIFACT_TYPE,
    _validated_completed_batch,
    convert_completed_journey_batch,
    load_completed_journey_batch_evidence,
)
from tests.agent_live.batch_orchestrator import (
    BATCH_RESULT_SCHEMA_VERSION,
    _load_journey_controller_bundle,
    _publish_journey_controller_bundle,
    recover_journey_controller_bundle,
    _rollback_journey_controller_bundle,
    validate_journey_controller_authority,
)
from tests.agent_live.coverage_evidence import (
    content_hash,
    create_pty_authority_signer,
    sign_controller_payload,
)


REVISION = {"commit": "a" * 40, "worktree_hash": "b" * 64}


class CompletedJourneyBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        for path in sorted(self.root.rglob("*"), reverse=True):
            path.chmod(0o700 if path.is_dir() else 0o600)
        self.temp.cleanup()

    @staticmethod
    def _sha256(path: Path) -> str:
        return __import__("hashlib").sha256(path.read_bytes()).hexdigest()

    def test_conversion_cleans_abandoned_staging_after_failure(self) -> None:
        destination = self.root / "published"

        def fail_conversion(**_kwargs):
            staging = self.root / ".published.staging-crashed"
            staging.mkdir()
            (staging / "partial.json").write_text(
                "{}\n",
                encoding="utf-8",
            )
            raise RuntimeError("injected conversion failure")

        with patch(
            "tests.agent_live.completed_journey_batch."
            "_convert_completed_journey_batch_locked",
            side_effect=fail_conversion,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "injected conversion failure",
            ):
                convert_completed_journey_batch(
                    manifest_path=self.root / "manifest.json",
                    result_index_path=self.root / "result.json",
                    output_dir=destination,
                    obligations=(),
                    revision=REVISION,
                    artifact_type=G4_ARTIFACT_TYPE,
                    round_id="round-1",
                    expected_authority_trust_root_id="trusted-root",
                    convert_one=lambda *_args: self.root / "unused",
                )

        self.assertFalse(tuple(
            self.root.glob(".published.staging-*")
        ))

    def test_journey_bundle_fsync_failure_requires_retry_reconciliation(
        self,
    ) -> None:
        evidence_path = self.root / "candidate.json"
        evidence_path.write_text('{"turns": []}\n', encoding="utf-8")
        authority = {"authority_id": "a" * 64}
        for failure_target in ("bundle", "parent"):
            with self.subTest(failure_target=failure_target):
                bundle = self.root / f"journey-{failure_target}"
                from tests.agent_live import batch_orchestrator as module

                original_fsync = module._fsync_parent_directory

                def fail_after_publish(path):
                    target = (
                        bundle
                        if failure_target == "bundle"
                        else bundle.parent
                    )
                    if bundle.exists() and path == target:
                        raise OSError("injected fsync failure")
                    original_fsync(path)

                with patch(
                    "tests.agent_live.batch_orchestrator."
                    "_fsync_parent_directory",
                    side_effect=fail_after_publish,
                ):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "durability is uncertain",
                    ):
                        _publish_journey_controller_bundle(
                            evidence_path=evidence_path,
                            authority=authority,
                            bundle_path=bundle,
                        )

                with self.assertRaisesRegex(
                    ValueError,
                    "durability reconciliation",
                ):
                    _load_journey_controller_bundle(bundle)
                self.assertEqual(
                    recover_journey_controller_bundle(
                        bundle,
                        expected_candidate={"turns": []},
                        expected_authority=authority,
                    ),
                    bundle,
                )
                self.assertEqual(
                    _publish_journey_controller_bundle(
                        evidence_path=evidence_path,
                        authority=authority,
                        bundle_path=bundle,
                    ),
                    bundle,
                )

    def test_journey_bundle_mismatched_retry_preserves_first_commit(
        self,
    ) -> None:
        first = self.root / "first.json"
        second = self.root / "second.json"
        first.write_text('{"turns": []}\n', encoding="utf-8")
        second.write_text(
            '{"turns": [{"turn_index": 1}]}\n',
            encoding="utf-8",
        )
        bundle = self.root / "journey-authority"
        authority = {"authority_id": "a" * 64}
        _publish_journey_controller_bundle(
            evidence_path=first,
            authority=authority,
            bundle_path=bundle,
        )
        committed_candidate = (
            bundle / "candidate.json"
        ).read_bytes()

        with self.assertRaisesRegex(
            FileExistsError,
            "another candidate",
        ):
            _publish_journey_controller_bundle(
                evidence_path=second,
                authority=authority,
                bundle_path=bundle,
            )

        self.assertEqual(
            (bundle / "candidate.json").read_bytes(),
            committed_candidate,
        )
        self.assertEqual(
            _load_journey_controller_bundle(bundle)[0].read_bytes(),
            first.read_bytes(),
        )

    def test_completed_batch_accepts_zero_turn_and_rejects_a_nonpassing_shard(
        self,
    ) -> None:
        runtime = self.root / "runtime"
        runtime.mkdir()
        evidence_path = runtime / "evidence.json"
        evidence_path.write_text(
            json.dumps({"turns": []}) + "\n",
            encoding="utf-8",
        )
        journey_result_path = runtime / "journey-result.json"
        journey_result_path.write_text(
            json.dumps({"evidence_path": str(evidence_path)}),
            encoding="utf-8",
        )
        transcript_path = runtime / "transcript.txt"
        transcript_path.write_text("", encoding="utf-8")
        signer = create_pty_authority_signer()
        shard = SimpleNamespace(
            shard_id="shard-1",
            lane="journey",
            obligation_id="obligation-1",
            target_hash="c" * 64,
            runtime_root=str(runtime),
            execution_id="e" * 64,
            schedule_id="s" * 64,
        )
        manifest = SimpleNamespace(
            revision=REVISION,
            shards=(shard,),
            batch_id="batch-1",
            manifest_id="manifest-1",
            pty_authority_trust_root_id=signer.trust_root_id,
            pty_authority_public_key_b64=signer.public_key_b64,
            controller_owned_execution=True,
        )
        authority_unsigned = {
            "batch_id": manifest.batch_id,
            "manifest_id": manifest.manifest_id,
            "shard_id": shard.shard_id,
            "execution_id": shard.execution_id,
            "schedule_id": shard.schedule_id,
            "candidate_artifact_sha256": self._sha256(evidence_path),
            "controller_turn_observations": [],
            "controller_turn_facts": [],
            "controller_initial_fact": {
                "initial_event": {"turn_index": 0},
                "initial_verification": {"terminal_outcome": {}},
            },
        }
        authority = {
            **authority_unsigned,
            "controller_signature_b64": sign_controller_payload(
                signer,
                authority_unsigned,
            ),
        }
        authority["authority_id"] = content_hash(authority)
        authority_path = runtime / "journey-controller-admission"
        _publish_journey_controller_bundle(
            evidence_path=evidence_path,
            authority=authority,
            bundle_path=authority_path,
        )
        evidence_digest = _load_journey_controller_bundle(
            authority_path
        )[3]
        signed_payload = {
            "batch_id": "batch-1",
            "manifest_id": "manifest-1",
            "revision": REVISION,
            "execution_authority_mode": "controller_owned_v1",
            "execution_status": "discovery_complete",
            "release_status": "not_evaluated",
            "scheduled": 1,
            "started": 1,
            "completed": 1,
            "discovery_attempt_ids": ["attempt-1"],
            "classification_counts": {"passed": 1},
            "batch_survivor_proof": {"cleaned": True},
            "shards": [{
                "shard_id": "shard-1",
                "classification": "passed",
                "target_hash": "c" * 64,
                "started_at_ns": 1,
                "finished_at_ns": 2,
                "response_hashes": [],
                "decision_hashes": [],
                "evidence_hashes": [evidence_digest],
                "schedule_result_hash": self._sha256(journey_result_path),
                "transcript_hash": self._sha256(transcript_path),
            }],
            "pty_authority_trust_root_id": signer.trust_root_id,
            "pty_authority_public_key_b64": signer.public_key_b64,
            "schema_version": BATCH_RESULT_SCHEMA_VERSION,
        }
        result = {
            **signed_payload,
            "controller_signature_b64": sign_controller_payload(
                signer,
                signed_payload,
            ),
        }
        result["index_id"] = content_hash(result)
        result_path = self.root / "result.json"
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text("{}", encoding="utf-8")
        manifest_path.chmod(0o400)
        result_path.write_text(json.dumps(result), encoding="utf-8")
        result_path.chmod(0o400)

        def load_bundle_snapshot(**kwargs):
            return _load_journey_controller_bundle(
                Path(kwargs["bundle_path"])
            )[2]

        with (
            patch(
                "tests.agent_live.completed_journey_batch.load_frozen_manifest",
                return_value=manifest,
            ),
            patch(
                "tests.agent_live.completed_journey_batch.validate_completed_shard_result",
            ),
            patch(
                "tests.agent_live.batch_orchestrator."
                "_validate_and_recompute_journey_controller_facts",
            ),
            patch(
                "tests.agent_live.completed_journey_batch."
                "load_validated_journey_controller_authority",
                side_effect=load_bundle_snapshot,
            ),
        ):
            _, sources, execution_ids, _ = _validated_completed_batch(
                manifest_path=manifest_path,
                result_index_path=result_path,
                expected_obligation_ids={"obligation-1"},
                revision=REVISION,
                expected_authority_trust_root_id=signer.trust_root_id,
            )
            self.assertEqual(
                sources["obligation-1"].runtime_root,
                runtime,
            )
            self.assertEqual(execution_ids["obligation-1"], shard.execution_id)
            with self.assertRaisesRegex(ValueError, "externally trusted"):
                _validated_completed_batch(
                    manifest_path=manifest_path,
                    result_index_path=result_path,
                    expected_obligation_ids={"obligation-1"},
                    revision=REVISION,
                    expected_authority_trust_root_id="wrong-root",
                )

            authority_file = authority_path / "authority.json"
            authority_file.chmod(0o600)
            tampered_authority = json.loads(
                authority_file.read_text(encoding="utf-8")
            )
            tampered_authority["controller_initial_fact"][
                "initial_event"
            ]["turn_index"] = 99
            authority_file.write_text(
                json.dumps(tampered_authority),
                encoding="utf-8",
            )
            authority_file.chmod(0o400)
            with self.assertRaisesRegex(
                ValueError,
                "commit marker is stale",
            ):
                _validated_completed_batch(
                    manifest_path=manifest_path,
                    result_index_path=result_path,
                    expected_obligation_ids={"obligation-1"},
                    revision=REVISION,
                    expected_authority_trust_root_id=signer.trust_root_id,
                )
            _rollback_journey_controller_bundle(authority_path)
            _publish_journey_controller_bundle(
                evidence_path=evidence_path,
                authority=authority,
                bundle_path=authority_path,
            )

            result_path.chmod(0o600)
            result["shards"][0]["classification"] = "product_failed"
            signed_payload = {
                key: value
                for key, value in result.items()
                if key not in {"index_id", "controller_signature_b64"}
            }
            result["controller_signature_b64"] = sign_controller_payload(
                signer,
                signed_payload,
            )
            result["index_id"] = content_hash({
                **signed_payload,
                "controller_signature_b64": result[
                    "controller_signature_b64"
                ],
            })
            result_path.write_text(json.dumps(result), encoding="utf-8")
            result_path.chmod(0o400)
            with self.assertRaisesRegex(ValueError, "not qualifying"):
                _validated_completed_batch(
                    manifest_path=manifest_path,
                    result_index_path=result_path,
                    expected_obligation_ids={"obligation-1"},
                    revision=REVISION,
                    expected_authority_trust_root_id=signer.trust_root_id,
                )

    @patch(
        "tests.agent_live.batch_orchestrator."
        "validate_journey_evidence_artifact",
    )
    @patch(
        "tests.agent_live.batch_orchestrator.load_verifier_registry",
        return_value=SimpleNamespace(registry_id="r" * 64),
    )
    @patch(
        "tests.agent_live.batch_orchestrator.validate_journey_schedule",
    )
    @patch(
        "tests.agent_live.batch_orchestrator.build_journey_schedule",
        return_value=SimpleNamespace(schedule_id="s" * 64),
    )
    @patch(
        "tests.agent_live.batch_orchestrator._journey_target",
        return_value=(object(), "tests.fake_registry"),
    )
    @patch(
        "tests.agent_live.batch_orchestrator."
        "_validate_and_recompute_journey_controller_facts"
    )
    def test_journey_authority_rejects_rehashed_reorder_missing_replay_and_resign(
        self,
        _recompute,
        _journey_target,
        _build_schedule,
        _validate_schedule,
        _load_registry,
        _validate_evidence,
    ) -> None:
        evidence_path = self.root / "journey-evidence.json"
        candidate_turns = [
            {"turn_index": 1, "result": "first"},
            {"turn_index": 2, "result": "second"},
        ]
        evidence_path.write_text(
            json.dumps({"turns": candidate_turns}) + "\n",
            encoding="utf-8",
        )
        (self.root / "target.json").write_text("{}\n", encoding="utf-8")
        signer = create_pty_authority_signer()
        shard = SimpleNamespace(
            shard_id="shard-1",
            execution_id="e" * 64,
            schedule_id="s" * 64,
            target_path=str(self.root / "target.json"),
            seed=7,
            verifier_registry_import="tests.fake_registry",
            verifier_registry_id="r" * 64,
        )
        manifest = SimpleNamespace(
            batch_id="batch-1",
            manifest_id="manifest-1",
            pty_authority_public_key_b64=signer.public_key_b64,
            revision=REVISION,
        )

        observations = []
        previous_hash = ""
        for ordinal in (1, 2):
            unsigned = {
                "shard_turn_ordinal": ordinal,
                "turn_index": ordinal,
                "previous_response_hash": content_hash(
                    f"previous-response-{ordinal}"
                ),
                "user_message_hash": content_hash(
                    f"user-message-{ordinal}"
                ),
                "approved_decision_hash": content_hash(
                    f"approved-decision-{ordinal}"
                ),
                "simulator_context_hash": content_hash(
                    f"simulator-context-{ordinal}"
                ),
                "approved_decision": {
                    "user_message": f"user-message-{ordinal}",
                },
                "simulator_context": {"turn_index": ordinal},
                "submission_sequence": ordinal * 2,
                "submitted_input_commitment": content_hash(
                    f"submitted-input-{ordinal}"
                ),
                "agent_response_hash": content_hash(
                    f"agent-response-{ordinal}"
                ),
                "baseline_runtime_event_hash": content_hash(
                    f"baseline-event-{ordinal}"
                ),
                "committed_runtime_event_hash": content_hash(
                    f"committed-event-{ordinal}"
                ),
                "terminal_runtime_event_id": f"terminal-{ordinal}",
                "terminal_runtime_event_sequence": ordinal,
                "terminal_outcome_hash": content_hash(
                    f"terminal-outcome-{ordinal}"
                ),
                "turn_result_hash": content_hash(
                    candidate_turns[ordinal - 1]
                ),
                "previous_controller_observation_hash": previous_hash,
            }
            receipt_hash = content_hash(unsigned)
            observations.append({
                **unsigned,
                "controller_observation_hash": receipt_hash,
            })
            previous_hash = receipt_hash

        unsigned_authority = {
            "batch_id": manifest.batch_id,
            "manifest_id": manifest.manifest_id,
            "shard_id": shard.shard_id,
            "execution_id": shard.execution_id,
            "schedule_id": shard.schedule_id,
            "candidate_artifact_sha256": self._sha256(evidence_path),
            "controller_turn_observations": observations,
            "controller_turn_facts": [
                {"turn_index": ordinal}
                for ordinal in (1, 2)
            ],
            "controller_initial_fact": {
                "initial_event": {"turn_index": 0},
                "initial_verification": {"terminal_outcome": {}},
            },
        }

        def signed_authority(
            payload,
            *,
            authority_signer=signer,
        ):
            value = {
                **payload,
                "controller_signature_b64": sign_controller_payload(
                    authority_signer,
                    payload,
                ),
            }
            value["authority_id"] = content_hash(value)
            return value

        def write_authority(name, payload):
            path = self.root / name
            _publish_journey_controller_bundle(
                evidence_path=evidence_path,
                authority=payload,
                bundle_path=path,
            )
            return path

        authority_path = write_authority(
            "authority-valid.json",
            signed_authority(unsigned_authority),
        )
        self.assertEqual(
            len(validate_journey_controller_authority(
                manifest=manifest,
                shard=shard,
                bundle_path=authority_path,
            )),
            64,
        )

        for name, rows in (
            ("reordered", list(reversed(observations))),
            ("missing", observations[:1]),
            ("duplicate", [observations[0], observations[0]]),
        ):
            rehashed_rows = []
            previous_hash = ""
            for ordinal, row in enumerate(rows, start=1):
                unsigned_row = {
                    **row,
                    "shard_turn_ordinal": ordinal,
                    "previous_controller_observation_hash": previous_hash,
                }
                unsigned_row.pop("controller_observation_hash", None)
                row_hash = content_hash(unsigned_row)
                rehashed_rows.append({
                    **unsigned_row,
                    "controller_observation_hash": row_hash,
                })
                previous_hash = row_hash
            forged = {
                **unsigned_authority,
                "controller_turn_observations": rehashed_rows,
            }
            forged_authority = signed_authority(forged)
            forged_authority["controller_signature_b64"] = (
                signed_authority(unsigned_authority)[
                    "controller_signature_b64"
                ]
            )
            forged_authority["authority_id"] = content_hash({
                key: value
                for key, value in forged_authority.items()
                if key != "authority_id"
            })
            with self.assertRaisesRegex(ValueError, "binding is invalid"):
                validate_journey_controller_authority(
                    manifest=manifest,
                    shard=shard,
                    bundle_path=write_authority(
                        f"authority-{name}.json",
                        forged_authority,
                    ),
                )
            with self.assertRaisesRegex(
                ValueError,
                "observation",
            ):
                validate_journey_controller_authority(
                    manifest=manifest,
                    shard=shard,
                    bundle_path=write_authority(
                        f"authority-{name}-trusted-resign.json",
                        signed_authority(forged),
                    ),
                )

        replayed_shard = SimpleNamespace(
            shard_id=shard.shard_id,
            execution_id="f" * 64,
            schedule_id=shard.schedule_id,
        )
        with self.assertRaisesRegex(ValueError, "binding is invalid"):
            validate_journey_controller_authority(
                manifest=manifest,
                shard=replayed_shard,
                bundle_path=authority_path,
            )

        attacker = create_pty_authority_signer()
        with self.assertRaisesRegex(ValueError, "binding is invalid"):
            validate_journey_controller_authority(
                manifest=manifest,
                shard=shard,
                bundle_path=write_authority(
                    "authority-attacker.json",
                    signed_authority(
                        unsigned_authority,
                        authority_signer=attacker,
                    ),
                ),
            )

    def test_conversion_publishes_a_declared_read_only_index(self) -> None:
        runtime = self.root / "runtime"
        runtime.mkdir()
        manifest_source = self.root / "batch-manifest.json"
        result_source = self.root / "batch-result.json"
        manifest_source.write_text("{}", encoding="utf-8")
        result_source.write_text("{}", encoding="utf-8")
        manifest_source.chmod(0o400)
        result_source.chmod(0o400)
        manifest = SimpleNamespace(
            manifest_id="manifest-1",
            batch_id="batch-1",
        )
        obligation = {
            "obligation_id": "obligation-1",
            "contract_hash": "c" * 64,
        }

        source = SimpleNamespace(runtime_root=runtime)

        def convert_one(_obligation, _source, evidence_path, checkpoint_diff):
            runtime_artifact = _source.runtime_root / "runtime-artifact.txt"
            runtime_artifact.write_text("runtime\n", encoding="utf-8")
            evidence_path.write_text(json.dumps({
                "obligation_id": "obligation-1",
                "obligation_contract_hash": "c" * 64,
                "evidence_id": "d" * 64,
                "execution": {"execution_id": "e" * 64},
                "artifacts": [{
                    "role": "runtime",
                    "path": str(runtime_artifact),
                    "sha256": __import__("hashlib").sha256(
                        runtime_artifact.read_bytes()
                    ).hexdigest(),
                }],
            }), encoding="utf-8")
            self.assertIsNotNone(checkpoint_diff)
            checkpoint_diff.write_text('{"diff": true}\n', encoding="utf-8")
            return evidence_path

        from tests.agent_live import completed_journey_batch as module

        with (
            patch(
                "tests.agent_live.completed_journey_batch._validated_completed_batch",
                return_value=(
                    manifest,
                    {"obligation-1": source},
                    {"obligation-1": "e" * 64},
                    {"index_id": "index-1"},
                ),
            ),
            patch(
                "tests.agent_live.completed_journey_batch.admit_product_obligation_evidence",
                return_value={
                    "complete": True,
                    "passed": 1,
                    "failed": 0,
                    "external": 0,
                    "not_run": 0,
                },
            ),
            patch(
                "tests.agent_live.completed_journey_batch."
                "_fsync_completed_batch_tree",
                wraps=module._fsync_completed_batch_tree,
            ) as fsync_tree,
        ):
            destination = self.root / "published"
            original_fsync = module._fsync_directory

            def fail_after_publish(path):
                if destination.exists() and path == destination.parent:
                    raise OSError("injected completed-batch fsync failure")
                original_fsync(path)

            with patch(
                "tests.agent_live.completed_journey_batch._fsync_directory",
                side_effect=fail_after_publish,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "durability is uncertain",
                ):
                    convert_completed_journey_batch(
                        manifest_path=manifest_source,
                        result_index_path=result_source,
                        output_dir=destination,
                        obligations=(obligation,),
                        revision=REVISION,
                        artifact_type=G4_ARTIFACT_TYPE,
                        round_id="round-1",
                        expected_authority_trust_root_id="trusted-root",
                        convert_one=convert_one,
                    )
                with self.assertRaisesRegex(
                    ValueError,
                    "reconciliation is required",
                ):
                    load_completed_journey_batch_evidence(
                        destination / "manifest.json",
                        obligations=(obligation,),
                        revision=REVISION,
                        artifact_type=G4_ARTIFACT_TYPE,
                        round_id="round-1",
                        expected_authority_trust_root_id="trusted-root",
                    )
                with self.assertRaisesRegex(
                    RuntimeError,
                    "reconciliation did not complete",
                ):
                    convert_completed_journey_batch(
                        manifest_path=manifest_source,
                        result_index_path=result_source,
                        output_dir=destination,
                        obligations=(obligation,),
                        revision=REVISION,
                        artifact_type=G4_ARTIFACT_TYPE,
                        round_id="round-1",
                        expected_authority_trust_root_id="trusted-root",
                        convert_one=lambda *_args: self.fail(
                            "uncertain recovery must not reconvert evidence"
                        ),
                    )
            index = convert_completed_journey_batch(
                manifest_path=manifest_source,
                result_index_path=result_source,
                output_dir=destination,
                obligations=(obligation,),
                revision=REVISION,
                artifact_type=G4_ARTIFACT_TYPE,
                round_id="round-1",
                expected_authority_trust_root_id="trusted-root",
                convert_one=convert_one,
            )
            paths = load_completed_journey_batch_evidence(
                index,
                obligations=(obligation,),
                revision=REVISION,
                artifact_type=G4_ARTIFACT_TYPE,
                round_id="round-1",
                expected_authority_trust_root_id="trusted-root",
            )
            recovered_index = convert_completed_journey_batch(
                manifest_path=manifest_source,
                result_index_path=result_source,
                output_dir=self.root / "published",
                obligations=(obligation,),
                revision=REVISION,
                artifact_type=G4_ARTIFACT_TYPE,
                round_id="round-1",
                expected_authority_trust_root_id="trusted-root",
                convert_one=lambda *_args: self.fail(
                    "recovery must not reconvert evidence"
                ),
            )
            self.assertEqual(recovered_index, index)
            (runtime / "runtime-artifact.txt").write_text(
                "tampered\n",
                encoding="utf-8",
            )
            self.assertEqual(
                len(load_completed_journey_batch_evidence(
                    index,
                    obligations=(obligation,),
                    revision=REVISION,
                    artifact_type=G4_ARTIFACT_TYPE,
                    round_id="round-1",
                    expected_authority_trust_root_id="trusted-root",
                )),
                1,
            )
            self.assertEqual(fsync_tree.call_count, 2)
        self.assertEqual(len(paths), 1)
        self.assertFalse(index.stat().st_mode & 0o222)
        self.assertFalse(index.parent.stat().st_mode & 0o222)
        self.assertFalse(
            (self.root / ".published.durability-uncertain.json").exists()
        )

    def test_idempotent_recovery_rejects_different_or_changed_sources(
        self,
    ) -> None:
        destination = self.root / "published"
        destination.mkdir()
        source_manifest = self.root / "source-manifest.json"
        source_result = self.root / "source-result.json"
        source_manifest.write_text('{"source": "manifest"}\n', encoding="utf-8")
        source_result.write_text('{"source": "result"}\n', encoding="utf-8")
        source_manifest.chmod(0o400)
        source_result.chmod(0o400)
        manifest = SimpleNamespace(manifest_id="manifest-1")
        result = {"index_id": "index-1"}
        destination_index = destination / "manifest.json"
        destination_index.write_text(json.dumps({
            "batch_manifest": {
                "path": str(source_manifest.resolve()),
                "sha256": self._sha256(source_manifest),
                "manifest_id": "manifest-1",
            },
            "batch_result": {
                "path": str(source_result.resolve()),
                "sha256": self._sha256(source_result),
                "index_id": "index-1",
            },
        }), encoding="utf-8")

        different_manifest = self.root / "different-manifest.json"
        different_manifest.write_bytes(source_manifest.read_bytes())
        different_manifest.chmod(0o400)
        obligation = {
            "obligation_id": "obligation-1",
            "contract_hash": "c" * 64,
        }
        common = {
            "output_dir": destination,
            "obligations": (obligation,),
            "revision": REVISION,
            "artifact_type": G4_ARTIFACT_TYPE,
            "round_id": "round-1",
            "expected_authority_trust_root_id": "trusted-root",
            "convert_one": lambda *_args: self.fail(
                "idempotent recovery must not reconvert evidence"
            ),
        }
        with (
            patch(
                "tests.agent_live.completed_journey_batch."
                "_validated_completed_batch",
                return_value=(manifest, {}, {}, result),
            ),
            patch(
                "tests.agent_live.completed_journey_batch."
                "load_completed_journey_batch_evidence",
                return_value=(),
            ),
        ):
            with self.assertRaisesRegex(
                FileExistsError,
                "not the same recoverable publication",
            ):
                convert_completed_journey_batch(
                    manifest_path=different_manifest,
                    result_index_path=source_result,
                    **common,
                )

            source_manifest.chmod(0o600)
            source_manifest.write_text(
                '{"source": "changed"}\n',
                encoding="utf-8",
            )
            source_manifest.chmod(0o400)
            with self.assertRaisesRegex(
                FileExistsError,
                "not the same recoverable publication",
            ):
                convert_completed_journey_batch(
                    manifest_path=source_manifest,
                    result_index_path=source_result,
                    **common,
                )

    def test_declared_index_rejects_evidence_tampering(self) -> None:
        evidence_root = self.root / "evidence-set"
        evidence_dir = evidence_root / "evidence"
        evidence_dir.mkdir(parents=True)
        runtime_artifact = self.root / "runtime-artifact.txt"
        runtime_artifact.write_text("runtime\n", encoding="utf-8")
        evidence = evidence_dir / "evidence.json"
        evidence.write_text(json.dumps({
            "obligation_id": "obligation-1",
            "obligation_contract_hash": "c" * 64,
            "evidence_id": "d" * 64,
            "execution": {"execution_id": "e" * 64},
            "artifacts": [{
                "role": "runtime",
                "path": str(runtime_artifact),
                "sha256": __import__("hashlib").sha256(
                    runtime_artifact.read_bytes()
                ).hexdigest(),
            }],
        }), encoding="utf-8")
        obligation = {
            "obligation_id": "obligation-1",
            "contract_hash": "c" * 64,
        }
        batch_manifest = self.root / "batch-manifest.json"
        batch_result = self.root / "batch-result.json"
        batch_manifest.write_text("{}", encoding="utf-8")
        batch_result.write_text("{}", encoding="utf-8")
        batch_manifest.chmod(0o400)
        batch_result.chmod(0o400)
        unsigned = {
            "schema_version": 1,
            "artifact_type": G4_ARTIFACT_TYPE,
            "revision_binding": REVISION,
            "round_id": "round-1",
            "authority_trust_root_id": "trusted-root",
            "batch_manifest": {
                "path": str(batch_manifest),
                "sha256": __import__("hashlib").sha256(
                    batch_manifest.read_bytes()
                ).hexdigest(),
                "manifest_id": "manifest-1",
            },
            "batch_result": {
                "path": str(batch_result),
                "sha256": __import__("hashlib").sha256(
                    batch_result.read_bytes()
                ).hexdigest(),
                "index_id": "index-1",
            },
            "obligation_count": 1,
            "evidence": [{
                "obligation_id": "obligation-1",
                "obligation_contract_hash": "c" * 64,
                "evidence_id": "d" * 64,
                "execution_id": "e" * 64,
                "path": str(evidence.resolve()),
                "sha256": __import__("hashlib").sha256(evidence.read_bytes()).hexdigest(),
            }],
        }
        index = evidence_root / "manifest.json"
        index.write_text(
            json.dumps({**unsigned, "index_hash": content_hash(unsigned)}),
            encoding="utf-8",
        )
        evidence.chmod(0o400)
        evidence_dir.chmod(0o500)
        index.chmod(0o400)
        evidence_root.chmod(0o500)
        manifest = SimpleNamespace(manifest_id="manifest-1")
        with (
            patch(
                "tests.agent_live.completed_journey_batch._validated_completed_batch",
                return_value=(
                    manifest,
                    {
                        "obligation-1": SimpleNamespace(
                            runtime_root=self.root
                        )
                    },
                    {"obligation-1": "e" * 64},
                    {"index_id": "index-1"},
                ),
            ),
            patch(
                "tests.agent_live.completed_journey_batch.admit_product_obligation_evidence",
                return_value={"complete": True},
            ),
        ):
            self.assertEqual(
                len(load_completed_journey_batch_evidence(
                    index,
                    obligations=(obligation,),
                    revision=REVISION,
                    artifact_type=G4_ARTIFACT_TYPE,
                    round_id="round-1",
                    expected_authority_trust_root_id="trusted-root",
                )),
                1,
            )
            with patch(
                "tests.agent_live.completed_journey_batch._validated_completed_batch",
                return_value=(
                    manifest,
                    {
                        "obligation-1": SimpleNamespace(
                            runtime_root=self.root
                        )
                    },
                    {"obligation-1": "f" * 64},
                    {"index_id": "index-1"},
                ),
            ), self.assertRaisesRegex(ValueError, "row binding"):
                load_completed_journey_batch_evidence(
                    index,
                    obligations=(obligation,),
                    revision=REVISION,
                    artifact_type=G4_ARTIFACT_TYPE,
                    round_id="round-1",
                    expected_authority_trust_root_id="trusted-root",
                )
            evidence.chmod(0o600)
            evidence.write_text('{"tampered": true}\n', encoding="utf-8")
            evidence.chmod(0o400)
            with self.assertRaisesRegex(ValueError, "row binding"):
                load_completed_journey_batch_evidence(
                    index,
                    obligations=(obligation,),
                    revision=REVISION,
                    artifact_type=G4_ARTIFACT_TYPE,
                    round_id="round-1",
                    expected_authority_trust_root_id="trusted-root",
                )

    def test_conversion_rejects_execution_identity_substitution_atomically(self) -> None:
        runtime = self.root / "runtime"
        runtime.mkdir()
        manifest_source = self.root / "batch-manifest.json"
        result_source = self.root / "batch-result.json"
        manifest_source.write_text("{}", encoding="utf-8")
        result_source.write_text("{}", encoding="utf-8")
        manifest_source.chmod(0o400)
        result_source.chmod(0o400)
        manifest = SimpleNamespace(
            manifest_id="manifest-1",
            batch_id="batch-1",
        )
        obligation = {
            "obligation_id": "obligation-1",
            "contract_hash": "c" * 64,
        }

        def convert_one(_obligation, _runtime, evidence_path, _checkpoint_diff):
            evidence_path.write_text(json.dumps({
                "obligation_id": "obligation-1",
                "obligation_contract_hash": "c" * 64,
                "evidence_id": "d" * 64,
                "execution": {"execution_id": "e" * 64},
            }), encoding="utf-8")
            return evidence_path

        destination = self.root / "published"
        with (
            patch(
                "tests.agent_live.completed_journey_batch._validated_completed_batch",
                return_value=(
                    manifest,
                    {"obligation-1": runtime},
                    {"obligation-1": "f" * 64},
                    {"index_id": "index-1"},
                ),
            ),
            self.assertRaisesRegex(ValueError, "identity"),
        ):
            convert_completed_journey_batch(
                manifest_path=manifest_source,
                result_index_path=result_source,
                output_dir=destination,
                obligations=(obligation,),
                revision=REVISION,
                artifact_type=G4_ARTIFACT_TYPE,
                round_id="round-1",
                expected_authority_trust_root_id="trusted-root",
                convert_one=convert_one,
            )
        self.assertFalse(destination.exists())

    def test_loader_rejects_symlinked_index(self) -> None:
        real = self.root / "manifest.json"
        real.write_text("{}", encoding="utf-8")
        real.chmod(0o400)
        link = self.root / "linked-manifest.json"
        link.symlink_to(real)
        with self.assertRaisesRegex(ValueError, "symlink"):
            load_completed_journey_batch_evidence(
                link,
                obligations=({"obligation_id": "obligation-1"},),
                revision=REVISION,
                artifact_type=G4_ARTIFACT_TYPE,
                round_id="round-1",
                expected_authority_trust_root_id="trusted-root",
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any

from tests.agent_live.generate_product_review_evidence import (
    generate_product_review_evidence,
)
from tests.agent_live.product_review_contract import (
    PRODUCT_REVIEW_SCHEMA_VERSION,
    content_hash,
    file_sha256,
)
from tests.agent_live.validate_product_review_evidence import (
    validate_product_review_evidence,
    validator_imports_generator,
)


REVISION = {"commit": "a" * 40, "worktree_hash": "b" * 64}


class ProductReviewContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        subprocess.run(("git", "init", "-q", str(self.repo)), check=True)
        (self.repo / ".gitignore").write_text("runtime.tmp\n", encoding="utf-8")
        (self.repo / "runtime.tmp").write_text("runtime\n", encoding="utf-8")
        self.binding = self.repo / "runtime-policy.txt"
        self.binding.write_text("runtime.tmp is ephemeral\n", encoding="utf-8")
        self.en_doc = self.repo / "docs" / "en" / "architecture.md"
        self.zh_doc = self.repo / "docs" / "zh" / "architecture.md"
        self.en_doc.parent.mkdir(parents=True)
        self.zh_doc.parent.mkdir(parents=True)
        self.en_doc.write_text("CONTROL_PLANE_FACT: one owner\n", encoding="utf-8")
        self.zh_doc.write_text("CONTROL_PLANE_FACT：唯一控制权\n", encoding="utf-8")
        self.migration_check = self.repo / "migration-check.txt"
        self.migration_check.write_text("checkpoint cutoff passed\n", encoding="utf-8")
        self.finding_evidence = self.repo / "finding-evidence.txt"
        self.finding_evidence.write_text("regression passed\n", encoding="utf-8")

        self.planner = self._source(
            "planner.json",
            "planner_runtime_receipt",
            {
                "turn_index": 1,
                "control_receipt": self._planner_control_receipt(),
                "planner_metrics": {
                    "latency_ms": 12.5,
                    "model_calls": 2,
                    "prompt_bytes": 2048,
                    "token_usage": {"availability": "not_supplied"},
                },
            },
        )
        self.migration = self._source(
            "migration.json",
            "migration_cutoff",
            {
                "current_schema_version": 23,
                "minimum_supported_schema_version": 12,
                "legacy_fixture_count": 3,
                "accepted_legacy_fixture_count": 0,
                "cutoff_revision": REVISION["commit"],
                "checks": [
                    {
                        "check_id": (
                            f"checkpoint.v{version}."
                            + (
                                "quarantined"
                                if version <= 11
                                else "migrated"
                                if version <= 22
                                else "current"
                                if version == 23
                                else "rejected-future"
                            )
                        ),
                        "outcome": "passed",
                        "evidence_path": str(self.migration_check),
                    }
                    for version in range(1, 25)
                ],
            },
        )
        self.docs = self._source(
            "docs.json",
            "bilingual_documentation",
            {
                "pairs": [{
                    "pair_id": "architecture.control-plane",
                    "en_path": str(self.en_doc),
                    "zh_path": str(self.zh_doc),
                    "facts": [{
                        "fact_id": "control-plane.owner",
                        "en_marker": "CONTROL_PLANE_FACT",
                        "zh_marker": "CONTROL_PLANE_FACT",
                    }],
                }],
            },
        )
        self.hygiene = self._source(
            "hygiene.json",
            "ignored_runtime_hygiene",
            {
                "repo_root": str(self.repo),
                "allowed_runtime_entries": [{
                    "path": "runtime.tmp",
                    "kind": "ignored_file",
                    "reason": "ephemeral runtime evidence",
                    "binding_path": str(self.binding),
                }],
                "declared_inputs": [],
            },
        )
        self.external_probe = self._source(
            "external-probe.json",
            "external_capability_probe",
            {
                "capability_id": "gemini.google_search",
                "probe_id": "probe.google-search-1",
                "outcome": "externally_blocked",
                "reason": "provider is not configured in this environment",
            },
        )
        self.external = self._source(
            "external.json",
            "external_capability",
            {
                "capability_id": "gemini.google_search",
                "state": "externally_blocked",
                "probe_path": str(self.external_probe),
            },
        )
        self.severity = self._source(
            "severity.json",
            "severity_ledger",
            {
                "source_reviews": [{
                    "source_id": "unit_product_review",
                    "finding_ids": ["g6.finding-001"],
                    "evidence_paths": [str(self.finding_evidence)],
                }],
                "findings": [{
                    "finding_id": "g6.finding-001",
                    "severity": "S2",
                    "owner": "agent-platform",
                    "status": "resolved",
                    "disposition": "fixed and covered by regression evidence",
                    "root_class": "control-authority",
                    "discovery_revision": "9" * 40,
                    "fix_revision": REVISION["commit"],
                    "evidence_paths": [str(self.finding_evidence)],
                }],
            },
        )
        fixture = (
            self.repo
            / "tools"
            / "fake-node"
            / "fixtures"
            / "demo"
            / "eth_chainId.json"
        )
        fixture.parent.mkdir(parents=True)
        fixture.write_text('{"result":"0x1"}\n', encoding="utf-8")
        template = self.repo / "config" / "chains" / "demo.json"
        template.parent.mkdir(parents=True)
        template.write_text(
            json.dumps({
                "_meta": {"adapter_family": "jsonrpc"},
                "rpc_methods": {
                    "single": "eth_chainId",
                    "mixed_weighted": [
                        {"method": "eth_chainId", "weight": 100}
                    ],
                },
            }),
            encoding="utf-8",
        )
        shell_manifest = (
            self.repo
            / "tests"
            / "agent_live"
            / "fixtures"
            / "linux_shell_gate_manifest.json"
        )
        shell_manifest.parent.mkdir(parents=True)
        shell_manifest.write_text(
            json.dumps({
                "schema_version": 1,
                "scope": "tracked_tests_recursive_shell",
                "entries": [],
            }),
            encoding="utf-8",
        )
        self.scope_manifest = shell_manifest.with_name(
            "product_review_scope.json"
        )
        self.scope_manifest.write_text(
            json.dumps({
                "schema_version": 2,
                "scope_id": "unit-g6-scope-v2",
                "documentation": {
                    "english_root": "docs/en",
                    "chinese_root": "docs/zh",
                    "paired_basenames": ["architecture.md"],
                    "reviewed_exemptions": [],
                    "required_facts": [{
                        "fact_id": "control-plane.owner",
                        "basename": "architecture.md",
                        "en_marker": "CONTROL_PLANE_FACT",
                        "zh_marker": "CONTROL_PLANE_FACT",
                    }],
                },
                "migration": {
                    "current_schema_version": 23,
                    "quarantine_versions": list(range(1, 12)),
                    "migrate_versions": list(range(12, 23)),
                    "current_versions": [23],
                    "future_version_probe": 24,
                },
                "external_capabilities": [{
                    "capability_id": "gemini.google_search",
                    "required_state": "available_or_externally_blocked",
                }],
                "severity_sources": ["unit_product_review"],
                "shell_gate_manifest": str(
                    shell_manifest.relative_to(self.repo)
                ),
                "fixture_policy": {
                    "fixture_root": "tools/fake-node/fixtures",
                    "chain_template_root": "config/chains",
                    "workload_membership": "all_single_and_mixed_weighted_methods",
                    "non_workload_classification": "auxiliary",
                    "strict_authenticity_without_capture_provenance": "scoped_limitation",
                },
            }),
            encoding="utf-8",
        )
        subprocess.run(
            ("git", "-C", str(self.repo), "add", "docs", "config", "tools", "tests"),
            check=True,
        )
        from tests.agent_live.product_review_scope import (
            compute_product_review_scope,
        )

        scope = compute_product_review_scope(
            self.repo,
            self.scope_manifest,
        )
        self.shell_receipt = self._source(
            "shell-receipt.json",
            "linux_shell_gate_receipt",
            {
                "execution": {
                    "manifest": scope["shell_gates"]["manifest"],
                    "manifest_sha256": scope["shell_gates"][
                        "manifest_sha256"
                    ],
                    "classified_denominator": scope["shell_gates"][
                        "denominator"
                    ],
                    "required_denominator": 0,
                    "passed": 0,
                    "failed": 0,
                    "results": [],
                    "status": "passed",
                }
            },
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _source(
        self,
        name: str,
        source_type: str,
        payload: dict[str, Any],
    ) -> Path:
        unsigned = {
            "schema_version": PRODUCT_REVIEW_SCHEMA_VERSION,
            "source_type": source_type,
            "revision": dict(REVISION),
            **payload,
        }
        path = self.root / name
        path.write_text(
            json.dumps({**unsigned, "source_hash": content_hash(unsigned)}),
            encoding="utf-8",
        )
        return path

    def _planner_control_receipt(self) -> dict[str, Any]:
        unsigned = {
            "receipt_type": "semantic_planner",
            "turn_index": 1,
            "input_hash": "4" * 64,
            "pending_contract_hash": "5" * 64,
            "resolver_invoked": True,
            "result_reason_hash": "6" * 64,
            "planned_action_types": ["select_fake_node"],
            "semantic_units": [{
                "unit_id": "unit-1",
                "disposition": "planned",
            }],
            "planner_metrics": {
                "latency_ms": 12.5,
                "model_calls": 2,
                "prompt_bytes": 2048,
            },
        }
        encoded = json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return {
            **unsigned,
            "receipt_id": hashlib.sha256(encoded).hexdigest(),
        }

    def _generate(self, name: str = "review") -> Path:
        return generate_product_review_evidence(
            output_dir=self.root / name,
            revision=REVISION,
            planner_receipt_paths=[self.planner],
            migration_cutoff_path=self.migration,
            bilingual_docs_path=self.docs,
            runtime_hygiene_path=self.hygiene,
            external_capability_paths=[self.external],
            severity_ledger_path=self.severity,
            shell_gate_receipt_path=self.shell_receipt,
            scope_repo_root=self.repo,
            scope_manifest_path=self.scope_manifest,
            generated_at="2026-07-25T00:00:00+00:00",
        )

    def _validate(self, manifest_path: Path, **kwargs):
        return validate_product_review_evidence(
            manifest_path,
            scope_repo_root=self.repo,
            scope_manifest_path=self.scope_manifest,
            **kwargs,
        )

    def _load(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _write(self, path: Path, value: dict[str, Any]) -> None:
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")

    def _rehash_source(self, path: Path, mutate: Any) -> dict[str, Any]:
        value = self._load(path)
        mutate(value)
        unsigned = dict(value)
        unsigned.pop("source_hash", None)
        value["source_hash"] = content_hash(unsigned)
        self._write(path, value)
        return value

    def _rebind_artifact_and_manifest(
        self,
        manifest_path: Path,
        role: str,
        mutate_payload: Any | None = None,
    ) -> None:
        manifest = self._load(manifest_path)
        row = next(item for item in manifest["artifacts"] if item["role"] == role)
        artifact_path = Path(row["path"])
        artifact = self._load(artifact_path)
        artifact["source_evidence"][0]["sha256"] = file_sha256(
            artifact["source_evidence"][0]["path"]
        )
        if mutate_payload is not None:
            mutate_payload(artifact["payload"])
        unsigned_artifact = dict(artifact)
        unsigned_artifact.pop("artifact_hash", None)
        artifact["artifact_hash"] = content_hash(unsigned_artifact)
        self._write(artifact_path, artifact)
        row["sha256"] = file_sha256(artifact_path)
        row["artifact_hash"] = artifact["artifact_hash"]
        unsigned_manifest = dict(manifest)
        unsigned_manifest.pop("manifest_hash", None)
        manifest["manifest_hash"] = content_hash(unsigned_manifest)
        self._write(manifest_path, manifest)

    def test_generator_publishes_evidence_but_only_validator_decides_gate(self) -> None:
        manifest_path = self._generate()
        manifest = self._load(manifest_path)
        self.assertNotIn("status", manifest)
        self.assertNotIn("passed", manifest)
        self.assertEqual(
            {row["role"] for row in manifest["artifacts"]},
            {
                "product_review_scope",
                "planner_metrics",
                "migration_cutoff",
                "bilingual_documentation",
                "ignored_runtime_hygiene",
                "external_capabilities",
                "severity_ledger",
                "linux_shell_gates",
                "fixture_provenance",
            },
        )

        result = self._validate(
            manifest_path,
            revision=REVISION,
        )
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["failed_checks"])

    def test_validator_does_not_import_generator(self) -> None:
        self.assertFalse(validator_imports_generator())

    def test_handwritten_boolean_review_json_fails_closed(self) -> None:
        path = self.root / "handwritten.json"
        self._write(path, {
            "schema_version": 1,
            "revision": REVISION,
            "planner_metrics": True,
            "migration_cutoff_verified": True,
            "bilingual_docs_verified": True,
            "external_capabilities": True,
            "review_hash": "0" * 64,
        })
        with self.assertRaisesRegex(ValueError, "manifest schema"):
            self._validate(path, revision=REVISION)

    def test_old_revision_fails_closed_even_with_valid_manifest_hash(self) -> None:
        manifest_path = self._generate()
        manifest = self._load(manifest_path)
        manifest["revision"] = {
            "commit": "c" * 40,
            "worktree_hash": "d" * 64,
        }
        unsigned = dict(manifest)
        unsigned.pop("manifest_hash")
        manifest["manifest_hash"] = content_hash(unsigned)
        self._write(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "stale"):
            self._validate(manifest_path, revision=REVISION)

    def test_tampered_artifact_hash_fails_closed(self) -> None:
        manifest_path = self._generate()
        manifest = self._load(manifest_path)
        artifact_path = Path(manifest["artifacts"][0]["path"])
        artifact = self._load(artifact_path)
        artifact["payload"]["receipt_count"] = 999
        self._write(artifact_path, artifact)
        with self.assertRaisesRegex(ValueError, "file hash drifted"):
            self._validate(manifest_path, revision=REVISION)

    def test_tampered_manifest_hash_fails_closed(self) -> None:
        manifest_path = self._generate()
        manifest = self._load(manifest_path)
        manifest["generated_at"] = "2099-01-01T00:00:00+00:00"
        self._write(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, "manifest hash drifted"):
            self._validate(manifest_path, revision=REVISION)

    def test_generator_implementation_drift_fails_closed(self) -> None:
        manifest_path = self._generate()
        drifted = self.root / "generator-copy.py"
        drifted.write_text("# different generator implementation\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "generator implementation binding"):
            self._validate(
                manifest_path,
                revision=REVISION,
                generator_implementation_path=drifted,
            )

    def test_validator_implementation_drift_fails_closed(self) -> None:
        manifest_path = self._generate()
        drifted = self.root / "validator-copy.py"
        drifted.write_text("# different validator implementation\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "validator implementation binding"):
            self._validate(
                manifest_path,
                revision=REVISION,
                validator_implementation_path=drifted,
            )

    def test_forged_external_state_fails_after_all_outer_hashes_are_rebound(self) -> None:
        manifest_path = self._generate()
        self._rehash_source(
            self.external,
            lambda value: value.update({
                "state": "available",
            }),
        )
        self._rebind_artifact_and_manifest(
            manifest_path,
            "external_capabilities",
            lambda payload: payload["capabilities"][0].update({
                "state": "available",
            }),
        )
        with self.assertRaisesRegex(ValueError, "state is forged"):
            self._validate(manifest_path, revision=REVISION)

    def test_missing_provider_usage_cannot_be_fabricated_as_zero(self) -> None:
        self._rehash_source(
            self.planner,
            lambda value: value["planner_metrics"]["token_usage"].update({
                "prompt_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }),
        )
        with self.assertRaisesRegex(ValueError, "cannot contain zero counts"):
            self._generate()

    def test_supplied_token_usage_is_derived_from_provider_evidence(self) -> None:
        control_receipt = self._load(self.planner)["control_receipt"]
        provider = self._source(
            "provider-usage.json",
            "provider_token_usage",
            {
                "receipt_id": control_receipt["receipt_id"],
                "prompt_tokens": 120,
                "output_tokens": 30,
                "total_tokens": 150,
            },
        )
        self._rehash_source(
            self.planner,
            lambda value: value["planner_metrics"].update({
                "token_usage": {
                    "availability": "supplied",
                    "provider_evidence_path": str(provider),
                }
            }),
        )
        manifest = self._load(self._generate())
        planner_path = Path(next(
            row["path"] for row in manifest["artifacts"]
            if row["role"] == "planner_metrics"
        ))
        usage = self._load(planner_path)["payload"]["token_usage"]
        self.assertEqual(usage["total_tokens"], 150)
        self.assertEqual(usage["provider_evidence"][0]["sha256"], file_sha256(provider))

    def test_unadmitted_planner_metrics_fail_closed(self) -> None:
        self._rehash_source(
            self.planner,
            lambda value: value["planner_metrics"].update({"model_calls": 99}),
        )
        with self.assertRaisesRegex(ValueError, "admitted receipt"):
            self._generate()

    def test_unavailable_token_usage_is_preserved_without_counts(self) -> None:
        self._rehash_source(
            self.planner,
            lambda value: value["planner_metrics"].update({
                "token_usage": {"availability": "unavailable"}
            }),
        )
        manifest_path = self._generate()
        manifest = self._load(manifest_path)
        planner_path = Path(next(
            row["path"] for row in manifest["artifacts"]
            if row["role"] == "planner_metrics"
        ))
        usage = self._load(planner_path)["payload"]["token_usage"]
        self.assertEqual(usage, {"availability": "unavailable"})

    def test_open_s1_and_undisposed_s2_each_fail_gate(self) -> None:
        for suffix, finding in (
            ("open-s1", {
                "finding_id": "g6.open-s1",
                "severity": "S1",
                "owner": "agent-platform",
                "status": "open",
                "disposition": "",
                "root_class": "control-authority",
                "discovery_revision": REVISION["commit"],
                "fix_revision": "",
                "evidence_paths": [str(self.finding_evidence)],
            }),
            ("open-s2", {
                "finding_id": "g6.open-s2",
                "severity": "S2",
                "owner": "agent-platform",
                "status": "open",
                "disposition": "",
                "root_class": "state-lifecycle",
                "discovery_revision": REVISION["commit"],
                "fix_revision": "",
                "evidence_paths": [str(self.finding_evidence)],
            }),
        ):
            source = self._load(self.severity)
            source["findings"] = [finding]
            source["source_reviews"][0]["finding_ids"] = [
                finding["finding_id"]
            ]
            unsigned = dict(source)
            unsigned.pop("source_hash", None)
            source["source_hash"] = content_hash(unsigned)
            self._write(self.severity, source)
            manifest_path = self._generate(suffix)
            result = self._validate(
                manifest_path,
                revision=REVISION,
            )
            self.assertEqual(result["status"], "failed")
            self.assertTrue(
                {"open_s0_s1", "undisposed_s2"} & set(result["failed_checks"])
            )

    def test_migration_cutoff_failure_is_derived_not_self_reported(self) -> None:
        self._rehash_source(
            self.migration,
            lambda value: value.update({"accepted_legacy_fixture_count": 1}),
        )
        result = self._validate(
            self._generate(),
            revision=REVISION,
        )
        self.assertEqual(result["status"], "failed")
        self.assertIn("migration_cutoff", result["failed_checks"])

    def test_caller_selected_migration_subset_cannot_close_g6(self) -> None:
        self._rehash_source(
            self.migration,
            lambda value: value["checks"].pop(),
        )
        manifest_path = self._generate()
        with self.assertRaisesRegex(ValueError, "migration check membership"):
            self._validate(manifest_path, revision=REVISION)

    def test_empty_severity_ledger_requires_complete_source_reviews(self) -> None:
        self._rehash_source(
            self.severity,
            lambda value: value.update({
                "findings": [],
                "source_reviews": [],
            }),
        )
        with self.assertRaisesRegex(ValueError, "source review membership"):
            self._generate()

    def test_bilingual_document_hash_or_fact_drift_fails_closed(self) -> None:
        manifest_path = self._generate()
        self.zh_doc.write_text("文档事实已被删除\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "fact drifted"):
            self._validate(manifest_path, revision=REVISION)

    def test_new_ignored_runtime_file_invalidates_hygiene_evidence(self) -> None:
        manifest_path = self._generate()
        with (self.repo / ".gitignore").open("a", encoding="utf-8") as handle:
            handle.write("unexpected.cache\n")
        (self.repo / "unexpected.cache").write_text("new runtime state\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "independent recomputation"):
            self._validate(manifest_path, revision=REVISION)

    def test_unavailable_external_capability_does_not_close_g6(self) -> None:
        self._rehash_source(
            self.external,
            lambda value: value.update({
                "state": "unavailable",
            }),
        )
        self._rehash_source(
            self.external_probe,
            lambda value: value.update({
                "outcome": "unavailable",
                "reason": "probe failed without an external blocker",
            }),
        )
        result = self._validate(
            self._generate(),
            revision=REVISION,
        )
        self.assertEqual(result["status"], "failed")
        self.assertIn("external_capabilities", result["failed_checks"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.agent_live import run_product_acceptance as acceptance


REVISION = {"commit": "a" * 40, "worktree_hash": "b" * 64}


class ProductAcceptanceGateWiringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.phase_root = self.root / "phase8"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_display_path_accepts_repository_and_external_outputs(self) -> None:
        repository_path = acceptance.REPO_ROOT / ".agent" / "evidence.json"
        external_path = self.root / "evidence.json"

        self.assertEqual(
            acceptance._display_path(repository_path),
            ".agent/evidence.json",
        )
        self.assertEqual(
            acceptance._display_path(external_path),
            str(external_path.resolve()),
        )

    def test_failed_g0_is_not_masked_by_later_incomplete_phase(self) -> None:
        self.assertEqual(
            acceptance._aggregate_status(
                requested_supported=True,
                g0_status="failed",
                phase_statuses=("passed", "incomplete"),
            ),
            "failed",
        )
        self.assertEqual(
            acceptance._aggregate_status(
                requested_supported=True,
                g0_status="passed",
                phase_statuses=("passed", "incomplete"),
            ),
            "incomplete",
        )
        self.assertEqual(
            acceptance._aggregate_status(
                requested_supported=True,
                g0_status="passed",
                phase_statuses=("passed", "passed"),
            ),
            "passed",
        )

    def test_g3_loads_only_the_declared_immutable_manifest(self) -> None:
        provider_path = self.phase_root / "g3" / "provider.json"
        manifest_path = (
            self.phase_root / "g3" / "evidence-set" / "manifest.json"
        )
        provider_path.parent.mkdir(parents=True)
        manifest_path.parent.mkdir(parents=True)
        provider_path.write_text('{"provider_id":"provider"}', encoding="utf-8")
        manifest_path.write_text('{"manifest":"typed"}', encoding="utf-8")
        loaded = {
            "obligation_count": 60,
            "exact_count": 15,
            "open_count": 45,
            "manifest_hash": "c" * 64,
        }

        with patch.object(
            acceptance,
            "load_retained_regression_evidence_set",
            return_value=loaded,
        ) as loader:
            result = acceptance._phase8_g3_gate(
                phase_root=self.phase_root,
                obligations=[{"obligation_id": "one"}],
                revision=REVISION,
            )

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["passed"], 60)
        loader.assert_called_once_with(
            manifest_path,
            provider={"provider_id": "provider"},
            obligations=[{"obligation_id": "one"}],
            revision=REVISION,
        )

    def test_g3_missing_or_rejected_manifest_fails_closed(self) -> None:
        missing = acceptance._phase8_g3_gate(
            phase_root=self.phase_root,
            obligations=[],
            revision=REVISION,
        )
        self.assertEqual(missing["status"], "incomplete")
        self.assertFalse(missing["complete"])

        provider_path = self.phase_root / "g3" / "provider.json"
        manifest_path = (
            self.phase_root / "g3" / "evidence-set" / "manifest.json"
        )
        provider_path.parent.mkdir(parents=True)
        manifest_path.parent.mkdir(parents=True)
        provider_path.write_text("{}", encoding="utf-8")
        manifest_path.write_text("{}", encoding="utf-8")
        with patch.object(
            acceptance,
            "load_retained_regression_evidence_set",
            side_effect=ValueError("hash drift"),
        ):
            rejected = acceptance._phase8_g3_gate(
                phase_root=self.phase_root,
                obligations=[],
                revision=REVISION,
            )
        self.assertEqual(rejected["status"], "failed")
        self.assertIn("hash drift", rejected["reason"])

    def test_g3_controller_contains_no_evidence_glob(self) -> None:
        source = inspect.getsource(acceptance._phase8_source)
        g3_section = source.split("g4_summary", 1)[0]
        self.assertNotIn("_json_files", g3_section)
        self.assertIn("_phase8_g3_gate", g3_section)

    def test_g6_runs_typed_generator_then_independent_validator(self) -> None:
        config_path = self._write_g6_config()
        events: list[str] = []

        def generate(**_kwargs):
            events.append("generate")
            return self.root / "generated-manifest.json"

        def validate(_path, *, revision):
            events.append("validate")
            self.assertEqual(revision, REVISION)
            return {
                "gate": "G6",
                "status": "passed",
                "manifest_hash": "d" * 64,
                "failed_checks": [],
            }

        with (
            patch.object(
                acceptance,
                "generate_product_review_evidence",
                side_effect=generate,
            ) as generator,
            patch.object(
                acceptance,
                "validate_product_review_evidence",
                side_effect=validate,
            ) as validator,
        ):
            result = acceptance._phase8_product_review(
                revision=REVISION,
                phase_root=self.phase_root,
                prerequisites_passed=True,
            )

        self.assertEqual(events, ["generate", "validate"])
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["generation"], "generated")
        generator.assert_called_once()
        validator.assert_called_once()
        self.assertEqual(result["config"], str(config_path))

    def test_g6_runs_a_fresh_generator_for_every_acceptance(self) -> None:
        self._write_g6_config()
        with (
            patch.object(
                acceptance,
                "generate_product_review_evidence",
                side_effect=(
                    self.root / "manifest-one.json",
                    self.root / "manifest-two.json",
                ),
            ) as generator,
            patch.object(
                acceptance,
                "validate_product_review_evidence",
                return_value={"gate": "G6", "status": "passed"},
            ) as validator,
        ):
            first = acceptance._phase8_product_review(
                revision=REVISION,
                phase_root=self.phase_root,
                prerequisites_passed=True,
            )
            second = acceptance._phase8_product_review(
                revision=REVISION,
                phase_root=self.phase_root,
                prerequisites_passed=True,
            )
        self.assertEqual(generator.call_count, 2)
        self.assertEqual(validator.call_count, 2)
        self.assertNotEqual(first["manifest"], second["manifest"])
        self.assertEqual(first["generation"], "generated")
        self.assertEqual(second["generation"], "generated")

    def test_g6_missing_invalid_or_rejected_evidence_fails_closed(self) -> None:
        missing = acceptance._phase8_product_review(
            revision=REVISION,
            phase_root=self.phase_root,
            prerequisites_passed=True,
        )
        self.assertEqual(missing["status"], "incomplete")

        config_path = self._write_g6_config()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["unexpected"] = True
        config_path.write_text(json.dumps(config), encoding="utf-8")
        invalid = acceptance._phase8_product_review(
            revision=REVISION,
            phase_root=self.phase_root,
            prerequisites_passed=True,
        )
        self.assertEqual(invalid["status"], "failed")
        self.assertIn("schema", invalid["reason"])

        config.pop("unexpected")
        config_path.write_text(json.dumps(config), encoding="utf-8")
        with (
            patch.object(
                acceptance,
                "generate_product_review_evidence",
                return_value=self.root / "manifest.json",
            ),
            patch.object(
                acceptance,
                "validate_product_review_evidence",
                return_value={
                    "gate": "G6",
                    "status": "failed",
                    "failed_checks": ["open_s0_s1"],
                },
            ),
        ):
            rejected = acceptance._phase8_product_review(
                revision=REVISION,
                phase_root=self.phase_root,
                prerequisites_passed=True,
            )
        self.assertEqual(rejected["status"], "failed")
        self.assertEqual(rejected["failed_checks"], ["open_s0_s1"])

        with (
            patch.object(
                acceptance,
                "generate_product_review_evidence",
                return_value=self.root / "manifest.json",
            ),
            patch.object(
                acceptance,
                "validate_product_review_evidence",
                return_value={"status": "passed"},
            ),
        ):
            malformed = acceptance._phase8_product_review(
                revision=REVISION,
                phase_root=self.phase_root,
                prerequisites_passed=True,
            )
        self.assertEqual(malformed["status"], "failed")
        self.assertIn("invalid decision", malformed["reason"])

    def test_g6_controller_has_no_legacy_boolean_review_policy(self) -> None:
        source = inspect.getsource(acceptance._phase8_product_review)
        for legacy_field in (
            "migration_cutoff_verified",
            "bilingual_docs_verified",
            "open_findings",
            "review_valid",
            "product-review.json",
        ):
            self.assertNotIn(legacy_field, source)
        self.assertIn("generate_product_review_evidence", source)
        self.assertIn("validate_product_review_evidence", source)

    def _write_g6_config(self) -> Path:
        path = self.phase_root / "g6" / "generator-config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({
                "revision": REVISION,
                "planner_receipt_paths": ["/evidence/planner.json"],
                "migration_cutoff_path": "/evidence/migration.json",
                "bilingual_docs_path": "/evidence/docs.json",
                "runtime_hygiene_path": "/evidence/hygiene.json",
                "external_capability_paths": ["/evidence/external.json"],
                "severity_ledger_path": "/evidence/severity.json",
            }),
            encoding="utf-8",
        )
        return path


if __name__ == "__main__":
    unittest.main()

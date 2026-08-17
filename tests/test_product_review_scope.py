from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.agent_live.product_review_scope import compute_product_review_scope


class ProductReviewScopeTest(unittest.TestCase):
    def test_repository_scope_recomputes_fixed_denominators(self) -> None:
        root = Path(__file__).resolve().parents[1]
        scope = compute_product_review_scope(root)

        self.assertEqual(scope["documentation"]["pair_denominator"], 13)
        self.assertEqual(scope["documentation"]["exemption_denominator"], 0)
        self.assertEqual(scope["documentation"]["fact_denominator"], 12)
        self.assertEqual(scope["migration"]["check_denominator"], 25)
        self.assertEqual(scope["external_capabilities"]["denominator"], 1)
        self.assertEqual(scope["severity"]["denominator"], 5)
        self.assertEqual(scope["shell_gates"]["denominator"], 51)
        self.assertEqual(scope["fixture_provenance"]["fixture_denominator"], 207)
        self.assertEqual(scope["fixture_provenance"]["workload_denominator"], 184)
        self.assertEqual(
            scope["fixture_provenance"]["strict_authenticity"],
            "scoped_limitation",
        )

    def test_unclassified_document_fails_closed(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = (
            root
            / "tests"
            / "agent_live"
            / "fixtures"
            / "product_review_scope.json"
        )
        with tempfile.TemporaryDirectory(dir=root) as tmpdir:
            manifest = json.loads(source.read_text(encoding="utf-8"))
            manifest["documentation"]["paired_basenames"].remove(
                "agent-cli-verification-guide.md"
            )
            path = Path(tmpdir) / "scope.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "membership drift"):
                compute_product_review_scope(root, path)

    def test_required_fact_marker_drift_fails_closed(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = (
            root
            / "tests"
            / "agent_live"
            / "fixtures"
            / "product_review_scope.json"
        )
        manifest = json.loads(source.read_text(encoding="utf-8"))
        manifest["documentation"]["required_facts"][0]["en_marker"] = (
            "caller-selected marker"
        )
        with tempfile.TemporaryDirectory(dir=root) as tmpdir:
            path = Path(tmpdir) / "scope.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fact drifted"):
                compute_product_review_scope(root, path)

    def test_schema_version_drift_fails_closed(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = (
            root
            / "tests"
            / "agent_live"
            / "fixtures"
            / "product_review_scope.json"
        )
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["migration"]["current_schema_version"] = 17
        with tempfile.TemporaryDirectory(dir=root) as tmpdir:
            path = Path(tmpdir) / "scope.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "migration membership"):
                compute_product_review_scope(root, path)


if __name__ == "__main__":
    unittest.main()

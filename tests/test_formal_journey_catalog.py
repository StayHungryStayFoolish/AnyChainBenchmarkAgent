from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from agent.harness.runtime_identity import repository_revision
from tests.agent_live.batch_orchestrator import (
    freeze_batch_manifest,
    validate_frozen_manifest,
)
from tests.agent_live.formal_journey_catalog import (
    FORMAL_JOURNEY_VERIFIER_REGISTRY,
    formal_journey_definitions,
)
from tests.agent_live.generate_harness_coverage_ledger import build_ledger


class FormalJourneyCatalogTest(unittest.TestCase):
    def test_catalog_has_eight_open_journeys_and_versioned_verifiers(self) -> None:
        rows = formal_journey_definitions()
        self.assertEqual(len(rows), 8)
        self.assertEqual(len({row["journey"]["journey_id"] for row in rows}), 8)
        self.assertTrue(FORMAL_JOURNEY_VERIFIER_REGISTRY.registry_id)
        for row in rows:
            self.assertEqual(row["lane"], "journey")
            encoded = json.dumps(row, ensure_ascii=False)
            self.assertNotIn("user_message", encoded)
            self.assertNotIn("transcript", encoded)

    def test_formal_manifest_requires_exactly_24_edge_and_8_journey_lanes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            subprocess.run(("git", "init", "-q"), cwd=root, check=True)
            subprocess.run(("git", "config", "user.email", "test@example.com"), cwd=root, check=True)
            subprocess.run(("git", "config", "user.name", "Test"), cwd=root, check=True)
            (root / ".gitignore").write_text(".agent/\n", encoding="utf-8")
            (root / "tracked.txt").write_text("frozen\n", encoding="utf-8")
            subprocess.run(("git", "add", "."), cwd=root, check=True)
            subprocess.run(("git", "commit", "-qm", "fixture"), cwd=root, check=True)
            revision = repository_revision(root)
            ledger = build_ledger(revision=revision)
            edge = next(
                row for row in ledger["edges"]
                if row.get("executable_scenario_ids")
            )
            targets = root / ".agent" / "targets"
            targets.mkdir(parents=True)
            for index in range(1, 25):
                (targets / f"{index:02d}.json").write_text(json.dumps({
                    "lane": "edge",
                    "targets": [{
                        "target_id": f"edge-{index}",
                        "edge_key": edge["edge_key"],
                        "persona": "operator",
                        "goal": "exercise one authoritative edge",
                        "scenario_id": edge["executable_scenario_ids"][0],
                    }],
                }), encoding="utf-8")
            for offset, row in enumerate(formal_journey_definitions(), start=25):
                (targets / f"{offset:02d}.json").write_text(
                    json.dumps(row, ensure_ascii=False), encoding="utf-8"
                )
            manifest = freeze_batch_manifest(
                repo_root=root,
                targets_dir=targets,
                manifest_path=root / ".agent" / "manifest.json",
                runtime_base=root / ".agent" / "runtime",
                formal_profile=True,
                worker_runtime="linux",
            )
            self.assertEqual([row.lane for row in manifest.shards].count("edge"), 24)
            self.assertEqual([row.lane for row in manifest.shards].count("journey"), 8)
            journey = manifest.shards[24]
            self.assertIn("tests.agent_live.journey_simulator_bridge", journey.command)
            self.assertEqual(
                journey.verifier_registry_id,
                FORMAL_JOURNEY_VERIFIER_REGISTRY.registry_id,
            )
            escaped = replace(
                manifest,
                shards=(
                    replace(manifest.shards[0], command=("python", "worker.py")),
                    *manifest.shards[1:],
                ),
            )
            with self.assertRaisesRegex(
                ValueError, "escaped its Linux worker implementation"
            ):
                validate_frozen_manifest(escaped)


if __name__ == "__main__":
    unittest.main()

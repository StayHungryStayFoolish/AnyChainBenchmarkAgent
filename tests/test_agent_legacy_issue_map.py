"""Contracts for the retired Agent issue/repair-document migration map."""

from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class AgentLegacyIssueMapTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from tests.agent_live import legacy_issue_map

        cls.mapping = legacy_issue_map

    def test_map_is_complete_and_self_validating(self) -> None:
        self.assertEqual(self.mapping.validate(), [])
        self.assertEqual(
            {item.legacy_id for item in self.mapping.KNOWN_ISSUES_A},
            {f"A.{number}" for number in range(1, 72)},
        )
        self.assertEqual(
            {item.legacy_id for item in self.mapping.REPAIR_PLAN_PHASES},
            {f"RP.{number}" for number in range(1, 31)},
        )
        mapped = {item.legacy_id for item in self.mapping.ALL_ITEMS}
        self.assertEqual(
            {
                item.legacy_id
                for item in self.mapping.KNOWN_ISSUES_OPEN
                if item.legacy_id.startswith("B.")
            },
            {
                *(f"B.{number}" for number in range(2, 16)),
                "B.19",
                "B.24",
                "B.26",
            },
        )
        self.assertTrue({f"C.{number}" for number in range(1, 8)} <= mapped)
        self.assertTrue({"D.1", "D.2", "D.3"} <= mapped)

    def test_non_open_dispositions_have_exact_current_test_locations(self) -> None:
        payload = self.mapping.report()
        for item in payload["items"]:
            if item["disposition"] not in {"guarded", "superseded"}:
                continue
            self.assertTrue(item["evidence"], item["legacy_id"])
            self.assertEqual(
                len(item["evidence_locations"]),
                len(item["evidence"]),
                f"{item['legacy_id']} evidence must resolve uniquely",
            )
            self.assertTrue(
                all("tests/" in location and "::test_" in location for location in item["evidence_locations"]),
                item["legacy_id"],
            )

    def test_groups_and_owner_vocabulary_match_current_architecture(self) -> None:
        from agent.workflows.group_registry import GROUP_OWNER

        boundary_owners = {"architecture", "coordinator", "llm_boundary", "terminal", "test_harness"}
        allowed_owners = set(GROUP_OWNER.values()) | boundary_owners
        for item in self.mapping.ALL_ITEMS:
            self.assertTrue(item.groups, item.legacy_id)
            self.assertLessEqual(set(item.groups), set(GROUP_OWNER), item.legacy_id)
            self.assertIn(item.owner, allowed_owners, item.legacy_id)

    def test_long_lived_docs_use_current_registry_and_ownership_boundary(self) -> None:
        from agent.workflows.group_registry import GROUP_OWNER

        documents = (
            REPO_ROOT / "docs" / "en" / "agent-handoff-product-verification.md",
            REPO_ROOT / "docs" / "en" / "adk-agent-architecture.md",
            REPO_ROOT / "docs" / "zh" / "adk-agent-architecture.md",
        )
        for path in documents:
            text = path.read_text(encoding="utf-8")
            self.assertIn("agent/workflows/group_registry.py", text, str(path))
            self.assertIn("tests/agent_live/legacy_issue_map.py", text, str(path))
            self.assertNotIn("agent/harness/groups.py", text, str(path))
            self.assertNotIn("agent/harness/nodes/", text, str(path))

        en = documents[1].read_text(encoding="utf-8")
        zh = documents[2].read_text(encoding="utf-8")
        for fact in ("20", "8", "LangGraph", "google_search"):
            self.assertIn(fact, en)
            self.assertIn(fact, zh)
        for position, (group, owner) in enumerate(GROUP_OWNER.items(), start=1):
            row = f"| {position} | `{group}` | `{owner}` |"
            self.assertIn(row, en, group)
            self.assertIn(row, zh, group)

    def test_long_lived_docs_do_not_restore_old_adk_ownership(self) -> None:
        documents = (
            REPO_ROOT / "docs" / "en" / "anychain-agent-ai-work-gate.md",
            REPO_ROOT / "docs" / "zh" / "anychain-agent-ai-work-gate.md",
            REPO_ROOT / "docs" / "en" / "secondary-development-guide.md",
            REPO_ROOT / "docs" / "zh" / "secondary-development-guide.md",
            REPO_ROOT / "docs" / "en" / "agent-cli-verification-guide.md",
        )
        forbidden = (
            "ADK compatibility bridge",
            "optional model/tool bridge",
            "ADK-native Agent runtime",
        )
        for path in documents:
            text = path.read_text(encoding="utf-8")
            self.assertIn("LangGraph Harness", text, str(path))
            self.assertIn("google_search", text, str(path))
            for stale_claim in forbidden:
                self.assertNotIn(stale_claim, text, str(path))

    def test_retired_dated_documents_stay_deleted(self) -> None:
        for name in (
            "2026-07-10-langgraph-harness-repair-plan.md",
            "2026-07-11-agent-known-issues.md",
        ):
            self.assertFalse((REPO_ROOT / ".agent" / "task-docs" / name).exists())


if __name__ == "__main__":
    unittest.main()

"""Public-repository marker scanner contract."""

from __future__ import annotations

import unittest

from tools.check_public_repo_markers import MARKER_RE


class PublicRepoMarkerTest(unittest.TestCase):
    def test_product_proposal_and_supported_provider_prose_are_public(self) -> None:
        allowed = (
            "Review the typed configuration proposal before applying it.",
            "DeepSeek, OpenAI, Claude, and Gemini are supported providers.",
        )

        for line in allowed:
            with self.subTest(line=line):
                self.assertIsNone(MARKER_RE.search(line))

    def test_internal_execution_markers_remain_rejected(self) -> None:
        rejected = (
            "ask a sub" + "agent to do this",
            "this is a hard" + " gate",
            "run fix" + "_wave before pre-" + "S0",
        )

        for line in rejected:
            with self.subTest(line=line):
                self.assertIsNotNone(MARKER_RE.search(line))


if __name__ == "__main__":
    unittest.main()

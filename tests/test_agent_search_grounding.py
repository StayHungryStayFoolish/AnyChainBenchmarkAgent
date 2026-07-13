"""Regression tests for `agent.llm.search_grounding`.

Covers the exact bug found via live dual-AI chaos analysis: the harness
(`agent/harness/groups.py`) reads `state["web_research"]["google_search_available"]`,
but the producer (formerly `agent/adk_app/tools/web_research.py`) emitted the
key `enabled` — so the branch could never see a true value, even with a
correctly configured Gemini + google_search setup. `WebResearchStatus`'s
field must stay named `google_search_available` on both sides of that
contract.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch


class SearchGroundingTest(unittest.TestCase):
    def test_web_research_status_dict_key_matches_harness_contract(self) -> None:
        from agent.llm.config import LLMConfig
        from agent.llm.search_grounding import web_research_status

        deepseek_config = LLMConfig(provider="deepseek", model="deepseek-chat", deepseek_api_key_present=True)
        status = web_research_status(deepseek_config)
        payload = status.as_dict()

        self.assertIn("google_search_available", payload)
        self.assertNotIn("enabled", payload)
        self.assertFalse(payload["google_search_available"])

    def test_run_google_search_grounding_returns_unavailable_for_non_gemini_provider(self) -> None:
        from agent.llm.config import LLMConfig
        from agent.llm.search_grounding import run_google_search_grounding

        deepseek_config = LLMConfig(provider="deepseek", model="deepseek-chat", deepseek_api_key_present=True)
        result = run_google_search_grounding("solana official node client", config=deepseek_config)

        self.assertFalse(result.available)
        self.assertEqual(result.query, "solana official node client")
        self.assertTrue(result.error)

    def test_run_google_search_grounding_degrades_gracefully_on_adk_failure(self) -> None:
        from agent.llm.config import LLMConfig
        from agent.llm.search_grounding import run_google_search_grounding

        gemini_config = LLMConfig(
            provider="gemini",
            model="gemini-2.5-flash",
            auth_mode="api_key",
            gemini_api_key_present=True,
        )
        with patch("agent.llm.search_grounding.web_research_status") as mocked_status:
            from agent.llm.search_grounding import WebResearchStatus

            mocked_status.return_value = WebResearchStatus(True, "gemini", "gemini-2.5-flash", "adk_google_search", "enabled")
            result = run_google_search_grounding("bsc official node client", config=gemini_config)

        # Real google-adk is not guaranteed to be importable in every test
        # environment; either a clean import failure or a real (mocked-away)
        # runner failure must degrade to a typed result, never raise.
        self.assertFalse(result.available)
        self.assertTrue(result.error)

    def test_get_google_search_tools_empty_when_unavailable(self) -> None:
        from agent.llm.config import LLMConfig
        from agent.llm.search_grounding import get_google_search_tools

        deepseek_config = LLMConfig(provider="deepseek", model="deepseek-chat", deepseek_api_key_present=True)
        self.assertEqual(get_google_search_tools(deepseek_config), [])


if __name__ == "__main__":
    unittest.main()

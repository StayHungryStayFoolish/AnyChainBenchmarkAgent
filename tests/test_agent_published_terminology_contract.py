"""Published Agent terminology and generated-path ownership contracts."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.diagnostics.adk_status import adk_status
from agent.diagnostics.doctor import run_doctor
from agent.harness.response_messages.orientation import MESSAGES as ORIENTATION_MESSAGES
from agent.harness.domains.orientation import consultation_fragment
from agent.harness.response_catalog import render_fragment
from agent.harness.state import new_state
from agent.knowledge.framework_capabilities import load_framework_capabilities
from agent.knowledge.framework_context import load_framework_context
from agent.knowledge.gap_analyzer import onboarding_plan
from agent.llm.config import LLMConfig
from agent.terminal.language import t
from agent.tools.schema import tool_schema


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_CORE_OWNERSHIP = ("ADK Agent", "ADK terminal", "ADK runtime")


class PublishedAgentTerminologyContractTests(unittest.TestCase):
    def test_terminal_labels_adk_as_optional_search_not_core_runtime(self) -> None:
        for language in ("en", "zh"):
            rendered = t(language, "adk", status="google-adk is not installed")
            self.assertIn("Gemini", rendered)
            self.assertIn("search", rendered.lower())
            self.assertNotIn("ADK runtime", rendered)

    def test_empty_model_output_does_not_assign_response_ownership_to_adk(self) -> None:
        for language in ("en", "zh"):
            rendered = t(language, "unknown")
            self.assertNotIn("ADK", rendered)
            self.assertTrue("model" in rendered.lower() or "模型" in rendered)

    def test_missing_optional_adk_does_not_report_core_runtime_failure(self) -> None:
        with patch("agent.diagnostics.adk_status.is_adk_available", return_value=False):
            rendered = adk_status().reason
        self.assertIn("optional", rendered.lower())
        self.assertIn("core", rendered.lower())
        self.assertIn("--with-google-search", rendered)
        self.assertNotIn("ADK runtime", rendered)

    def test_structured_doctor_uses_current_agent_ownership(self) -> None:
        config = LLMConfig(provider="openai", model="test-model")
        discovery = {
            "cloud": {},
            "deployment": {},
            "host": {},
            "network": {},
            "disks": {},
            "dependencies": {"missing_required": [], "missing_optional": []},
            "warnings": [],
        }
        with (
            patch("agent.diagnostics.doctor.load_agent_environment", return_value={}),
            patch("agent.diagnostics.doctor.load_llm_config", return_value=config),
            patch(
                "agent.diagnostics.doctor.load_framework_capabilities",
                return_value={
                    "chain_count": 0,
                    "family_count": 0,
                    "unique_rpc_method_count": 0,
                    "fake_node": {"fixture_file_count": 0},
                },
            ),
        ):
            report = run_doctor(discovery)
        published = json.dumps(
            {"warnings": report["warnings"], "next_actions": report["next_actions"]},
            ensure_ascii=False,
        )
        for phrase in FORBIDDEN_CORE_OWNERSHIP:
            self.assertNotIn(phrase, published)
        self.assertIn("AnyChain Agent", published)

    def test_model_facing_framework_context_names_langgraph_runtime(self) -> None:
        with (
            patch(
                "agent.knowledge.framework_context.load_framework_capabilities",
                return_value={"fake_node": {}},
            ),
            patch(
                "agent.knowledge.framework_context.load_or_build_framework_index",
                return_value={},
            ),
            patch("agent.knowledge.framework_context._doc_index", return_value=[]),
        ):
            context = load_framework_context(ROOT)
        runtime = str(context["identity"]["agent_runtime"])
        self.assertIn("LangGraph Harness", runtime)
        for phrase in FORBIDDEN_CORE_OWNERSHIP:
            self.assertNotIn(phrase, runtime)

    def test_install_tool_schema_distinguishes_core_runtime_and_compatibility_path(self) -> None:
        install = next(
            item["function"]
            for item in tool_schema()["tools"]
            if item["function"]["name"] == "install_dependencies"
        )
        properties = install["parameters"]["properties"]
        runtime_description = properties["include_agent_runtime"]["description"]
        venv_description = properties["adk_venv"]["description"]
        self.assertIn("core Agent runtime", runtime_description)
        self.assertNotIn("Reinstall/update Google ADK", runtime_description)
        self.assertIn("compatibility", venv_description.lower())

    def test_unknown_chain_onboarding_uses_existing_template_starter(self) -> None:
        steps = onboarding_plan(
            "example-chain",
            [],
            [{"type": "chain_template", "severity": "blocker", "message": "missing"}],
        )
        published = "\n".join(steps)
        expected = "config/chain_template.json.bak"
        self.assertIn(expected, published)
        self.assertNotIn("config/chains/chain_template.json.bak", published)
        self.assertTrue((ROOT / expected).is_file())

    def test_bsc_native_metrics_stay_inside_sync_observe_contract(self) -> None:
        capabilities = load_framework_capabilities(ROOT)
        sync_observe = next(
            mode for mode in capabilities["run_modes"] if mode["id"] == "sync_observe"
        )
        self.assertIn("BSC v1.7.x", " ".join(sync_observe["optional"]))
        self.assertIn("without RPC workload", sync_observe["purpose"])
        self.assertIn("Vegeta", sync_observe["purpose"])

        for language in ("en", "zh"):
            requirements = ORIENTATION_MESSAGES[
                "harness.orientation.consultation.requirements"
            ][language]
            comparison = ORIENTATION_MESSAGES[
                "harness.orientation.consultation.mode_comparison"
            ][language]
            self.assertIn("BSC v1.7.x", requirements)
            self.assertIn("Prometheus endpoint", requirements)
            self.assertIn("Vegeta", requirements)
            self.assertIn("BSC v1.7.x", comparison)
            self.assertIn("TPS", comparison)

    def test_client_metric_profiles_are_projected_from_structured_config(self) -> None:
        capabilities = load_framework_capabilities(ROOT)
        profiles = capabilities["client_metric_profiles"]
        bsc = next(profile for profile in profiles if profile["profile_id"] == "bsc_v1_7")

        self.assertEqual(bsc["chains"], ["bsc"])
        self.assertEqual(bsc["metrics_path"], "/debug/metrics/prometheus")
        self.assertIn('chain_mgasps{quantile="0.5"}', bsc["native_samples"])
        self.assertIn("TPS", bsc["report_kpis"])
        self.assertIn("empty_block_rate", bsc["report_kpis"])
        self.assertEqual(
            bsc["requirements"][0]["id"],
            "client_prometheus_endpoint_enabled",
        )
        self.assertTrue(bsc["requirements"][0]["zh"])
        self.assertTrue(bsc["missing_endpoint"]["workflow_continues"])
        self.assertTrue(bsc["missing_endpoint"]["shared_timeline_available"])
        self.assertIn("can continue", bsc["missing_endpoint"]["en"])
        self.assertIn("可以继续", bsc["missing_endpoint"]["zh"])

    def test_sync_observe_metric_consultation_uses_profile_inventory(self) -> None:
        state = new_state("bsc-metric-consultation", language="en")
        state["framework_summary"] = load_framework_capabilities(ROOT)

        fragment = consultation_fragment(
            state,
            {"topic": "sync_observe_metrics", "subject": "bsc"},
        )
        rendered = render_fragment(fragment, "en").text

        self.assertEqual(
            fragment.message_id,
            "harness.orientation.consultation.sync_observe_metrics_profile",
        )
        self.assertIn('chain_mgasps{quantile="0.5"}', rendered)
        self.assertIn("active block import", rendered)
        self.assertIn("TPS", rendered)
        self.assertIn("empty_block_rate", rendered)
        self.assertIn("does not run Vegeta", rendered)
        self.assertIn("can continue", rendered)
        self.assertIn("N/A", rendered)

    def test_sync_observe_metric_consultation_does_not_claim_other_evm_profiles(self) -> None:
        state = new_state("ethereum-metric-consultation", language="zh")
        state["framework_summary"] = load_framework_capabilities(ROOT)

        fragment = consultation_fragment(
            state,
            {"topic": "sync_observe_metrics", "subject": "ethereum"},
        )
        rendered = render_fragment(fragment, "zh").text

        self.assertEqual(
            fragment.message_id,
            "harness.orientation.consultation.sync_observe_metrics_generic",
        )
        self.assertIn("ethereum", rendered)
        self.assertIn("没有为该链注册", rendered)
        self.assertIn("不走 Vegeta", rendered)

        descriptive = consultation_fragment(
            state,
            {"topic": "sync_observe_metrics", "subject": "Ethereum Geth node"},
        )
        descriptive_text = render_fragment(descriptive, "en").text
        self.assertIn("Ethereum Geth node", descriptive_text)
        self.assertNotIn("`bsc`", descriptive_text)

        bsc_fragment = consultation_fragment(
            state,
            {"topic": "sync_observe_metrics", "subject": "bsc"},
        )
        bsc_zh = render_fragment(bsc_fragment, "zh").text
        self.assertIn("已启用客户端 Prometheus endpoint", bsc_zh)
        self.assertNotIn("enabled client Prometheus endpoint", bsc_zh)

    def test_sync_observe_metric_consultation_uses_confirmed_chain_subject(self) -> None:
        state = new_state("current-bsc-metric-consultation", language="en")
        state["framework_summary"] = load_framework_capabilities(ROOT)
        state["chain_identity"] = {
            "raw": "BNB",
            "canonical": "bsc",
            "status": "confirmed",
        }

        fragment = consultation_fragment(
            state,
            {"topic": "sync_observe_metrics"},
        )

        self.assertEqual(
            fragment.message_id,
            "harness.orientation.consultation.sync_observe_metrics_profile",
        )


if __name__ == "__main__":
    unittest.main()

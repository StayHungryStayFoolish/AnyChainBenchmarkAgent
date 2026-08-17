#!/usr/bin/env python3
"""Contracts for client-native Prometheus metric profiles."""

from __future__ import annotations

import math
import unittest

from monitoring.client_metric_profiles import (
    collect_client_metrics,
    parse_prometheus_samples,
    select_sample,
)


class PrometheusParserContractTest(unittest.TestCase):
    def test_selects_exact_summary_quantile_independent_of_order(self) -> None:
        samples = parse_prometheus_samples(
            """
            chain_mgasps {quantile="0.95"} 700
            chain_mgasps_count 12
            chain_mgasps{quantile="0.5"} 500
            """
        )

        self.assertEqual(
            select_sample(samples, "chain_mgasps", {"quantile": "0.5"}),
            500.0,
        )
        self.assertEqual(
            select_sample(samples, "chain_mgasps", {"quantile": "0.95"}),
            700.0,
        )

    def test_rejects_non_finite_and_ambiguous_exact_samples(self) -> None:
        samples = parse_prometheus_samples(
            """
            chain_head_block NaN
            chain_head_finalized +Inf
            chain_head_justified 100
            chain_head_justified 101
            """
        )

        self.assertIsNone(select_sample(samples, "chain_head_block"))
        self.assertIsNone(select_sample(samples, "chain_head_finalized"))
        self.assertIsNone(select_sample(samples, "chain_head_justified"))

    def test_decodes_only_prometheus_label_escapes(self) -> None:
        samples = parse_prometheus_samples(
            'client_info{note="BNB 节点",path="a\\\\b\\\"c\\nd"} 1\n'
        )

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].labels["note"], "BNB 节点")
        self.assertEqual(samples[0].labels["path"], 'a\\b"c\nd')


class BscMetricProfileContractTest(unittest.TestCase):
    def test_collects_complete_v171_contract_with_units(self) -> None:
        result = collect_client_metrics(
            "bsc",
            """
            # TYPE chain_mgasps summary
            chain_mgasps_count 42
            chain_mgasps{quantile="0.95"} 620.0
            chain_mgasps {quantile="0.5"} 567.23
            # TYPE chain_inserts summary
            chain_inserts_count 44
            chain_inserts{quantile="0.5"} 348190000
            chain_insert_txsize 1769
            chain_insert_gasused 140500000
            chain_head_block 5000
            chain_head_justified 4999
            chain_head_finalized 4998
            """,
        )

        self.assertEqual(result["client_metric_profile"], "bsc_v1_7")
        self.assertEqual(result["client_metric_quality"], "complete")
        self.assertEqual(result["client_import_mgas_per_sec_p50"], 567.23)
        self.assertTrue(math.isclose(result["client_block_insert_ms_p50"], 348.19))
        self.assertEqual(result["client_import_observation_count"], 42.0)
        self.assertEqual(result["client_inserted_blocks_count"], 44.0)
        self.assertEqual(result["client_block_tx_count"], 1769.0)
        self.assertEqual(result["client_block_gas_used"], 140500000.0)
        self.assertEqual(result["client_head_block"], 5000.0)
        self.assertEqual(result["client_justified_block"], 4999.0)
        self.assertEqual(result["client_finalized_block"], 4998.0)
        self.assertEqual(result["execution_metric_source"], 'chain_mgasps{quantile="0.5"}')
        self.assertEqual(result["execution_metric_status"], "available")

    def test_declared_but_unupdated_insert_gauge_is_not_a_bsc_fallback(self) -> None:
        result = collect_client_metrics("bsc", "chain_insert_mgasps 0\n")

        self.assertEqual(result["client_metric_profile"], "bsc_v1_7")
        self.assertEqual(result["client_metric_quality"], "unavailable")
        self.assertIsNone(result["client_import_mgas_per_sec_p50"])
        self.assertIsNone(result["execution_mgas_per_sec"])
        self.assertEqual(result["execution_metric_status"], "unavailable")

    def test_partial_bsc_scrape_preserves_available_raw_observations(self) -> None:
        result = collect_client_metrics(
            "bsc",
            """
            chain_insert_txsize 11
            chain_insert_gasused 900000
            chain_head_block 101
            chain_head_justified 100
            chain_head_finalized 99
            """,
        )

        self.assertEqual(result["client_metric_quality"], "partial")
        self.assertEqual(result["client_block_tx_count"], 11.0)
        self.assertEqual(result["client_block_gas_used"], 900000.0)
        self.assertIsNone(result["client_import_mgas_per_sec_p50"])

    def test_non_bsc_chain_does_not_inherit_bsc_profile(self) -> None:
        result = collect_client_metrics(
            "ethereum",
            'chain_mgasps{quantile="0.5"} 12.5\nchain_head_block 100\n',
        )

        self.assertEqual(result["client_metric_profile"], "none")
        self.assertEqual(result["client_metric_quality"], "unsupported")


if __name__ == "__main__":
    unittest.main()

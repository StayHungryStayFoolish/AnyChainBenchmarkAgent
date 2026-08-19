#!/usr/bin/env python3
"""Window analytics contracts for BSC sync-observe reports."""

from __future__ import annotations

import unittest

import pandas as pd

from visualization.bsc_sync_metrics import calculate_bsc_sync_kpis


class BscSyncMetricsTest(unittest.TestCase):
    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": [
                    "2026-08-17 00:00:00",
                    "2026-08-17 00:00:01",
                    "2026-08-17 00:00:02",
                    "2026-08-17 00:00:03",
                ],
                "client_metric_profile": ["bsc_v1_7"] * 4,
                "client_metric_quality": ["complete"] * 4,
                "client_head_block": [100, 101, 102, 103],
                "client_justified_block": [99, 100, 100, 102],
                "client_finalized_block": [98, 99, 99, 101],
                "client_block_tx_count": [5, 10, 20, 30],
                "client_block_gas_used": [500_000, 1_000_000, 2_000_000, 3_000_000],
                "client_block_insert_ms_p50": [100, 200, 300, 400],
                "client_import_mgas_per_sec_p50": [100, 200, 300, 400],
                "node_process_cpu_pct": [10, 20, 30, 40],
                "node_process_rss_mib": [1024, 1536, 2048, 2560],
                "node_process_memory_pct": [12.5, 18.75, 25.0, 31.25],
                "mem_used": [2048, 2048, 2048, 2048],
                "mem_usage": [25, 25, 25, 25],
            }
        )

    def test_calculates_complete_window_kpis(self) -> None:
        result = calculate_bsc_sync_kpis(self._frame())

        self.assertTrue(result["available"])
        self.assertEqual(result["quality"], "complete")
        self.assertEqual(result["duration_seconds"], 3.0)
        self.assertEqual(result["block_delta"], 3.0)
        self.assertEqual(result["block_rate"], 1.0)
        self.assertEqual(result["sample_coverage_pct"], 100.0)
        self.assertEqual(result["transactions"], 60.0)
        self.assertFalse(result["transactions_estimated"])
        self.assertEqual(result["tps"], 20.0)
        self.assertEqual(result["avg_tx_per_block"], 20.0)
        self.assertEqual(result["empty_block_rate_pct"], 0.0)
        self.assertEqual(result["avg_block_gas_used_mgas"], 2.0)
        self.assertEqual(result["block_gas_used_per_sec_mgas"], 2.0)
        self.assertEqual(result["avg_gas_per_tx"], 100_000.0)
        self.assertEqual(result["block_insert_ms_p50"], 300.0)
        self.assertEqual(result["import_mgas_per_sec_p50"], 300.0)
        self.assertEqual(result["justified_lag_p50"], 1.0)
        self.assertEqual(result["justified_lag_p90"], 2.0)
        self.assertEqual(result["justified_lag_p99"], 2.0)
        self.assertEqual(result["finalized_lag_p90"], 3.0)
        self.assertEqual(result["process_cpu_avg_pct"], 25.0)
        self.assertEqual(result["process_memory_rss_avg_gib"], 1.75)
        self.assertEqual(result["process_memory_avg_pct"], 21.875)
        self.assertEqual(result["memory_used_avg_gib"], 2.0)

    def test_marks_skipped_blocks_as_estimated(self) -> None:
        df = self._frame().iloc[[0, 3]].copy()
        result = calculate_bsc_sync_kpis(df)

        self.assertEqual(result["quality"], "estimated")
        self.assertAlmostEqual(result["sample_coverage_pct"], 100.0 / 3.0)
        self.assertEqual(result["transactions"], 90.0)
        self.assertTrue(result["transactions_estimated"])
        self.assertEqual(result["tps"], 30.0)

    def test_average_gas_per_transaction_keeps_zero_transaction_blocks(self) -> None:
        df = self._frame()
        df.loc[1, "client_block_tx_count"] = 0
        df.loc[1, "client_block_gas_used"] = 500_000
        result = calculate_bsc_sync_kpis(df)

        self.assertEqual(result["avg_gas_per_tx"], 5_500_000 / 50)
        self.assertAlmostEqual(result["empty_block_rate_pct"], 100.0 / 3.0)

    def test_empty_block_rate_is_unavailable_without_transaction_samples(self) -> None:
        df = self._frame()
        df.loc[1:, "client_block_tx_count"] = None

        result = calculate_bsc_sync_kpis(df)

        self.assertIsNone(result["empty_block_rate_pct"])

    def test_ignores_duplicate_scrapes_of_the_same_head(self) -> None:
        df = pd.concat([self._frame(), self._frame().iloc[[3]]], ignore_index=True)
        df.loc[4, "timestamp"] = "2026-08-17 00:00:04"
        result = calculate_bsc_sync_kpis(df)

        self.assertEqual(result["observed_imported_blocks"], 3)
        self.assertEqual(result["transactions"], 60.0)

    def test_returns_unavailable_for_old_or_non_bsc_csv(self) -> None:
        self.assertFalse(calculate_bsc_sync_kpis(pd.DataFrame())["available"])
        self.assertFalse(
            calculate_bsc_sync_kpis(pd.DataFrame({"client_metric_profile": ["none"]}))["available"]
        )


if __name__ == "__main__":
    unittest.main()

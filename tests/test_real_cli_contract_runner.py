"""Focused contracts for the deterministic real-CLI evidence producer."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.agent_live.execute_real_cli_contract_ledger import (
    _prepare_execution_case,
    _require_target_contract,
    _runtime_environment,
    eligible_edges,
    select_edges,
)
from tests.agent_live.dynamic_dual_ai_chaos import ChaosRunConfig
from tests.agent_live.generate_harness_coverage_ledger import (
    build_ledger,
    contract_variant_hash,
)
from tests.agent_live.harness_contract_scenarios import question_scenarios


class RealCliContractRunnerTest(unittest.TestCase):
    def test_runtime_environment_freezes_configured_provider_identity(self) -> None:
        root = Path("/workspace/.agent/real-cli/unit")
        config = ChaosRunConfig(
            repo_root=Path("/workspace"),
            command=("agent",),
            provider="deepseek",
            model="deepseek-v4-pro",
            runtime_root=root,
            runtime_root_in_process=root,
        )
        environment = _runtime_environment(config, root)

        self.assertEqual(environment["LLM_PROVIDER"], "deepseek")
        self.assertEqual(environment["LLM_MODEL"], "deepseek-v4-pro")
        self.assertEqual(
            environment["AGENT_CONFIG_LOCAL"],
            "config/agent_config.local.sh",
        )

    def test_unknown_setup_capability_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "unsupported execution-case setup"):
            _prepare_execution_case(("unknown",), {})

    def test_prepared_plan_setup_uses_product_plan_output(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "approved.json"
            product_plan = {"plan_id": "plan-product", "execution": {"command": ["true"]}}

            def prepare(**kwargs):
                generated = Path(kwargs["output_dir"]) / "plan-product.json"
                generated.write_text(json.dumps(product_plan), encoding="utf-8")
                return {
                    "status": "ok",
                    "data": {
                        "plan": product_plan,
                        "plan_file": str(generated),
                        "preflight": {"passed": True},
                    },
                }

            with patch(
                "agent.runners.benchmark_pipeline.prepare_benchmark_run",
                side_effect=prepare,
            ) as mocked:
                _prepare_execution_case(
                    ("prepared_real_node_plan",),
                    {"plan_file": str(target)},
                )

            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), product_plan)
            kwargs = mocked.call_args.kwargs
            self.assertEqual(kwargs["target_rpc_url"], "http://geth-dev:8545")
            self.assertEqual(kwargs["rpc_methods"], ["eth_blockNumber"])
            self.assertEqual(kwargs["duration_seconds"], 3)

    def test_prepared_plan_setup_rejects_blocked_product_plan(self) -> None:
        with tempfile.TemporaryDirectory() as root, patch(
            "agent.runners.benchmark_pipeline.prepare_benchmark_run",
            return_value={
                "status": "blocked",
                "data": {"plan": {}, "plan_file": "", "preflight": {"passed": False}},
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "approved executable plan"):
                _prepare_execution_case(
                    ("prepared_real_node_plan",),
                    {"plan_file": str(Path(root) / "approved.json")},
                )

    def test_eligible_rows_require_real_cli_without_dynamic_and_have_one_reviewed_case(self) -> None:
        ledger = {
            "edges": [
                {
                    "edge_key": "eligible",
                    "execution_case_ids": ["case-1"],
                    "evidence": {
                        "real_cli": {"required": True},
                        "dynamic_dual_ai": {"required": False},
                    },
                },
                {
                    "edge_key": "dynamic",
                    "execution_case_ids": ["case-2"],
                    "evidence": {
                        "real_cli": {"required": True},
                        "dynamic_dual_ai": {"required": True},
                    },
                },
                {
                    "edge_key": "unseeded",
                    "execution_case_ids": [],
                    "evidence": {
                        "real_cli": {"required": True},
                        "dynamic_dual_ai": {"required": False},
                    },
                },
            ]
        }
        self.assertEqual([edge["edge_key"] for edge in eligible_edges(ledger)], ["eligible"])

    def test_target_contract_must_match_question_and_contract_hash(self) -> None:
        scenario = next(item for item in question_scenarios("en") if item.scenario_id == "opening")
        edge = {
            "question_id": scenario.question["id"],
            "contract_hash": contract_variant_hash(scenario.question),
        }
        _require_target_contract(edge, scenario.question)
        with self.assertRaisesRegex(RuntimeError, "question"):
            _require_target_contract({**edge, "question_id": "other"}, scenario.question)
        with self.assertRaisesRegex(RuntimeError, "contract hash"):
            _require_target_contract({**edge, "contract_hash": "wrong"}, scenario.question)

    def test_shards_are_stable_disjoint_and_exhaustive(self) -> None:
        edges = [
            {
                "edge_key": f"edge-{index}",
                "execution_case_ids": [f"case-{index}"],
                "evidence": {
                    "real_cli": {"required": True},
                    "dynamic_dual_ai": {"required": False},
                },
            }
            for index in range(11)
        ]
        ledger = {"edges": edges}
        shards = [select_edges(ledger, shard_index=index, shard_count=4) for index in range(4)]
        observed = [str(edge["edge_key"]) for shard in shards for edge in shard]
        self.assertEqual(len(observed), len(set(observed)))
        self.assertEqual(set(observed), {str(edge["edge_key"]) for edge in edges})
        self.assertEqual(shards[0], select_edges(ledger, shard_index=0, shard_count=4))

    def test_invalid_shard_parameters_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 1"):
            select_edges({"edges": []}, shard_count=0)
        with self.assertRaisesRegex(ValueError, "shard_index"):
            select_edges({"edges": []}, shard_index=2, shard_count=2)

    def test_authoritative_ledger_exposes_implemented_runner_and_excludes_whitespace(self) -> None:
        ledger = build_ledger(revision={"commit": "test", "worktree_hash": "a" * 64})
        self.assertEqual(ledger["runner_contracts"]["real_cli"]["status"], "implemented")
        self.assertTrue(ledger["runner_contracts"]["real_cli"]["producer"])
        self.assertTrue(all(
            not edge["evidence"]["real_cli"]["required"]
            for edge in ledger["edges"]
            if edge["input_class"] == "empty_whitespace"
        ))
        fixed_required = [
            edge for edge in ledger["edges"]
            if edge["evidence"]["real_cli"]["required"]
            and not edge["evidence"]["dynamic_dual_ai"]["required"]
        ]
        self.assertTrue(fixed_required)
        self.assertTrue(all(len(edge["execution_case_ids"]) == 1 for edge in fixed_required))
        self.assertTrue(all(edge["expected_admitted"] is not None for edge in fixed_required if edge["edge_type"] == "manual_input"))


if __name__ == "__main__":
    unittest.main()

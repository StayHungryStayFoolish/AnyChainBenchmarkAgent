from __future__ import annotations

from copy import deepcopy
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.agent_live.coverage_evidence import content_hash
from tests.agent_live.retained_regression_obligations import (
    RETAINED_REGRESSION_OBLIGATION_COUNT,
    RETAINED_REGRESSION_VARIANTS,
    build_retained_regression_obligations,
    validate_retained_regression_obligations,
)
from tests.agent_live.runtime_checkpoint import reviewed_scenario


REPO_ROOT = Path(__file__).resolve().parents[1]
REVISION = {"commit": "a" * 40, "worktree_hash": "b" * 64}


class RetainedRegressionObligationCatalogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.obligations = build_retained_regression_obligations(
            repo_root=REPO_ROOT,
            revision=REVISION,
        )

    def test_builds_exactly_four_stable_obligations_per_sanitized_case(self) -> None:
        first = self.obligations
        second = build_retained_regression_obligations(
            repo_root=REPO_ROOT,
            revision=REVISION,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), RETAINED_REGRESSION_OBLIGATION_COUNT)
        self.assertEqual(len({row["obligation_id"] for row in first}), 60)
        self.assertEqual(
            {row["case_id"] for row in first},
            {f"RR-{index:03d}" for index in range(1, 16)},
        )
        for case_id in {row["case_id"] for row in first}:
            self.assertEqual(
                {row["variant"] for row in first if row["case_id"] == case_id},
                set(RETAINED_REGRESSION_VARIANTS),
            )

    def test_contracts_bind_revision_fixture_seed_and_machine_postconditions(self) -> None:
        for row in self.obligations:
            with self.subTest(obligation_id=row["obligation_id"]):
                self.assertEqual(row["revision_binding"]["revision"], REVISION)
                self.assertTrue(row["revision_binding"]["source_fixture"]["sanitized"])
                self.assertTrue(
                    row["revision_binding"]["source_fixture"][
                        "variant_contract_sha256"
                    ]
                )
                self.assertTrue(row["seed_contract"]["scenario_state_fingerprint"])
                self.assertTrue(row["verifier_contract"]["required_postcondition_ids"])
                self.assertFalse(
                    row["verifier_contract"]["fixture_expected_text_is_verifier"]
                )
                self.assertNotIn("expected", row["verifier_contract"])
                self.assertNotIn("expected_text", row["verifier_contract"])
                self.assertEqual(row["execution_status"], "not_run")
                stimulus = row["stimulus_contract"]
                self.assertEqual(
                    stimulus["source_contract_hash"],
                    content_hash(stimulus["source_contract"]),
                )
                self.assertEqual(
                    stimulus["variant_contract"]["variant"],
                    row["variant"],
                )
                if row["variant"] == "exact":
                    self.assertTrue(row["stimulus_contract"]["turns"])
                else:
                    self.assertNotIn("turns", row["stimulus_contract"])
                    self.assertTrue(
                        row["stimulus_contract"]["constraints"][
                            "read_each_complete_agent_response"
                        ]
                    )

    def test_validation_rejects_stale_immutable_variant_contract(self) -> None:
        rows = list(deepcopy(self.obligations))
        rows[0]["stimulus_contract"]["source_contract"][
            "source_turns_hash"
        ] = "0" * 64
        with self.assertRaisesRegex(ValueError, "contract semantics|immutable stimulus"):
            validate_retained_regression_obligations(rows, revision=REVISION)

        rows = list(deepcopy(self.obligations))
        open_row = next(row for row in rows if row["variant"] == "negative")
        open_row["stimulus_contract"]["variant_contract"][
            "relation"
        ] = "isomorphic_meaning"
        with self.assertRaisesRegex(ValueError, "variant contract semantics"):
            validate_retained_regression_obligations(rows, revision=REVISION)

    def test_reviewed_seeds_satisfy_network_and_backtrack_preconditions(self) -> None:
        network = reviewed_scenario("network_interface")
        self.assertEqual(network.question["id"], "network_interface")
        self.assertEqual(
            [option["value"] for option in network.question["options"]],
            ["eth0"],
        )
        self.assertEqual(
            network.seed_state["discovery"]["network"]["default_interface"],
            "eth0",
        )

        qps = reviewed_scenario("qps_mode")
        self.assertEqual(qps.question["id"], "benchmark_mode")
        self.assertEqual(qps.seed_state["group_history"], ["workload_rpc"])
        self.assertEqual(qps.seed_state["chain_identity"]["canonical"], "bsc")
        self.assertEqual(qps.seed_state["rpc_mode"], "single")
        self.assertTrue(qps.seed_state["workload"]["confirmed"])
        self.assertEqual(
            qps.seed_state["confirmed_config"]["NETWORK_INTERFACE"],
            "eth0",
        )

        rpc_schema = reviewed_scenario("custom_needs_schema_evidence")
        self.assertEqual(
            rpc_schema.seed_state["custom_rpc"]["endpoint"],
            "http://geth-dev:8545",
        )
        self.assertEqual(
            rpc_schema.seed_state["custom_rpc"]["catalog"]["draft"][
                "validation_endpoint"
            ],
            "http://geth-dev:8545",
        )

        multi_intent = reviewed_scenario("action_change_group")
        self.assertEqual(
            multi_intent.seed_state["chain_identity"]["canonical"],
            "ethereum",
        )

    def test_validation_fails_closed_for_missing_and_duplicate_obligations(self) -> None:
        missing = list(deepcopy(self.obligations[:-1]))
        with self.assertRaisesRegex(ValueError, "exactly 60"):
            validate_retained_regression_obligations(
                missing,
                revision=REVISION,
            )

        duplicate = list(deepcopy(self.obligations))
        duplicate[-1] = deepcopy(duplicate[0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_retained_regression_obligations(
                duplicate,
                revision=REVISION,
            )

    def test_validation_fails_closed_for_unknown_variant(self) -> None:
        rows = list(deepcopy(self.obligations))
        rows[0]["variant"] = "paraphrase"
        with self.assertRaisesRegex(ValueError, "unknown retained regression variant"):
            validate_retained_regression_obligations(rows, revision=REVISION)

    def test_validation_fails_closed_for_non_executable_seed(self) -> None:
        rows = list(deepcopy(self.obligations))
        with patch(
            "tests.agent_live.retained_regression_obligations.reviewed_scenario",
            side_effect=ValueError("not executable"),
        ):
            with self.assertRaisesRegex(ValueError, "seed is not executable"):
                validate_retained_regression_obligations(rows, revision=REVISION)

    def test_validation_rejects_natural_language_expected_as_verifier(self) -> None:
        rows = list(deepcopy(self.obligations))
        rows[0]["verifier_contract"]["expected"] = (
            "Copying the fixture expectation here is not verification."
        )
        with self.assertRaisesRegex(ValueError, "expected text leaked"):
            validate_retained_regression_obligations(rows, revision=REVISION)

    def test_validation_rejects_stale_revision_and_contract_identity(self) -> None:
        with self.assertRaisesRegex(ValueError, "stale retained regression revision"):
            validate_retained_regression_obligations(
                deepcopy(self.obligations),
                revision={"commit": "c" * 40, "worktree_hash": "d" * 64},
            )

        rows = list(deepcopy(self.obligations))
        rows[0]["severity"] = "S9"
        with self.assertRaisesRegex(ValueError, "severity"):
            validate_retained_regression_obligations(rows, revision=REVISION)

        rows = list(deepcopy(self.obligations))
        rows[0]["execution_status"] = "passed"
        with self.assertRaisesRegex(ValueError, "cannot claim execution"):
            validate_retained_regression_obligations(rows, revision=REVISION)


if __name__ == "__main__":
    unittest.main()

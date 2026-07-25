"""End-to-end contracts for Case1/Case2 custom RPC runtime closure."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch


def _custom_rpc_catalog(methods: list[dict], *, chain: str = "", **workflow: object) -> dict:
    return {
        **workflow,
        "catalog": {
            "contract_version": 1,
            "revision": 1,
            "chain": chain,
            "methods": methods,
            "finished": bool(methods),
        },
    }


class CustomRpcRuntimeClosureTest(unittest.TestCase):
    def test_shell_loader_accepts_matching_job_local_chain_template(self) -> None:
        override_payload = {
            "_meta": {"adapter_family": "jsonrpc", "job_local_override": True},
            "chain_type": "case2-runtime-only",
            "rpc_url": "LOCAL_RPC_URL",
            "rpc_methods": {
                "single": "eth_chainId",
                "mixed": "eth_chainId",
                "mixed_weighted": [{"method": "eth_chainId", "weight": 100}],
            },
            "param_formats": {"eth_chainId": "no_params"},
            "param_spec": {"eth_chainId": {"transport": "jsonrpc_list", "params": []}},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            override = root / "override.json"
            override.write_text(json.dumps(override_payload), encoding="utf-8")
            env = os.environ.copy()
            env.update({
                "BLOCKCHAIN_NODE": "case2-runtime-only",
                "CHAIN_CONFIG_OVERRIDE_FILE": str(override),
                "LOCAL_RPC_URL": "http://127.0.0.1:8545",
                "MAINNET_RPC_URL": "",
                "RPC_MODE": "single",
                "BLOCKCHAIN_BENCHMARK_DATA_DIR": str(root / "data"),
                "MEMORY_SHARE_DIR": str(root / "memory"),
            })
            result = subprocess.run(
                ["bash", "-lc", "source config/config_loader.sh; printf '%s|%s|%s|%s' \"$BLOCKCHAIN_NODE\" \"$CURRENT_RPC_METHODS_STRING\" \"${MAINNET_RPC_URL:-}\" \"$ACTIVE_CHAIN_TEMPLATE_FILE\""],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"case2-runtime-only|eth_chainId||{override}", result.stdout)
        self.assertNotIn("using default Solana endpoint", result.stderr)

    def test_shell_loader_rejects_mismatched_job_local_chain_template(self) -> None:
        override_payload = {
            "_meta": {"adapter_family": "jsonrpc", "job_local_override": True},
            "chain_type": "different-chain",
            "rpc_methods": {"single": "eth_chainId"},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            override = root / "override.json"
            override.write_text(json.dumps(override_payload), encoding="utf-8")
            env = os.environ.copy()
            env.update({
                "BLOCKCHAIN_NODE": "case2-runtime-only",
                "CHAIN_CONFIG_OVERRIDE_FILE": str(override),
                "MAINNET_RPC_URL": "",
                "BLOCKCHAIN_BENCHMARK_DATA_DIR": str(root / "data"),
                "MEMORY_SHARE_DIR": str(root / "memory"),
            })
            result = subprocess.run(
                ["bash", "-lc", "source config/config_loader.sh"],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("override is malformed or does not match", result.stderr)

    def test_no_parameter_workload_generates_neutral_seed_without_target_address(self) -> None:
        from tools.fetch_active_accounts import select_target_seed, workload_requires_target_seed

        config = {
            "chain_type": "case2-runtime-only",
            "rpc_methods": {"single": "eth_chainId"},
            "param_formats": {"eth_chainId": "no_params"},
            "param_spec": {"eth_chainId": {"transport": "jsonrpc_list", "params": []}},
        }
        with patch.dict(os.environ, {"RPC_MODE": "single"}, clear=False):
            self.assertFalse(workload_requires_target_seed(config))
            self.assertEqual(select_target_seed(config, ""), "__ANYCHAIN_NO_TARGET__")

    def test_parameterized_workload_still_requires_an_explicit_target_seed(self) -> None:
        from tools.fetch_active_accounts import select_target_seed, workload_requires_target_seed

        config = {
            "chain_type": "case2-runtime-only",
            "rpc_methods": {"single": "eth_getBalance"},
            "param_spec": {
                "eth_getBalance": {
                    "transport": "jsonrpc_list",
                    "params": [{"name": "address", "example": "${TARGET_ADDRESS}"}],
                }
            },
        }
        with patch.dict(os.environ, {"RPC_MODE": "single"}, clear=False):
            self.assertTrue(workload_requires_target_seed(config))
            with self.assertRaisesRegex(ValueError, "No target seed is available"):
                select_target_seed(config, "")

    def test_versioned_catalog_cannot_be_overwritten_by_legacy_projections(self) -> None:
        from agent.harness.domains.rpc_catalog import ensure_catalog, validated_contracts_view
        from agent.harness.state import new_state

        canonical = {"method": "catalog_method", "params": [], "chain": "bsc"}
        state = new_state("catalog-single-owner")
        state["chain_identity"] = {
            "canonical": "bsc",
            "validated_methods": [{"method": "identity_projection", "params": []}],
        }
        state["custom_rpc"] = {
            "catalog": {
                "contract_version": 1,
                "revision": 7,
                "chain": "bsc",
                "methods": [canonical],
                "finished": True,
            },
            "validated_methods": [{"method": "legacy_projection", "params": []}],
        }

        catalog = ensure_catalog(state)

        self.assertEqual(catalog["methods"], [canonical])
        self.assertEqual(validated_contracts_view(state), [canonical])
        self.assertNotIn("validated_methods", state["custom_rpc"])
        self.assertNotIn("validated_methods", state["chain_identity"])

    def test_legacy_checkpoint_catalog_migrates_once_and_removes_mirrors(self) -> None:
        from agent.harness.domains.rpc_catalog import ensure_catalog, validated_contracts_view
        from agent.harness.state import new_state

        state = new_state("legacy-catalog-migration")
        state["chain_identity"] = {"canonical": "bsc"}
        state["custom_rpc"] = {
            "status": "method_validated_next",
            "validated_methods": [{"method": "eth_accounts", "params": []}],
            "schema_draft": {"method": "next_method", "params_json": []},
        }

        first = ensure_catalog(state)
        first_revision = first["revision"]
        second = ensure_catalog(state)

        self.assertEqual(validated_contracts_view(state), [{"method": "eth_accounts", "params": []}])
        self.assertEqual(second["revision"], first_revision)
        self.assertNotIn("validated_methods", state["custom_rpc"])
        self.assertNotIn("schema_draft", state["custom_rpc"])
        self.assertEqual(
            [item["event"] for item in state["audit_events"]].count("legacy_rpc_catalog_migrated"),
            1,
        )

    def test_rpc_mode_change_preserves_verified_catalog_and_clears_only_effective_workload(self) -> None:
        from agent.harness.state import new_state
        from agent.harness.transitions import invalidate_for_rpc_mode_change

        state = new_state("catalog-mode-switch")
        zero_param = {
            "method": "demo_zero",
            "params": [],
            "schema": {"params": [], "response_summary": "hex string"},
            "observed_response": {"sample": '{"result":"0x1"}'},
            "validation_endpoint": "http://validation.invalid",
            "evidence_file": "zero.json",
        }
        multi_param = {
            "method": "demo_multi",
            "params": ["0xabc", "latest"],
            "schema": {
                "params": [
                    {
                        "index": 0,
                        "name": "address",
                        "json_type": "string",
                        "semantic_type": "account_address",
                        "meaning": "account to inspect",
                        "required": True,
                        "example": "0xabc",
                    },
                    {
                        "index": 1,
                        "name": "block_tag",
                        "json_type": "string",
                        "semantic_type": "block_reference",
                        "meaning": "state version",
                        "required": True,
                        "example": "latest",
                    },
                ],
                "response_summary": "quantity hex string",
            },
            "validation_endpoint": "http://validation.invalid",
            "evidence_file": "multi.json",
        }
        state["custom_rpc"] = _custom_rpc_catalog([zero_param, multi_param], chain="bsc", **{
            "status": "needs_weights",
            "scope": "mixed_replace",
            "weights": {"demo_zero": 40, "demo_multi": 60},
            "requested_workload": {"scope": "mixed_replace"},
            "job_local_override": True,
            "endpoint": "http://validation.invalid",
            "endpoint_ready": True,
        })
        state["custom_rpc"]["methods"] = ["demo_zero", "demo_multi"]
        state["workload"] = {
            "confirmed": True,
            "methods": ["demo_zero", "demo_multi"],
            "mixed_weights": {"demo_zero": 40, "demo_multi": 60},
        }

        invalidate_for_rpc_mode_change(state)

        self.assertEqual(state["workload"], {})
        from agent.harness.domains.rpc_catalog import validated_contracts_view
        self.assertEqual(validated_contracts_view(state), [zero_param, multi_param])
        self.assertEqual(state["custom_rpc"]["status"], "method_validated_next")
        self.assertEqual(state["custom_rpc"]["endpoint"], "http://validation.invalid")
        self.assertTrue(state["custom_rpc"]["endpoint_ready"])
        for key in ("scope", "weights", "methods", "requested_workload", "job_local_override"):
            self.assertNotIn(key, state["custom_rpc"])

    def test_chain_change_retains_only_catalog_contracts_with_matching_provenance(self) -> None:
        from agent.harness.state import new_state
        from agent.harness.transitions import invalidate_for_chain_change

        state = new_state("catalog-chain-switch")
        state["chain_identity"] = {"canonical": "old-chain"}
        state["custom_rpc"] = _custom_rpc_catalog([
            {"method": "old_method", "params": [], "chain": "old-chain"},
            {
                "method": "new_method",
                "params": ["0xabc", "latest"],
                "chain": "new-chain",
                "schema": {"params": [{"name": "address"}, {"name": "block_tag"}]},
            },
        ], chain="old-chain", **{
            "endpoint": "http://old-chain.invalid",
            "endpoint_ready": True,
            "scope": "mixed_replace",
        })

        invalidate_for_chain_change(state, new_chain="new-chain")

        self.assertEqual(
            state["custom_rpc"]["catalog"]["methods"],
            [{
                "method": "new_method",
                "params": ["0xabc", "latest"],
                "chain": "new-chain",
                "schema": {"params": [{"name": "address"}, {"name": "block_tag"}]},
            }],
        )
        self.assertEqual(state["custom_rpc"]["catalog"]["chain"], "new-chain")
        self.assertNotIn("scope", state["custom_rpc"])
        self.assertNotIn("endpoint", state["custom_rpc"])
        self.assertNotIn("endpoint_ready", state["custom_rpc"])
        self.assertEqual(state["workload"], {})

    def test_confirmed_chain_change_uses_typed_candidate_provenance(self) -> None:
        from agent.harness.state import new_state
        from agent.harness.transitions import invalidate_for_chain_change

        state = new_state("catalog-confirmed-chain-switch")
        state["chain_identity"] = {
            "canonical": "old-chain",
            "change_candidate": {
                "raw": "user spelling",
                "resolution": {"canonical_chain_name": "new-chain"},
            },
        }
        state["custom_rpc"] = _custom_rpc_catalog([
                {"method": "old_method", "params": [], "chain": "old-chain"},
                {"method": "new_method", "params": {"height": "latest"}, "chain": "new-chain"},
        ], chain="old-chain")

        invalidate_for_chain_change(state)

        self.assertEqual(
            state["custom_rpc"]["catalog"]["methods"],
            [{"method": "new_method", "params": {"height": "latest"}, "chain": "new-chain"}],
        )

    def test_endpoint_change_strips_only_mismatched_probe_evidence(self) -> None:
        from agent.harness.state import new_state
        from agent.harness.transitions import invalidate_for_endpoint_change

        state = new_state("catalog-endpoint-switch")
        matching = {
            "method": "demo_zero",
            "params": [],
            "schema": {"params": []},
            "validation_endpoint": "http://new.invalid",
            "evidence_file": "new.json",
            "observed_response": {"sample": '{"result":"0x1"}'},
        }
        stale = {
            "method": "demo_multi",
            "params": ["0xabc", "latest"],
            "schema": {"params": [{"semantic_type": "account_address"}, {"semantic_type": "block_reference"}]},
            "validation_endpoint": "http://old.invalid",
            "evidence_file": "old.json",
            "observed_response": {"sample": '{"result":"0x2"}'},
            "final_endpoint": "http://final.invalid",
            "final_endpoint_evidence_file": "final.json",
        }
        state["custom_rpc"] = _custom_rpc_catalog([matching, stale], chain="bsc", **{
            "endpoint": "http://old.invalid",
            "endpoint_ready": True,
        })

        invalidate_for_endpoint_change(state, "http://new.invalid", role="validation")

        self.assertEqual(state["custom_rpc"]["catalog"]["methods"][0], matching)
        retained_schema = state["custom_rpc"]["catalog"]["methods"][1]
        self.assertEqual(retained_schema["params"], ["0xabc", "latest"])
        self.assertEqual(retained_schema["schema"], stale["schema"])
        for key in ("validation_endpoint", "evidence_file", "observed_response"):
            self.assertNotIn(key, retained_schema)
        self.assertEqual(retained_schema["final_endpoint"], "http://final.invalid")
        self.assertEqual(state["custom_rpc"]["endpoint"], "http://new.invalid")
        self.assertNotIn("endpoint_ready", state["custom_rpc"])

        invalidate_for_endpoint_change(state, "http://other-final.invalid", role="final")
        self.assertNotIn("final_endpoint", retained_schema)
        self.assertNotIn("final_endpoint_evidence_file", retained_schema)

    def test_case1_derives_adapter_family_and_generates_exact_custom_target(self) -> None:
        from agent.planners.strategy_planner import materialize_custom_rpc_template
        from tools.chain_adapters import get_adapter
        from tools.chain_adapters.cli import _get_param_format

        template = materialize_custom_rpc_template(
            chain="bsc",
            adapter_family="",
            rpc_mode="single",
            workload={"methods": ["eth_accounts"]},
            validated_methods=[{"method": "eth_accounts", "params": []}],
        )

        self.assertEqual(template["_meta"]["adapter_family"], "jsonrpc")
        self.assertEqual(template["rpc_methods"]["single"], "eth_accounts")
        self.assertEqual(template["proxy_extraction"]["extractors"][0]["protocol"], "json_rpc")
        with tempfile.TemporaryDirectory() as tmpdir:
            override = Path(tmpdir) / "override.json"
            override.write_text(json.dumps(template), encoding="utf-8")
            with patch.dict(
                os.environ,
                {"CHAIN_CONFIG_OVERRIDE_FILE": str(override), "BLOCKCHAIN_NODE": "bsc"},
                clear=False,
            ):
                target = get_adapter("bsc").build_vegeta_target(
                    "eth_accounts",
                    "0xfallback",
                    "http://node.invalid",
                    _get_param_format("bsc", "eth_accounts"),
                )
        body = json.loads(base64.b64decode(target["body"]).decode("utf-8"))
        self.assertEqual(body["method"], "eth_accounts")
        self.assertEqual(body["params"], [])

    def test_case2_template_generates_exact_zero_positional_and_object_wire_params(self) -> None:
        from agent.planners.strategy_planner import materialize_custom_rpc_template
        from tools.chain_adapters import get_adapter
        from tools.chain_adapters.cli import _get_param_format

        contracts = [
            {"method": "demo_zero", "params": []},
            {"method": "demo_multi", "params": ["0xabc", 7, True]},
            {"method": "demo_object", "params": {"owner": "0xabc", "limit": 3}},
        ]
        workload = {
            "methods": ["demo_zero", "demo_multi", "demo_object"],
            "mixed_weights": {"demo_zero": 34, "demo_multi": 33, "demo_object": 33},
        }
        template = materialize_custom_rpc_template(
            chain="case2-runtime-only",
            adapter_family="jsonrpc",
            rpc_mode="mixed",
            workload=workload,
            validated_methods=contracts,
        )

        self.assertEqual(template["_meta"]["adapter_family"], "jsonrpc")
        self.assertEqual(template["rpc_methods"]["mixed_weighted"], [
            {"method": "demo_zero", "weight": 34},
            {"method": "demo_multi", "weight": 33},
            {"method": "demo_object", "weight": 33},
        ])
        self.assertEqual(template["proxy_extraction"]["extractors"][0]["protocol"], "json_rpc")
        with tempfile.TemporaryDirectory() as tmpdir:
            override = Path(tmpdir) / "override.json"
            override.write_text(json.dumps(template), encoding="utf-8")
            with patch.dict(os.environ, {"CHAIN_CONFIG_OVERRIDE_FILE": str(override), "BLOCKCHAIN_NODE": "case2-runtime-only"}, clear=False):
                adapter = get_adapter("case2-runtime-only")
                observed = {}
                for contract in contracts:
                    method = contract["method"]
                    target = adapter.build_vegeta_target(
                        method,
                        "0xfallback",
                        "http://node.invalid",
                        _get_param_format("case2-runtime-only", method),
                    )
                    body = json.loads(base64.b64decode(target["body"]).decode("utf-8"))
                    observed[method] = body["params"]

        self.assertEqual(observed, {item["method"]: item["params"] for item in contracts})

    def test_case2_template_rejects_any_selected_method_without_a_validated_contract(self) -> None:
        from agent.planners.strategy_planner import materialize_custom_rpc_template

        template = materialize_custom_rpc_template(
            chain="case2-incomplete",
            adapter_family="jsonrpc",
            rpc_mode="mixed",
            workload={
                "methods": ["demo_a", "demo_b"],
                "mixed_weights": {"demo_a": 50, "demo_b": 50},
            },
            validated_methods=[{"method": "demo_a", "params": []}],
        )

        self.assertEqual(template, {})

    def test_wire_facts_override_model_metadata_and_object_fields_match_by_name(self) -> None:
        from agent.harness.domains.chain_rpc_support import _merge_rpc_schema_draft

        draft = _merge_rpc_schema_draft(
            method="demo_object",
            params_json={"limit": 3, "owner": "0xabc"},
            extracted={
                "params": [
                    {"name": "owner", "json_type": "number", "example": 999, "semantic_type": "account_address"},
                    {"name": "invented", "json_type": "string", "example": "bad"},
                    {"name": "limit", "json_type": "string", "example": "bad", "semantic_type": "page_limit"},
                ]
            },
        )

        self.assertEqual([item["name"] for item in draft["params"]], ["limit", "owner"])
        self.assertEqual([item["json_type"] for item in draft["params"]], ["number", "string"])
        self.assertEqual([item["example"] for item in draft["params"]], [3, "0xabc"])
        self.assertEqual([item["semantic_type"] for item in draft["params"]], ["page_limit", "account_address"])

        zero = _merge_rpc_schema_draft(
            method="demo_zero",
            params_json=[],
            extracted={"params": [{"name": "invented", "json_type": "string", "example": "bad"}]},
        )
        self.assertEqual(zero["params"], [])

        positional = _merge_rpc_schema_draft(
            method="demo_positional",
            params_json=["0xabc", 7],
            extracted={"params": [
                {"name": "owner", "json_type": "number", "example": "wrong"},
                {"name": "limit", "json_type": "string", "example": "wrong"},
            ]},
        )
        self.assertEqual([item["name"] for item in positional["params"]], ["owner", "limit"])
        self.assertEqual([item["json_type"] for item in positional["params"]], ["string", "number"])
        self.assertEqual([item["example"] for item in positional["params"]], ["0xabc", 7])

    def test_catalog_strict_identity_accepts_protocol_request_but_rejects_recovery_prose(self) -> None:
        from agent.harness.domains.rpc_endpoint import _apply_method_answer
        from agent.harness.domains.rpc_catalog import draft_view, strict_method_identity
        from agent.harness.state import new_state

        self.assertEqual(strict_method_identity("eth_chainId", adapter_family="jsonrpc"), "eth_chainId")
        self.assertEqual(strict_method_identity("please recover the previous method", adapter_family="jsonrpc"), "")
        self.assertEqual(strict_method_identity("GET /v1/blocks", adapter_family="rest"), "GET /v1/blocks")

        state = new_state("strict-method")
        state["chain_identity"] = {"canonical": "bsc", "adapter_family": "jsonrpc"}
        state["custom_rpc"] = {"status": "needs_method", "endpoint_ready": True}
        state = deepcopy(state)
        _apply_method_answer(
            state,
            "custom_rpc",
            "please recover the previous method",
            responses=[],
        )
        self.assertEqual(state["custom_rpc"]["status"], "needs_method")
        self.assertFalse(draft_view(state).get("method"))

        request = '{"jsonrpc":"2.0","id":1,"method":"vendor$exact","params":[]}'
        with patch(
            "agent.harness.domains.rpc_endpoint.extract_rpc_schema_from_evidence",
            return_value={
                "status": "draft",
                "method": "vendor$exact",
                "params": [],
            },
        ):
            _apply_method_answer(state, "custom_rpc", request, responses=[])
        self.assertEqual(draft_view(state)["method"], "vendor$exact")

    def test_versioned_catalog_confirms_zero_positional_and_object_params_semantically(self) -> None:
        from agent.harness.domains.chain_rpc_support import _merge_rpc_schema_draft
        from agent.harness.domains.rpc_catalog import (
            RPC_CATALOG_VERSION,
            catalog_view,
            confirm_parameter,
            confirm_request,
            correct_draft,
            draft_view,
            next_parameter_to_confirm,
        )
        from agent.harness.state import new_state

        cases = [
            ("demo_zero", [], []),
            (
                "demo_multi",
                ["0xabc", "latest"],
                [
                    {"name": "address", "semantic_type": "account_address", "encoding": "20-byte hex", "meaning": "account to query", "required": True},
                    {"name": "block", "semantic_type": "block_reference", "encoding": "tag", "meaning": "state version", "required": True},
                ],
            ),
            (
                "demo_object",
                {"owner": "0xabc", "limit": 3},
                [
                    {"name": "owner", "semantic_type": "account_address", "encoding": "20-byte hex", "meaning": "result owner", "required": True},
                    {"name": "limit", "semantic_type": "page_limit", "encoding": "integer", "meaning": "maximum rows", "required": False},
                ],
            ),
        ]
        for method, params_json, semantics in cases:
            with self.subTest(method=method):
                state = new_state(f"catalog-{method}")
                state["chain_identity"] = {"canonical": "case-independent", "adapter_family": "jsonrpc"}
                draft = _merge_rpc_schema_draft(
                    method=method,
                    params_json=params_json,
                    extracted={"params": semantics, "response_summary": "unknown"},
                )
                correct_draft(state, draft)
                while (index := next_parameter_to_confirm(state)) is not None:
                    self.assertTrue(confirm_parameter(state, index, True).accepted)
                self.assertTrue(confirm_request(state, True).accepted)
                observed = draft_view(state)
                self.assertEqual(observed["params_json"], params_json)
                self.assertEqual(observed["phase"], "probe_confirmation")
                self.assertTrue(observed["request_confirmed"])
                self.assertEqual(catalog_view(state)["contract_version"], RPC_CATALOG_VERSION)

    def test_case1_and_case2_single_method_factories_never_build_empty_options(self) -> None:
        from agent.harness.domains.chain_rpc import question_for_chain_rpc
        from agent.harness.state import new_state

        case1 = new_state("case1-single")
        case1["chain_identity"] = {"canonical": "bsc", "status": "confirmed"}
        case1["custom_rpc"] = _custom_rpc_catalog(
            [{"method": "demo_a"}, {"method": "demo_b"}],
            chain="bsc",
            status="needs_single_method",
        )
        case2 = new_state("case2-single")
        case2["chain_identity"] = {
            "canonical": "new-chain",
            "case": "case2",
            "status": "existing_family_needs_single_method",
        }
        case2["custom_rpc"] = _custom_rpc_catalog(
            [{"method": "demo_a"}, {"method": "demo_b"}],
            chain="new-chain",
        )
        for state in (case1, case2):
            with self.subTest(case=state["thread_id"]):
                question = question_for_chain_rpc(state, "endpoint_process")
                self.assertIsNotNone(question)
                self.assertEqual([item["value"] for item in question["options"]], ["demo_a", "demo_b"])

    def test_compiled_graph_probe_failure_preserves_confirmation_and_retries_without_invariant_error(self) -> None:
        from agent.harness.domains.chain_rpc import question_for_chain_rpc
        from agent.harness.domains.rpc_catalog import confirm_request, correct_draft, draft_view
        from agent.harness.state import new_state
        from tests.agent_live.graph_turn import invoke_product_graph_turn

        state = new_state("probe-failure-contract")
        state.update({
            "target_mode": "real-node",
            "workflow_mode": "rpc_benchmark",
            "active_group": "endpoint_process",
            "chain_identity": {"canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed"},
            "custom_rpc": {"status": "schema_needs_confirmation", "endpoint": "http://probe.invalid", "endpoint_ready": True},
        })
        correct_draft(state, {
            "method": "eth_chainId",
            "params": [],
            "params_json": [],
            "response_summary": "unknown",
            "validation_endpoint": "http://probe.invalid",
        })
        self.assertTrue(confirm_request(state, True).accepted)
        state["pending_question"] = question_for_chain_rpc(state, "endpoint_process")
        state["last_user_input"] = "Y"

        failure = {"ready": False, "status": "failed", "error": "method not found", "evidence_file": "probe-failed.json"}
        with patch("agent.harness.domains.rpc_endpoint.validate_rpc_endpoint", return_value=failure):
            result = invoke_product_graph_turn(state)

        self.assertEqual(result["custom_rpc"]["status"], "probe_failed")
        self.assertTrue(draft_view(result)["request_confirmed"])
        self.assertEqual(result["pending_question"]["id"], "custom_rpc_probe_confirm")

    def test_adding_second_method_resets_only_in_progress_evidence(self) -> None:
        from agent.harness.domains.rpc_workload import _apply_continue
        from agent.harness.state import new_state

        state = new_state("method-isolation")
        state["custom_rpc"] = _custom_rpc_catalog(
            [{"method": "demo_a", "params": []}],
            status="method_validated_next",
            schema_evidence="request-a",
        )
        state["custom_rpc"]["catalog"]["draft"] = {
            "method": "demo_a",
            "params_json": [],
            "evidence": [{"content": "request-a"}, {"content": "response-a"}],
            "observed_response": {"sample": '{"result":"a"}'},
            "response_confirmed": True,
        }

        state = deepcopy(state)
        _apply_continue(
            state,
            "custom_rpc_continue",
            "add_another",
            responses=[],
        )

        self.assertEqual(state["custom_rpc"]["catalog"]["methods"], [{"method": "demo_a", "params": []}])
        self.assertEqual(state["custom_rpc"]["catalog"]["draft"], {})

    def test_new_method_or_validation_endpoint_resets_only_unfinished_evidence(self) -> None:
        from agent.harness.domains.rpc_endpoint import _apply_endpoint_answer, _apply_method_answer
        from agent.harness.state import new_state

        state = new_state("method-change-isolation")
        state["chain_identity"] = {"canonical": "bsc", "adapter_family": "jsonrpc"}
        state["custom_rpc"] = _custom_rpc_catalog([{"method": "demo_complete", "params": []}], chain="bsc", **{
            "endpoint": "http://first.invalid",
            "endpoint_ready": True,
        })
        state["custom_rpc"]["catalog"]["draft"] = {
            "method": "demo_in_progress",
            "evidence": [{"content": "stale request"}],
        }

        state = deepcopy(state)
        with patch(
            "agent.harness.domains.rpc_endpoint.validate_rpc_endpoint",
            return_value={"ready": True, "status": "ok", "evidence_file": "second-endpoint.json"},
        ):
            _apply_endpoint_answer(
                state,
                {
                    "endpoint_role": "validation",
                    "rpc_case": "custom_rpc",
                    "config_field": "",
                },
                "http://second.invalid",
                responses=[],
            )

        self.assertEqual(state["custom_rpc"]["catalog"]["methods"], [{"method": "demo_complete", "params": []}])
        self.assertEqual(state["custom_rpc"]["catalog"]["draft"], {})
        _apply_method_answer(state, "custom_rpc", "demo_next", responses=[])
        self.assertEqual(state["custom_rpc"]["catalog"]["draft"]["method"], "demo_next")
        self.assertEqual(state["custom_rpc"]["catalog"]["methods"], [{"method": "demo_complete", "params": []}])

    def test_response_contract_detects_expected_observed_shape_mismatch(self) -> None:
        from agent.harness.domains.rpc_endpoint import _response_contract_conflicts

        conflicts = _response_contract_conflicts(
            {
                "response_summary": "result object",
                "response_fields": [{"name": "height", "type": "number"}],
            },
            '{"jsonrpc":"2.0","result":{"height":"0x10"}}',
        )

        self.assertTrue(any("height expected number" in item for item in conflicts))

    def test_response_summary_does_not_control_json_type_validation(self) -> None:
        from agent.harness.domains.rpc_endpoint import _response_contract_conflicts

        observed = '{"jsonrpc":"2.0","result":[]}'
        self.assertEqual(
            _response_contract_conflicts(
                {"response_summary": "hex string returned by the method"},
                observed,
            ),
            [],
        )
        self.assertIn(
            "response contract expected string, observed array",
            _response_contract_conflicts(
                {
                    "response_summary": "human-readable explanation",
                    "response_json_type": "string",
                },
                observed,
            ),
        )

    def test_rpc_params_require_structured_wire_evidence(self) -> None:
        from agent.harness.input_values import extract_rpc_params_or_request

        self.assertEqual(
            extract_rpc_params_or_request(
                "The method is eth_blockNumber and it has no params."
            ),
            ("", None),
        )
        self.assertEqual(extract_rpc_params_or_request("[]"), ("", []))

    def test_validated_method_records_its_own_probe_endpoint(self) -> None:
        from agent.harness.domains.rpc_endpoint import _probe_schema
        from agent.harness.domains.rpc_catalog import confirm_request, confirm_response, correct_draft
        from agent.harness.state import new_state

        state = new_state("method-endpoint-evidence")
        state["chain_identity"] = {"canonical": "bsc", "adapter_family": "jsonrpc"}
        state["custom_rpc"] = {
            "endpoint": "http://sample-one.invalid",
            "method": "demo_a",
        }
        state = deepcopy(state)
        correct_draft(state, {
            "method": "demo_a",
            "params": [],
            "params_json": [],
            "response_summary": "hex string",
            "validation_endpoint": "http://sample-one.invalid",
        })
        self.assertTrue(confirm_request(state, True).accepted)
        self.assertTrue(confirm_response(state, True).accepted)
        probe = {
            "ready": True,
            "status": "ok",
            "evidence_file": "demo-a.json",
            "checks": [{
                "name": "method_probe:demo_a",
                "response_sample": '{"jsonrpc":"2.0","result":"0x1"}',
                "response_shape_hash": "shape-a",
                "http_status": 200,
            }],
        }
        with patch("agent.harness.domains.rpc_endpoint.validate_rpc_endpoint", return_value=probe):
            _probe_schema(state, "custom_rpc", [], responses=[])

        contract = state["custom_rpc"]["catalog"]["methods"][0]
        self.assertEqual(contract["validation_endpoint"], "http://sample-one.invalid")
        self.assertEqual(contract["params"], [])

    def test_final_endpoint_replays_every_selected_custom_method_with_exact_params(self) -> None:
        from agent.harness.domains.rpc_endpoint import _apply_endpoint_answer
        from agent.harness.state import new_state

        state = new_state("final-endpoint")
        state.update({
            "chain_identity": {"canonical": "bsc", "adapter_family": "jsonrpc"},
            "workload": {"methods": ["demo_a", "demo_b"]},
            "custom_rpc": _custom_rpc_catalog([
                {"method": "demo_a", "params": []},
                {"method": "demo_b", "params": {"owner": "0xabc"}},
            ], chain="bsc"),
        })
        state = deepcopy(state)
        def probe(**kwargs):
            method = kwargs["methods"][0]
            return {"ready": True, "status": "ok", "evidence_file": f"{method}.json"}

        with patch("agent.harness.domains.rpc_endpoint.validate_rpc_endpoint", side_effect=probe) as validate:
            _apply_endpoint_answer(
                state,
                {
                    "endpoint_role": "final_benchmark",
                    "rpc_case": "runtime",
                    "config_field": "LOCAL_RPC_URL",
                },
                "http://final.invalid",
                responses=[],
            )

        self.assertEqual(validate.call_count, 2)
        self.assertEqual(
            [(call.kwargs["methods"], call.kwargs["method_params"]) for call in validate.call_args_list],
            [(["demo_a"], {"demo_a": []}), (["demo_b"], {"demo_b": {"owner": "0xabc"}})],
        )
        self.assertEqual(state["confirmed_config"]["LOCAL_RPC_URL"], "http://final.invalid")
        self.assertTrue(all(item["final_endpoint"] == "http://final.invalid" for item in state["custom_rpc"]["catalog"]["methods"]))
        self.assertEqual(
            [item["final_endpoint_evidence_file"] for item in state["custom_rpc"]["catalog"]["methods"]],
            ["demo_a.json", "demo_b.json"],
        )

    def test_final_endpoint_blocks_when_any_selected_custom_method_fails(self) -> None:
        from agent.harness.domains.rpc_endpoint import _apply_endpoint_answer
        from agent.harness.state import new_state

        state = new_state("final-endpoint-failure")
        state.update({
            "chain_identity": {"canonical": "bsc", "adapter_family": "jsonrpc"},
            "workload": {"methods": ["demo_a", "demo_b"]},
            "custom_rpc": _custom_rpc_catalog([
                {"method": "demo_a", "params": []},
                {"method": "demo_b", "params": ["0xabc"]},
            ], chain="bsc"),
        })

        state = deepcopy(state)
        def probe(**kwargs):
            method = kwargs["methods"][0]
            return {
                "ready": method == "demo_a",
                "status": "ok" if method == "demo_a" else "failed",
                "error": "method rejected" if method == "demo_b" else "",
                "evidence_file": f"{method}.json",
            }

        with patch("agent.harness.domains.rpc_endpoint.validate_rpc_endpoint", side_effect=probe):
            _apply_endpoint_answer(
                state,
                {
                    "endpoint_role": "final_benchmark",
                    "rpc_case": "runtime",
                    "config_field": "LOCAL_RPC_URL",
                },
                "http://final.invalid",
                responses=[],
            )

        self.assertNotIn("LOCAL_RPC_URL", state.get("confirmed_config", {}))
        self.assertFalse(state["endpoint_evidence"]["local_rpc_url_ready"])
        self.assertIn("demo_b", state["endpoint_evidence"]["final_custom_rpc_probe"]["selected_methods"])
        self.assertTrue(all("final_endpoint" not in item for item in state["custom_rpc"]["catalog"]["methods"]))

    def test_weights_require_positive_complete_final_workload_for_replace_and_add(self) -> None:
        from agent.harness.domains.rpc_workload import _apply_weights
        from agent.harness.state import new_state

        replace = new_state("replace-weights")
        replace["chain_identity"] = {"canonical": "bsc"}
        replace["custom_rpc"] = _custom_rpc_catalog([{"method": "demo_a"}, {"method": "demo_b"}], scope="mixed_replace")
        replace = deepcopy(replace)
        _apply_weights(
            replace,
            "custom_rpc_weights",
            '{"demo_a":0,"demo_b":100}',
            responses=[],
        )
        self.assertFalse((replace.get("workload") or {}).get("confirmed"))

        _apply_weights(
            replace,
            "custom_rpc_weights",
            '{"demo_a":50.0,"demo_b":50.0}',
            responses=[],
        )
        self.assertFalse((replace.get("workload") or {}).get("confirmed"))

        add = new_state("add-weights")
        add["chain_identity"] = {"canonical": "bsc"}
        add["custom_rpc"] = _custom_rpc_catalog([{"method": "demo_custom"}], scope="mixed_add")
        add = deepcopy(add)
        incomplete = {"demo_custom": 100}
        _apply_weights(
            add,
            "custom_rpc_weights",
            json.dumps(incomplete),
            responses=[],
        )
        self.assertFalse((add.get("workload") or {}).get("confirmed"))

        final = {
            "eth_getBalance": 20,
            "eth_getTransactionCount": 20,
            "eth_blockNumber": 20,
            "eth_gasPrice": 20,
            "demo_custom": 20,
        }
        _apply_weights(
            add,
            "custom_rpc_weights",
            json.dumps(final),
            responses=[],
        )
        self.assertTrue(add["workload"]["confirmed"])
        self.assertFalse(add["workload"]["replace_defaults"])
        self.assertEqual(add["workload"]["mixed_weights"], final)

    def test_weight_syntax_authority_accepts_yaml_json_and_env_equivalently(self) -> None:
        from agent.harness.input_values import parse_weight_spec

        expected = {"eth_blockNumber": 40, "eth_gasPrice": 60}
        inputs = (
            "weights:\n  eth_blockNumber: 40\n  eth_gasPrice: 60",
            '{"weights":{"eth_blockNumber":40,"eth_gasPrice":60}}',
            "eth_blockNumber=40,eth_gasPrice=60",
            "Please change it to eth_blockNumber=40,eth_gasPrice=60",
        )

        for value in inputs:
            with self.subTest(value=value):
                self.assertEqual(parse_weight_spec(value), expected)

        for invalid in (
            "weights:\n  eth_blockNumber: 40.5\n  eth_gasPrice: 59.5",
            '{"eth_blockNumber":true,"eth_gasPrice":99}',
        ):
            with self.subTest(invalid=invalid):
                self.assertEqual(parse_weight_spec(invalid), {})

    def test_yaml_weights_reach_the_same_workload_domain_contract(self) -> None:
        from agent.harness.domains.rpc_workload import _apply_weights
        from agent.harness.state import new_state

        state = new_state("yaml-weights")
        state["chain_identity"] = {"canonical": "bsc"}
        state["custom_rpc"] = _custom_rpc_catalog([], scope="mixed_replace")
        state = deepcopy(state)

        _apply_weights(
            state,
            "custom_rpc_weights",
            "weights:\n  eth_getBalance: 25\n  eth_getTransactionCount: 25\n  eth_blockNumber: 25\n  eth_gasPrice: 25",
            responses=[],
        )

        self.assertTrue(state["workload"]["confirmed"])
        self.assertEqual(sum(state["workload"]["mixed_weights"].values()), 100)

    def test_typed_and_manual_mixed_replace_share_the_same_effective_contract(self) -> None:
        from agent.harness.domains.rpc_workload import _apply_requested_workload, _apply_weights
        from agent.harness.state import new_state

        weights = {"demo_custom": 60, "eth_blockNumber": 40}
        manual = new_state("manual-replace")
        manual["chain_identity"] = {"canonical": "bsc"}
        manual["custom_rpc"] = _custom_rpc_catalog(
            [{"method": "demo_custom", "params": []}], scope="mixed_replace",
        )
        manual = deepcopy(manual)
        _apply_weights(
            manual,
            "custom_rpc_weights",
            json.dumps(weights),
            responses=[],
        )

        typed = new_state("typed-replace")
        typed["chain_identity"] = {"canonical": "bsc"}
        typed["custom_rpc"] = _custom_rpc_catalog([{"method": "demo_custom", "params": []}], **{
            "requested_workload": {
                "scope": "mixed_replace",
                "weights": weights,
                "finish_methods": True,
            },
        })

        typed = deepcopy(typed)
        self.assertTrue(_apply_requested_workload(typed, responses=[]))
        self.assertEqual(manual["workload"]["mixed_weights"], weights)
        self.assertEqual(typed["workload"]["mixed_weights"], weights)
        self.assertTrue(manual["workload"]["replace_defaults"])
        self.assertTrue(typed["workload"]["replace_defaults"])

    def test_execution_runtime_persists_complete_case2_override_and_uses_it_for_preflight(self) -> None:
        from agent.harness.domains.execution_runtime import _prepare_benchmark_with_runtime_contract

        state = {
            "chain_identity": {
                "canonical": "case2-execution",
                "adapter_family": "jsonrpc",
            },
            "custom_rpc": _custom_rpc_catalog(
                [{"method": "demo_object", "params": {"height": "latest"}}],
                chain="case2-execution",
            ),
            "workload": {
                "confirmed": True,
                "job_local_override": True,
                "methods": ["demo_object"],
            },
            "rpc_mode": "single",
            "target_mode": "real-node",
            "confirmed_config": {"LOCAL_RPC_URL": "http://node.invalid"},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_file = Path(tmpdir) / "plan.json"
            plan_file.write_text("{}\n", encoding="utf-8")
            prepared = {
                "status": "blocked",
                "warnings": ["chain_template_exists", "keep-this-warning"],
                "data": {
                    "plan": {"chain": "case2-execution", "artifacts": {}},
                    "plan_file": str(plan_file),
                    "preflight": {"blockers": ["chain_template_exists"]},
                }
            }
            preflight = {
                "passed": False,
                "checks": [
                    {"name": "chain_template_exists", "passed": False, "detail": "missing"},
                    {"name": "chain_template_json_valid", "passed": False, "detail": "missing"},
                    {"name": "runtime_contract", "passed": True, "detail": "ok"},
                ],
                "blockers": ["chain_template_exists", "chain_template_json_valid"],
            }
            with (
                patch("agent.runners.application_service.prepare_benchmark_run", return_value=prepared),
                patch("agent.runners.application_service.run_preflight", return_value=preflight) as run_preflight,
            ):
                result = _prepare_benchmark_with_runtime_contract(state)

            plan = result["data"]["plan"]
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["warnings"], ["keep-this-warning"])
            self.assertTrue(result["data"]["preflight"]["passed"])
            self.assertEqual(plan["chain_config_override"]["rpc_methods"]["single"], "demo_object")
            self.assertEqual(
                plan["chain_config_override"]["param_spec"]["demo_object"]["fields"],
                {"height": {"literal": "latest"}},
            )
            self.assertEqual(run_preflight.call_args.args[0], plan)
            self.assertEqual(
                json.loads(plan_file.read_text(encoding="utf-8")),
                {},
                "runtime custom-RPC materialization must not overwrite the approved source plan",
            )


if __name__ == "__main__":
    unittest.main()

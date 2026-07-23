"""Tests for the unexposed LangGraph Harness skeleton."""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch


LANGGRAPH_AVAILABLE = importlib.util.find_spec("langgraph") is not None


def _admitted_mock_plan(state, text, payload):
    """Mint the same immutable receipts as the production intent boundary."""

    from agent.harness.action_registry import (
        ACTION_BY_TYPE,
        build_admission_transaction_hash,
        build_proposal_field_receipt,
    )

    actions = deepcopy(list(payload.get("actions") or []))
    for action in actions:
        if not isinstance(action, dict):
            continue
        spec = ACTION_BY_TYPE.get(str(action.get("type") or ""))
        if (
            (spec is not None and "source_evidence" in spec.required_arguments)
            or str(action.get("type") or "") == "start_custom_rpc"
        ):
            action.setdefault("source_evidence", str(text))
    pending = dict(state.get("pending_question") or {})
    options = [item for item in pending.get("options") or [] if isinstance(item, dict)]
    selected_options: dict[int, dict] = {}
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            continue
        option = {}
        if str(action.get("type") or "") == "answer_pending":
            selected = action.get("selected_value")
            option = next((item for item in options if item.get("value") == selected), {})
        else:
            spec = ACTION_BY_TYPE.get(str(action.get("type") or ""))
            if spec is not None and spec.pending_option_admission:
                option = next((
                    item
                    for item in options
                    if isinstance(item.get("action"), dict)
                    and str(item["action"].get("type") or "") == str(action.get("type") or "")
                    and all(
                        key == "type" or action.get(key) == value
                        for key, value in item["action"].items()
                    )
                ), {})
        if not option:
            continue
        selected = option.get("value")
        actions[index] = {
            "type": "answer_pending",
            "answer": selected,
            "selected_value": selected,
            "source_evidence": str(action.get("source_evidence") or text),
            "pending_option_semantic_verified": True,
            "semantic_purpose_verified": True,
            "confidence": str(action.get("confidence") or "medium"),
        }
        selected_options[index] = option
    thread_id = str(state.get("thread_id") or "default")
    session_id = str((state.get("session") or {}).get("id") or thread_id)
    turn_index = int(state.get("turn_index") or 0)
    action_ids = [f"test-admission-{index}" for index in range(len(actions))]
    semantic_units = [{
        "unit_id": f"test-unit-{index}",
        "clause_id": f"test-clause-{index}",
        "source_text": str(text),
        "disposition": "action",
        "action_indexes": [index],
    } for index in range(len(actions))]
    transaction_hash = build_admission_transaction_hash(
        thread_id=thread_id,
        session_id=session_id,
        submitted_turn_index=turn_index,
        actions=actions,
        semantic_units=semantic_units,
        admission_action_ids=action_ids,
    )
    for index, action in enumerate(actions):
        action["_admission_action_id"] = action_ids[index]
        action["_transaction_action_ids"] = list(action_ids)
        action["_plan_transaction_hash"] = transaction_hash
        if str(action.get("type") or "") != "propose_config_values":
            continue
        receipts = {}
        source_text = str(text)
        for field, value in dict(action.get("config_values") or {}).items():
            receipts[field] = build_proposal_field_receipt(
                thread_id=thread_id,
                session_id=session_id,
                submitted_turn_index=turn_index,
                transaction_hash=transaction_hash,
                admission_action_id=action_ids[index],
                config_field=field,
                canonical_value=value,
                source_unit_id=semantic_units[index]["unit_id"],
                source_unit_text=source_text,
                source_quote=source_text,
            )
        action["_proposal_transaction_hashes"] = [transaction_hash]
        action["_proposal_field_receipts"] = receipts
    pending_choice_contracts = [
        {
            "action_index": index,
            "admission_action_id": action_ids[index],
            "question": {
                "id": str(pending.get("id") or ""),
                "group": str(pending.get("group") or ""),
                "contract_version": pending.get("contract_version"),
            },
            "option": {
                "id": str(option.get("id") or ""),
                "selected_value": option.get("value"),
            },
            "semantic_units": [{
                "unit_id": semantic_units[index]["unit_id"],
                "clause_id": semantic_units[index]["clause_id"],
                "source_text": semantic_units[index]["source_text"],
            }],
        }
        for index, option in selected_options.items()
    ]
    return {
        "actions": actions,
        "pending_choice_contracts": pending_choice_contracts,
    }


def _admitted_mock_resolver(payload):
    return lambda state, text: _admitted_mock_plan(state, text, payload)


def _invoke_with_admitted_actions(process_turn, state, actions):
    """Execute a graph-contract turn without making a live model call."""

    payload = {"actions": actions}
    with patch(
        "agent.harness.coordinator.resolve_action_queue",
        side_effect=_admitted_mock_resolver(payload),
    ):
        return process_turn(state)


def _invoke_with_rpc_evidence(process_turn, state, evidence):
    return _invoke_with_admitted_actions(
        process_turn,
        state,
        [{
            "type": "rpc_catalog_command",
            "catalog_command": "append_evidence",
            "rpc_schema_evidence": evidence,
            "source_evidence": evidence,
            "confidence": "high",
        }],
    )


def _commit_result(state, result, *, owner: str):
    """Commit a typed domain result through the coordinator authority."""

    from agent.harness.coordinator import _apply_handler_result
    from agent.harness.domains.rpc_catalog import migrate_legacy_catalog

    migrate_legacy_catalog(state)
    return _apply_handler_result(state, result, owner=owner)


def _catalog_draft(state):
    from agent.harness.domains.rpc_catalog import draft_view

    return draft_view(state)


def _catalog_methods(state):
    from agent.harness.domains.rpc_catalog import validated_contracts_view

    return validated_contracts_view(state)


@unittest.skipUnless(LANGGRAPH_AVAILABLE, "langgraph is not installed in this Python environment")
class LangGraphHarnessSkeletonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        agent_runtime_dir = Path(__file__).resolve().parents[1] / ".agent"
        agent_runtime_dir.mkdir(parents=True, exist_ok=True)
        cls._endpoint_evidence_tmp = tempfile.TemporaryDirectory(
            prefix="test-endpoint-evidence-",
            dir=agent_runtime_dir,
        )
        cls._endpoint_evidence_patch = patch(
            "agent.validators.endpoint_probe.EVIDENCE_DIR",
            Path(cls._endpoint_evidence_tmp.name) / "endpoint-probes",
        )
        cls._endpoint_evidence_patch.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._endpoint_evidence_patch.stop()
        cls._endpoint_evidence_tmp.cleanup()
        super().tearDownClass()

    _RPC_CONFIRMATION_QUESTIONS = {
        "custom_rpc_parameter_confirm",
        "custom_rpc_schema_confirm",
        "custom_rpc_response_confirm",
        "custom_rpc_probe_confirm",
        "new_chain_parameter_confirm",
        "new_chain_schema_confirm",
        "new_chain_response_confirm",
        "new_chain_probe_confirm",
    }

    def _confirm_catalog_through_graph(self, state, process_turn):
        for _ in range(12):
            question_id = str((state.get("pending_question") or {}).get("id") or "")
            if question_id not in self._RPC_CONFIRMATION_QUESTIONS:
                return state
            state["last_user_input"] = "Y"
            state = process_turn(state)
        self.fail("RPC catalog confirmation did not reach a stable next question")

    def _confirm_catalog_through_domain(self, state):
        from agent.harness.domains.chain_rpc import apply_chain_rpc_answer, question_for_chain_rpc

        for _ in range(12):
            question = question_for_chain_rpc(state, "endpoint_process")
            if not question or question.get("id") not in self._RPC_CONFIRMATION_QUESTIONS:
                return state
            result = apply_chain_rpc_answer(state, question, True, "Y")
            state = _commit_result(state, result, owner="chain_rpc")
        self.fail("RPC catalog domain confirmation did not reach a stable next question")

    def test_unresolved_target_mode_preserves_same_turn_mutations(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.orientation import opening_question
        from agent.harness.state import new_state

        state = new_state("unresolved-mode-transaction", language="en")
        state["active_group"] = "opening"
        state["pending_question"] = opening_question(state)
        text = "Test BNB with mixed workload, quick QPS, and local Grafana"
        state["last_user_input"] = text
        actions = {"actions": [
            {
                "type": "choose_chain",
                "chain_text": "BNB",
                "source_evidence": text,
                "chain_selection_semantic_verified": True,
                "semantic_purpose_verified": True,
                "confidence": "high",
            },
            {
                "type": "set_rpc_mode",
                "rpc_mode": "mixed",
                "mutation_explicit": True,
                "source_evidence": "mixed workload",
                "semantic_purpose_verified": True,
                "confidence": "high",
            },
            {
                "type": "set_qps_mode",
                "qps_mode": "quick",
                "mutation_explicit": True,
                "source_evidence": "quick QPS",
                "semantic_purpose_verified": True,
                "confidence": "high",
            },
            {
                "type": "set_observability",
                "observability_mode": "local",
                "mutation_explicit": True,
                "source_evidence": "local Grafana",
                "semantic_purpose_verified": True,
                "confidence": "high",
            },
        ]}

        with patch("agent.harness.coordinator.resolve_action_queue", return_value=actions):
            result = process_turn(state)

        self.assertEqual(result["chain_identity"]["canonical"], "bsc")
        self.assertEqual(result["pending_question"]["id"], "target_mode_select")
        self.assertEqual(
            [item["type"] for item in result["action_queue"]],
            ["set_rpc_mode", "set_qps_mode", "set_observability"],
        )
        self.assertIsNone(result.get("observability_mode"))

        result["last_user_input"] = "fake-node"
        result = process_turn(result)

        self.assertEqual(result["target_mode"], "fake-node")
        self.assertEqual(result["rpc_mode"], "mixed")
        self.assertEqual(result["pending_question"]["id"], "workload_confirm")
        self.assertEqual(
            [item["type"] for item in result["action_queue"]],
            ["set_qps_mode", "set_observability"],
        )

    def test_cross_group_qps_change_survives_target_mode_prerequisite(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("cross-group-prerequisite", language="en")
        state.update({
            "active_group": "endpoint_process",
            "custom_rpc": {"status": "needs_endpoint"},
            "pending_question": {
                "id": "custom_rpc_endpoint",
                "group": "endpoint_process",
                "kind": "url",
                "manual_input_allowed": True,
                "validation": {"value_type": "url"},
            },
            "last_user_input": "Switch the QPS profile to quick.",
        })
        actions = {"actions": [{
            "type": "set_qps_mode",
            "qps_mode": "quick",
            "mutation_explicit": True,
            "source_evidence": "QPS profile to quick",
            "semantic_purpose_verified": True,
            "confidence": "high",
        }]}

        with patch("agent.harness.coordinator.resolve_action_queue", return_value=actions):
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "target_mode_select")
        self.assertEqual(
            [(item["type"], item.get("qps_mode")) for item in result["action_queue"]],
            [("set_qps_mode", "quick")],
        )
        self.assertFalse(result.get("qps_profile"))

        result["last_user_input"] = "fake-node"
        result = process_turn(result)

        self.assertEqual(result["target_mode"], "fake-node")
        self.assertEqual(result["qps_profile"]["mode"], "quick")
        self.assertEqual(result["pending_question"]["id"], "qps_profile_confirm")
        self.assertEqual(result["action_queue"], [])

    def test_single_free_text_resolver_is_the_action_queue(self) -> None:
        """Architecture audit: the older single-action resolver generation

        (`resolve_intent_action`/`_system_prompt`/`_payload`, backed by
        `_route_single_action`) was a strict subset of the action-queue path
        and was deleted so free-text turns route through exactly one resolver.
        Guard against reintroducing a second resolver that could drift again.
        """

        from agent.harness import coordinator, intent

        self.assertTrue(hasattr(intent, "resolve_action_queue"))
        self.assertFalse(hasattr(intent, "resolve_intent_action"))
        self.assertFalse(hasattr(intent, "_system_prompt"))
        self.assertFalse(hasattr(intent, "_payload"))
        self.assertFalse(hasattr(coordinator, "_route_single_action"))

    def test_opening_turn_returns_typed_pending_question(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoints.sqlite"
            runtime = AnyChainGraphRuntime(thread_id="unit-thread", checkpoint_path=checkpoint_path)
            with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
                resolver.return_value = {"actions": [{"type": "greeting", "confidence": "high"}]}
                state = runtime.invoke("Hi", language="en")

        self.assertEqual(state["active_group"], "opening")
        self.assertEqual(state["pending_question"]["id"], "opening_next_action")
        self.assertEqual(state["pending_question"]["kind"], "numbered_choice")
        self.assertFalse(state["pending_question"]["manual_input_allowed"])
        opening = state["visible_response"][0]
        self.assertIn("AnyChain Benchmark Agent", opening)
        self.assertIn("State a test goal", opening)

    def test_group_order_prioritizes_chain_after_target_mode(self) -> None:
        from agent.harness.state import DEFAULT_GROUP_ORDER

        self.assertLess(DEFAULT_GROUP_ORDER.index("chain_identity"), DEFAULT_GROUP_ORDER.index("provider_deployment"))

    def test_graph_state_records_session_metadata(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AnyChainGraphRuntime(
                thread_id="metadata-thread",
                checkpoint_path=Path(tmpdir) / "checkpoints.sqlite",
                session_purpose="chaos",
            )
            state = runtime.snapshot()

        self.assertEqual(state["session"]["id"], "metadata-thread")
        self.assertEqual(state["session"]["purpose"], "chaos")
        self.assertIn("created_at", state["session"])
        self.assertIn("updated_at", state["session"])

    def test_graph_runtime_reset_does_not_checkpoint_invocation_context(self) -> None:
        """Reset clears workflow data without persisting startup read models.

        Discovery and framework facts are re-injected by the Linux terminal on
        each invocation. Persisting them would leak stale machine context into
        a later session; only workflow-owned job/report receipts survive.
        """

        from agent.harness.graph import AnyChainGraphRuntime

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AnyChainGraphRuntime(thread_id="reset-thread", checkpoint_path=Path(tmpdir) / "checkpoints.sqlite")
            runtime._persist_state(
                {
                    "discovery": {"cloud": {"provider": "gcp"}},
                    "framework_summary": {"chain_count": 36},
                    "job": {"job_id": "job_demo", "status": "completed"},
                    "report_context": {"requested_job_id": "job_demo"},
                    "confirmed_config": {"BLOCKCHAIN_NODE": "bsc", "CLOUD_REGION": "us-1"},
                    "target_mode": "fake-node",
                    "chain_identity": {"raw": "bsc", "canonical": "bsc", "status": "confirmed"},
                }
            )
            state = runtime.reset(language="en")

        self.assertEqual(state.get("discovery"), {})
        self.assertEqual(state.get("framework_summary"), {})
        self.assertEqual(state.get("job", {}).get("job_id"), "job_demo")
        self.assertEqual(state.get("report_context", {}).get("requested_job_id"), "job_demo")
        self.assertEqual(state.get("confirmed_config"), {})
        self.assertEqual(state.get("target_mode"), "")
        self.assertEqual(state.get("chain_identity"), {})
        self.assertEqual(state.get("audit_events", [])[-1].get("event"), "workflow_reset")

    def test_agent_graph_state_typed_fields_match_new_state_keys(self) -> None:
        """A field present in `new_state()` but missing from the

        `AgentGraphState` TypedDict is silently dropped by LangGraph's
        `StateGraph` on every `graph.invoke()` round-trip, because
        `StateGraph(AgentGraphState)` only creates channels for fields the
        TypedDict declares. This exact bug shipped with the first
        `advanced_tuning` implementation: it worked perfectly when
        the old coordinator compatibility function was called directly but
        silently reset to empty on every real turn
        through `AnyChainGraphRuntime`, because `advanced_tuning` was never
        added to the `AgentGraphState` class body. No amount of testing
        against a serial compatibility function can catch this class of bug — only
        a real `AnyChainGraphRuntime` round-trip (see
        `test_advanced_tuning_state_survives_langgraph_checkpoint` below)
        or this static field-parity check can. Keep both.
        """

        from agent.harness.state import AgentGraphState, new_state

        typed_fields = set(AgentGraphState.__annotations__.keys())
        new_state_fields = set(new_state("parity-check-thread").keys())
        self.assertEqual(
            new_state_fields - typed_fields,
            set(),
            "Field(s) returned by new_state() are missing from the AgentGraphState "
            "TypedDict and will be silently dropped by LangGraph on every real turn.",
        )
        self.assertEqual(
            typed_fields - new_state_fields,
            set(),
            "Field(s) declared on AgentGraphState are never initialized by new_state().",
        )

    def test_advanced_tuning_state_survives_langgraph_checkpoint(self) -> None:
        """Real `AnyChainGraphRuntime` checkpoint round-trip, not a node-only call

        call, per the note on `test_agent_graph_state_typed_fields_match_new_state_keys`.
        """

        from agent.harness.graph import AnyChainGraphRuntime
        from agent.harness.state import new_state

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoints.sqlite"
            runtime = AnyChainGraphRuntime(thread_id="unit-thread", checkpoint_path=checkpoint_path)
            initial = new_state("unit-thread")
            initial["target_mode"] = "fake-node"
            initial["workflow_mode"] = "rpc_benchmark"
            initial["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed"}
            initial["confirmed_config"] = {
                "CLOUD_REGION": "us-1",
                "CLOUD_ZONE": "us-1-z",
                "MACHINE_TYPE": "n2",
                "LEDGER_DEVICE": "vda",
                "DATA_VOL_TYPE": "hyperdisk-balanced",
                "DATA_VOL_SIZE": "926",
                "DATA_VOL_MAX_IOPS": "20000",
                "DATA_VOL_MAX_THROUGHPUT": "1000",
                "has_accounts_device": False,
                "NETWORK_INTERFACE": "eth0",
                "NETWORK_MAX_BANDWIDTH_GBPS": "100",
            }
            initial["rpc_mode"] = "single"
            initial["workload"] = {"confirmed": True, "choice": "default"}
            initial["qps_profile"] = {"mode": "quick", "confirmed": True}
            initial["observability"] = {"mode": "disabled"}
            initial["active_group"] = "advanced_tuning"
            initial["pending_question"] = {
                "id": "advanced_tuning_confirm",
                "group": "advanced_tuning",
                "kind": "yes_no",
                "field": "advanced_tuning_confirmed",
                "options": [{"label": "Y", "value": True}, {"label": "N", "value": False}],
                "manual_input_allowed": False,
            }
            runtime.graph.update_state({"configurable": {"thread_id": "unit-thread"}}, initial)

            state = runtime.invoke("N", language="en")
            self.assertEqual(state["pending_question"]["id"], "advanced_tuning_adjust_field")
            self.assertEqual(state["advanced_tuning"], {"default_decision_made": True, "confirmed": False})

            state = runtime.invoke("5", language="en")

        self.assertEqual(state["pending_question"]["id"], "advanced_tuning_adjust_value")
        self.assertEqual(state["advanced_tuning"]["adjust_field"], "BOTTLENECK_CPU_THRESHOLD")

    def test_endpoint_probe_evidence_includes_runtime_session_metadata(self) -> None:
        from agent.validators.endpoint_probe import validate_rpc_endpoint

        evidence_path = Path(__file__).resolve().parents[1] / ".agent" / "evidence" / "endpoint-probes" / "unit-session.json"
        env_patch = {
            "ANYCHAIN_AGENT_SESSION_ID": "unit-session",
            "ANYCHAIN_AGENT_SESSION_PURPOSE": "chaos",
            "ANYCHAIN_AGENT_CHECKPOINT_PATH": "/tmp/anychain-unit.sqlite",
        }
        with (
            patch.dict(os.environ, env_patch, clear=False),
            patch("agent.validators.endpoint_probe._write_evidence", return_value=evidence_path),
        ):
            result = validate_rpc_endpoint("bsc", "not-a-url")

        self.assertEqual(result["session"]["id"], "unit-session")
        self.assertEqual(result["session"]["purpose"], "chaos")
        self.assertEqual(result["session"]["checkpoint_path"], "/tmp/anychain-unit.sqlite")

    def test_call_request_handles_empty_body_get_without_typeerror(self) -> None:
        """A GET-shaped request (empty-string body) must not crash `_call_request`.

        Regression for a live chaos finding: `body.encode(...) if isinstance(body,
        str) and body else body` falls through to `else body` for an empty
        string (falsy but still a `str`), passing the literal `""` as `data` to
        `urllib.request.Request`. urllib requires `data` to be `None`/bytes, so
        every GET-method probe raised "TypeError: POST data should be bytes...",
        regardless of endpoint health. GET is the default method for the
        bitcoin_jsonrpc/rest/hedera_dual/tendermint families and some substrate
        mixed methods, so this broke real-node/custom-RPC validation for
        roughly half of all configured chains.
        """

        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        from agent.validators.endpoint_probe import _call_request

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                body = b'{"ok": true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: Any) -> None:  # noqa: D401
                pass

        server = HTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            status, sample = _call_request(
                {"method": "GET", "url": f"http://127.0.0.1:{server.server_port}/", "headers": {}, "body": ""},
                timeout=5.0,
            )
        finally:
            server.shutdown()
            thread.join(timeout=2)
            server.server_close()
        self.assertEqual(status, 200)
        self.assertIn("ok", sample)

    def test_new_chain_endpoint_validation_works_for_generic_pop_families_without_a_template(self) -> None:
        """Case 2 ("new chain in an existing adapter family") was structurally

        broken for every family except `jsonrpc`: `health_probe_methods`
        returned `(None, {})` for any other family, so a template-less chain
        (Case 2's defining condition -- it has no `config/chains/<chain>.json`
        yet) ended up with zero probe methods. `_should_use_generic_jsonrpc_probe`
        then fell through to the template-requiring `_probe_health`/
        `_probe_method` path (`tools/chain_adapters/cli.py health-probe`),
        which raises `FileNotFoundError` for a chain with no template --
        Case 2 endpoint validation could never pass for `substrate`,
        `bitcoin_jsonrpc`, `tendermint`, `rest`, or `hedera_dual`, regardless
        of the real endpoint's health. Found live testing a genuinely new
        `substrate` chain (Moonriver) end to end against a real public RPC.

        Fixed for the two families whose transport is a plain POST JSON-RPC
        call (`substrate`, `bitcoin_jsonrpc`) by giving `health_probe_methods`
        a safe universal fallback method for a template-less chain, same as
        the existing `jsonrpc`/`eth_chainId` design.
        `tendermint`/`rest`/`hedera_dual` are GET/REST-shaped transports the
        generic probe does not build requests for; left as an open item
        (known-issues.md) rather than rushed into this fix.
        """

        import json
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        from agent.validators.endpoint_probe import health_probe_methods, validate_rpc_endpoint

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
                body = json.dumps({"jsonrpc": "2.0", "id": payload.get("id", 1), "result": f"ok:{payload.get('method')}"}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: Any) -> None:  # noqa: D401
                pass

        server = HTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        endpoint = f"http://127.0.0.1:{server.server_port}/"
        try:
            for chain, family, expected_method in (
                ("moonriver-unit-test", "substrate", "system_chain"),
                ("dash-unit-test", "bitcoin_jsonrpc", "getblockchaininfo"),
            ):
                methods, params = health_probe_methods(chain, family)
                self.assertEqual(methods, [expected_method], family)
                result = validate_rpc_endpoint(
                    chain=chain,
                    endpoint=endpoint,
                    methods=methods,
                    adapter_family=family,
                    method_params=params,
                    timeout=3.0,
                )
                self.assertTrue(result["ready"], f"{family}: {result.get('blockers')}")
                self.assertEqual(result["status"], "ok")
        finally:
            server.shutdown()
            thread.join(timeout=2)
            server.server_close()

    def test_new_chain_endpoint_validation_works_for_rest_shaped_families_without_a_template(self) -> None:
        """B.25 (known-issues.md): Case 2 endpoint validation for the

        GET-path/REST-shaped families (`rest`/`tendermint`/`hedera_dual`) was
        left unfixed by the POST-JSON-RPC fix above, since the generic
        JSON-RPC prober cannot build GET requests. Found live testing two
        genuinely new chains: Juno (`tendermint`, `https://juno-api.polkachu.com`)
        and Stellar (`rest`, `https://horizon.stellar.org`) both reproduced the
        identical `CalledProcessError`/`FileNotFoundError` crash.

        Fixed with a family-tiered strategy: `tendermint` gets a genuinely
        universal safe default (`GET /cosmos/base/tendermint/v1beta1/blocks/latest`,
        present on every Cosmos-SDK chain's own SDK-provided REST module,
        confirmed live on cosmos-hub and Juno); `rest`/`hedera_dual` have no
        such universal path (Algorand/Aptos/Cardano/Tezos/TON/Hedera all speak
        completely different REST APIs) and fall back to a bare reachability
        check instead of crashing. Once the user supplies a real GET/POST
        method (the natural next step for any of the three families), the new
        generic REST prober validates it directly.
        """

        import json
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        from agent.validators.endpoint_probe import validate_rpc_endpoint

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/cosmos/base/tendermint/v1beta1/blocks/latest":
                    body = json.dumps({"block": {"header": {"height": "123"}}}).encode()
                    self.send_response(200)
                elif self.path == "/cosmos/staking/v1beta1/pool":
                    body = json.dumps({"pool": {"bonded_tokens": "1"}}).encode()
                    self.send_response(200)
                elif self.path == "/v2/accounts/ABC123":
                    body = json.dumps({"address": "ABC123"}).encode()
                    self.send_response(200)
                else:
                    body = b"not found"
                    self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: Any) -> None:  # noqa: D401
                pass

        server = HTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        endpoint = f"http://127.0.0.1:{server.server_port}"
        try:
            # tendermint: the universal safe default health path succeeds.
            result = validate_rpc_endpoint(chain="juno-unit-test", endpoint=endpoint, adapter_family="tendermint", timeout=3.0)
            self.assertTrue(result["ready"], result.get("blockers"))
            self.assertEqual(result["selected_method"], "GET /cosmos/base/tendermint/v1beta1/blocks/latest")

            # rest (no universal default): a bare reachability check succeeds
            # even against a path this server 404s -- "the server responded"
            # is the whole point, not "this exact path exists."
            result = validate_rpc_endpoint(chain="stellar-unit-test", endpoint=endpoint, adapter_family="rest", timeout=3.0)
            self.assertTrue(result["ready"], result.get("blockers"))
            self.assertIn("bare reachability", " ".join(result.get("warnings") or []))

            # hedera_dual: same bare-reachability fallback as rest.
            result = validate_rpc_endpoint(chain="hedera-dual-unit-test", endpoint=endpoint, adapter_family="hedera_dual", timeout=3.0)
            self.assertTrue(result["ready"], result.get("blockers"))

            # A user-supplied custom GET method on a template-less tendermint
            # chain validates via the generic REST prober, not a crash.
            result = validate_rpc_endpoint(
                chain="juno-unit-test",
                endpoint=endpoint,
                methods=["GET /cosmos/staking/v1beta1/pool"],
                adapter_family="tendermint",
                method_params={},
                timeout=3.0,
            )
            self.assertTrue(result["ready"], result.get("blockers"))

            # Same for a template-less `rest` chain with a path placeholder,
            # filled from schema-evidence-style params.
            result = validate_rpc_endpoint(
                chain="algorand-clone-unit-test",
                endpoint=endpoint,
                methods=["GET /v2/accounts/{address}"],
                adapter_family="rest",
                method_params={"GET /v2/accounts/{address}": ["ABC123"]},
                timeout=3.0,
            )
            self.assertTrue(result["ready"], result.get("blockers"))
        finally:
            server.shutdown()
            thread.join(timeout=2)
            server.server_close()

    def test_evm_endpoint_attestation_rejects_wrong_chain_and_normalizes_hex(self) -> None:
        import json

        from agent.validators.endpoint_probe import validate_rpc_endpoint

        compatible = (200, json.dumps({"jsonrpc": "2.0", "id": 1, "result": False}))
        with patch("agent.validators.endpoint_probe._call_request", side_effect=[
            compatible,
            compatible,
            (200, json.dumps({"jsonrpc": "2.0", "id": 1, "result": "0x1"})),
        ]):
            rejected = validate_rpc_endpoint(
                chain="bsc",
                endpoint="https://ethereum.example.invalid",
                methods=["eth_syncing"],
                adapter_family="jsonrpc",
                method_params={"eth_syncing": []},
            )
        self.assertFalse(rejected["ready"])
        self.assertEqual(rejected["status"], "chain_identity_mismatch")
        self.assertEqual(rejected["identity"]["expected"], "56")
        self.assertEqual(rejected["identity"]["observed"], "1")
        self.assertIn("CHAIN_IDENTITY_MISMATCH", rejected["error"])

        with patch("agent.validators.endpoint_probe._call_request", side_effect=[
            compatible,
            compatible,
            (200, json.dumps({"jsonrpc": "2.0", "id": 1, "result": "0x38"})),
        ]):
            accepted = validate_rpc_endpoint(
                chain="bsc",
                endpoint="https://bsc.example.invalid",
                methods=["eth_syncing"],
                adapter_family="jsonrpc",
                method_params={"eth_syncing": []},
            )
        self.assertTrue(accepted["ready"], accepted.get("blockers"))
        self.assertTrue(accepted["identity"]["verified"])
        self.assertTrue(accepted["attestation_fingerprint"])

    def test_custom_rpc_method_questions_declare_incremental_continuation(self) -> None:
        from agent.harness.domains.chain_rpc import question_for_chain_rpc
        from agent.harness.state import new_state

        custom = new_state("custom-method-continuation", language="en")
        custom["active_group"] = "endpoint_process"
        custom["chain_identity"] = {"canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed"}
        custom["custom_rpc"] = {"status": "needs_method", "endpoint_ready": True}
        question = question_for_chain_rpc(custom, "endpoint_process")
        self.assertEqual(question["id"], "custom_rpc_method")
        self.assertIn("Missing later evidence does not undo", question["completion_effect"])

        new_chain = new_state("new-chain-method-continuation", language="en")
        new_chain["active_group"] = "endpoint_process"
        new_chain["chain_identity"] = {
            "canonical": "new-evm",
            "adapter_family": "jsonrpc",
            "status": "existing_family_needs_method",
        }
        question = question_for_chain_rpc(new_chain, "endpoint_process")
        self.assertEqual(question["id"], "new_chain_method")
        self.assertIn("Missing later evidence does not undo", question["completion_effect"])

    def test_stable_identity_response_sample_requires_semantic_equality(self) -> None:
        from agent.harness.domains.rpc_endpoint import _response_contract_conflicts

        draft = {"response_sample": {"jsonrpc": "2.0", "id": 1, "result": "0x38"}}
        observed = '{"jsonrpc":"2.0","id":1,"result":"0x1"}'
        self.assertIn(
            "stable response mismatch: expected 56, observed 1",
            _response_contract_conflicts(draft, observed, stable_result="1"),
        )

    def test_chain_change_clears_sync_endpoint_and_attestation(self) -> None:
        from agent.harness.state import new_state
        from agent.harness.transitions import invalidate_for_chain_change

        state = new_state("chain-change-attestation", language="en")
        state["confirmed_config"] = {
            "SYNC_OBSERVE_RPC_URL": "https://old.example.invalid",
            "LOCAL_RPC_URL": "https://old.example.invalid",
            "MAINNET_RPC_URL": "https://old-mainnet.example.invalid",
        }
        state["endpoint_evidence"] = {
            "sync_rpc_url_ready": True,
            "sync_rpc_url_probe": {"attestation_fingerprint": "old"},
        }
        invalidate_for_chain_change(state, new_chain="ethereum")
        self.assertNotIn("SYNC_OBSERVE_RPC_URL", state["confirmed_config"])
        self.assertNotIn("LOCAL_RPC_URL", state["confirmed_config"])
        self.assertNotIn("MAINNET_RPC_URL", state["confirmed_config"])
        self.assertEqual(state["endpoint_evidence"], {})

    def test_target_mode_choice_asks_chain_before_provider_values(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoints.sqlite"
            runtime = AnyChainGraphRuntime(thread_id="unit-thread", checkpoint_path=checkpoint_path)
            with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
                resolver.return_value = {"actions": [{"type": "greeting", "confidence": "high"}]}
                runtime.invoke("Hi", language="en")
            state = runtime.invoke("1", language="en")

        self.assertEqual(state["target_mode"], "fake-node")
        self.assertEqual(state["active_group"], "chain_identity")
        self.assertEqual(state["pending_question"]["id"], "chain")

    def test_chain_turn_can_set_target_mode_in_same_intent(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.orientation import opening_question
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["active_group"] = "opening"
        state["last_user_input"] = "我想观察 BNB 节点同步，不想打 RPC 压测流量"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "choose_target_mode", "target_mode": "sync-observe", "target_mode_explicit": True, "source_evidence": "观察 BNB 节点同步", "confidence": "high"},
                    {"type": "choose_chain", "chain_text": "BNB", "source_evidence": "BNB", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["target_mode"], "sync-observe")
        self.assertEqual(result["workflow_mode"], "sync_observe")
        self.assertEqual(result["chain_identity"]["canonical"], "bsc")
        self.assertEqual(result["confirmed_config"]["BLOCKCHAIN_NODE"], "bsc")
        self.assertEqual(result["active_group"], "provider_deployment")

    def test_target_mode_turn_uses_one_complete_typed_plan(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["active_group"] = "opening"
        state["last_user_input"] = "我想观察 BNB 节点同步，不想打 RPC 压测流量"

        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={
                "actions": [
                    {"type": "choose_target_mode", "target_mode": "sync-observe", "target_mode_explicit": True, "source_evidence": "观察 BNB 节点同步", "confidence": "high"},
                    {"type": "choose_chain", "chain_text": "BNB", "source_evidence": "BNB", "confidence": "high"},
                ]
            },
        ):
            result = process_turn(state)

        self.assertEqual(result["target_mode"], "sync-observe")
        self.assertEqual(result["workflow_mode"], "sync_observe")
        self.assertEqual(result["chain_identity"]["canonical"], "bsc")
        self.assertEqual(result["pending_question"]["id"], "CLOUD_REGION")

    def test_multi_action_turn_sets_nonblocking_groups_then_falls_back(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["active_group"] = "opening"
        state["last_user_input"] = "我要用 fake-node 测试 BNB，用 mixed，quick QPS，并开启本地 Grafana"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "choose_target_mode", "target_mode": "fake-node", "target_mode_explicit": True, "source_evidence": "fake-node", "confidence": "high"},
                    {"type": "choose_chain", "chain_text": "BNB", "source_evidence": "BNB", "confidence": "high"},
                    {"type": "set_rpc_mode", "rpc_mode": "mixed", "mutation_explicit": True, "source_evidence": "mixed", "confidence": "high"},
                    {"type": "set_qps_mode", "qps_mode": "quick", "mutation_explicit": True, "source_evidence": "quick", "confidence": "high"},
                    {"type": "set_observability", "observability_mode": "local", "mutation_explicit": True, "source_evidence": "本地 Grafana", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["target_mode"], "fake-node")
        self.assertEqual(result["chain_identity"]["canonical"], "bsc")
        self.assertEqual(result["rpc_mode"], "mixed")
        self.assertEqual(result["pending_question"]["id"], "workload_confirm")
        self.assertTrue(result["pending_question"].get("resume_action_queue"))
        self.assertEqual(len(result["completed_actions"]), 3)

        result["last_user_input"] = "1"
        resumed = process_turn(result)

        self.assertEqual(resumed["qps_profile"]["mode"], "quick")
        self.assertEqual(resumed["pending_question"]["id"], "qps_profile_confirm")
        self.assertEqual(len(resumed["action_queue"]), 1)

        resumed["last_user_input"] = "Y"
        completed = process_turn(resumed)
        self.assertEqual(completed["observability"]["mode"], "local")
        self.assertEqual(completed["pending_question"]["id"], "CLOUD_REGION")

    def test_capability_action_does_not_swallow_benchmark_goal_in_queue(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["active_group"] = "opening"
        state["last_user_input"] = "我想先随便跑通一下框架，但不确定 fake-node 还是 real-node；可能测 BNB，也想看看支持哪些链"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "ask_capabilities", "confidence": "high"},
                    {"type": "choose_chain", "chain_text": "BNB", "source_evidence": "BNB", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        visible = "\n".join(result.get("visible_response") or [])
        self.assertIn("当前框架", visible)
        self.assertNotIn("{'chain':", visible)
        self.assertIn("已确认链为 `bsc`", visible)
        self.assertEqual(result["chain_identity"]["canonical"], "bsc")
        self.assertEqual(result.get("target_mode"), "")
        self.assertEqual(result["pending_question"]["id"], "target_mode_select")
        self.assertEqual(
            [item.get("type") for item in result["completed_actions"]],
            ["choose_chain", "request_target_mode_selection"],
        )

    def test_consultation_return_cannot_reopen_confirmed_target_mode(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.chain_rpc_questions import _chain_question
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "chain_identity"
        state["pending_question"] = _chain_question(state)
        state["last_user_input"] = "回到 benchmark 设置"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {
                        "type": "request_target_mode_selection",
                        "source_evidence": "回到 benchmark 设置",
                        "confidence": "high",
                    }
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["target_mode"], "fake-node")
        self.assertEqual(result["active_group"], "chain_identity")
        self.assertEqual(result["pending_question"]["id"], "chain")

    def test_explicit_group_navigation_can_reopen_confirmed_target_mode(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.chain_rpc_questions import _chain_question
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "chain_identity"
        state["pending_question"] = _chain_question(state)
        state["last_user_input"] = "切换目标模式"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {
                        "type": "change_group",
                        "group": "target_mode",
                        "navigation_explicit": True,
                        "source_evidence": "切换目标模式",
                        "confidence": "high",
                    }
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["target_mode"], "fake-node")
        self.assertEqual(result["active_group"], "target_mode")
        self.assertEqual(result["pending_question"]["id"], "target_mode_select")

    def test_resume_current_flow_renders_pending_without_changing_group(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.chain_rpc_questions import _chain_question
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "chain_identity"
        state["pending_question"] = _chain_question(state)
        state["last_user_input"] = "回到 benchmark 设置"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [{
                    "type": "resume_current_flow",
                    "source_evidence": "回到 benchmark 设置",
                    "confidence": "high",
                }]
            }
            result = process_turn(state)

        self.assertEqual(result["target_mode"], "fake-node")
        self.assertEqual(result["active_group"], "chain_identity")
        self.assertEqual(result["pending_question"]["id"], "chain")
        self.assertIn("哪条链", "\n".join(result.get("visible_response") or []))
        self.assertEqual(
            [item.get("type") for item in (result.get("turn_context") or {}).get("admitted_actions") or []],
            ["resume_current_flow"],
        )
        self.assertNotIn("不像当前问题的答案", "\n".join(result.get("visible_response") or []))

    def test_registered_resume_semantic_selects_the_declared_resume_option(self) -> None:
        from agent.harness.domains.orientation import resume_question
        from agent.harness.domains.performance import question_for_performance
        from agent.harness.state import new_state
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn

        state = new_state("resume-semantic", language="en")
        state.update({
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "qps_profile": {"mode": "quick", "confirmed": False, "default_decision_made": False},
        })
        saved_question = question_for_performance(state, "qps_profile") or {}
        state["resume_context"] = {
            "active_group": "qps_profile",
            "pending_question": saved_question,
        }
        deferred = [
            {
                "action_id": "deferred-chain",
                "type": "choose_chain",
                "chain_text": "BNB",
                "source_evidence": "keep BNB",
            },
            {
                "action_id": "deferred-observability",
                "type": "set_observability",
                "observability_mode": "local",
                "mutation_explicit": True,
                "source_evidence": "keep local observability",
            },
        ]
        state["action_queue"] = deferred
        state["pending_question"] = resume_question(state)
        state["active_group"] = "opening"
        state["last_user_input"] = "Continue the saved configuration."

        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value=_admitted_mock_plan(state, state["last_user_input"], {"actions": [{
                "type": "answer_pending",
                "selected_value": "continue",
                "source_evidence": "Continue the saved configuration",
                "confidence": "high",
            }]}),
        ):
            result = process_turn(state)

        self.assertEqual(result["active_group"], "qps_profile")
        self.assertEqual(result["pending_question"]["id"], "qps_profile_confirm")
        self.assertEqual(result.get("resume_context"), {})
        self.assertEqual(result.get("action_queue"), deferred)
        admitted = (result.get("turn_context") or {}).get("admitted_actions") or []
        self.assertTrue(admitted)
        self.assertTrue(all(item.get("type") == "answer_pending" for item in admitted))

    def test_opening_menu_info_option_shows_capability_content_not_a_dead_loop(self) -> None:
        """Selecting the opening menu's 4th numbered option ("learn supported

        chains, RPC methods, and extension paths", value `"info"`) used to be
        a complete dead loop: `group == "opening"`'s handler set `active_group`
        to `report_artifact_analysis` but never appended any content, and
        since `target_mode` was still unset, the very same turn's
        `_ask_next_blocking_question` call recomputed the next group via
        routing and got "opening" again (routing's first check is `if not
        target_mode: return "opening"`) -- the user saw the identical opening
        menu again with zero information ever shown, repeatable indefinitely.

        The free-text equivalent (the `ask_capabilities` action, e.g. "what
        chains do you support") already rendered real content correctly; this
        was two disconnected code paths for the same stated capability, only
        one of which worked. Found via live manual testing (not caught by
        prior dual-AI chaos: chaos turns were consistently generated as
        free-text stress cases for the LLM resolver, never as a literal
        numbered-menu selection of a purely informational option).
        """

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.orientation import opening_question
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["active_group"] = "opening"
        state["pending_question"] = opening_question(state)
        state["last_user_input"] = "4"
        result = process_turn(state)

        visible = "\n".join(result.get("visible_response") or [])
        self.assertIn("当前框架", visible)
        self.assertEqual(result.get("target_mode"), "")
        # Content was shown and the turn stopped -- it must not have silently
        # re-rendered the identical opening menu a second time in this same
        # response (the old dead-loop behavior).
        self.assertEqual(visible.count("了解支持的链"), 0)

        # The natural next step is still correctly "opening" (unchanged target
        # mode), so the next real turn will offer the same menu again -- this
        # one-time info display must not have permanently broken navigation.
        from agent.harness.coordinator import _next_group
        self.assertEqual(_next_group(result), "opening")

    def test_opening_menu_sync_observe_label_names_the_framework_term(self) -> None:
        """Found live via user manual testing (2026-07-13): the opening

        target-mode menu's fake-node/real-node options keep the literal
        framework term in their Chinese label ("启动 fake-node 测试", "启动
        real-node 测试"), but the sync-observe option was fully translated
        away as "观察节点同步" with no mention of `sync-observe` anywhere --
        inconsistent with the other two options, and confusing because every
        other place in the harness (prompts, status dumps, docs) refers to
        this mode by its literal name. Fixed so all three mode options name
        their framework term consistently.
        """

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.orientation import opening_question
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        result = process_turn(state)
        pending = result.get("pending_question") or {}
        labels = [str(option.get("label") or "") for option in pending.get("options") or []]
        sync_observe_label = next((label for label in labels if "sync-observe" in label), "")
        self.assertTrue(sync_observe_label, f"no sync-observe option named the term: {labels}")

    def test_analyze_report_action_returns_visible_job_entrypoint(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["active_group"] = "opening"
        state["job"] = {"job_id": "job_demo", "status": "completed"}
        state["last_user_input"] = "看最近任务报告"

        with (
            patch("agent.harness.coordinator.resolve_action_queue") as resolver,
            patch("agent.harness.domains.analysis.resume_job") as resume,
            patch("agent.harness.domains.analysis.list_jobs", return_value=[]),
        ):
            resolver.return_value = {"actions": [{"type": "analyze_report", "confidence": "high"}]}
            resume.return_value = {
                "job_id": "job_demo",
                "status": "completed",
                "run_dir": ".agent/jobs/job_demo",
                "runtime_env_file": ".agent/jobs/job_demo/runtime.env",
                "artifact_index": ".agent/jobs/job_demo/artifact_index.json",
                "next_actions": ["status", "analyze", "artifact-qa"],
            }
            result = process_turn(state)

        visible = "\n".join(result.get("visible_response") or [])
        self.assertIn("job_demo", visible)
        self.assertIn("artifact_index", visible)
        self.assertEqual(result.get("pending_question"), {})

    def test_report_consultation_does_not_discard_same_turn_qps_change(self) -> None:
        from agent.harness.coordinator import _process_action_queue
        from agent.harness.state import new_state

        state = new_state("report-and-change", language="en")
        state.update({
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "chain_identity": {"canonical": "bsc", "status": "confirmed"},
            "job": {"job_id": "job_demo", "status": "completed"},
            "active_group": "qps_profile",
        })
        actions = [
            {"type": "analyze_report", "confidence": "high", "subject": "latest"},
            {
                "type": "set_qps_mode",
                "confidence": "high",
                "qps_mode": "quick",
                "mutation_explicit": True,
                "source_evidence": "quick",
            },
        ]
        with (
            patch("agent.harness.domains.analysis.resume_job", return_value={
                "job_id": "job_demo",
                "status": "completed",
                "run_dir": ".agent/jobs/job_demo",
                "artifacts": {},
                "plan": {},
            }),
            patch("agent.harness.domains.analysis.list_jobs", return_value=[]),
        ):
            result = _process_action_queue(state, actions, "analyze latest and set quick")

        self.assertEqual((result.get("qps_profile") or {}).get("mode"), "quick")
        self.assertEqual((result.get("pending_question") or {}).get("id"), "qps_profile_confirm")
        self.assertIn("job_demo", "\n".join(result.get("visible_response") or []))

    def test_report_consultation_runs_before_custom_rpc_and_leaves_rpc_question_active(self) -> None:
        from agent.harness.coordinator import _process_action_queue
        from agent.harness.state import new_state

        state = new_state("report-and-custom-rpc", language="en")
        state.update({
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "chain_identity": {"canonical": "bsc", "status": "confirmed", "adapter_family": "jsonrpc"},
            "job": {"job_id": "job_demo", "status": "completed"},
            "rpc_mode": "single",
            "qps_profile": {"mode": "quick", "confirmed": True},
            "confirmed_config": {"LEDGER_DEVICE": "vda", "NETWORK_INTERFACE": "eth0"},
            "active_group": "advanced_tuning",
        })
        actions = [
            {"type": "rpc_catalog_command", "catalog_command": "enter", "source_evidence": "enter custom RPC setup", "confidence": "high"},
            {"type": "analyze_report", "confidence": "high", "subject": "latest"},
        ]
        with (
            patch("agent.harness.domains.analysis.resume_job", return_value={
                "job_id": "job_demo",
                "status": "completed",
                "run_dir": ".agent/jobs/job_demo",
                "artifacts": {},
                "plan": {},
            }),
            patch("agent.harness.domains.analysis.list_jobs", return_value=[]),
        ):
            result = _process_action_queue(state, actions, "analyze latest, then enter custom RPC setup")

        self.assertEqual((result.get("custom_rpc") or {}).get("status"), "needs_endpoint")
        self.assertEqual((result.get("pending_question") or {}).get("id"), "custom_rpc_endpoint")
        self.assertEqual((result.get("qps_profile") or {}).get("mode"), "quick")
        self.assertEqual((result.get("confirmed_config") or {}).get("LEDGER_DEVICE"), "vda")
        self.assertIn("job_demo", "\n".join(result.get("visible_response") or []))

    def test_compound_rpc_custom_qps_and_observability_respects_domain_dependencies(self) -> None:
        from agent.harness.coordinator import _process_action_queue
        from agent.harness.state import new_state

        state = new_state("compound-domain-order", language="en")
        state.update({
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "chain_identity": {"canonical": "bsc", "status": "confirmed", "adapter_family": "jsonrpc"},
            "confirmed_config": {"LEDGER_DEVICE": "vda", "NETWORK_INTERFACE": "eth0"},
            "active_group": "workload_rpc",
        })
        actions = [
            {"type": "set_rpc_mode", "rpc_mode": "single", "mutation_explicit": True, "source_evidence": "single", "confidence": "high"},
            {"type": "set_qps_mode", "qps_mode": "quick", "mutation_explicit": True, "source_evidence": "quick", "confidence": "high"},
            {"type": "set_observability", "observability_mode": "disabled", "mutation_explicit": True, "source_evidence": "observability disabled", "confidence": "high"},
            {"type": "rpc_catalog_command", "catalog_command": "enter", "source_evidence": "custom RPC", "confidence": "high"},
        ]

        result = _process_action_queue(state, actions, "single, custom RPC, quick, observability disabled")

        self.assertEqual((result.get("custom_rpc") or {}).get("status"), "needs_endpoint")
        self.assertEqual((result.get("pending_question") or {}).get("id"), "custom_rpc_endpoint")
        self.assertEqual(
            [item.get("type") for item in result.get("action_queue") or []],
            ["set_qps_mode", "set_observability"],
        )
        prompt = str((result.get("pending_question") or {}).get("prompt") or "")
        self.assertEqual(sum(prompt in item for item in result.get("visible_response") or []), 1)

    def test_consultation_does_not_let_mutation_bypass_blocking_question(self) -> None:
        from agent.harness.coordinator import _process_action_queue
        from agent.harness.domains.chain_rpc import question_for_chain_rpc
        from agent.harness.state import new_state

        state = new_state("consultation-barrier", language="en")
        state.update({
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "chain_identity": {"canonical": "bsc", "status": "confirmed"},
            "rpc_mode": "single",
            "active_group": "workload_rpc",
        })
        state["pending_question"] = question_for_chain_rpc(state, "workload_rpc") or {}
        state["pending_question"]["created_turn_index"] = 1
        state["turn_index"] = 1
        actions = [
            {"type": "answer_opening_question", "topic": "current_config", "confidence": "high"},
            {
                "type": "set_qps_mode",
                "qps_mode": "quick",
                "confidence": "high",
                "mutation_explicit": True,
                "source_evidence": "quick",
            },
        ]

        result = _process_action_queue(state, actions, "what will run, then use quick")

        self.assertEqual((result.get("pending_question") or {}).get("id"), "workload_confirm")
        self.assertEqual([item["type"] for item in result.get("action_queue") or []], ["set_qps_mode"])
        self.assertFalse(result.get("qps_profile"))
        visible = "\n".join(result.get("visible_response") or [])
        self.assertIn("Use defaults", visible)
        self.assertIn("Add custom RPC method", visible)

    def test_new_turn_explicit_action_can_interrupt_older_blocking_question(self) -> None:
        from agent.harness.coordinator import _process_action_queue
        from agent.harness.domains.performance import question_for_performance
        from agent.harness.state import new_state

        state = new_state("new-turn-interruption", language="en")
        state.update({
            "turn_index": 4,
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "chain_identity": {"canonical": "bsc", "status": "confirmed"},
            "rpc_mode": "single",
            "workload": {"confirmed": True, "methods": ["eth_getBalance"], "weights": {}},
            "qps_profile": {"mode": "quick", "confirmed": False},
            "active_group": "qps_profile",
        })
        state["pending_question"] = question_for_performance(state, "qps_profile") or {}
        state["pending_question"]["created_turn_index"] = 3

        result = _process_action_queue(
            state,
            [{"type": "rpc_catalog_command", "catalog_command": "enter", "source_evidence": "Take me to custom RPC", "confidence": "high"}],
            "Take me to custom RPC",
        )

        self.assertEqual(result.get("active_group"), "endpoint_process")
        self.assertEqual((result.get("custom_rpc") or {}).get("status"), "needs_endpoint")
        self.assertTrue(result.get("interruption_stack"))

    def test_named_detour_preserves_and_resumes_partial_custom_rpc_state(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.chain_rpc import question_for_chain_rpc
        from agent.harness.state import new_state

        state = new_state("custom-rpc-detour-resume", language="en")
        state.update({
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "chain_identity": {
                "raw": "bsc",
                "canonical": "bsc",
                "adapter_family": "jsonrpc",
                "status": "confirmed",
            },
            "active_group": "endpoint_process",
            "custom_rpc": {
                "status": "needs_method",
                "endpoint": "http://geth-dev:8545",
                "endpoint_ready": True,
            },
        })
        state["pending_question"] = question_for_chain_rpc(state, "endpoint_process") or {}
        state["last_user_input"] = "Visit observability settings first."
        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{
                "type": "change_group",
                "group": "observability",
                "navigation_explicit": True,
                "source_evidence": "Visit observability settings first.",
            }]},
        ):
            detour = process_turn(state)

        self.assertEqual(detour["pending_question"]["group"], "observability")
        self.assertEqual(detour["custom_rpc"]["status"], "needs_method")
        self.assertEqual(detour["interruption_stack"][-1]["question_id"], "custom_rpc_method")

        detour["last_user_input"] = "1"
        resumed = process_turn(detour)

        self.assertEqual(resumed["observability"]["mode"], "disabled")
        self.assertEqual(resumed["custom_rpc"]["status"], "needs_method")
        self.assertEqual(resumed["pending_question"]["id"], "custom_rpc_method")

    def test_named_detour_resumes_registered_field_reconfiguration_question(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.environment import question_for_environment_field
        from agent.harness.state import new_state

        state = new_state("field-detour-resume", language="en")
        state.update({
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "active_group": "provider_deployment",
            "confirmed_config": {"CLOUD_REGION": "us-east1"},
        })
        state["pending_question"] = question_for_environment_field(
            state,
            "provider_deployment",
            "CLOUD_REGION",
        ) or {}
        state["last_user_input"] = "Visit observability settings first."
        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{
                "type": "change_group",
                "group": "observability",
                "navigation_explicit": True,
                "source_evidence": state["last_user_input"],
            }]},
        ):
            detour = process_turn(state)

        self.assertEqual(detour["pending_question"]["id"], "observability_mode")
        self.assertEqual(detour["interruption_stack"][-1]["field"], "CLOUD_REGION")
        detour["last_user_input"] = "1"
        resumed = process_turn(detour)
        self.assertEqual(resumed["pending_question"]["id"], "CLOUD_REGION")
        self.assertEqual(resumed["pending_question"]["field"], "CLOUD_REGION")
        self.assertEqual(resumed["confirmed_config"]["CLOUD_REGION"], "us-east1")

    def test_named_detour_resumes_case2_schema_evidence_question(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.chain_rpc import question_for_chain_rpc
        from agent.harness.state import new_state

        state = new_state("case2-detour-resume", language="en")
        state.update({
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "active_group": "endpoint_process",
            "chain_identity": {
                "raw": "flow",
                "canonical": "flow",
                "adapter_family": "jsonrpc",
                "status": "existing_family_needs_schema_evidence",
                "case": "case2",
                "candidate_method": "eth_blockNumber",
            },
            "endpoint_evidence": {
                "candidate_endpoint": "http://fake-node:19000",
                "candidate_endpoint_ready": True,
            },
        })
        state["pending_question"] = question_for_chain_rpc(state, "endpoint_process") or {}
        self.assertEqual(state["pending_question"]["id"], "new_chain_schema_evidence")
        state["last_user_input"] = "Visit observability settings first."
        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{
                "type": "change_group",
                "group": "observability",
                "navigation_explicit": True,
                "source_evidence": state["last_user_input"],
            }]},
        ):
            detour = process_turn(state)

        self.assertEqual(detour["pending_question"]["id"], "observability_mode")
        self.assertEqual(detour["interruption_stack"][-1]["question_id"], "new_chain_schema_evidence")
        detour["last_user_input"] = "1"
        resumed = process_turn(detour)
        self.assertEqual(resumed["pending_question"]["id"], "new_chain_schema_evidence")
        self.assertEqual(_catalog_draft(resumed)["method"], "eth_blockNumber")

    def test_case2_endpoint_answer_preserves_future_method_and_named_detour(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.chain_rpc import question_for_chain_rpc
        from agent.harness.state import new_state

        source = (
            "Use http://fake-node:19000 for validation. "
            "The method is eth_blockNumber with no params. "
            "Before the response schema, take me to observability settings."
        )
        state = new_state("case2-answer-future-actions", language="en")
        state.update({
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "active_group": "endpoint_process",
            "chain_identity": {
                "raw": "flow",
                "canonical": "flow",
                "adapter_family": "jsonrpc",
                "status": "existing_family_needs_endpoint",
                "case": "case2",
            },
            "last_user_input": source,
        })
        state["pending_question"] = question_for_chain_rpc(state, "endpoint_process") or {}
        self.assertEqual(state["pending_question"]["id"], "new_chain_endpoint")
        actions = [
            {
                "type": "answer_pending",
                "answer": "http://fake-node:19000",
                "source_evidence": "Use http://fake-node:19000 for validation.",
                "confidence": "high",
            },
            {
                "type": "rpc_catalog_command",
                "catalog_command": "set_method",
                "rpc_method": "eth_blockNumber",
                "source_evidence": "The method is eth_blockNumber with no params.",
                "confidence": "high",
            },
            {
                "type": "change_group",
                "group": "observability",
                "navigation_explicit": True,
                "group_navigation_semantic_verified": True,
                "source_evidence": "take me to observability settings",
                "confidence": "high",
            },
        ]
        probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}
        with (
            patch(
                "agent.harness.coordinator.resolve_action_queue",
                side_effect=_admitted_mock_resolver({"actions": actions}),
            ),
            patch("agent.harness.domains.rpc_endpoint.validate_rpc_endpoint", return_value=probe),
        ):
            detour = process_turn(state)

        self.assertEqual(detour["pending_question"]["id"], "observability_mode")
        self.assertEqual(_catalog_draft(detour)["method"], "eth_blockNumber")
        self.assertEqual(
            [row["question_id"] for row in detour["interruption_stack"]],
            ["new_chain_schema_evidence"],
        )
        detour["last_user_input"] = "1"
        resumed = process_turn(detour)
        self.assertEqual(resumed["pending_question"]["id"], "new_chain_schema_evidence")
        self.assertEqual(_catalog_draft(resumed)["method"], "eth_blockNumber")
        self.assertEqual(resumed["interruption_stack"], [])

    def test_go_back_consumes_interruption_before_domain_resume_fallback(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.chain_rpc import question_for_chain_rpc
        from agent.harness.state import new_state

        state = new_state("go-back-interruption-owner", language="en")
        state.update({
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "chain_identity": {"canonical": "bsc", "status": "confirmed"},
            "active_group": "endpoint_process",
            "custom_rpc": {"status": "needs_method", "endpoint_ready": True},
            "confirmed_config": {"CLOUD_REGION": "us-east1"},
            "interruption_stack": [{
                "group": "provider_deployment",
                "question_id": "CLOUD_REGION",
                "field": "CLOUD_REGION",
                "reason": "explicit_navigation",
            }],
        })
        state["pending_question"] = question_for_chain_rpc(state, "endpoint_process") or {}
        state["last_user_input"] = "Go back."
        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{
                "type": "go_back",
                "source_evidence": state["last_user_input"],
            }]},
        ):
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "CLOUD_REGION")
        self.assertEqual(result["active_group"], "provider_deployment")
        self.assertEqual(result["interruption_stack"], [])
        self.assertEqual(result["custom_rpc"], {})

    def test_deferred_actions_survive_next_turn_and_resume_after_barrier_answer(self) -> None:
        from agent.harness.coordinator import _process_action_queue
        from agent.harness.domains.chain_rpc import question_for_chain_rpc
        from agent.harness.state import new_state

        state = new_state("durable-queue", language="en")
        state.update({
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "chain_identity": {"canonical": "bsc", "status": "confirmed"},
            "rpc_mode": "single",
            "active_group": "workload_rpc",
            "action_queue": [
                {"type": "set_qps_mode", "qps_mode": "quick", "mutation_explicit": True, "source_evidence": "quick", "confidence": "high"},
                {"type": "set_observability", "observability_mode": "disabled", "mutation_explicit": True, "source_evidence": "observability disabled", "confidence": "high"},
            ],
        })
        state["pending_question"] = question_for_chain_rpc(state, "workload_rpc") or {}

        result = _process_action_queue(
            state,
            [{"type": "use_default_workload", "confidence": "high"}],
            "Use defaults",
        )

        self.assertTrue((result.get("workload") or {}).get("confirmed"))
        self.assertEqual((result.get("qps_profile") or {}).get("mode"), "quick")
        self.assertEqual((result.get("pending_question") or {}).get("id"), "qps_profile_confirm")
        self.assertEqual(
            [item["type"] for item in result.get("action_queue") or []],
            ["set_observability"],
        )

    def test_new_mutation_supersedes_deferred_value_in_same_dimension(self) -> None:
        from agent.harness.coordinator import _merge_durable_action_queue

        merged = _merge_durable_action_queue(
            [{"type": "set_qps_mode", "qps_mode": "quick"}],
            [{"type": "set_qps_mode", "qps_mode": "standard"}],
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["qps_mode"], "standard")

    def test_job_reference_after_evidence_paste_does_not_reanalyze_stale_evidence(self) -> None:
        """A turn naming a specific job must not be hijacked by a stale evidence_buffer entry.

        Regression for a live chaos finding: `evidence_buffer` only ever grows
        (nothing clears it), and the deterministic gate at the top of
        `process_turn` re-analyzes `evidence_buffer[-1]` for ANY text matching a
        broad keyword list ("分析", "why", "fix", ...) whenever the buffer is
        non-empty — with no exception for a later turn that explicitly names a
        real, different job by id ("分析 job_2026...而失败") or asks about "the
        latest job" ("分析最新的 job"). Both were swallowed into re-analyzing the
        FIRST pasted evidence forever, ignoring the named job entirely.
        """

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["active_group"] = "opening"
        state["evidence_buffer"] = [{"text": "Traceback (most recent call last):\nRuntimeError('endpoint probe failed: connection refused')"}]

        # Still routes to stale-evidence analysis when there's no job reference —
        # existing behavior for a genuine follow-up about the pasted evidence.
        state["last_user_input"] = "分析一下这个原因"
        with (
            patch("agent.harness.coordinator.resolve_action_queue", return_value={"actions": [{
                "type": "analyze_evidence",
                "evidence": state["evidence_buffer"][-1]["text"],
                "confidence": "high",
            }]}),
            patch(
                "agent.harness.domains.analysis.analyze_evidence_with_model",
                return_value="connection refused",
            ),
        ):
            no_job_result = process_turn(state)
        self.assertIn("connection refused", "\n".join(no_job_result.get("visible_response") or []))

        # A turn naming "the latest job" must fall through to normal routing
        # (the resolver's analyze_report action), not the stale evidence.
        job_state = new_state("unit-thread-2", language="zh")
        job_state["active_group"] = "opening"
        job_state["evidence_buffer"] = [{"text": "Traceback (most recent call last):\nRuntimeError('endpoint probe failed: connection refused')"}]
        job_state["job"] = {"job_id": "job_20260712163828_76ac0a38", "status": "failed"}
        job_state["last_user_input"] = "分析最新的 job"
        with (
            patch("agent.harness.coordinator.resolve_action_queue") as resolver,
            patch("agent.harness.domains.analysis.resume_job") as resume,
            patch("agent.harness.domains.analysis.list_jobs", return_value=[]),
        ):
            resolver.return_value = {"actions": [{"type": "analyze_report", "confidence": "high"}]}
            resume.return_value = {
                "job_id": "job_20260712163828_76ac0a38",
                "status": "failed",
                "run_dir": ".agent/jobs/job_20260712163828_76ac0a38",
                "runtime_env_file": ".agent/jobs/job_20260712163828_76ac0a38/runtime.env",
                "artifact_index": ".agent/jobs/job_20260712163828_76ac0a38/artifact_index.json",
                "next_actions": ["status", "analyze"],
            }
            job_result = process_turn(job_state)
        job_visible = "\n".join(job_result.get("visible_response") or [])
        self.assertIn("job_20260712163828_76ac0a38", job_visible)
        self.assertNotIn("connection refused", job_visible)

        # Explicitly naming a specific job id must also bypass the stale evidence.
        explicit_state = new_state("unit-thread-3", language="zh")
        explicit_state["active_group"] = "opening"
        explicit_state["evidence_buffer"] = [{"text": "Traceback (most recent call last):\nRuntimeError('endpoint probe failed: connection refused')"}]
        explicit_state["last_user_input"] = "分析 job_20260712163828_76ac0a38 为什么失败"
        with (
            patch("agent.harness.coordinator.resolve_action_queue") as resolver,
            patch("agent.harness.domains.analysis.resume_job") as resume,
        ):
            resolver.return_value = {"actions": [{"type": "analyze_report", "confidence": "high"}]}
            resume.return_value = {
                "job_id": "job_20260712163828_76ac0a38",
                "status": "failed",
                "run_dir": ".agent/jobs/job_20260712163828_76ac0a38",
                "runtime_env_file": ".agent/jobs/job_20260712163828_76ac0a38/runtime.env",
                "artifact_index": ".agent/jobs/job_20260712163828_76ac0a38/artifact_index.json",
                "next_actions": ["status", "analyze"],
            }
            explicit_result = process_turn(explicit_state)
        explicit_visible = "\n".join(explicit_result.get("visible_response") or [])
        self.assertIn("job_20260712163828_76ac0a38", explicit_visible)
        self.assertNotIn("connection refused", explicit_visible)

    def test_endpoint_pending_treats_error_line_as_evidence_not_url_answer(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "sync-observe"
        state["workflow_mode"] = "sync_observe"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["active_group"] = "endpoint_process"
        state["pending_question"] = {
            "group": "endpoint_process",
            "id": "SYNC_OBSERVE_RPC_URL",
            "field": "SYNC_OBSERVE_RPC_URL",
            "kind": "url",
            "manual_input_allowed": True,
            "prompt": "Provide endpoint.",
        }
        state["last_user_input"] = "RuntimeError: prometheus exporter port 9108 already in use"

        with (
            patch("agent.harness.domains.rpc_endpoint.validate_rpc_endpoint") as probe,
            patch("agent.harness.coordinator.resolve_action_queue") as resolver,
            patch("agent.harness.domains.analysis.analyze_evidence_with_model", return_value="证据分析完成"),
        ):
            resolver.return_value = {
                "actions": [{
                    "type": "analyze_evidence",
                    "evidence": state["last_user_input"],
                    "confidence": "high",
                }]
            }
            result = process_turn(state)

        probe.assert_not_called()
        self.assertIn("evidence_buffer", result)
        self.assertEqual(result["pending_question"]["id"], "SYNC_OBSERVE_RPC_URL")
        self.assertIn("证据分析完成", "\n".join(result.get("visible_response") or []))

    def test_final_endpoint_pending_requires_bare_endpoint_not_complex_intent(self) -> None:
        from agent.harness.coordinator import _answer_fits_pending

        pending = {
            "id": "LOCAL_RPC_URL",
            "group": "endpoint_process",
            "kind": "url",
            "field": "LOCAL_RPC_URL",
            "manual_input_allowed": True,
        }

        self.assertTrue(_answer_fits_pending("https://example.invalid/rpc", pending))
        self.assertFalse(_answer_fits_pending(
            "我要改测 Flow，endpoint 用 https://example.invalid/rpc，这只是验证 method，不是最终压测 endpoint",
            pending,
        ))

    def test_multiline_config_proposal_requires_confirmation_before_apply(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "solana", "canonical": "solana", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "solana"}
        state["active_group"] = "provider_deployment"
        state["last_user_input"] = """这是从环境里复制出来的：
cloud:
  region: asia-east1
  zone: asia-east1-c
machine: n2-standard-16
disk:
  ledger: vda
  size_gib: 926
"""

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {
                        "type": "propose_config_values",
                        "source_format": "yaml",
                        "config_values": {
                            "CLOUD_REGION": "asia-east1",
                            "CLOUD_ZONE": "asia-east1-c",
                            "MACHINE_TYPE": "n2-standard-16",
                            "LEDGER_DEVICE": "vda",
                            "DATA_VOL_SIZE": "926",
                        },
                        "unmapped_values": {"disk.ledger": "vda"},
                        "confidence": "high",
                    }
                ]
            }
            resolver.side_effect = _admitted_mock_resolver(resolver.return_value)
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "inferred_config_review")
        self.assertNotIn("CLOUD_REGION", result["confirmed_config"])
        self.assertNotIn("LEDGER_DEVICE", result["confirmed_config"])
        self.assertIn("CLOUD_REGION", result["visible_response"][0])
        proposal = result["inferred_config"]["pending_review"]
        self.assertEqual(proposal["config_values"]["LEDGER_DEVICE"], "vda")
        self.assertEqual(proposal["unmapped_values"], {})

        result["last_user_input"] = "Y"
        result = process_turn(result)

        self.assertEqual(result["confirmed_config"]["CLOUD_REGION"], "asia-east1")
        self.assertEqual(result["confirmed_config"]["CLOUD_ZONE"], "asia-east1-c")
        self.assertEqual(result["confirmed_config"]["MACHINE_TYPE"], "n2-standard-16")
        self.assertEqual(result["confirmed_config"]["LEDGER_DEVICE"], "vda")
        self.assertEqual(result["confirmed_config"]["DATA_VOL_SIZE"], "926")
        self.assertEqual(result["pending_question"]["id"], "DATA_VOL_TYPE")
        self.assertIn("accepted_reviews", result["inferred_config"])

    def test_multiline_config_paste_is_preserved_by_the_single_typed_plan(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["active_group"] = "opening"
        state["last_user_input"] = """我要用 fake-node 测试 BNB，用 mixed，QPS quick，并开启本地 Grafana。下面是从环境里复制的配置：
cloud:
  region: asia-east1
  zone: asia-east1-c
  machine_type: n2-standard-16
disk:
  ledger_device: vda
  data_vol_type: hyperdisk-balanced
  data_vol_size: 926
  data_vol_max_iops: 20000
  data_vol_max_throughput: 1000
network:
  interface: eth0
  bandwidth_gbps: 100"""

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "choose_target_mode", "target_mode": "fake-node", "target_mode_explicit": True, "source_evidence": "fake-node", "confidence": "high"},
                    {"type": "choose_chain", "chain_text": "BNB", "source_evidence": "BNB", "confidence": "high"},
                    {
                        "type": "propose_config_values",
                        "source_format": "yaml",
                        "config_values": {
                            "CLOUD_REGION": "asia-east1",
                            "CLOUD_ZONE": "asia-east1-c",
                            "MACHINE_TYPE": "n2-standard-16",
                            "LEDGER_DEVICE": "vda",
                            "DATA_VOL_TYPE": "hyperdisk-balanced",
                            "DATA_VOL_SIZE": "926",
                            "DATA_VOL_MAX_IOPS": "20000",
                            "DATA_VOL_MAX_THROUGHPUT": "1000",
                            "NETWORK_INTERFACE": "eth0",
                            "NETWORK_MAX_BANDWIDTH_GBPS": "100",
                        },
                        "unmapped_values": {},
                        "confidence": "high",
                    },
                ]
            }
            resolver.side_effect = _admitted_mock_resolver(resolver.return_value)
            result = process_turn(state)

        self.assertEqual(result["target_mode"], "fake-node")
        self.assertEqual(result["chain_identity"]["canonical"], "bsc")
        self.assertEqual(result["pending_question"]["id"], "inferred_config_review")
        self.assertNotIn("CLOUD_REGION", result["confirmed_config"])
        rendered = "\n".join(result["visible_response"])
        self.assertIn("CLOUD_REGION", rendered)
        self.assertIn("DATA_VOL_MAX_IOPS", rendered)

    def test_explicit_structured_assignment_uses_the_same_review_transaction(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("structured-pending-review", language="en")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "solana",
            "canonical": "solana",
            "status": "confirmed",
            "case": "known",
        }
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "solana"}
        state["active_group"] = "accounts_disk"
        state["pending_question"] = {
            "id": "ACCOUNTS_DEVICE",
            "group": "accounts_disk",
            "field": "ACCOUNTS_DEVICE",
            "kind": "manual",
            "prompt": "Enter the accounts device.",
            "manual_input_allowed": True,
            "contract_version": 1,
        }
        state["last_user_input"] = '{"ACCOUNTS_DEVICE":"/dev/nvme1n1"}'

        result = _invoke_with_admitted_actions(
            process_turn,
            state,
            [{
                "type": "propose_config_values",
                "source_format": "json",
                "config_values": {"ACCOUNTS_DEVICE": "/dev/nvme1n1"},
                "unmapped_values": {},
                "source_evidence": state["last_user_input"],
                "confidence": "high",
            }],
        )

        self.assertEqual(result["pending_question"]["id"], "inferred_config_review")
        self.assertNotIn("ACCOUNTS_DEVICE", result["confirmed_config"])
        self.assertEqual(
            result["inferred_config"]["pending_review"]["config_values"],
            {"ACCOUNTS_DEVICE": "/dev/nvme1n1"},
        )
        self.assertEqual(
            [
                item.get("type")
                for item in (result.get("turn_context") or {}).get("admitted_actions") or []
            ],
            ["propose_config_values"],
        )

    def test_standalone_yaml_config_uses_the_same_review_transaction(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("yaml-pending-review", language="en")
        state["active_group"] = "provider_deployment"
        state["pending_question"] = {
            "id": "CLOUD_REGION",
            "group": "provider_deployment",
            "field": "CLOUD_REGION",
            "kind": "manual_value",
            "prompt": "Enter the cloud region.",
            "manual_input_allowed": True,
            "contract_version": 1,
        }
        state["last_user_input"] = "CLOUD_REGION: us-1\nCLOUD_ZONE: us-1-z"

        result = _invoke_with_admitted_actions(
            process_turn,
            state,
            [{
                "type": "propose_config_values",
                "source_format": "yaml",
                "config_values": {"CLOUD_REGION": "us-1", "CLOUD_ZONE": "us-1-z"},
                "unmapped_values": {},
                "source_evidence": state["last_user_input"],
                "confidence": "high",
            }],
        )

        self.assertEqual(result["pending_question"]["id"], "inferred_config_review")
        self.assertEqual(
            result["inferred_config"]["pending_review"]["config_values"],
            {"CLOUD_REGION": "us-1", "CLOUD_ZONE": "us-1-z"},
        )
        self.assertNotIn("CLOUD_REGION", result["confirmed_config"])
        self.assertEqual(
            [
                item.get("type")
                for item in (result.get("turn_context") or {}).get("admitted_actions") or []
            ],
            ["propose_config_values"],
        )

    def test_mixed_prose_and_structured_config_reaches_the_semantic_planner(self) -> None:
        from unittest.mock import patch

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("mixed-structured-planner", language="en")
        state["active_group"] = "provider_deployment"
        state["pending_question"] = {
            "id": "CLOUD_REGION",
            "group": "provider_deployment",
            "field": "CLOUD_REGION",
            "kind": "manual_value",
            "prompt": "Enter the cloud region.",
            "manual_input_allowed": True,
            "contract_version": 1,
        }
        state["last_user_input"] = 'Use these values, then explain fake-node.\n{"CLOUD_REGION":"us-1"}'

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = _admitted_mock_plan(state, state["last_user_input"], {
                "actions": [{
                    "type": "answer_opening_question",
                    "topic": "mode_comparison",
                    "source_evidence": "explain fake-node",
                    "confidence": "high",
                }]
            })
            result = process_turn(state)

        resolver.assert_called_once()
        self.assertNotIn("CLOUD_REGION", result["confirmed_config"])

    def test_standalone_json_rpc_payload_is_not_captured_as_configuration(self) -> None:
        from unittest.mock import patch

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("json-rpc-not-config", language="en")
        state["active_group"] = "workload_rpc"
        state["last_user_input"] = '{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}'

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": []}
            result = process_turn(state)

        resolver.assert_called_once()
        self.assertNotEqual((result.get("pending_question") or {}).get("id"), "inferred_config_review")

    def test_qps_override_and_accounts_presence_can_be_set_from_one_freeform_turn(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {
            "BLOCKCHAIN_NODE": "bsc",
            "CLOUD_REGION": "asia-east1",
            "CLOUD_ZONE": "asia-east1-c",
            "MACHINE_TYPE": "n2-standard-16",
            "LEDGER_DEVICE": "vda",
            "DATA_VOL_TYPE": "hyperdisk-balanced",
            "DATA_VOL_MAX_IOPS": "20000",
            "DATA_VOL_MAX_THROUGHPUT": "1000",
        }
        state["qps_profile"] = {"mode": "quick", "confirmed": False, "default_decision_made": True}
        state["active_group"] = "qps_profile"
        state["pending_question"] = {
            "id": "qps_adjust_field",
            "group": "qps_profile",
            "kind": "numbered_choice",
            "field": "qps_adjust_field",
            "options": [
                {"label": "INITIAL_QPS / 起始 QPS", "value": "INITIAL_QPS"},
                {"label": "完成调整", "value": "done"},
            ],
            "manual_input_allowed": False,
        }
        state["last_user_input"] = "把 initial qps 设成 5，然后我没有 accounts 盘，回去继续磁盘"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "set_qps_override", "qps_overrides": {"INITIAL_QPS": 5}, "confidence": "high"},
                    {"type": "set_accounts_presence", "has_accounts_device": False, "confidence": "high"},
                    {"type": "change_group", "group": "ledger_disk", "confidence": "high", "selection_contract_verified": True},
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["qps_profile"]["overrides"]["INITIAL_QPS"], "5")
        self.assertTrue(result["qps_profile"]["confirmed"])
        self.assertFalse(result["confirmed_config"]["has_accounts_device"])
        self.assertEqual(result["pending_question"]["id"], "DATA_VOL_SIZE")

    def test_jump_to_completed_optional_group_reopens_its_owned_question(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {
            "BLOCKCHAIN_NODE": "bsc",
            "CLOUD_REGION": "asia-east1",
            "CLOUD_ZONE": "asia-east1-c",
            "MACHINE_TYPE": "n2-standard-16",
            "LEDGER_DEVICE": "vda",
            "DATA_VOL_TYPE": "hyperdisk-balanced",
            "has_accounts_device": False,
        }
        state["active_group"] = "qps_profile"
        state["last_user_input"] = "回到 accounts 配置"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "change_group", "group": "accounts_disk", "navigation_explicit": True, "source_evidence": "回到 accounts 配置", "confidence": "high"}]}
            result = process_turn(state)

        self.assertEqual(result["active_group"], "accounts_disk")
        self.assertEqual(result["pending_question"]["id"], "has_accounts_device")
        self.assertFalse(result["confirmed_config"]["has_accounts_device"])

    def test_optional_endpoint_navigation_does_not_advance_fallback_in_same_turn(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("empty-auxiliary-navigation", language="en")
        state.update({
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "active_group": "opening",
            "chain_identity": {
                "raw": "bsc",
                "canonical": "bsc",
                "status": "confirmed",
                "case": "known",
            },
            "confirmed_config": {"BLOCKCHAIN_NODE": "bsc"},
            "last_user_input": (
                "Take me to the optional chain endpoint and credential settings "
                "without changing any confirmed value."
            ),
        })
        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{
                "type": "change_group",
                "group": "chain_auxiliary_endpoints",
                "navigation_explicit": True,
                "source_evidence": state["last_user_input"],
                "confidence": "high",
            }]},
        ):
            result = process_turn(state)

        self.assertEqual(result["active_group"], "chain_auxiliary_endpoints")
        self.assertFalse(result.get("pending_question"))
        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("`chain_auxiliary_endpoints`", text)
        self.assertNotIn("CLOUD_REGION", text)

    def test_config_proposal_rejection_does_not_mutate_confirmed_config(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["active_group"] = "provider_deployment"
        state["last_user_input"] = "CLOUD_REGION=us-1\nCLOUD_ZONE=us-1-z\nunknown_flag=true"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {
                        "type": "propose_config_values",
                        "source_format": "env",
                        "config_values": {"CLOUD_REGION": "us-1", "CLOUD_ZONE": "us-1-z"},
                        "unmapped_values": {"unknown_flag": True},
                        "confidence": "high",
                    }
                ]
            }
            resolver.side_effect = _admitted_mock_resolver(resolver.return_value)
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "inferred_config_review")
        result["last_user_input"] = "N"
        result = process_turn(result)

        self.assertEqual(result["confirmed_config"], {})
        self.assertNotIn("pending_review", result.get("inferred_config", {}))
        self.assertIn("Discarded", result["visible_response"][0])

    def test_inferred_config_review_is_blocking_until_user_confirms(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["inferred_config"] = {
            "config_values": {"CLOUD_REGION": "asia-east1"},
            "unmapped_values": {},
            "source_format": "mixed",
            "reason": "",
        }
        state["pending_question"] = {
            "id": "inferred_config_review",
            "group": "provider_deployment",
            "kind": "yes_no",
            "field": "inferred_config_review",
            "options": [{"label": "Y", "value": True}, {"label": "N", "value": False}],
            "manual_input_allowed": False,
        }
        state["last_user_input"] = 'endpoint 是 http://fake-node:19000，method 是 eth_chainId，请求 {"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}'

        with patch("agent.harness.coordinator.resolve_action_queue", return_value={"actions": [{"type": "unknown", "confidence": "low"}]}):
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "inferred_config_review")
        self.assertNotIn("CLOUD_REGION", result["confirmed_config"])
        self.assertNotIn("jsonrpc", result.get("inferred_config", {}).get("pending_review", {}).get("unmapped_values", {}))

    def test_verbose_yes_applies_inferred_config_through_typed_pending_answer(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.environment import config_proposal_review_question
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        proposal = {
            "config_values": {"CLOUD_REGION": "asia-east1", "MACHINE_TYPE": "c3-standard-8"},
            "unmapped_values": {},
            "source_format": "yaml",
            "reason": "pasted deployment notes",
        }
        state["inferred_config"] = {"pending_review": proposal}
        state["pending_question"] = config_proposal_review_question(
            "provider_deployment", proposal, language="en"
        )
        state["last_user_input"] = "Yes, apply only those inferred values and then continue."
        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            side_effect=_admitted_mock_resolver({
                "actions": [{
                    "type": "answer_pending",
                    "selected_value": True,
                    "answer": "Y",
                    "pending_option_semantic_verified": True,
                    "semantic_purpose_verified": True,
                    "confidence": "high",
                    "source_evidence": state["last_user_input"],
                }]
            }),
        ):
            result = process_turn(state)

        self.assertEqual(result["confirmed_config"]["CLOUD_REGION"], "asia-east1")
        self.assertEqual(result["confirmed_config"]["MACHINE_TYPE"], "c3-standard-8")
        self.assertNotEqual((result.get("pending_question") or {}).get("id"), "inferred_config_review")

    def test_config_proposal_saves_endpoint_as_candidate_not_validated_config(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["target_mode"] = "real-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "ethereum", "canonical": "ethereum", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "ethereum"}
        state["active_group"] = "endpoint_process"
        state["last_user_input"] = '{"LOCAL_RPC_URL":"https://node.example","RPC_MODE":"mixed"}'

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {
                        "type": "set_rpc_mode",
                        "rpc_mode": "mixed",
                        "mutation_explicit": True,
                        "source_evidence": '"RPC_MODE":"mixed"',
                        "confidence": "high",
                    },
                    {
                        "type": "propose_config_values",
                        "source_format": "json",
                        "config_values": {"LOCAL_RPC_URL": "https://node.example", "RPC_MODE": "mixed"},
                        "confidence": "high",
                    }
                ]
            }
            resolver.side_effect = _admitted_mock_resolver(resolver.return_value)
            result = process_turn(state)

        result["last_user_input"] = "Y"
        result = process_turn(result)

        self.assertEqual(result["rpc_mode"], "mixed")
        self.assertEqual(result["endpoint_evidence"]["proposed_values"]["LOCAL_RPC_URL"], "https://node.example")
        self.assertNotIn("LOCAL_RPC_URL", result["confirmed_config"])
        self.assertNotEqual(result.get("endpoint_evidence", {}).get("local_rpc_url_ready"), True)

    def test_environment_proposal_excludes_other_workflow_dimensions(self) -> None:
        from agent.harness.domains.environment import build_config_proposal

        proposal = build_config_proposal({
            "config_values": {
                "chain": "bsc",
                "target_mode": "fake-node",
                "rpc_mode": "mixed",
                "qps_mode": "quick",
                "CLOUD_REGION": "test-region",
            },
            "unmapped_values": {
                "payload.chain": "bsc",
                "payload.target_mode": "fake-node",
            },
        })

        self.assertEqual(proposal["config_values"], {"CLOUD_REGION": "test-region"})
        self.assertEqual(proposal["unmapped_values"], {})

    def test_environment_proposal_rejects_prose_url_as_structured_key(self) -> None:
        from agent.harness.domains.environment import (
            build_config_proposal,
            extract_structured_config_proposal,
        )

        text = (
            "Use Ethereum. Keep http://geth-dev:8545 as both the validation "
            "endpoint and final LOCAL_RPC_URL."
        )
        proposal = build_config_proposal({
            "config_values": {},
            "unmapped_values": {
                "Use Ethereum. Keep http": "//geth-dev:8545 as both endpoints",
            },
            "source_text": text,
        })

        self.assertEqual(proposal["unmapped_values"], {})
        self.assertIsNone(extract_structured_config_proposal(text))

    def test_environment_proposal_preserves_structural_unknown_yaml_key(self) -> None:
        from agent.harness.domains.environment import extract_structured_config_proposal

        proposal = extract_structured_config_proposal(
            "cloud_region: asia-east1\nteam.custom_limit: 42"
        )

        self.assertIsNotNone(proposal)
        self.assertEqual(proposal["config_values"]["CLOUD_REGION"], "asia-east1")
        self.assertEqual(proposal["unmapped_values"]["team.custom_limit"], "42")

    def test_nested_structured_config_leaf_maps_once_and_preserves_unknown_sibling(self) -> None:
        from agent.harness.domains.environment import (
            build_config_proposal,
            extract_structured_input_candidates,
        )

        candidates = extract_structured_input_candidates(
            "environment:\n"
            "  CLOUD_REGION: asia-east1\n"
            "  team_ticket: INC-12345"
        )

        self.assertEqual(candidates["config_values"], {"CLOUD_REGION": "asia-east1"})
        self.assertEqual(candidates["unmapped_values"], {"environment.team_ticket": "INC-12345"})
        self.assertEqual(candidates["source_format"], "yaml")

        proposal = build_config_proposal({
            "config_values": {"CLOUD_REGION": "asia-east1"},
            "unmapped_values": {
                "environment.CLOUD_REGION": "must-not-override",
                "environment.team_ticket": "INC-12345",
            },
        })
        self.assertEqual(proposal["config_values"], {"CLOUD_REGION": "asia-east1"})
        self.assertEqual(proposal["unmapped_values"], {"environment.team_ticket": "INC-12345"})

    def test_nested_json_config_leaf_uses_same_schema_resolution(self) -> None:
        from agent.harness.domains.environment import extract_structured_input_candidates

        candidates = extract_structured_input_candidates(
            '{"deployment":{"NETWORK_INTERFACE":"eth0"}}'
        )

        self.assertEqual(candidates["config_values"], {"NETWORK_INTERFACE": "eth0"})
        self.assertEqual(candidates["unmapped_values"], {})
        self.assertEqual(candidates["source_format"], "json")

    def test_structured_candidates_separate_workflow_and_unknown_fields(self) -> None:
        from agent.harness.domains.environment import extract_structured_input_candidates

        candidates = extract_structured_input_candidates(
            "copied config:\n"
            "export CLOUD_REGION=asia-east1\n"
            "RPC_MODE=single\n"
            "unrelated_ticket=INC-12345"
        )

        self.assertEqual(candidates["config_values"], {"CLOUD_REGION": "asia-east1"})
        self.assertEqual(candidates["workflow_values"], {"RPC_MODE": "single"})
        self.assertEqual(candidates["unmapped_values"], {"UNRELATED_TICKET": "INC-12345"})
        self.assertEqual(candidates["source_format"], "env")

    def test_structured_candidates_report_mixed_source_syntax(self) -> None:
        from agent.harness.domains.environment import extract_structured_input_candidates

        candidates = extract_structured_input_candidates(
            '{"CLOUD_REGION":"asia-east1"}\nNETWORK_INTERFACE=eth0'
        )

        self.assertEqual(
            candidates["config_values"],
            {"CLOUD_REGION": "asia-east1", "NETWORK_INTERFACE": "eth0"},
        )
        self.assertEqual(candidates["source_format"], "mixed")

    def test_action_payload_exposes_clause_scoped_structured_candidates(self) -> None:
        from agent.harness.intent import _action_queue_payload
        from agent.harness.state import new_state

        payload = _action_queue_payload(
            new_state("unit-thread", language="en"),
            "Use these values:\nCLOUD_REGION=asia-east1\nRPC_MODE=mixed",
        )

        self.assertEqual(len(payload["structured_candidates"]), 1)
        self.assertEqual(payload["structured_candidates"][0]["clause_id"], "clause-1")
        self.assertEqual(
            payload["structured_candidates"][0]["workflow_values"],
            {"RPC_MODE": "mixed"},
        )

    def test_error_only_key_value_block_is_not_a_configuration_candidate(self) -> None:
        from agent.harness.intent import _action_queue_payload
        from agent.harness.state import new_state

        payload = _action_queue_payload(
            new_state("unit-thread", language="en"),
            'Traceback (most recent call last):\n  File "runner.py", line 4\nRuntimeError: endpoint timeout',
        )

        self.assertEqual(payload["structured_candidates"], [])

    def test_unknown_only_config_proposal_cannot_take_environment_control(self) -> None:
        from agent.harness.contracts import ActionProposal
        from agent.harness.domains.environment import apply_environment_action
        from agent.harness.state import new_state

        result = apply_environment_action(
            new_state("unit-thread", language="en"),
            ActionProposal(
                action_id="proposal-1",
                action_type="propose_config_values",
                arguments={"config_values": {}, "unmapped_values": {"RuntimeError": "endpoint timeout"}},
                confidence="high",
            ),
        )

        self.assertEqual(result.completion, "completed")
        self.assertIsNone(result.pending_question)
        self.assertTrue(result.delta.is_empty())

    def test_request_only_rpc_evidence_cannot_claim_a_response_contract(self) -> None:
        from agent.harness.domains.rpc_endpoint import _request_only_schema_evidence

        self.assertTrue(_request_only_schema_evidence(
            [
                "eth_blockNumber",
                '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}',
            ],
            "eth_blockNumber",
        ))
        self.assertFalse(_request_only_schema_evidence(
            [
                '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}',
                '{"jsonrpc":"2.0","id":1,"result":"0x10"}',
            ],
            "eth_blockNumber",
        ))

    def test_config_proposal_normalizes_units_and_accounts_absence(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["active_group"] = "provider_deployment"
        state["last_user_input"] = "pasted environment facts"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {
                        "type": "propose_config_values",
                        "source_format": "mixed",
                        "config_values": {
                            "CLOUD_REGION": "asia-east1",
                            "DATA_VOL_TYPE": "hyperdisk-balanced,",
                            "DATA_VOL_SIZE": "926GiB",
                            "DATA_VOL_MAX_IOPS": "20000 IOPS",
                            "DATA_VOL_MAX_THROUGHPUT": "1000 MiB/s",
                            "ACCOUNTS_DEVICE": "None",
                            "NETWORK_MAX_BANDWIDTH_GBPS": "100Gbps",
                        },
                        "confidence": "high",
                    }
                ]
            }
            resolver.side_effect = _admitted_mock_resolver(resolver.return_value)
            result = process_turn(state)

        prompt = "\n".join(result.get("visible_response") or [])
        self.assertIn("DATA_VOL_SIZE: `926`", prompt)
        self.assertIn("HAS_ACCOUNTS_DEVICE", prompt)
        self.assertNotIn("ACCOUNTS_DEVICE: `None`", prompt)

        result["last_user_input"] = "Y"
        result = process_turn(result)

        self.assertEqual(result["confirmed_config"]["DATA_VOL_TYPE"], "hyperdisk-balanced")
        self.assertEqual(result["confirmed_config"]["DATA_VOL_SIZE"], "926")
        self.assertEqual(result["confirmed_config"]["DATA_VOL_MAX_IOPS"], "20000")
        self.assertEqual(result["confirmed_config"]["DATA_VOL_MAX_THROUGHPUT"], "1000")
        self.assertEqual(result["confirmed_config"]["NETWORK_MAX_BANDWIDTH_GBPS"], "100")
        self.assertIs(result["confirmed_config"]["has_accounts_device"], False)
        self.assertNotIn("ACCOUNTS_DEVICE", result["confirmed_config"])

    def test_config_proposal_derives_accounts_absence_when_model_outputs_empty_device(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["active_group"] = "provider_deployment"
        state["last_user_input"] = "环境信息：没有 accounts，ledger=vda"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {
                        "type": "propose_config_values",
                        "source_format": "mixed",
                        "config_values": {"ACCOUNTS_DEVICE": "", "LEDGER_DEVICE": "vda"},
                        "confidence": "high",
                    }
                ]
            }
            resolver.side_effect = _admitted_mock_resolver(resolver.return_value)
            result = process_turn(state)

        prompt = "\n".join(result.get("visible_response") or [])
        self.assertIn("HAS_ACCOUNTS_DEVICE: `False`", prompt)
        self.assertNotIn("ACCOUNTS_DEVICE: ``", prompt)

    def test_config_review_then_continues_to_custom_rpc_action(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["active_group"] = "provider_deployment"
        state["last_user_input"] = "我要加自定义 RPC method eth_chainId，endpoint http://fake-node:19000，region asia-east1"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {
                        "type": "propose_config_values",
                        "source_format": "mixed",
                        "config_values": {"CLOUD_REGION": "asia-east1"},
                        "confidence": "high",
                    },
                    {
                        "type": "start_custom_rpc",
                        "rpc_method": "eth_chainId",
                        "rpc_endpoint": "http://fake-node:19000",
                        "confidence": "high",
                    },
                ]
            }
            resolver.side_effect = _admitted_mock_resolver(resolver.return_value)
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "inferred_config_review")

        ok_probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}
        schema = {"status": "draft", "method": "eth_blockNumber", "params": [], "params_json": [], "response_summary": "hex block height", "confidence": "high"}
        with patch("agent.harness.domains.rpc_endpoint.extract_rpc_schema_from_evidence", return_value=schema), patch("agent.harness.domains.rpc_endpoint.validate_rpc_endpoint", return_value=ok_probe):
            result["last_user_input"] = "Y"
            result = process_turn(result)

        self.assertEqual(result["confirmed_config"]["CLOUD_REGION"], "asia-east1")
        self.assertNotIn("LOCAL_RPC_URL", result["confirmed_config"])
        self.assertEqual(_catalog_draft(result)["method"], "eth_chainId")
        self.assertTrue(result["custom_rpc"]["endpoint_ready"])
        self.assertEqual(result["custom_rpc"]["status"], "needs_schema_evidence")
        self.assertEqual(result["pending_question"]["id"], "custom_rpc_schema_evidence")
        self.assertIn("`eth_chainId`", result["pending_question"]["prompt"])
        self.assertIn("custom_rpc_endpoint_probe", result["endpoint_evidence"])

    def test_config_review_is_prioritized_before_custom_rpc_even_if_model_orders_it_later(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["active_group"] = "provider_deployment"
        state["last_user_input"] = "我要加自定义 RPC method eth_chainId，endpoint http://fake-node:19000，region asia-east1"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {
                        "type": "start_custom_rpc",
                        "rpc_method": "eth_chainId",
                        "rpc_endpoint": "http://fake-node:19000",
                        "confidence": "high",
                    },
                    {
                        "type": "propose_config_values",
                        "source_format": "mixed",
                        "config_values": {"CLOUD_REGION": "asia-east1"},
                        "confidence": "high",
                    },
                ]
            }
            resolver.side_effect = _admitted_mock_resolver(resolver.return_value)
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "inferred_config_review")
        self.assertEqual(result["action_queue"][0]["type"], "rpc_catalog_command")
        self.assertNotIn("endpoint_ready", result.get("custom_rpc", {}))

    def test_custom_rpc_schema_answer_clears_stale_queue_control(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["active_group"] = "endpoint_process"
        state["custom_rpc"] = {
            "status": "needs_schema_evidence",
            "endpoint": "http://fake-node:19000",
            "endpoint_ready": True,
            "method": "eth_chainId",
        }
        state["pending_question"] = {
            "group": "endpoint_process",
            "id": "custom_rpc_schema_evidence",
            "field": "custom_rpc_schema_evidence",
            "kind": "evidence",
            "manual_input_allowed": True,
            "accepted_action_types": ["rpc_catalog_command"],
            "manual_action": {
                "type": "rpc_catalog_command",
                "catalog_command": "append_evidence",
                "value_argument": "rpc_schema_evidence",
                "use_complete_turn": True,
            },
            "queue_barrier": True,
        }
        state["action_queue"] = [
            {
                "type": "propose_config_values",
                "config_values": {"CLOUD_REGION": "asia-east1"},
                "confidence": "high",
            }
        ]
        state["last_user_input"] = "[]"

        ok_probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}
        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{
                "type": "rpc_catalog_command",
                "catalog_command": "append_evidence",
                "rpc_schema_evidence": state["last_user_input"],
                "source_evidence": state["last_user_input"],
                "confidence": "high",
            }]},
        ), patch("agent.harness.domains.rpc_endpoint.extract_rpc_schema_from_evidence", return_value={"status": "draft", "method": "eth_chainId", "params": [], "params_json": [], "response_summary": "hex chain id", "confidence": "high"}), patch("agent.harness.domains.rpc_endpoint.validate_rpc_endpoint", return_value=ok_probe):
            result = process_turn(state)
            self.assertEqual(result["custom_rpc"]["status"], "schema_needs_confirmation")
            self.assertEqual(result["pending_question"]["id"], "custom_rpc_schema_confirm")
            result = self._confirm_catalog_through_graph(result, process_turn)

        self.assertEqual(result["custom_rpc"]["status"], "method_validated_next")
        self.assertEqual(result["pending_question"]["id"], "custom_rpc_continue")
        self.assertEqual(result.get("action_queue"), [])
        self.assertNotIn("CLOUD_REGION", result["confirmed_config"])
        prompt = str(result["pending_question"].get("prompt") or "")
        self.assertEqual(sum(prompt in item for item in result.get("visible_response") or []), 1)

    def test_explicit_config_assignment_reviews_named_field_not_current_pending(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc", "CLOUD_REGION": "asia-east1"}
        state["active_group"] = "provider_deployment"
        state["pending_question"] = {
            "group": "provider_deployment",
            "id": "CLOUD_ZONE",
            "field": "CLOUD_ZONE",
            "kind": "manual_value",
            "manual_input_allowed": True,
            "prompt": "Confirm CLOUD_ZONE.",
        }
        state["last_user_input"] = "MACHINE_TYPE=n2-standard-16"

        result = _invoke_with_admitted_actions(
            process_turn,
            state,
            [{
                "type": "propose_config_values",
                "source_format": "env",
                "config_values": {"MACHINE_TYPE": "n2-standard-16"},
                "unmapped_values": {},
                "source_evidence": state["last_user_input"],
                "confidence": "high",
            }],
        )

        self.assertNotIn("MACHINE_TYPE", result["confirmed_config"])
        self.assertNotEqual(result["confirmed_config"].get("CLOUD_ZONE"), "MACHINE_TYPE=n2-standard-16")
        self.assertEqual(result["pending_question"]["id"], "inferred_config_review")
        self.assertEqual(
            result["inferred_config"]["pending_review"]["config_values"],
            {"MACHINE_TYPE": "n2-standard-16"},
        )

        result["last_user_input"] = "Y"
        result = process_turn(result)

        self.assertEqual(result["confirmed_config"].get("MACHINE_TYPE"), "n2-standard-16")
        self.assertEqual(result["pending_question"]["id"], "CLOUD_ZONE")

    def test_comma_separated_config_assignments_apply_atomically_after_review(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
        state["active_group"] = "provider_deployment"
        state["last_user_input"] = "CLOUD_REGION=us-1, CLOUD_ZONE=us-1-z, MACHINE_TYPE=n2"

        result = _invoke_with_admitted_actions(
            process_turn,
            state,
            [{
                "type": "propose_config_values",
                "source_format": "env",
                "config_values": {
                    "CLOUD_REGION": "us-1",
                    "CLOUD_ZONE": "us-1-z",
                    "MACHINE_TYPE": "n2",
                },
                "unmapped_values": {},
                "source_evidence": state["last_user_input"],
                "confidence": "high",
            }],
        )

        self.assertNotIn("CLOUD_REGION", result["confirmed_config"])
        self.assertEqual(result["pending_question"]["id"], "inferred_config_review")

        result["last_user_input"] = "Y"
        result = process_turn(result)

        self.assertEqual(result["confirmed_config"].get("CLOUD_REGION"), "us-1")
        self.assertEqual(result["confirmed_config"].get("CLOUD_ZONE"), "us-1-z")
        self.assertEqual(result["confirmed_config"].get("MACHINE_TYPE"), "n2")
        self.assertEqual(result["pending_question"]["id"], "LEDGER_DEVICE")

    def test_structured_config_field_families_share_review_ownership(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        cases = (
            ("CLOUD_REGION", "asia-east1", "CLOUD_REGION=asia-east1"),
            ("DATA_VOL_MAX_IOPS", "20000", "DATA_VOL_MAX_IOPS=20000"),
            ("BLOCKCHAIN_PROCESS_NAMES", "geth", "BLOCKCHAIN_PROCESS_NAMES=geth"),
            ("LOCAL_RPC_URL", "http://node:8545", "LOCAL_RPC_URL=http://node:8545"),
            ("RPC_API_KEY", "unit-secret-value", "RPC_API_KEY=unit-secret-value"),
            ("SYNC_OBSERVE_DURATION_SECONDS", "600", "sync_observe_duration_seconds: 600"),
        )
        for field, value, user_text in cases:
            with self.subTest(field=field):
                state = new_state(f"structured-{field}", language="en")
                state["target_mode"] = "sync-observe" if field.startswith("SYNC_OBSERVE") else "real-node"
                state["workflow_mode"] = "sync_observe" if field.startswith("SYNC_OBSERVE") else "rpc_benchmark"
                state["active_group"] = "provider_deployment"
                state["pending_question"] = {
                    "group": "provider_deployment",
                    "id": "CLOUD_ZONE",
                    "field": "CLOUD_ZONE",
                    "kind": "manual_value",
                    "manual_input_allowed": True,
                    "prompt": "Confirm CLOUD_ZONE.",
                }
                state["last_user_input"] = user_text
                proposal = {
                    "type": "propose_config_values",
                    "source_format": "yaml" if ":" in user_text and "=" not in user_text else "env",
                    "config_values": {field: value},
                    "source_evidence": user_text,
                    "confidence": "high",
                }
                with patch(
                    "agent.harness.coordinator.resolve_action_queue",
                    side_effect=_admitted_mock_resolver({"actions": [proposal]}),
                ):
                    result = process_turn(state)

                self.assertEqual(result["pending_question"]["id"], "inferred_config_review")
                self.assertEqual(result["input_shape"], "structured")
                self.assertEqual(result["inferred_config"]["pending_review"]["config_values"][field], value)
                self.assertNotIn(field, result["confirmed_config"])
                if field == "RPC_API_KEY":
                    response = "\n".join(result.get("visible_response") or [])
                    self.assertNotIn(value, response)
                    self.assertIn("***REDACTED***", response)

    def test_assignments_embedded_in_natural_language_do_not_bypass_other_actions(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["last_user_input"] = (
            "Run BNB on fake-node. Review: CLOUD_REGION=us-east1, "
            "CLOUD_ZONE=us-east1-b, MACHINE_TYPE=n2-standard-8."
        )
        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {
                        "type": "choose_target_mode",
                        "target_mode": "fake-node",
                        "target_mode_explicit": True,
                        "source_evidence": "fake-node",
                        "confidence": "high",
                    },
                    {
                        "type": "choose_chain",
                        "chain_text": "BNB",
                        "source_evidence": "BNB",
                        "confidence": "high",
                    },
                    {
                        "type": "propose_config_values",
                        "source_format": "env",
                        "config_values": {
                            "CLOUD_REGION": "us-east1",
                            "CLOUD_ZONE": "us-east1-b",
                            "MACHINE_TYPE": "n2-standard-8",
                        },
                        "unmapped_values": {},
                        "confidence": "high",
                    },
                ]
            }
            resolver.side_effect = _admitted_mock_resolver(resolver.return_value)
            result = process_turn(state)

        resolver.assert_called_once()
        self.assertEqual(result.get("target_mode"), "fake-node")
        self.assertEqual((result.get("chain_identity") or {}).get("canonical"), "bsc")
        self.assertEqual(result.get("pending_question", {}).get("id"), "inferred_config_review")

    def test_declining_preflight_smoke_pauses_instead_of_looping(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["active_group"] = "preflight_smoke_execution"
        state["pending_question"] = {
            "group": "preflight_smoke_execution",
            "id": "preflight_smoke_confirm",
            "field": "preflight_smoke_confirmed",
            "kind": "yes_no",
            "manual_input_allowed": False,
            "options": [{"label": "Y", "value": True}, {"label": "N", "value": False}],
            "prompt": "配置已收集。是否运行 preflight 和 smoke？",
        }
        state["last_user_input"] = "n"

        result = process_turn(state)

        self.assertEqual(result.get("pending_question"), {})
        self.assertFalse(result["preflight"]["approved"])
        self.assertIn("已暂停", "\n".join(result.get("visible_response") or []))

    def test_queue_rejects_target_mode_not_explicit_in_user_text(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["active_group"] = "opening"
        state["last_user_input"] = "我要用 fake-node 测试 BNB，用 mixed，QPS quick，并开启本地 Grafana"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {
                        "type": "choose_target_mode",
                        "target_mode": "real-node",
                        "target_mode_explicit": False,
                        "confidence": "high",
                        "reason": "bad model default",
                    },
                    {"type": "choose_chain", "chain_text": "BNB", "source_evidence": "BNB", "confidence": "high"},
                    {"type": "set_rpc_mode", "rpc_mode": "mixed", "mutation_explicit": True, "source_evidence": "mixed", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        self.assertNotEqual(result.get("target_mode"), "real-node")
        self.assertEqual(result["chain_identity"]["canonical"], "bsc")
        self.assertEqual(result["rpc_mode"], "")
        self.assertEqual(result["pending_question"]["id"], "target_mode_select")
        self.assertTrue(result.get("action_queue"))

    def test_opening_option_contract_does_not_trust_inferred_mode_action(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.orientation import opening_question
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["active_group"] = "opening"
        state["pending_question"] = opening_question(state)
        state["last_user_input"] = "我要测试 BNB，用 mixed，QPS quick，并开启本地 Grafana"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {
                        "type": "choose_target_mode",
                        "target_mode": "real-node",
                        "target_mode_explicit": True,
                        "source_evidence": state["last_user_input"],
                        "confidence": "medium",
                    },
                    {"type": "choose_chain", "chain_text": "BNB", "source_evidence": "BNB", "confidence": "high"},
                    {"type": "set_rpc_mode", "rpc_mode": "mixed", "mutation_explicit": True, "source_evidence": "mixed", "confidence": "high"},
                    {"type": "set_qps_mode", "qps_mode": "quick", "mutation_explicit": True, "source_evidence": "QPS quick", "confidence": "high"},
                    {"type": "set_observability", "observability_mode": "local", "mutation_explicit": True, "source_evidence": "开启本地 Grafana", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        self.assertEqual(result.get("target_mode"), "")
        self.assertEqual(result["pending_question"]["id"], "target_mode_select")
        self.assertEqual((result.get("chain_identity") or {}).get("canonical"), "bsc")
        queued = {str(item.get("type") or "") for item in result.get("action_queue") or []}
        self.assertTrue({"set_rpc_mode", "set_qps_mode", "set_observability"}.issubset(queued))
        self.assertIsNone((result.get("observability") or {}).get("mode"))

    def test_return_to_origin_group_does_not_preempt_new_blocking_group(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "ledger_disk"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {
            "BLOCKCHAIN_NODE": "bsc",
            "CLOUD_REGION": "asia-east1",
            "CLOUD_ZONE": "asia-east1-c",
            "MACHINE_TYPE": "n2-standard-16",
            "NETWORK_INTERFACE": "eth0",
            "NETWORK_MAX_BANDWIDTH_GBPS": "100",
        }
        state["pending_question"] = {
            "id": "LEDGER_DEVICE",
            "group": "ledger_disk",
            "kind": "device",
            "field": "LEDGER_DEVICE",
            "options": [{"label": "vda", "value": "vda"}, {"label": "vdb", "value": "vdb"}],
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "先配置 QPS quick，然后回到磁盘配置"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "change_group", "group": "qps_profile", "navigation_explicit": True, "source_evidence": "配置 QPS quick", "confidence": "high"},
                    {"type": "set_qps_mode", "qps_mode": "quick", "mutation_explicit": True, "source_evidence": "QPS quick", "confidence": "high"},
                    {"type": "change_group", "group": "ledger_disk", "navigation_explicit": True, "source_evidence": "回到磁盘配置", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["qps_profile"]["mode"], "quick")
        self.assertEqual(result["pending_question"]["id"], "qps_profile_confirm")
        self.assertEqual(result["active_group"], "qps_profile")

    def test_pending_question_can_route_to_observability_without_being_swallowed(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "provider_deployment"
        state["chain_identity"] = {"raw": "solana", "canonical": "solana", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "solana"}
        state["pending_question"] = {
            "id": "CLOUD_REGION",
            "group": "provider_deployment",
            "kind": "manual_value",
            "field": "CLOUD_REGION",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "先确认一下，本地 Grafana 不开"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "change_group", "group": "observability", "navigation_explicit": True, "source_evidence": "本地 Grafana 不开", "confidence": "high"},
                    {"type": "set_observability", "observability_mode": "disabled", "mutation_explicit": True, "source_evidence": "本地 Grafana 不开", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["observability"]["mode"], "disabled")
        self.assertEqual(result["pending_question"]["id"], "CLOUD_REGION")
        self.assertIn("已禁用", "\n".join(result["visible_response"]))

    def test_completed_observability_group_reopens_owned_question_when_revisited(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "provider_deployment"
        state["chain_identity"] = {"raw": "solana", "canonical": "solana", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "solana"}
        state["observability"] = {"mode": "disabled"}
        state["pending_question"] = {
            "id": "CLOUD_REGION",
            "group": "provider_deployment",
            "kind": "manual_value",
            "field": "CLOUD_REGION",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "回到可观测性配置"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "change_group", "group": "observability", "navigation_explicit": True, "source_evidence": "回到可观测性配置", "confidence": "high"}]}
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "observability_mode")
        self.assertEqual(result["observability"]["mode"], "disabled")

    def test_sync_observe_multi_action_clarifies_invalid_demo_source_transactionally(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["active_group"] = "opening"
        state["last_user_input"] = "我想观察 BNB 节点同步，不打 RPC 压测，并只做流程 demo"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {
                        "type": "clarify_unresolved",
                        "clauses": ["只做流程 demo 不能作为 sync-observe 的真实数据源"],
                        "confidence": "high",
                    },
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["target_mode"], "")
        self.assertEqual(result["workflow_mode"], "")
        self.assertEqual(result["chain_identity"], {})
        self.assertFalse(result.get("sync_observe", {}).get("source"))
        self.assertIn("只做流程 demo", "\n".join(result.get("visible_response") or []))

    def test_legacy_sync_observe_demo_checkpoint_cannot_auto_execute(self) -> None:
        """Live-found design gap (2026-07-13, user manual testing, fixed per

        explicit user direction): `stop_condition` (run-until-stopped/fixed
        duration/stop-when-synced) and `duration_seconds` only make sense for
        a real, indefinite-length sync process -- there is none for a
        plumbing-only demo. The old flow still asked the user to pick among
        them, then observability, then advanced tuning, before finally
        routing through the *same* preflight/submit path as a real
        observation -- so "just a demo" quietly submitted a real job with
        nothing real to observe, after 4 extra questions whose semantics did
        not apply. Fixed so confirming the demo disclaimer auto-fills safe
        internal defaults for all four and runs preflight/smoke immediately.
        """

        from unittest.mock import patch

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "sync-observe"
        state["workflow_mode"] = "sync_observe"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["sync_observe"] = {"source": "demo_only"}
        state["active_group"] = "sync_observe"
        state["pending_question"] = {
            "id": "sync_observe_demo_ack",
            "group": "sync_observe",
            "kind": "yes_no",
            "field": "sync_observe_demo_ack",
            "options": [{"label": "Y", "value": True}, {"label": "N", "value": False}],
            "manual_input_allowed": False,
        }
        state["last_user_input"] = "y"

        with patch("agent.harness.domains.execution_runtime.execution_service.execute") as execute:
            result = process_turn(state)

        execute.assert_not_called()
        self.assertFalse(result.get("preflight", {}).get("approved"))

    def test_paused_action_queue_resumes_after_confirmation(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "provider_deployment"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["last_user_input"] = "换成 eth，使用 real-node，QPS quick"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "choose_target_mode", "target_mode": "real-node", "target_mode_explicit": True, "source_evidence": "real-node", "confidence": "high"},
                    {"type": "change_chain", "chain_text": "eth", "source_evidence": "eth", "confidence": "high"},
                    {"type": "set_qps_mode", "qps_mode": "quick", "mutation_explicit": True, "source_evidence": "QPS quick", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "target_mode_change_confirm")
        self.assertEqual(result["target_mode_change_candidate"], "real-node")
        self.assertEqual(len(result["action_queue"]), 2)

        result["last_user_input"] = "Y"
        result = process_turn(result)

        self.assertEqual(result["pending_question"]["id"], "chain_change_confirm")
        self.assertEqual(result["target_mode"], "real-node")
        self.assertEqual(len(result["action_queue"]), 1)

        result["last_user_input"] = "Y"
        result = process_turn(result)

        self.assertEqual(result["chain_identity"]["canonical"], "ethereum")
        self.assertEqual(result["qps_profile"]["mode"], "quick")
        self.assertEqual(result["pending_question"]["id"], "qps_profile_confirm")

    def test_numbered_answer_to_yes_no_confirm_applies_without_llm(self) -> None:
        """A numbered/label answer to a yes/no confirm ("1"/"2"/"Y"/"N") must be

        treated as a direct answer, not sent to the free-text/LLM resolver.
        Regression for a live dual-AI chaos failure: "1" to a target-mode switch
        confirm was rejected as "not an answer" because `_answer_fits_pending`
        only accepted y/yes/n/no, so the digit fell through to the resolver which
        could not map a lone "1" back to an option.
        """

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        def _confirm_state() -> dict:
            state = new_state("unit-thread", language="zh")
            state["target_mode"] = "sync-observe"
            state["workflow_mode"] = "sync_observe"
            state["active_group"] = "target_mode"
            state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
            state["target_mode_change_candidate"] = "fake-node"
            state["pending_question"] = {
                "id": "target_mode_change_confirm",
                "group": "target_mode",
                "kind": "yes_no",
                "field": "target_mode_change_confirmed",
                "options": [{"label": "Y", "value": True}, {"label": "N", "value": False}],
                "interrupted_group": "provider_deployment",
            }
            return state

        # "1" == first option (Y): the LLM choice resolver must NOT be consulted.
        state = _confirm_state()
        state["last_user_input"] = "1"
        with patch("agent.harness.coordinator.resolve_action_queue", return_value={"actions": []}) as resolver:
            result = process_turn(state)
        resolver.assert_not_called()
        self.assertEqual(result["target_mode"], "fake-node")
        self.assertEqual(result["workflow_mode"], "rpc_benchmark")

        # "2" == second option (N): declines, keeps the original mode.
        state = _confirm_state()
        state["last_user_input"] = "2"
        with patch("agent.harness.coordinator.resolve_action_queue", return_value={"actions": []}) as resolver:
            result = process_turn(state)
        resolver.assert_not_called()
        self.assertEqual(result["target_mode"], "sync-observe")

    def test_custom_rpc_weights_reject_unknown_method_but_allow_template_default(self) -> None:
        """Custom-RPC mixed weights must reject a method that is neither a

        validated custom method nor a chain template default (a typo/garbage such
        as "eth_fooBar"), while still allowing a genuine template-default method.
        Regression for a live chaos finding: "eth_fooBar=50" was accepted just
        because the weights summed to 100, creating a non-existent benchmark method.
        """

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        def _weights_state(answer: str) -> dict:
            state = new_state("unit-thread", language="zh")
            state["target_mode"] = "fake-node"
            state["workflow_mode"] = "rpc_benchmark"
            state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
            state["custom_rpc"] = {
                "status": "needs_weights",
                "scope": "mixed_replace",
                "validated_methods": [{"method": "eth_getBlockByNumber"}, {"method": "eth_getBalance"}],
            }
            state["pending_question"] = {
                "id": "custom_rpc_weights",
                "group": "endpoint_process",
                "kind": "manual_value",
                "field": "custom_rpc_weights",
                "manual_input_allowed": True,
                "validation": {"input_mode": "rpc_weights"},
            }
            state["last_user_input"] = answer
            return state

        def _answer_weights(answer: str) -> dict:
            with patch(
                "agent.harness.coordinator.resolve_action_queue",
                return_value={"actions": [{
                    "type": "answer_pending",
                    "answer": answer,
                    "source_evidence": answer,
                    "confidence": "high",
                }]},
            ):
                return process_turn(_weights_state(answer))

        # Garbage method -> rejected (kept at needs_weights).
        rejected = _answer_weights("eth_getBlockByNumber=50,eth_fooBar=50")
        self.assertEqual(rejected["custom_rpc"]["status"], "needs_weights")
        # mixed_replace accepts only methods validated for this runtime workload.
        accepted = _answer_weights("eth_getBlockByNumber=50,eth_getBalance=50")
        self.assertEqual(accepted["custom_rpc"]["status"], "validated")

        # Empty allowed-set (no validated custom methods AND no template — e.g. a
        # job-local chain with no template): every named method is unknown and
        # must be rejected, not silently accepted because the weights sum to 100.
        def _empty_allowed_state(answer: str) -> dict:
            state = _weights_state(answer)
            state["chain_identity"] = {"raw": "zzz-unknown", "canonical": "zzz-unknown", "status": "confirmed", "case": "known"}
            state["custom_rpc"] = {"status": "needs_weights", "scope": "mixed_replace", "validated_methods": []}
            return state

        answer = "eth_fooBar=100"
        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{
                "type": "answer_pending",
                "answer": answer,
                "source_evidence": answer,
                "confidence": "high",
            }]},
        ):
            empty = process_turn(_empty_allowed_state(answer))
        self.assertEqual(empty["custom_rpc"]["status"], "needs_weights")

    def test_weight_parser_accepts_embedded_structured_mapping_and_rejects_conflicts(self) -> None:
        from agent.harness.input_values import parse_weight_spec

        expected = {"eth_blockNumber": 65, "eth_gasPrice": 35}
        self.assertEqual(parse_weight_spec('{"eth_blockNumber":65,"eth_gasPrice":35}'), expected)
        self.assertEqual(
            parse_weight_spec('Use these weights:\n```json\n{"eth_blockNumber":65,"eth_gasPrice":35}\n```'),
            expected,
        )
        self.assertEqual(
            parse_weight_spec('Review this JSON: {"weights":{"eth_blockNumber":"65","eth_gasPrice":35}}'),
            expected,
        )
        self.assertEqual(parse_weight_spec("eth_blockNumber: 65\neth_gasPrice: 35"), expected)
        self.assertEqual(parse_weight_spec("eth_blockNumber=65,eth_gasPrice=35"), expected)
        self.assertEqual(
            parse_weight_spec(
                '{"eth_blockNumber":65,"eth_gasPrice":35}\n'
                '{"eth_blockNumber":50,"eth_gasPrice":50}'
            ),
            {},
        )
        self.assertEqual(parse_weight_spec('{"eth_blockNumber":true,"eth_gasPrice":35}'), {})

    def test_case1_and_case2_weights_accept_fenced_json_through_shared_domain(self) -> None:
        from agent.harness.domains.chain_rpc import apply_chain_rpc_answer
        from agent.harness.domains.rpc_catalog import migrate_legacy_catalog
        from agent.harness.state import new_state

        def apply(state: dict, question_id: str) -> dict:
            state["pending_question"] = {
                "id": question_id,
                "group": "endpoint_process",
                "kind": "manual_value",
                "field": question_id,
                "manual_input_allowed": True,
                "validation": {"input_mode": "rpc_weights"},
            }
            answer = 'Use these weights:\n```json\n{"eth_blockNumber":65,"eth_gasPrice":35}\n```'
            outcome = apply_chain_rpc_answer(state, state["pending_question"], answer, answer)
            return _commit_result(state, outcome, owner="chain_rpc")

        case1 = new_state("case1-json-weights", language="en")
        case1["target_mode"] = "real-node"
        case1["workflow_mode"] = "rpc_benchmark"
        case1["chain_identity"] = {"canonical": "bsc", "status": "confirmed", "case": "known"}
        case1["custom_rpc"] = {
            "status": "needs_weights",
            "scope": "mixed_replace",
            "validated_methods": [
                {"method": "eth_blockNumber", "params": []},
                {"method": "eth_gasPrice", "params": []},
            ],
        }
        migrate_legacy_catalog(case1)
        case1 = apply(case1, "custom_rpc_weights")
        self.assertEqual(case1["workload"]["mixed_weights"], {"eth_blockNumber": 65, "eth_gasPrice": 35})

        case2 = new_state("case2-json-weights", language="en")
        case2["target_mode"] = "real-node"
        case2["workflow_mode"] = "rpc_benchmark"
        case2["chain_identity"] = {
            "canonical": "flow-evm",
            "adapter_family": "jsonrpc",
            "status": "existing_family_needs_weights",
            "case": "case2",
            "workload_scope": "mixed_replace",
            "validated_methods": [
                {"method": "eth_blockNumber", "params": []},
                {"method": "eth_gasPrice", "params": []},
            ],
        }
        migrate_legacy_catalog(case2)
        case2 = apply(case2, "new_chain_custom_weights")
        self.assertEqual(case2["workload"]["mixed_weights"], {"eth_blockNumber": 65, "eth_gasPrice": 35})

    def test_qps_override_rejects_invalid_values(self) -> None:
        """QPS override values must be validated: positive integers, MAX >= INITIAL.

        Regression for a live chaos failure: `set_qps_override` stored negative /
        zero / inverted / non-integer values silently (nothing validated them
        downstream), so an invalid QPS config could reach preflight/execution.
        """

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.performance import validate_qps_overrides
        from agent.harness.state import new_state

        # Pure-function validation.
        self.assertTrue(validate_qps_overrides("standard", {"INITIAL_QPS": "-100"}))
        self.assertTrue(validate_qps_overrides("standard", {"MAX_QPS": "0"}))
        self.assertTrue(validate_qps_overrides("standard", {"INITIAL_QPS": "5000", "MAX_QPS": "100"}))
        self.assertTrue(validate_qps_overrides("standard", {"QPS_STEP": "2.5"}))
        self.assertTrue(validate_qps_overrides("standard", {"DURATION": "abc"}))
        self.assertFalse(validate_qps_overrides("standard", {"INITIAL_QPS": "1000", "MAX_QPS": "5000", "QPS_STEP": "500"}))
        self.assertFalse(validate_qps_overrides("standard", {"MAX_QPS": "50000"}))

        # End-to-end: an invalid override must not confirm the QPS profile.
        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "qps_profile"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
        state["qps_profile"] = {"mode": "standard"}
        state["last_user_input"] = "把 INITIAL_QPS 设成 -100"
        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "set_qps_override", "qps_overrides": {"INITIAL_QPS": "-100"}, "confidence": "high"}]}
            result = process_turn(state)
        self.assertFalse((result.get("qps_profile") or {}).get("confirmed"))
        self.assertFalse((result.get("qps_profile") or {}).get("overrides"))

        # Cross-turn invariant: MAX_QPS < a previously-stored INITIAL_QPS must be
        # rejected even though each turn's dict is individually valid (the merged
        # view is what carries the invariant).
        cross = new_state("unit-thread-2", language="zh")
        cross["target_mode"] = "fake-node"
        cross["workflow_mode"] = "rpc_benchmark"
        cross["active_group"] = "qps_profile"
        cross["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
        cross["qps_profile"] = {"mode": "standard", "overrides": {"INITIAL_QPS": "1000"}}
        cross["last_user_input"] = "把 MAX_QPS 设成 100"
        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "set_qps_override", "qps_overrides": {"MAX_QPS": "100"}, "confidence": "high"}]}
            cross_result = process_turn(cross)
        self.assertFalse((cross_result.get("qps_profile") or {}).get("confirmed"))
        # The invalid MAX_QPS must not have been merged into the stored overrides.
        self.assertNotEqual((cross_result.get("qps_profile") or {}).get("overrides", {}).get("MAX_QPS"), "100")

    def test_qps_override_validates_against_mode_defaults_not_just_overrides(self) -> None:
        """A single-sided QPS override must be checked against the mode's baseline.

        Regression for a live chaos failure: on `intensive` mode (baseline
        INITIAL_QPS=50000), overriding only MAX_QPS to 100 was accepted and the
        flow advanced straight past QPS to RPC mode — an inverted profile
        (MAX_QPS 100 < the still-in-effect INITIAL_QPS 50000) reached the
        confirmed state uncaught, because both `set_qps_override` and the
        `qps_adjust_value` deterministic path validated only the `overrides`
        dict in isolation, never merging in the mode's own defaults for the
        side that was never overridden.
        """

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.questions import manual_question
        from agent.harness.state import new_state

        # Path 1: free-text `set_qps_override` action.
        state = new_state("unit-thread-qps-mode-default", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "qps_profile"
        state["chain_identity"] = {"raw": "solana", "canonical": "solana", "status": "confirmed", "case": "known"}
        state["qps_profile"] = {"mode": "intensive"}
        state["last_user_input"] = "把 MAX_QPS 设成 100"
        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{"type": "set_qps_override", "qps_overrides": {"MAX_QPS": "100"}, "confidence": "high"}]},
        ):
            result = process_turn(state)
        self.assertFalse((result.get("qps_profile") or {}).get("confirmed"))
        self.assertNotEqual((result.get("qps_profile") or {}).get("overrides", {}).get("MAX_QPS"), "100")

        # Path 2: deterministic qps_adjust_value answer (the numbered-menu flow).
        adjust_state = new_state("unit-thread-qps-mode-default-2", language="zh")
        adjust_state["target_mode"] = "fake-node"
        adjust_state["workflow_mode"] = "rpc_benchmark"
        adjust_state["active_group"] = "qps_profile"
        adjust_state["chain_identity"] = {"raw": "solana", "canonical": "solana", "status": "confirmed", "case": "known"}
        adjust_state["qps_profile"] = {"mode": "intensive", "adjust_field": "MAX_QPS"}
        adjust_state["pending_question"] = manual_question(
            "qps_profile", "qps_adjust_value", "请输入 MAX_QPS 的值。", field="qps_adjust_value"
        )
        adjust_state["last_user_input"] = "100"
        adjust_result = process_turn(adjust_state)
        self.assertNotEqual((adjust_result.get("qps_profile") or {}).get("overrides", {}).get("MAX_QPS"), "100")
        self.assertIsNotNone(adjust_result.get("pending_question"))  # re-asked, not advanced to RPC mode

    def test_environment_readiness_question_answered_from_discovery(self) -> None:
        """"can my machine run this / is my env ready" must answer from startup

        discovery (status + detected specs + deps), not a generic requirements
        checklist. Regression for a real transcript where "我这台机器能不能跑" got
        a generic workflow reply despite the data being computed at startup.
        """

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.orientation import answer_consultation
        from agent.harness.state import new_state

        # Content builder answers from startup discovery (status + detected specs).
        state = new_state("unit-thread", language="zh")
        state["discovery"] = {
            "dependencies": {"missing_required": [], "missing_optional": ["docker"]},
            "cloud": {"provider": "gcp", "machine_type": "e2-standard-4"},
            "host": {"cpu_count": 4, "memory_gib": 15.62, "os": "linux"},
        }
        ready = answer_consultation(state, {"topic": "environment_readiness"})
        self.assertIn("e2-standard-4", ready)
        self.assertIn("就绪", ready)

        state["discovery"]["dependencies"]["missing_required"] = ["vegeta"]
        not_ready = answer_consultation(state, {"topic": "environment_readiness"})
        self.assertIn("vegeta", not_ready)

        # Conforming contract: the resolver types the readiness question as
        # answer_opening_question topic=environment_readiness; the harness routes
        # it to the discovery-based response, not a generic requirements checklist.
        # (Resolver prompt rule lives in intent.py; exercised live, not mocked.)
        routed_state = new_state("unit-thread-2", language="zh")
        routed_state["discovery"] = {
            "dependencies": {"missing_required": [], "missing_optional": []},
            "cloud": {"provider": "gcp", "machine_type": "e2-standard-4"},
            "host": {"cpu_count": 4, "memory_gib": 15.62, "os": "linux"},
        }
        routed_state["last_user_input"] = "我这台机器能不能跑"
        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{
                "type": "answer_opening_question",
                "topic": "environment_readiness",
                "confidence": "high",
            }]},
        ):
            routed = process_turn(routed_state)
        routed_text = "\n".join(routed.get("visible_response") or [])
        self.assertIn("e2-standard-4", routed_text)

    def test_provider_metadata_confirm_stores_detected_value_not_literal(self) -> None:
        """CLOUD_REGION/ZONE/MACHINE_TYPE must store the DETECTED value when the

        user accepts it, never the user's literal sentence. Regression for a live
        chaos bug: answering "用检测到的就行" to the bare CLOUD_REGION prompt stored
        the Chinese phrase as the region (garbage report metadata), because the
        prompt promised "use the detected value" but nothing bound it.
        """

        from agent.harness.coordinator import _question_for_group
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        def _mk() -> dict:
            s = new_state("t", language="zh")
            s["target_mode"] = "fake-node"
            s["workflow_mode"] = "rpc_benchmark"
            s["chain_identity"] = {"raw": "solana", "canonical": "solana", "status": "confirmed", "case": "known"}
            s["active_group"] = "provider_deployment"
            s["discovery"] = {"cloud": {"provider": "gcp", "region": "us-central1", "zone": "us-central1-a", "machine_type": "e2-standard-4"}}
            return s

        # The group now confirms the detected value (Y/N), naming it in the prompt.
        q = _question_for_group(_mk(), "provider_deployment")
        self.assertEqual(q["kind"], "yes_no")
        self.assertEqual(q["field"], "CLOUD_REGION")
        self.assertIn("us-central1", q["prompt"])

        # Accepting (numbered or Y) stores the detected value, not any literal text.
        for answer in ("1", "y"):
            s = _mk()
            s["pending_question"] = q
            s["last_user_input"] = answer
            with patch("agent.harness.coordinator.resolve_action_queue", return_value={"actions": []}):
                r = process_turn(s)
            self.assertEqual((r.get("confirmed_config") or {}).get("CLOUD_REGION"), "us-central1")

        # Declining (N) asks for a custom value next turn, which is stored verbatim.
        s = _mk()
        s["pending_question"] = q
        s["last_user_input"] = "n"
        with patch("agent.harness.coordinator.resolve_action_queue", return_value={"actions": []}):
            r = process_turn(s)
        self.assertTrue((r.get("inferred_config") or {}).get("CLOUD_REGION_manual_required"))
        self.assertIsNone((r.get("confirmed_config") or {}).get("CLOUD_REGION"))
        q2 = _question_for_group(r, "provider_deployment")
        self.assertEqual(q2["kind"], "manual_value")
        r["pending_question"] = q2
        r["last_user_input"] = "asia-east1"
        with patch("agent.harness.coordinator.resolve_action_queue", return_value={"actions": []}):
            r2 = process_turn(r)
        self.assertEqual((r2.get("confirmed_config") or {}).get("CLOUD_REGION"), "asia-east1")

    def test_choice_contract_accepts_valid_manual_replacement_atomically(self) -> None:
        """A choice that advertises manual input must not discard its value."""

        from agent.harness.coordinator import (
            _action_answers_pending_contract,
            _dispatch_pending_action,
        )
        from agent.harness.domains.environment import question_for_environment
        from agent.harness.state import new_state

        state = new_state("manual-choice", language="en")
        state.update({
            "target_mode": "real-node",
            "workflow_mode": "rpc_benchmark",
            "active_group": "ledger_disk",
            "confirmed_config": {
                "LEDGER_DEVICE": "vda",
                "DATA_VOL_TYPE": "hyperdisk-balanced",
            },
            "discovery": {
                "disks": {
                    "candidates": [
                        {"name": "vda", "size": "926.3G", "type": "disk"},
                    ],
                },
            },
        })
        question = question_for_environment(state, "ledger_disk")
        self.assertEqual(question["kind"], "yes_no")
        self.assertTrue(question["manual_input_allowed"])
        state["pending_question"] = question
        state["last_user_input"] = "Use 1000 GiB instead."
        action = {
            "type": "answer_pending",
            "answer": "1000",
            "source_evidence": "1000",
            "semantic_purpose_verified": True,
        }

        self.assertTrue(_action_answers_pending_contract(state, action))
        result = _dispatch_pending_action(state, action)

        self.assertEqual((result.get("confirmed_config") or {}).get("DATA_VOL_SIZE"), "1000")
        self.assertFalse((result.get("inferred_config") or {}).get("DATA_VOL_SIZE_manual_required"))

    def test_optional_chain_auxiliary_field_accepts_plain_english_decline(self) -> None:
        """B.18 (known-issues.md): an optional `chain_auxiliary_endpoints`

        field (e.g. `RPC_API_KEY`, described to the user as optional) rejected
        a plain-English decline ("skip it, I don't have one") as "that reply
        does not look like an answer", while the bare word "none" was
        accepted immediately after -- `_is_plain_scalar_answer`'s "single
        word, no punctuation" gate has no natural-language matching, unlike
        the broader NL matching `has_accounts_device` already used
        (`_text_mentions_no_accounts_disk`/`_text_mentions_yes_accounts_disk`).
        Fixed by mirroring that same pattern for optional auxiliary fields.
        """

        from agent.harness.coordinator import _question_for_group
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("t", language="en")
        state["target_mode"] = "real-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "dogecoin", "canonical": "dogecoin", "status": "confirmed", "case": "known"}
        state["active_group"] = "chain_auxiliary_endpoints"

        question = _question_for_group(state, "chain_auxiliary_endpoints")
        self.assertEqual(question["id"], "RPC_API_KEY")
        self.assertEqual(question["kind"], "manual_value")

        state["pending_question"] = question
        state["last_user_input"] = "skip it, I don't have one"
        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{"type": "answer_pending", "answer": "skip", "selected_value": "none", "source_evidence": "skip it, I don't have one", "semantic_purpose_verified": True, "confidence": "high"}]},
        ):
            result = process_turn(state)

        self.assertEqual((result.get("confirmed_config") or {}).get("RPC_API_KEY"), "none")
        # It must not have stored the user's literal sentence as the "key".
        self.assertNotIn("skip", str((result.get("confirmed_config") or {}).get("RPC_API_KEY") or ""))

    def test_config_field_explanation_answers_from_runtime_contract(self) -> None:
        """Asking what a config field means / whether it affects results must be

        answered concretely from the runtime field contract (purpose, inference,
        applies-to), not with a generic "paste the field and I'll explain" reply.
        Regression for a live chaos turn: during the DATA_VOL_TYPE prompt, "这个
        磁盘类型会不会影响压测结果" returned the canned config placeholder.
        """

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.orientation import config_field_explanation
        from agent.harness.state import new_state

        # Resolver names the field in subject.
        by_subject = config_field_explanation(new_state("t", language="zh"), "DATA_VOL_TYPE", "zh")
        self.assertIsNotNone(by_subject)
        self.assertIn("DATA_VOL_TYPE", by_subject)
        self.assertIn("作用", by_subject)  # concrete purpose, not a placeholder

        # No subject -> fall back to the active pending question's field.
        pending_state = new_state("t2", language="zh")
        pending_state["pending_question"] = {"id": "DATA_VOL_TYPE", "group": "ledger_disk", "field": "DATA_VOL_TYPE"}
        by_pending = config_field_explanation(pending_state, "", "zh")
        self.assertIsNotNone(by_pending)
        self.assertIn("DATA_VOL_TYPE", by_pending)

        # Unknown field -> None (handler keeps its generic fallback).
        self.assertFalse(config_field_explanation(new_state("t3", language="zh"), "NOT_A_FIELD", "zh"))

        # End-to-end via the typed config_explanation topic during a pending field.
        state = new_state("t4", language="zh")
        state["pending_question"] = {"id": "DATA_VOL_TYPE", "group": "ledger_disk", "field": "DATA_VOL_TYPE"}
        state["last_user_input"] = "这个磁盘类型会不会影响压测结果"
        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{
                "type": "answer_opening_question",
                "topic": "config_explanation",
                "subject": "DATA_VOL_TYPE",
                "confidence": "high",
            }]},
        ):
            result = process_turn(state)
        response = "\n".join(result.get("visible_response") or [])
        self.assertIn("DATA_VOL_TYPE", response)
        self.assertNotIn("贴出当前问题", response)  # not the canned placeholder

    def test_config_field_explanation_covers_sync_observe_only_fields(self) -> None:
        """Field explanations must cover every catalog, not just the baseline set.

        Regression for a live chaos turn: asking "系统里默认支持哪几种停止条件" during
        sync-observe returned the canned "paste the field" placeholder because
        `_config_field_knowledge` only searched `RUNTIME_BASELINE_FIELDS`, which
        excludes `SYNC_OBSERVE_FIELDS` (a separate catalog per
        `agent/knowledge/entry_contract.py`). Also covers fields with no 1:1 env
        var (`field.env == ""`), which must display their logical key instead of
        an empty `` `` `` pair.
        """

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.orientation import config_field_explanation
        from agent.harness.state import new_state

        by_subject = config_field_explanation(new_state("t", language="zh"), "sync_observe_stop_condition", "zh")
        self.assertIsNotNone(by_subject)
        self.assertIn("sync_observe_stop_condition", by_subject)  # falls back to key, not `` ``
        self.assertIn("作用", by_subject)
        self.assertNotIn("``", by_subject)

        # The resolver's `subject` is free-text guessed from phrasing, not a
        # canonical key lookup, and produced the pluralized
        # "sync_observe_stop_conditions" for "系统里默认支持哪几种停止条件" live —
        # must still resolve to the singular registered field.
        by_pluralized_subject = config_field_explanation(
            new_state("t1b", language="zh"), "sync_observe_stop_conditions", "zh"
        )
        self.assertIsNotNone(by_pluralized_subject)
        self.assertIn("sync_observe_stop_condition", by_pluralized_subject)

        # Unrelated free text must still miss (normalization isn't so loose it
        # matches anything).
        self.assertFalse(config_field_explanation(new_state("t1c", language="zh"), "totally_unrelated_thing", "zh"))

        state = new_state("t2", language="zh")
        state["pending_question"] = {"id": "sync_observe_stop_condition", "group": "sync_observe", "field": "sync_observe_stop_condition"}
        state["last_user_input"] = "系统里默认支持哪几种停止条件"
        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{
                "type": "answer_opening_question",
                "topic": "config_explanation",
                "subject": "sync_observe_stop_conditions",
                "confidence": "high",
            }]},
        ):
            result = process_turn(state)
        response = "\n".join(result.get("visible_response") or [])
        self.assertIn("sync_observe_stop_condition", response)
        self.assertNotIn("贴出当前问题", response)  # not the canned placeholder

    def test_config_field_explanation_covers_qps_and_advanced_tuning_fields(self) -> None:
        """Field explanations must also cover QPS-profile and advanced_tuning sub-fields.

        Regression for a live chaos finding: asking "QPS_STEP是干什么用的" or
        "MAX_LATENCY_THRESHOLD这个阈值具体是干嘛的" still returned the canned
        placeholder — `#31`'s fix widened `_config_field_knowledge` to search
        `ALL_RUNTIME_FIELDS`, but the QPS-profile sub-fields (`INITIAL_QPS`,
        `MAX_QPS`, `QPS_STEP`, `DURATION`) and every `advanced_tuning` threshold
        field had no `RuntimeField` entry in any catalog at all, so validation
        (`#36`/`#39`) and explanation drifted apart for the same field families.
        """

        from agent.harness.domains.orientation import config_field_explanation
        from agent.harness.state import new_state

        for field in ("QPS_STEP", "INITIAL_QPS", "MAX_QPS", "MAX_LATENCY_THRESHOLD", "BOTTLENECK_CPU_THRESHOLD"):
            explanation = config_field_explanation(new_state(f"t-{field}", language="zh"), field, "zh")
            self.assertIsNotNone(explanation, f"no field explanation for {field}")
            self.assertIn(field, explanation)
            self.assertIn("作用", explanation)

    def test_reason_label_does_not_leak_raw_internal_status(self) -> None:
        """`_reason_label` must not echo a raw internal status enum verbatim.

        Regression for a live chaos finding: after a custom-RPC schema
        validation failure, the "recommended next action" message included the
        raw, untranslated reason string
        "continue custom RPC workflow: needs_schema_evidence" — `_reason_label`'s
        `mapping` dict had no entry for this `routing.next_group_and_reason`
        f-string pattern (or its new-chain-validation sibling), so it fell
        through to echoing the internal status enum verbatim in a user-facing
        sentence.
        """

        from agent.harness.oracle import _reason_label

        for reason, forbidden in (
            ("continue custom RPC workflow: needs_schema_evidence", "needs_schema_evidence"),
            ("continue new-chain validation: existing_family_needs_method", "existing_family_needs_method"),
        ):
            for language in ("zh", "en"):
                label = _reason_label(reason, language)
                self.assertNotIn(forbidden, label)
                self.assertTrue(label.strip())

    def test_execution_status_prefers_live_job_status_over_stale_snapshot(self) -> None:
        """B.20 (known-issues.md): `_execution_status` read `state["job"]`,

        a one-time snapshot written at submission time
        (`agent/harness/domains/execution_runtime.py`) and never refreshed. A
        `current_config`/status-dump response could claim `job_running` for a
        job that had actually long since finished, contradicting the
        deterministic `status`/`jobs`/`logs` commands (which already read
        correctly from disk via `job_manager`). Fixed by looking up the live
        on-disk status by `job_id` via `job_manager.get_job` and preferring it
        over the stale snapshot.
        """

        from unittest.mock import patch

        from agent.harness.oracle import compute_next_action

        state = {"job": {"job_id": "job_demo", "status": "running"}}
        with patch("agent.harness.oracle.get_job", return_value={"status": "failed"}):
            action = compute_next_action(state)
        self.assertEqual(action.execution_status, "job_failed")

        # If the live lookup fails (e.g. the job directory is gone), fall
        # back to the snapshot rather than raising.
        with patch("agent.harness.oracle.get_job", side_effect=FileNotFoundError("gone")):
            action = compute_next_action(state)
        self.assertEqual(action.execution_status, "job_running")

    def test_config_status_not_forced_complete_by_a_stale_unrelated_job(self) -> None:
        """Live-found regression (2026-07-13, user manual testing): `job`/

        `latest_job_id` deliberately survive a full reset
        (`RESET_PRESERVED_KEYS`) so `analyze_report`/`status` keep working for
        the last completed job. `compute_next_action` used to force
        `config_status` to "complete" whenever `execution_status` showed any
        job status at all, with no check that the job belonged to the
        in-progress workflow -- so a brand-new sync-observe setup with a
        leftover `job_failed` from an earlier, unrelated run reported
        `config_status: complete` in the same response that also named a real
        unmet next blocking question (`choose sync-observe stop condition`),
        a directly self-contradictory status dump.
        """

        from unittest.mock import patch

        from agent.harness.oracle import compute_next_action

        state = {
            "target_mode": "sync-observe",
            "workflow_mode": "sync_observe",
            "chain_identity": {"status": "confirmed", "canonical": "bsc"},
            "confirmed_config": {
                "CLOUD_REGION": "us-central1",
                "CLOUD_ZONE": "us-central1-a",
                "MACHINE_TYPE": "e2-standard-4",
                "LEDGER_DEVICE": "sda",
                "DATA_VOL_TYPE": "pd-ssd",
                "DATA_VOL_SIZE": "10",
                "DATA_VOL_MAX_IOPS": "3000",
                "DATA_VOL_MAX_THROUGHPUT": "125",
                "has_accounts_device": False,
                "NETWORK_INTERFACE": "ens4",
                "NETWORK_MAX_BANDWIDTH_GBPS": "10",
            },
            "sync_observe": {"source": "demo_only", "demo_acknowledged": True},
            "job": {"job_id": "job_old_unrelated", "status": "failed"},
        }
        with patch("agent.harness.oracle.get_job", return_value={"status": "failed"}):
            action = compute_next_action(state)
        self.assertEqual(action.execution_status, "job_failed")
        self.assertEqual(action.config_status, "incomplete")
        self.assertEqual(action.next_blocking_group, "sync_observe")
        self.assertTrue(action.blockers)

    def test_endpoint_question_context_explains_field_not_generic_state_dump(self) -> None:
        """"Why do I need this field, can I skip it" for LOCAL_RPC_URL/

        SYNC_OBSERVE_RPC_URL must get the field's actual purpose, not a generic
        state dump. Regression for a live chaos finding: `_pending_context_response`
        had real explanatory text for `new_chain_endpoint`/`custom_rpc_endpoint`
        but fell through to `format_current_context` (an unrelated state-summary
        dump) for `LOCAL_RPC_URL`/`SYNC_OBSERVE_RPC_URL`, even though
        `_pending_endpoint_context_question` explicitly routes both into this
        same handler. `SYNC_OBSERVE_RPC_URL` also had no `RuntimeField` entry at
        all before this fix, so the underlying field-explanation lookup could
        not have found it regardless of routing.
        """

        from agent.harness.domains.orientation import pending_context_response
        from agent.harness.state import new_state

        state = new_state("t", language="zh")
        for question_id in ("LOCAL_RPC_URL", "SYNC_OBSERVE_RPC_URL"):
            question = {"id": question_id, "group": "endpoint_process", "field": question_id}
            response = pending_context_response(state, question)
            self.assertIn(question_id, response)
            self.assertIn("作用", response)
            # Not format_current_context's generic "current pending question is X" fallback.
            self.assertNotIn("当前待确认问题是", response)

    def test_sync_observe_duration_rejects_invalid_values(self) -> None:
        """sync-observe duration_seconds must be validated like QPS overrides.

        Regression for a live chaos turn: after choosing the "fixed duration"
        stop condition, answering "-100" to the duration prompt was accepted
        with no validation and the flow advanced straight to observability —
        a negative duration would reach `strategy_planner`'s
        `--duration <value>` CLI arg unchecked.
        """

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.questions import manual_question
        from agent.harness.state import new_state

        def _make_state() -> dict:
            state = new_state("unit-thread-duration", language="zh")
            state["target_mode"] = "sync-observe"
            state["workflow_mode"] = "sync_observe"
            state["active_group"] = "sync_observe"
            state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
            state["sync_observe"] = {"source": "endpoint_only", "stop_condition": "duration"}
            state["pending_question"] = manual_question(
                "sync_observe",
                "sync_observe_duration_seconds",
                "请输入 sync-observe 观察时长，单位秒。",
                field="sync_observe_duration_seconds",
                validation={"value_type": "positive_number"},
            )
            return state

        for bad in ("-100", "0", "abc"):
            state = _make_state()
            state["last_user_input"] = bad
            result = process_turn(state)
            self.assertEqual((result.get("sync_observe") or {}).get("duration_seconds"), None)
            self.assertIsNotNone(result.get("pending_question"))  # re-asked, not advanced to observability
            response = "\n".join(result.get("visible_response") or [])
            self.assertIn("无效", response)

        good_state = _make_state()
        good_state["last_user_input"] = "600"
        good_result = process_turn(good_state)
        self.assertEqual((good_result.get("sync_observe") or {}).get("duration_seconds"), 600)

    def test_exporter_observability_mode_discloses_scrape_port(self) -> None:
        """Choosing exporter-only observability must tell the user the scrape target.

        Regression for a live chaos turn: picking "只要exporter" (exporter-only,
        meant to integrate with the user's EXISTING Prometheus) confirmed the
        mode and moved straight to advanced-tuning defaults without ever saying
        which port/URL the user's Prometheus should scrape — defeating the
        entire point of exporter mode. Covers both the free-text
        `set_observability` action path and the deterministic numbered-choice
        pending-answer path.
        """

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.questions import manual_question
        from agent.harness.state import new_state

        state = new_state("unit-thread-exporter", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "observability"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
        state["last_user_input"] = "第三个，只要exporter"
        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{"type": "set_observability", "observability_mode": "exporter", "mutation_explicit": True, "source_evidence": "只要exporter", "confidence": "high"}]},
        ):
            result = process_turn(state)
        self.assertEqual((result.get("observability") or {}).get("mode"), "exporter")
        response = "\n".join(result.get("visible_response") or [])
        self.assertIn("9108", response)
        self.assertIn("Prometheus", response)

        # Deterministic numbered-choice path must disclose the same thing.
        pending_state = new_state("unit-thread-exporter-2", language="zh")
        pending_state["target_mode"] = "fake-node"
        pending_state["workflow_mode"] = "rpc_benchmark"
        pending_state["active_group"] = "observability"
        pending_state["pending_question"] = manual_question(
            "observability", "observability_mode", "请选择可观测性模式。", field="observability_mode", kind="numbered_choice"
        )
        pending_state["pending_question"]["options"] = [
            {"label": "禁用", "value": "disabled"},
            {"label": "本地 Prometheus/Grafana", "value": "local"},
            {"label": "仅 exporter，对接已有 Prometheus", "value": "exporter"},
        ]
        pending_state["last_user_input"] = "3"
        pending_result = process_turn(pending_state)
        self.assertEqual((pending_result.get("observability") or {}).get("mode"), "exporter")
        pending_response = "\n".join(pending_result.get("visible_response") or [])
        self.assertIn("9108", pending_response)

    def test_sync_observe_stop_condition_and_duration_correctable_via_nl(self) -> None:
        """sync-observe stop_condition/duration must be re-settable via natural language.

        Regression for a live chaos dead-end: once `sync_observe_stop_condition`
        and `sync_observe_duration_seconds` were answered, there was no way to
        correct them afterward — `SYNC_OBSERVE_STOP_CONDITION`/
        `SYNC_OBSERVE_DURATION_SECONDS` were absent from every field-mapping
        allowlist (`CONFIRMABLE_CONFIG_FIELDS`/`SPECIAL_CONFIG_FIELDS`), so a
        correction like "把观察时长改成 600" was always classified "unmapped" and
        silently discarded, even though the resolver correctly extracted it.
        Also covers the invalid-duration path so an invalid correction can't
        sneak back in through this second entry point.
        """

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        def _make_state(config_values: dict) -> dict:
            state = new_state("unit-thread-sync-nl-fix", language="zh")
            state["target_mode"] = "sync-observe"
            state["workflow_mode"] = "sync_observe"
            state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
            state["sync_observe"] = {"source": "endpoint_only", "stop_condition": "duration", "duration_seconds": "-100"}
            state["inferred_config"] = {"pending_review": {
                "config_values": config_values,
                "unmapped_values": {},
                "source_format": "mixed",
                "reason": "user requested a duration change",
            }}
            state["pending_question"] = {
                "id": "inferred_config_review",
                "group": "sync_observe",
                "kind": "yes_no",
                "field": "inferred_config_review",
                "options": [{"label": "Y", "value": True}, {"label": "N", "value": False}],
            }
            state["last_user_input"] = "Y"
            return state

        good = _make_state({"SYNC_OBSERVE_STOP_CONDITION": "duration", "SYNC_OBSERVE_DURATION_SECONDS": "600"})
        good_result = process_turn(good)
        self.assertEqual((good_result.get("sync_observe") or {}).get("duration_seconds"), 600)
        self.assertEqual((good_result.get("sync_observe") or {}).get("stop_condition"), "duration")

        bad = _make_state({"SYNC_OBSERVE_DURATION_SECONDS": "-50"})
        bad_result = process_turn(bad)
        # The stale -100 must not be silently replaced with another invalid value.
        self.assertEqual((bad_result.get("sync_observe") or {}).get("duration_seconds"), "-100")
        self.assertIn("无效", "\n".join(bad_result.get("visible_response") or []))

        # Live chaos turn: the resolver extracted "SYNC_OBSERVE_DURATION" (no
        # "_SECONDS" suffix) for "把 sync observe 观察时长改成 600" — must still
        # resolve to the same field, not fall into unmapped_values.
        aliased = _make_state({"SYNC_OBSERVE_DURATION": "600"})
        aliased_result = process_turn(aliased)
        self.assertEqual((aliased_result.get("sync_observe") or {}).get("duration_seconds"), 600)

    def test_declined_yes_no_with_trailing_intent_uses_typed_pending_action(self) -> None:
        """A Y/N answer with trailing intent is resolved as an ordered typed plan.

        A complete turn containing more than the exact local ``N`` token goes
        through the model planner. The planner must bind the explicit decline
        to the active typed question with source evidence; the QPS owner then
        opens its adjustment subflow without invented navigation actions.
        """

        from unittest.mock import patch as _patch

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread-declined-yn", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "solana", "canonical": "solana", "status": "confirmed", "case": "known"}
        state["active_group"] = "qps_profile"
        state["qps_profile"] = {"mode": "intensive", "default_decision_made": False}
        state["pending_question"] = {
            "id": "qps_profile_confirm",
            "group": "qps_profile",
            "kind": "yes_no",
            "field": "qps_profile_confirmed",
            "options": [{"label": "Y", "value": True}, {"label": "N", "value": False}],
            "manual_input_allowed": False,
        }
        state["last_user_input"] = "N，我要调一下"

        with (
            _patch(
                "agent.harness.coordinator.resolve_action_queue",
                return_value=_admitted_mock_plan(state, state["last_user_input"], {"actions": [
                    {"type": "answer_pending", "answer": "N", "selected_value": False, "source_evidence": "N", "pending_option_semantic_verified": True, "semantic_purpose_verified": True, "confidence": "high"},
                ]}),
            ) as resolver,
        ):
            result = process_turn(state)
        resolver.assert_called_once()

        self.assertFalse((result.get("qps_profile") or {}).get("confirmed"))
        self.assertEqual(result.get("pending_question", {}).get("id"), "qps_adjust_field")
        # The harness must actually render the next question, not leave
        # visible_response empty (which the terminal would mask with a generic
        # "ADK returned nothing" fallback).
        self.assertTrue("\n".join(result.get("visible_response") or []).strip())

    def test_fuzzy_matched_false_decline_does_not_invert_to_confirm(self) -> None:
        """A fuzzy-matched decline (`selected_value=False`) must not silently

        execute the confirmed action. Regression for a critical live chaos
        finding: declining `preflight_smoke_confirm` with a non-leading-token
        phrase ("nope, let me adjust something first") is resolved correctly by
        typed action plan to `selected_value=False`, but the deterministic
        dispatch at `process_turn` applies it via
        `_apply_pending_answer(state, str(selected), pending)` — round-tripping
        the Python `False` through `str()` into the literal text "False". Inside
        `_coerce_answer`'s generic option-matching loop, `str(option.get("value")
        or "")` collapsed the "N" option's `False` value to `""` (since `False`
        is falsy) before stringifying, so "false" never matched any candidate
        and the loop fell through to `return raw`, returning the **string**
        "False". `bool("False")` is `True` in Python, so
        `preflight_smoke_execution`'s `if value: return
        run_approved_preflight_and_smoke(state)` silently inverted an explicit
        decline into **actually submitting a real benchmark job** — reproduced
        live against a real BSC endpoint during this exact test sweep. The same
        `is False` identity-check pattern is used by several other yes/no
        confirms (`target_mode_change_confirm`, `chain_change_confirm`, ...),
        so this was a systemic risk, not specific to preflight.
        """

        from agent.harness.coordinator import _coerce_answer
        from agent.harness.questions import choice_question
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        preflight_question = choice_question(
            "preflight_smoke_execution",
            "preflight_smoke_confirm",
            "Run preflight and smoke?",
            field="preflight_smoke_confirmed",
            kind="yes_no",
            options=[
                {"label": "Y", "value": True, "action": {"type": "approve_preflight_smoke"}},
                {"label": "N", "value": False, "action": {"type": "reject_preflight_smoke"}},
            ],
        )
        # Pure-function check: the exact round-trip the dispatch performs.
        self.assertIs(_coerce_answer(str(False), preflight_question), False)
        self.assertIs(_coerce_answer(str(True), preflight_question), True)

        state = new_state("unit-thread-false-decline", language="en")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
        state["active_group"] = "preflight_smoke_execution"
        state["preflight"] = {}
        state["pending_question"] = dict(preflight_question)
        state["last_user_input"] = "nope, let me adjust something first"

        with (
            patch(
                "agent.harness.coordinator.resolve_action_queue",
                return_value={"actions": [{"type": "answer_pending", "answer": "nope", "selected_value": False, "confidence": "high"}]},
            ),
            patch("agent.harness.domains.execution_runtime.execution_service.execute") as execute,
        ):
            result = process_turn(state)

        execute.assert_not_called()  # must NOT execute on a decline
        self.assertFalse((result.get("preflight") or {}).get("approved"))

    def test_disk_and_network_numeric_fields_reject_negative_values(self) -> None:
        """DATA_VOL_SIZE/MAX_IOPS/MAX_THROUGHPUT, their ACCOUNTS_VOL_* twins, and

        NETWORK_MAX_BANDWIDTH_GBPS must reject negative/zero/non-numeric values
        via the direct manual-question answer path. Regression for a live chaos
        finding: typing "-50" to "Confirm DATA_VOL_SIZE in GiB" (and the same for
        MAX_IOPS, ACCOUNTS_VOL_SIZE, NETWORK_MAX_BANDWIDTH_GBPS) was accepted with
        zero validation and the flow advanced to the next question regardless.
        """

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.questions import manual_question
        from agent.harness.state import new_state

        def _answer(group: str, field: str, answer: str) -> dict:
            state = new_state(f"unit-thread-disk-neg-{field}-{answer}", language="zh")
            state["target_mode"] = "fake-node"
            state["workflow_mode"] = "rpc_benchmark"
            state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
            state["active_group"] = group
            state["pending_question"] = manual_question(
                group,
                field,
                f"Confirm {field}.",
                field=field,
                validation={"value_type": "positive_number"},
            )
            state["last_user_input"] = answer
            return process_turn(state)

        for field in ("DATA_VOL_SIZE", "DATA_VOL_MAX_IOPS", "DATA_VOL_MAX_THROUGHPUT"):
            for bad in ("-50", "0", "abc"):
                result = _answer("ledger_disk", field, bad)
                self.assertNotIn(field, result.get("confirmed_config") or {})
                self.assertIn("无效", "\n".join(result.get("visible_response") or []))
            good = _answer("ledger_disk", field, "20")
            self.assertEqual(good.get("confirmed_config", {}).get(field), "20")

        for field in ("ACCOUNTS_VOL_SIZE", "ACCOUNTS_VOL_MAX_IOPS", "ACCOUNTS_VOL_MAX_THROUGHPUT"):
            bad = _answer("accounts_disk", field, "-20")
            self.assertNotIn(field, bad.get("confirmed_config") or {})

        bad_net = _answer("network", "NETWORK_MAX_BANDWIDTH_GBPS", "-5")
        self.assertNotIn("NETWORK_MAX_BANDWIDTH_GBPS", bad_net.get("confirmed_config") or {})
        good_net = _answer("network", "NETWORK_MAX_BANDWIDTH_GBPS", "10")
        self.assertEqual(good_net.get("confirmed_config", {}).get("NETWORK_MAX_BANDWIDTH_GBPS"), "10")

    def test_paste_inferred_disk_values_preserve_sign_and_reject_negative(self) -> None:
        """The paste-inferred-config path must not silently launder a negative

        disk/network value into a positive one. Regression: `_first_number_text`
        (used by `_normalize_proposed_config_value` for these fields) extracts
        digits only, so a pasted "-50" for DATA_VOL_MAX_IOPS became the applied
        value "50" instead of being rejected — worse than doing nothing, since it
        changes the value without telling the user.
        """

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread-paste-disk-neg", language="zh")
        state["inferred_config"] = {"pending_review": {
            "config_values": {"DATA_VOL_MAX_IOPS": "-50", "NETWORK_MAX_BANDWIDTH_GBPS": "20"},
            "unmapped_values": {},
            "source_format": "mixed",
            "reason": "",
        }}
        state["pending_question"] = {
            "id": "inferred_config_review",
            "group": "ledger_disk",
            "kind": "yes_no",
            "field": "inferred_config_review",
            "options": [{"label": "Y", "value": True}, {"label": "N", "value": False}],
            "accepted_action_types": ["answer_pending", "propose_config_values"],
        }
        state["last_user_input"] = "Y"
        result = process_turn(state)
        self.assertNotIn("DATA_VOL_MAX_IOPS", result.get("confirmed_config") or {})
        self.assertNotEqual(result.get("confirmed_config", {}).get("DATA_VOL_MAX_IOPS"), "50")
        self.assertEqual(result.get("confirmed_config", {}).get("NETWORK_MAX_BANDWIDTH_GBPS"), "20")
        self.assertIn("无效", "\n".join(result.get("visible_response") or []))

    def test_advanced_tuning_adjust_value_rejects_invalid(self) -> None:
        """advanced_tuning threshold/interval overrides must be validated:

        positive numbers, and percentage fields capped at 100. Regression for a
        live chaos finding: MAX_LATENCY_THRESHOLD=-500 and
        BOTTLENECK_CPU_THRESHOLD=150 (a percentage field) were both accepted with
        zero validation, reaching the confirmed profile uncaught.
        """

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.performance import valid_advanced_value
        from agent.harness.questions import manual_question
        from agent.harness.state import new_state

        self.assertFalse(valid_advanced_value("MAX_LATENCY_THRESHOLD", "-500"))
        self.assertFalse(valid_advanced_value("BOTTLENECK_CPU_THRESHOLD", "150"))
        self.assertTrue(valid_advanced_value("BOTTLENECK_CPU_THRESHOLD", "85"))
        self.assertTrue(valid_advanced_value("MAX_LATENCY_THRESHOLD", "500"))

        def _adjust(adjust_field: str, answer: str) -> dict:
            state = new_state(f"unit-thread-tuning-{adjust_field}-{answer}", language="zh")
            state["target_mode"] = "fake-node"
            state["workflow_mode"] = "rpc_benchmark"
            state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
            state["active_group"] = "advanced_tuning"
            state["advanced_tuning"] = {"default_decision_made": True, "confirmed": False, "adjust_field": adjust_field}
            state["pending_question"] = manual_question(
                "advanced_tuning",
                "advanced_tuning_adjust_value",
                f"Enter the value for {adjust_field}.",
                field="advanced_tuning_adjust_value",
                validation={"value_type": "positive_number"},
            )
            state["last_user_input"] = answer
            return process_turn(state)

        bad = _adjust("MAX_LATENCY_THRESHOLD", "-500")
        self.assertNotIn("MAX_LATENCY_THRESHOLD", (bad.get("advanced_tuning") or {}).get("overrides", {}))
        self.assertEqual(bad.get("advanced_tuning", {}).get("adjust_field"), "MAX_LATENCY_THRESHOLD")  # re-asked
        self.assertIn("无效", "\n".join(bad.get("visible_response") or []))

        bad_pct = _adjust("BOTTLENECK_CPU_THRESHOLD", "150")
        self.assertNotIn("BOTTLENECK_CPU_THRESHOLD", (bad_pct.get("advanced_tuning") or {}).get("overrides", {}))

        good = _adjust("MAX_LATENCY_THRESHOLD", "800")
        self.assertEqual(good.get("advanced_tuning", {}).get("overrides", {}).get("MAX_LATENCY_THRESHOLD"), "800")
        self.assertIsNone(good.get("advanced_tuning", {}).get("adjust_field"))

    def test_local_observability_mode_discloses_ports(self) -> None:
        """Choosing observability `local` mode must disclose the ports it starts.

        Regression for a live chaos finding: picking "local" (starts exporter,
        Prometheus, and Grafana on this host) confirmed the mode and moved
        straight to advanced-tuning defaults without ever telling the user which
        ports would be bound — same class of gap as exporter mode's missing
        scrape-target disclosure, but for local mode's own services (relevant for
        port-conflict awareness on the host).
        """

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.questions import manual_question
        from agent.harness.state import new_state

        state = new_state("unit-thread-local-obs", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
        state["last_user_input"] = "second one, local Prometheus/Grafana"
        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{"type": "set_observability", "observability_mode": "local", "mutation_explicit": True, "source_evidence": "local Prometheus/Grafana", "confidence": "high"}]},
        ):
            result = process_turn(state)
        self.assertEqual((result.get("observability") or {}).get("mode"), "local")
        response = "\n".join(result.get("visible_response") or [])
        self.assertIn("9091", response)
        self.assertIn("3001", response)

        pending_state = new_state("unit-thread-local-obs-2", language="zh")
        pending_state["target_mode"] = "fake-node"
        pending_state["workflow_mode"] = "rpc_benchmark"
        pending_state["pending_question"] = manual_question(
            "observability", "observability_mode", "Choose observability mode.", field="observability_mode", kind="numbered_choice"
        )
        pending_state["pending_question"]["options"] = [
            {"label": "Disabled", "value": "disabled"},
            {"label": "Local Prometheus/Grafana", "value": "local"},
            {"label": "Exporter only", "value": "exporter"},
        ]
        pending_state["last_user_input"] = "2"
        pending_result = process_turn(pending_state)
        self.assertEqual((pending_result.get("observability") or {}).get("mode"), "local")
        self.assertIn("9091", "\n".join(pending_result.get("visible_response") or []))

    def test_accept_recommendation_starts_recommended_setup(self) -> None:
        """Accepting a recommendation must start it, not re-print it.

        Regression for a real transcript: after "recommend the simplest test",
        the user's "好，那就按你推荐的来" re-printed the same recommendation instead
        of starting solana fake-node. A recommendation now sets an actionable
        `accept_recommendation` yes/no; "y" applies the recommended chain+mode.
        """

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        def _recommendation_state() -> dict:
            state = new_state("unit-thread", language="zh")
            state["active_group"] = "opening"
            state["last_user_input"] = "推荐一个最简单的闭环测试"
            return state

        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{"type": "answer_opening_question", "topic": "recommendation", "confidence": "high"}]},
        ):
            offered = process_turn(_recommendation_state())
        self.assertEqual(offered["pending_question"]["id"], "accept_recommendation")

        offered["last_user_input"] = "y"
        accepted = process_turn(offered)
        self.assertEqual(accepted["target_mode"], "fake-node")
        self.assertEqual(accepted["pending_question"]["id"], "chain")
        self.assertFalse((accepted.get("chain_identity") or {}).get("canonical"))

        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{"type": "answer_opening_question", "topic": "recommendation", "confidence": "high"}]},
        ):
            offered = process_turn(_recommendation_state())
        offered["last_user_input"] = "2"
        declined = process_turn(offered)
        self.assertNotEqual(declined.get("target_mode"), "fake-node")

    def test_custom_rpc_probe_endpoint_is_not_the_benchmark_endpoint(self) -> None:
        """A custom-RPC method's probe/pasted endpoint is evidence only. On

        real-node, the final benchmark LOCAL_RPC_URL must still be asked
        separately — no matter how many custom methods were added, and even
        though a candidate probe endpoint exists. (Users often paste method
        content whose URL is not the endpoint they want to test.)
        """

        from agent.harness.coordinator import _question_for_group
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "real-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
        state["rpc_mode"] = "mixed"
        state["workload"] = {"confirmed": True}
        state["custom_rpc"] = {"status": "validated", "validated_methods": [{"method": "eth_getBalance"}]}
        # A custom-RPC probe endpoint exists as evidence, but the final endpoint
        # is NOT confirmed.
        state["endpoint_evidence"] = {
            "candidate_endpoint": "https://bsc-dataseed.binance.org/",
            "candidate_endpoint_ready": True,
        }
        question = _question_for_group(state, "endpoint_process")
        self.assertIsNotNone(question)
        self.assertEqual(question["id"], "LOCAL_RPC_URL")

    def test_custom_rpc_schema_failure_reprompts_for_schema_evidence(self) -> None:
        """B.22 (known-issues.md): a custom-RPC schema validation failure must

        leave a real re-ask question active, not just a bare error message.
        `_validate_rpc_schema` used to write `custom_rpc["params"]` *before*
        knowing whether validation would succeed, and never cleared it on
        failure. The `custom_rpc_schema_evidence` question only renders while
        `"params" not in custom_rpc`, so a single failed attempt permanently
        defeated that gate for the rest of the session -- recoverable only by
        chance (free-text routing), never via a formal pending question.
        """

        from unittest.mock import patch

        from agent.harness.coordinator import _question_for_group
        from agent.harness.domains.chain_rpc import apply_chain_rpc_answer
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "real-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["custom_rpc"] = {
            "status": "schema_needs_confirmation",
            "endpoint": "https://bsc-rpc.publicnode.com",
            "endpoint_ready": True,
            "method": "eth_getBalance",
            "schema_draft": {
                "status": "draft",
                "method": "eth_getBalance",
                "params": ["not-a-valid-address"],
                "params_json": ["not-a-valid-address"],
                "transport": "jsonrpc",
            },
        }
        question = {
            "id": "custom_rpc_schema_confirm",
            "group": "endpoint_process",
            "field": "custom_rpc_schema_confirm",
        }

        with patch("agent.harness.domains.rpc_endpoint.validate_rpc_endpoint", return_value={"ready": False, "error": "bad params", "evidence_file": ""}):
            outcome = apply_chain_rpc_answer(state, question, True, "Y")
            state = _commit_result(state, outcome, owner="chain_rpc")

        self.assertNotIn("params", state["custom_rpc"])
        question = _question_for_group(state, "endpoint_process")
        self.assertIsNotNone(question)
        self.assertEqual(question["id"], "custom_rpc_schema_evidence")

    def test_chain_workload_question_answered_not_selected(self) -> None:
        """"bsc 有哪些 rpc workload" is a question about a chain's default methods,

        not a chain selection. It must be answered with that chain's workload and
        must NOT select the chain — even when the resolver misfires to
        choose_chain (which it reliably does live). Regression for a real
        transcript where such questions either selected the chain or dumped the
        generic chain list.
        """

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        # Conforming contract: the resolver types the question as
        # answer_opening_question topic=supported_chains with the chain in
        # `subject`. The harness must answer it from that chain's template and
        # must NOT select the chain. (Resolver prompt rule lives in intent.py; it
        # is exercised live by dual-AI chaos, not mocked here.)
        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = "bsc 有哪些 rpc workload"
        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{
                "type": "answer_opening_question",
                "topic": "supported_chains",
                "subject": "bsc",
                "confidence": "high",
            }]},
        ):
            result = process_turn(state)
        response = "\n".join(result.get("visible_response") or [])
        self.assertIn("eth_getBalance", response)  # bsc default single method
        self.assertFalse((result.get("chain_identity") or {}).get("canonical"))  # chain NOT selected

        # A chain-specific custom-RPC how-to types as topic=extension + subject.
        howto_state = new_state("unit-thread-2", language="zh")
        howto_state["last_user_input"] = "bsc 的自定义 rpc method 如何添加"
        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{
                "type": "answer_opening_question",
                "topic": "extension",
                "subject": "bsc",
                "confidence": "high",
            }]},
        ):
            howto = process_turn(howto_state)
        howto_text = "\n".join(howto.get("visible_response") or [])
        self.assertIn("endpoint", howto_text.lower())
        self.assertFalse((howto.get("chain_identity") or {}).get("canonical"))  # chain NOT selected

    def test_unknown_chain_consultation_is_specific_and_preserves_pending_question(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "real-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "provider_deployment"
        state["pending_question"] = {
            "id": "CLOUD_REGION",
            "group": "provider_deployment",
            "kind": "manual_value",
            "field": "CLOUD_REGION",
            "prompt": "请输入 CLOUD_REGION。",
            "manual_input_allowed": True,
            "options": [],
            "accepted_action_types": ["answer_pending"],
            "validation": {"value_type": "scalar_token"},
        }
        state["last_user_input"] = "Mina 使用 GraphQL 吗？这个协议你支持吗？"

        actions = {
            "actions": [
                {"type": "answer_opening_question", "topic": "capabilities", "confidence": "high"},
                {
                    "type": "answer_opening_question",
                    "topic": "supported_chains",
                    "subject": "Mina",
                    "source_evidence": "Mina 使用 GraphQL 吗？这个协议你支持吗？",
                    "confidence": "high",
                },
                {"type": "answer_opening_question", "topic": "supported_chains", "subject": "Mina GraphQL", "confidence": "high"},
            ]
        }
        resolution = {
            "chain_exists": True,
            "canonical_chain_name": "mina",
            "adapter_family": "unsupported",
            "protocol_or_api": "GraphQL",
            "evidence_summary": "Mina exposes a GraphQL API.",
            "confidence": "high",
        }
        with (
            patch("agent.harness.coordinator.resolve_action_queue", return_value=actions),
            patch("agent.harness.domains.orientation.research_chain_identity", return_value=resolution) as research,
        ):
            result = process_turn(state)

        response = "\n".join(result.get("visible_response") or [])
        self.assertIn("`mina`", response)
        self.assertIn("GraphQL", response)
        self.assertIn("未进行互联网核实", response)
        self.assertNotIn("36 chains", response)
        self.assertIn("CLOUD_REGION", response)
        self.assertEqual(result["pending_question"]["id"], "CLOUD_REGION")
        self.assertEqual(result["target_mode"], "real-node")
        self.assertFalse((result.get("chain_identity") or {}).get("canonical"))
        self.assertEqual(research.call_count, 1)
        self.assertEqual(research.call_args.args[1], "Mina")

    def test_opening_consultation_cannot_launder_model_selected_menu_value(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.orientation import opening_question
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["active_group"] = "opening"
        state["pending_question"] = opening_question(state)
        state["last_user_input"] = "你好"
        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={
                "actions": [
                    {
                        "type": "answer_pending",
                        "answer": "info",
                        "selected_value": "info",
                        "source_evidence": "你好",
                        "semantic_purpose_verified": True,
                        "confidence": "high",
                    },
                    {
                        "type": "answer_opening_question",
                        "topic": "identity",
                        "source_evidence": "你好",
                        "semantic_purpose_verified": True,
                        "confidence": "high",
                    },
                ]
            },
        ):
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "opening_next_action")
        self.assertIn("AnyChain Benchmark Agent", "\n".join(result.get("visible_response") or []))
        self.assertFalse(
            any(item.get("type") == "answer_pending" for item in result.get("completed_actions") or [])
        )

    def test_pending_answer_removes_equivalent_duplicate_setter(self) -> None:
        from agent.harness.coordinator import _validate_action_plan
        from agent.harness.domains.chain_rpc import question_for_chain_rpc
        from agent.harness.state import new_state

        state = new_state("dedupe-pending", language="en")
        state.update({
            "target_mode": "fake-node",
            "chain_identity": {"canonical": "bsc", "status": "confirmed"},
            "active_group": "workload_rpc",
        })
        state["pending_question"] = question_for_chain_rpc(state, "workload_rpc") or {}
        admitted = _admitted_mock_plan(state, "single quick", {"actions": [
            {"type": "answer_pending", "selected_value": "single", "source_evidence": "single"},
        ]})
        state.setdefault("turn_context", {})["pending_choice_contracts"] = admitted[
            "pending_choice_contracts"
        ]
        actions = admitted["actions"] + [
            {"type": "set_rpc_mode", "rpc_mode": "single", "source_evidence": "single"},
            {"type": "set_qps_mode", "qps_mode": "quick", "source_evidence": "quick"},
        ]

        validated = _validate_action_plan(state, actions)
        self.assertEqual([item["type"] for item in validated], ["answer_pending", "set_qps_mode"])

    def test_multiline_analysis_reaches_typed_planner_before_evidence_domain(self) -> None:
        """Multiline is transport shape; the typed planner owns its semantics."""

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        report = (
            "这是我压测 solana 的结果，帮我分析为什么成功率低：\n"
            "Success ratio: 62.30%\n"
            "37.7% => 429 Too Many Requests\n"
            "CPU: node process avg 780%"
        )
        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = report
        with (
            patch(
                "agent.harness.coordinator.resolve_action_queue",
                return_value={"actions": [{
                    "type": "analyze_evidence",
                    "evidence": report,
                    "confidence": "high",
                }]},
            ) as resolver,
            patch("agent.harness.domains.analysis.analyze_evidence_with_model", return_value="analysis complete"),
        ):
            result = process_turn(state)
        self.assertEqual(resolver.call_args.args[1], report)
        self.assertFalse((result or {}).get("evidence_collection"))
        self.assertTrue((result or {}).get("evidence_buffer"))

    def test_failure_language_without_log_structure_reaches_intent_resolver(self) -> None:
        """Ordinary questions about a failed job are not log paste evidence."""

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        text = (
            "I'm new here. That failed job is not mine. What can you do, and "
            "can I start a clean BSC fake-node check without losing the detected disk information?"
        )
        state = new_state("unit-thread", language="en")
        state["last_user_input"] = text
        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [], "reason": "consultation"},
        ) as resolver:
            result = process_turn(state)

        self.assertEqual(resolver.call_args.args[1], text)
        self.assertFalse((result or {}).get("evidence_collection"))

    def test_structural_single_line_error_reaches_typed_planner(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        text = "ERROR: endpoint probe failed with HTTP 404"
        state["last_user_input"] = text
        with (
            patch(
                "agent.harness.coordinator.resolve_action_queue",
                return_value={"actions": [{
                    "type": "analyze_evidence",
                    "evidence": text,
                    "confidence": "high",
                }]},
            ) as resolver,
            patch("agent.harness.domains.analysis.analyze_evidence_with_model", return_value="analysis complete"),
        ):
            result = process_turn(state)

        self.assertEqual(resolver.call_args.args[1], text)
        self.assertFalse((result or {}).get("evidence_collection"))
        self.assertTrue((result or {}).get("evidence_buffer"))

    def test_multiline_configuration_is_not_preclassified_as_error_evidence(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        text = '{\n  "chain": "bsc",\n  "target_mode": "fake-node",\n  "qps_mode": "quick"\n}\nDo not execute.'
        state = new_state("unit-thread", language="en")
        state["last_user_input"] = text
        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [
                {
                    "type": "choose_target_mode",
                    "target_mode": "fake-node",
                    "target_mode_explicit": True,
                    "source_evidence": '"target_mode": "fake-node"',
                    "confidence": "high",
                },
                {
                    "type": "choose_chain",
                    "chain_text": "bsc",
                    "source_evidence": '"chain": "bsc"',
                    "confidence": "high",
                },
                {
                    "type": "set_qps_mode",
                    "qps_mode": "quick",
                    "mutation_explicit": True,
                    "source_evidence": '"qps_mode": "quick"',
                    "confidence": "high",
                },
            ]},
        ) as resolver:
            result = process_turn(state)

        self.assertEqual(resolver.call_args.args[1], text)
        self.assertFalse((result or {}).get("evidence_collection"))
        self.assertEqual((result or {}).get("target_mode"), "fake-node")
        self.assertEqual(((result or {}).get("chain_identity") or {}).get("canonical"), "bsc")

    def test_noop_same_chain_change_preserves_active_question(self) -> None:
        """A no-op "change" to the already-selected chain must not drop an active

        pending question. Regression for a live chaos failure: meaningless input
        ("🚀🚀🚀") that the resolver misread as choose_chain(bsc) wiped the active
        benchmark_mode question and derailed the flow.
        """

        from agent.harness.contracts import ActionProposal
        from agent.harness.domains.chain_rpc import apply_chain_rpc_action
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
        state["active_group"] = "qps_profile"
        state["pending_question"] = {
            "id": "benchmark_mode",
            "group": "qps_profile",
            "kind": "numbered_choice",
            "options": [{"label": "quick", "value": "quick"}, {"label": "standard", "value": "standard"}],
        }

        outcome = apply_chain_rpc_action(
            state,
            ActionProposal("same-chain", "change_chain", {"chain_text": "bsc"}, "high"),
        )
        result = _commit_result(state, outcome, owner="chain_rpc")
        self.assertEqual(result["pending_question"]["id"], "benchmark_mode")

    def test_preflight_group_not_offered_when_config_incomplete(self) -> None:
        """A jump to the preflight/execution group must not offer the run

        confirmation ("配置已收集，是否运行?") when the config is not actually
        ready. Regression for a live chaos failure: "run preflight/smoke now" on a
        real-node with no endpoint/workload/QPS reached preflight_smoke_confirm
        and falsely claimed config was collected.
        """

        from agent.harness.coordinator import _question_for_group
        from agent.harness.state import new_state

        # Real-node with only the chain confirmed — nowhere near ready.
        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "real-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
        self.assertIsNone(_question_for_group(state, "preflight_smoke_execution"))

    def test_sync_observe_group_completes_instead_of_looping(self) -> None:
        """Once sync-observe has a source, validated endpoint, and stop condition

        (plus duration when applicable), its group must return no further
        question so the harness advances to observability/preflight. Regression
        for a live chaos failure: `_question_for_group` re-asked the stop
        condition unconditionally, so sync-observe looped forever and could never
        reach execution.
        """

        from agent.harness.coordinator import _question_for_group
        from agent.harness.state import new_state

        def _sync_state(sync: dict) -> dict:
            state = new_state("unit-thread", language="en")
            state["workflow_mode"] = "sync_observe"
            state["endpoint_evidence"] = {"sync_rpc_url_ready": True}
            state["confirmed_config"] = {
                "SYNC_OBSERVE_RPC_URL": "http://node:8545",
                "MAINNET_RPC_URL_REVIEWED": True,
            }
            state["sync_observe"] = sync
            return state

        # Fully configured -> no more questions (group complete).
        self.assertIsNone(
            _question_for_group(_sync_state({"source": "endpoint_only", "stop_condition": "until_synced"}), "sync_observe")
        )
        self.assertIsNone(
            _question_for_group(
                _sync_state({"source": "endpoint_only", "stop_condition": "duration", "duration_seconds": "300"}),
                "sync_observe",
            )
        )
        # Still-missing pieces are asked exactly once.
        self.assertEqual(
            _question_for_group(_sync_state({"source": "endpoint_only"}), "sync_observe")["id"],
            "sync_observe_stop_condition",
        )
        self.assertEqual(
            _question_for_group(_sync_state({"source": "endpoint_only", "stop_condition": "duration"}), "sync_observe")["id"],
            "sync_observe_duration_seconds",
        )

    def test_endpoint_probe_resolves_env_placeholder_sample_address(self) -> None:
        """The endpoint probe must expand `${VAR:-default}` template placeholders

        before using a sample address. Regression for a live chaos failure: the
        raw `${TARGET_ADDRESS:-0x...}` string was sent as the eth_getBalance
        argument, so a healthy public endpoint was wrongly failed
        ("hex string without 0x prefix").
        """

        from unittest.mock import patch

        from agent.validators.endpoint_probe import _resolve_env_placeholder, _sample_address

        # No env var set -> use the default from the placeholder.
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("ZZ_TEST_ADDR", None)
            self.assertEqual(_resolve_env_placeholder("${ZZ_TEST_ADDR:-0xABC}"), "0xABC")
        # Env var set -> it wins.
        with patch.dict("os.environ", {"ZZ_TEST_ADDR": "0xFEED"}, clear=False):
            self.assertEqual(_resolve_env_placeholder("${ZZ_TEST_ADDR:-0xABC}"), "0xFEED")
        # A plain value is returned unchanged.
        self.assertEqual(_resolve_env_placeholder("0xdeadbeef"), "0xdeadbeef")
        # A real EVM template resolves to a usable 0x address, never a placeholder.
        addr = _sample_address("bsc")
        self.assertTrue(addr.startswith("0x"))
        self.assertNotIn("${", addr)

    def test_custom_rpc_method_extracts_from_pasted_content_with_url(self) -> None:
        """Pasting copied RPC content (a curl command / JSON-RPC body that contains

        a URL) at the custom-method step must extract the method+params from the
        JSON body, NOT be rejected for containing an endpoint URL. Regression for a
        live chaos failure: a pasted `curl https://... --data '{"method":...}'` was
        rejected as "looks like an endpoint".
        """

        from unittest.mock import patch

        from agent.harness.domains.chain_rpc import apply_chain_rpc_answer
        from agent.harness.state import new_state

        def _probe_ok(*args, **kwargs):
            return {"ready": True, "evidence_file": "x.json", "safe_method": "eth_getBlockByNumber", "checks": []}

        state = new_state("t", language="zh")
        state["chain_identity"] = {"canonical": "bsc", "status": "confirmed", "case": "known"}
        state["custom_rpc"] = {"status": "needs_method", "endpoint": "https://bsc-rpc.publicnode.com", "endpoint_ready": True}
        q = {"id": "custom_rpc_method", "group": "endpoint_process", "kind": "manual_value", "field": "custom_rpc_method", "manual_input_allowed": True}
        blob = 'curl https://some-other-node.example.com -X POST --data \'{"jsonrpc":"2.0","method":"eth_getBlockByNumber","params":["latest",false],"id":1}\''
        with patch("agent.harness.domains.rpc_endpoint.validate_rpc_endpoint", side_effect=_probe_ok):
            outcome = apply_chain_rpc_answer(state, q, blob, blob)
            result = _commit_result(state, outcome, owner="chain_rpc")
        self.assertEqual(_catalog_draft(result).get("method"), "eth_getBlockByNumber")
        response = "\n".join(result.get("visible_response") or [])
        self.assertNotIn("这看起来像 endpoint", response)  # not the rejection message

        # A bare URL with no JSON-RPC body is still rejected as not-a-method.
        state2 = new_state("t2", language="zh")
        state2["chain_identity"] = {"canonical": "bsc", "status": "confirmed", "case": "known"}
        state2["custom_rpc"] = {"status": "needs_method", "endpoint": "https://bsc-rpc.publicnode.com", "endpoint_ready": True}
        rejected_outcome = apply_chain_rpc_answer(
            state2,
            q,
            "https://docs.example.com/api/eth_getLogs",
            "https://docs.example.com/api/eth_getLogs",
        )
        rejected = _commit_result(state2, rejected_outcome, owner="chain_rpc")
        self.assertIn("严格 method grammar", "\n".join(rejected.get("visible_response") or []))

    def test_custom_rpc_schema_extraction_grounds_with_google_search_unconditionally(self) -> None:
        """B.1 (known-issues.md), second call site: `extract_rpc_schema_from_evidence`'s

        draft used to include a `needs_google_search` self-judgment signal
        (mirroring the chain-identity resolver's field of the same name), but
        the confirmed framework design is that adding a custom RPC method
        always re-verifies with google_search once it is available --
        unconditionally, not gated by the underlying LLM's own confidence
        (even a fully-confident draft still gets the search). The field was
        removed from the resolver schema/prompt entirely rather than kept as
        an unused decision point. `_augment_schema_draft_with_search` grounds
        every draft with a real `run_google_search_grounding()` call whenever
        `google_search_available` is true, and surfaces the result in the
        schema confirmation prompt shown to the user.
        """

        from unittest.mock import patch

        from agent.harness.domains.chain_rpc import apply_chain_rpc_answer
        from agent.harness.state import new_state
        from agent.llm.search_grounding import SearchGroundingResult

        state = new_state("t", language="en")
        state["chain_identity"] = {"canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["web_research"] = {"google_search_available": True}
        state["custom_rpc"] = {"status": "needs_schema_evidence", "endpoint": "https://bsc-rpc.publicnode.com", "endpoint_ready": True, "method": "eth_call"}
        q = {"id": "custom_rpc_schema_evidence", "group": "endpoint_process", "kind": "evidence", "field": "custom_rpc_schema_evidence", "manual_input_allowed": True}

        # High confidence, no `needs_google_search` field at all -- search
        # must still run, proving it is not gated by that removed field.
        draft = {
            "status": "draft",
            "evidence_kind": "docs_excerpt",
            "transport": "jsonrpc",
            "method": "eth_call",
            "params": [{"index": 0, "name": "callObject", "type": "object", "meaning": "call parameters", "example": {}, "required": True}],
            "params_json": [{}],
            "response_summary": "returns call result",
            "confidence": "high",
        }
        with (
            patch("agent.harness.domains.rpc_endpoint.extract_rpc_schema_from_evidence", return_value=dict(draft)),
            patch(
                "agent.harness.domains.rpc_endpoint.run_google_search_grounding",
                return_value=SearchGroundingResult(available=True, query="q", text_summary="eth_call takes a call object and a block tag per the official JSON-RPC spec."),
            ) as grounding,
        ):
            outcome = apply_chain_rpc_answer(
                state,
                q,
                "the docs mention a call object but I'm not sure of the exact shape",
                "the docs mention a call object but I'm not sure of the exact shape",
            )
            result = _commit_result(state, outcome, owner="chain_rpc")

        grounding.assert_called_once()
        prompt = "\n".join(result.get("visible_response") or [])
        self.assertIn("eth_call takes a call object", prompt)

        # Search unavailable -> never called, LLM draft used as-is.
        state2 = new_state("t2", language="en")
        state2["chain_identity"] = {"canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state2["web_research"] = {"google_search_available": False}
        state2["custom_rpc"] = {"status": "needs_schema_evidence", "endpoint": "https://bsc-rpc.publicnode.com", "endpoint_ready": True, "method": "eth_call"}
        with (
            patch("agent.harness.domains.rpc_endpoint.extract_rpc_schema_from_evidence", return_value=dict(draft)),
            patch("agent.harness.domains.rpc_endpoint.run_google_search_grounding") as grounding2,
        ):
            apply_chain_rpc_answer(
                state2,
                q,
                "the docs mention a call object but I'm not sure of the exact shape",
                "the docs mention a call object but I'm not sure of the exact shape",
            )
        grounding2.assert_not_called()

    def test_custom_rpc_method_accepts_get_path_for_rest_shaped_chain_family(self) -> None:
        """`GET /path`-shaped custom RPC methods must be accepted, not rejected,

        for a chain whose own adapter family is REST-shaped (rest/tendermint/
        hedera_dual) -- every default method on a `cosmos-hub`/`algorand`/
        `hedera` chain already looks exactly like this, so it is the *correct*
        format for that family, not a mistake. Found live testing Case 1
        (custom RPC method on an already-supported chain) on `cosmos-hub`:
        `custom_rpc_method` unconditionally rejected any `GET /...` answer via
        `_looks_like_rest_path_or_doc_method`, with no adapter-family
        awareness, even though `cosmos-hub`'s own template methods are all
        `GET /cosmos/...` paths. The same unconditional check also existed on
        the Case 2 (`new_chain_method`) path.
        """

        from agent.harness.domains.chain_rpc import apply_chain_rpc_answer
        from agent.harness.state import new_state

        q = {"id": "custom_rpc_method", "group": "endpoint_process", "kind": "manual_value", "field": "custom_rpc_method", "manual_input_allowed": True}

        # tendermint (cosmos-hub): a GET path is the correct format -- accepted.
        state = new_state("t", language="zh")
        state["chain_identity"] = {"canonical": "cosmos-hub", "adapter_family": "tendermint", "status": "confirmed", "case": "known"}
        state["custom_rpc"] = {"status": "needs_method", "endpoint": "https://cosmos-rest.publicnode.com", "endpoint_ready": True}
        outcome = apply_chain_rpc_answer(
            state,
            q,
            "GET /cosmos/staking/v1beta1/pool",
            "GET /cosmos/staking/v1beta1/pool",
        )
        result = _commit_result(state, outcome, owner="chain_rpc")
        self.assertEqual(_catalog_draft(result).get("method"), "GET /cosmos/staking/v1beta1/pool")
        self.assertEqual((result.get("custom_rpc") or {}).get("status"), "needs_schema_evidence")
        response = "\n".join(result.get("visible_response") or [])
        self.assertNotIn("这看起来像 endpoint", response)

        # jsonrpc (bsc): the same GET-path answer is still a real mistake -- rejected.
        state2 = new_state("t2", language="zh")
        state2["chain_identity"] = {"canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state2["custom_rpc"] = {"status": "needs_method", "endpoint": "https://bsc-rpc.publicnode.com", "endpoint_ready": True}
        rejected_outcome = apply_chain_rpc_answer(
            state2,
            q,
            "GET /cosmos/staking/v1beta1/pool",
            "GET /cosmos/staking/v1beta1/pool",
        )
        rejected = _commit_result(state2, rejected_outcome, owner="chain_rpc")
        rejected_custom = rejected.get("custom_rpc") or {}
        self.assertEqual(rejected_custom.get("status"), "needs_method")
        self.assertFalse(rejected_custom.get("method"))

    def test_analyze_latest_job_uses_disk_latest_not_stale_hint(self) -> None:
        """"Analyze the latest job" must analyze the most recent job on disk, not a

        stale `latest_job_id` startup hint. Regression for a live chaos failure:
        after submitting a new smoke job, "分析最近 job" kept analyzing the older
        running job detected at startup (the injected hint was never refreshed).
        """

        from unittest.mock import patch

        from agent.harness.domains.analysis import report_artifact_entry_response as _report_artifact_entry_response
        from agent.harness.state import new_state

        state = new_state("t", language="zh")
        state["job"] = {"job_id": "job_STALE_running", "status": "running"}
        newest = [{"job_id": "job_NEWEST_failed", "status": "failed"}]
        summary = {"status": "failed", "run_dir": "x", "artifact_index": "i.json", "runtime_env_file": "r.env", "next_actions": ["status", "analyze"]}
        with patch("agent.harness.domains.analysis.list_jobs", return_value=newest), patch("agent.harness.domains.analysis.resume_job", return_value=summary):
            out = _report_artifact_entry_response(state)
        self.assertIn("job_NEWEST_failed", out)

        # Falls back to the current workflow-owned job receipt when disk history is unavailable.
        with patch("agent.harness.domains.analysis.list_jobs", return_value=[]), patch("agent.harness.domains.analysis.resume_job", return_value=summary) as rj:
            _report_artifact_entry_response(state)
            rj.assert_called_with("job_STALE_running")

    def test_analyze_report_includes_real_log_excerpt_not_just_paths(self) -> None:
        """B.21 (known-issues.md): `_report_artifact_entry_response` used to

        only list artifact *paths* (run_dir, artifact_index, runtime.env),
        even when the same turn explicitly asked to interpret them ("跟我解释
        一下这个报告，帮我看看瓶颈在哪") -- it never read `benchmark.log` or the
        artifact index inline, only suggesting a follow-up ask. Fixed by
        reading the job's real log via `tail_job_log` and surfacing the most
        relevant lines (error/failure markers for a failed job) directly in
        the response.
        """

        from unittest.mock import patch

        from agent.harness.domains.analysis import report_artifact_entry_response as _report_artifact_entry_response
        from agent.harness.state import new_state

        state = new_state("t", language="zh")
        jobs = [{"job_id": "job_REAL_failed", "status": "failed"}]
        summary = {"status": "failed", "run_dir": "x", "artifact_index": "i.json", "runtime_env_file": "r.env", "next_actions": ["logs", "artifact-qa"]}
        log = {
            "job_id": "job_REAL_failed",
            "log_file": "x/benchmark.log",
            "exists": True,
            "lines": [
                "Starting execution: blockchain_node_benchmark.sh",
                "Preparing configuration...",
                "❌ --fake-node: Go toolchain is required to build fake-node, but go was not found",
                "Executing framework cleanup...",
            ],
        }
        with (
            patch("agent.harness.domains.analysis.list_jobs", return_value=jobs),
            patch("agent.harness.domains.analysis.resume_job", return_value=summary),
            patch("agent.harness.domains.analysis.tail_job_log", return_value=log),
        ):
            out = _report_artifact_entry_response(state)

        self.assertIn("Go toolchain is required", out)
        self.assertNotIn("Preparing configuration", out)  # only the relevant line, not the whole tail

    def test_prepare_kwargs_are_all_accepted_by_prepare_benchmark_run(self) -> None:
        """Every key `_prepare_kwargs` produces must be a parameter of

        `prepare_benchmark_run`. Regression for a live crash: a new kwarg
        (`sync_observe_local_attribution`) was added to the harness side but not to
        the pipeline signature, so preflight raised TypeError — masked by the
        generic "model call failed" message.
        """

        import inspect

        from agent.harness.domains.execution_runtime import _prepare_kwargs
        from agent.harness.state import new_state
        from agent.runners.benchmark_pipeline import prepare_benchmark_run

        accepted = set(inspect.signature(prepare_benchmark_run).parameters)
        for mode, wf in (("fake-node", "rpc_benchmark"), ("real-node", "rpc_benchmark"), ("sync-observe", "sync_observe")):
            state = new_state("t", language="zh")
            state["target_mode"] = mode
            state["workflow_mode"] = wf
            state["rpc_mode"] = "mixed"
            state["chain_identity"] = {"canonical": "ethereum", "status": "confirmed", "case": "known"}
            state["workload"] = {"confirmed": True, "choice": "default"}
            state["sync_observe"] = {"source": "endpoint_only", "local_attribution_available": False, "stop_condition": "duration", "duration_seconds": 600}
            unknown = set(_prepare_kwargs(state)) - accepted
            self.assertEqual(unknown, set(), f"{mode}: kwargs not accepted by prepare_benchmark_run: {unknown}")

    def test_sync_observe_endpoint_only_waives_node_process_identity(self) -> None:
        """Endpoint-only sync-observe (a remote node, no local process) must not

        require `node_process_identity` (local CPU/thread attribution). Regression
        for a live chaos deadlock: the endpoint-only flow never asks for it, but
        the checklist blanket-required it, so preflight was offered then blocked.
        """

        from agent.planners.config_checklist import build_configuration_checklist

        plan = {"chain": "ethereum", "use_fake_node": False, "workflow_type": "sync_observe", "materialized_config": {}, "chain_template_requirements": {}}
        base = {"chain": "ethereum", "workflow_type": "sync_observe", "sync_observe_stop_condition": "duration"}

        # Local node process present -> attribution required.
        local = build_configuration_checklist({**base, "sync_observe_source": "existing_local_node"}, plan)
        self.assertIn("node_process_identity", local["missing_blockers"])
        # Endpoint-only (no local process) -> waived.
        endpoint_only = build_configuration_checklist({**base, "sync_observe_source": "endpoint_only"}, plan)
        self.assertNotIn("node_process_identity", endpoint_only["missing_blockers"])

    def test_sync_observe_demo_only_does_not_waive_real_node_requirements(self) -> None:
        """demo_only has neither a local process nor a real endpoint at all --

        it must waive both `node_process_identity` and `mainnet_rpc_url_reviewed`,
        not just the former. Regression for a live bug: the checklist only ever
        knew how to waive `node_process_identity` (for `endpoint_only`), so
        confirming the demo_only disclaimer and auto-running preflight always
        failed with `missing: mainnet_rpc_url_reviewed, node_process_identity`.
        """

        from agent.planners.config_checklist import build_configuration_checklist

        plan = {"chain": "ethereum", "use_fake_node": False, "workflow_type": "sync_observe", "materialized_config": {}, "chain_template_requirements": {}}
        base = {"chain": "ethereum", "workflow_type": "sync_observe", "sync_observe_stop_condition": "duration"}

        demo_only = build_configuration_checklist({**base, "sync_observe_source": "demo_only"}, plan)
        self.assertIn("node_process_identity", demo_only["missing_blockers"])
        self.assertIn("mainnet_rpc_url_reviewed", demo_only["missing_blockers"])

    def test_mixed_default_workload_confirms_weights_for_preflight(self) -> None:
        """A confirmed mixed workload (default template weights or validated custom

        weights) must satisfy `mixed_weights_confirmed`. Regression for a live
        chaos failure: choosing "use defaults" for a mixed workload left the flag
        unset, so preflight was offered ("配置已收集") but then blocked on
        `mixed_weights_confirmed` with no way forward.
        """

        from agent.harness.domains.execution_runtime import _prepare_kwargs
        from agent.harness.state import new_state

        state = new_state("t", language="zh")
        state["target_mode"] = "real-node"
        state["rpc_mode"] = "mixed"
        state["chain_identity"] = {"canonical": "solana", "status": "confirmed", "case": "known"}
        state["workload"] = {"confirmed": True, "choice": "default"}
        kwargs = _prepare_kwargs(state)
        self.assertIn("mixed_weights_confirmed", kwargs["confirmations"])

        # Not confirmed yet -> flag absent (still correctly blocking).
        state["workload"] = {"confirmed": False}
        self.assertNotIn("mixed_weights_confirmed", _prepare_kwargs(state)["confirmations"])

        # single mode never needs the mixed-weights confirmation.
        state["rpc_mode"] = "single"
        state["workload"] = {"confirmed": True, "choice": "default"}
        self.assertNotIn("mixed_weights_confirmed", _prepare_kwargs(state)["confirmations"])

    def test_adjust_mixed_weights_skips_endpoint_revalidation(self) -> None:
        """B.23 (known-issues.md): choosing "adjust mixed weights" for a

        chain's own default methods used to be forced through the exact same
        endpoint/method (re-)validation cycle as "add a genuinely new custom
        RPC method" -- the `workload_choice` handler's `else` branch treated
        `value == "weights"` and `value == "custom_rpc"` identically. This
        wastes a real network round-trip re-probing methods that are already
        known-good (they are the chain template's own validated defaults) and
        mislabels a "just reweight what's already there" action as "adding a
        custom method." Fixed with a genuine separate branch that jumps
        straight to the weight-entry question, which already supports
        weighting template-default methods (merges validated custom methods,
        empty here, with the chain's own template methods).
        """

        from agent.harness.coordinator import _question_for_group
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("t", language="zh")
        state["target_mode"] = "real-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["rpc_mode"] = "mixed"
        state["active_group"] = "workload_rpc"
        state["pending_question"] = {
            "id": "workload_confirm",
            "group": "workload_rpc",
            "kind": "numbered_choice",
            "field": "workload_choice",
            "options": [
                {"label": "使用默认值", "value": "default"},
                {"label": "添加自定义 RPC method", "value": "custom_rpc"},
                {"label": "调整 mixed 权重", "value": "weights"},
                {"label": "更换链或目标模式", "value": "change_target"},
            ],
            "manual_input_allowed": False,
        }
        state["last_user_input"] = "3"
        result = process_turn(state)

        # Must go straight to the weight question -- NOT the endpoint
        # question a genuinely new custom method would require.
        self.assertEqual(result["pending_question"]["id"], "custom_rpc_weights")
        question = _question_for_group(result, "endpoint_process")
        self.assertEqual(question["id"], "custom_rpc_weights")
        self.assertIn("eth_getBalance", question["prompt"])

        # Real weights over the chain's own template-default methods (no
        # custom method was ever validated) must be accepted directly.
        result["last_user_input"] = "eth_getBalance=40,eth_getTransactionCount=20,eth_blockNumber=20,eth_gasPrice=20"
        final = process_turn(result)
        self.assertEqual(final["rpc_mode"], "mixed")
        self.assertTrue(final["workload"]["confirmed"])
        self.assertEqual(final["custom_rpc"]["status"], "validated")

    def test_typed_weight_question_owns_assignment_json_and_yaml_answers(self) -> None:
        from agent.harness.domains.chain_rpc import question_for_chain_rpc
        from agent.harness.state import new_state
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn

        answers = (
            "eth_getBalance=40,eth_getTransactionCount=20,eth_blockNumber=20,eth_gasPrice=20",
            '{"eth_getBalance":40,"eth_getTransactionCount":20,"eth_blockNumber":20,"eth_gasPrice":20}',
            "eth_getBalance: 40\neth_getTransactionCount: 20\neth_blockNumber: 20\neth_gasPrice: 20",
        )
        for answer in answers:
            with self.subTest(answer=answer):
                state = new_state("typed-weight-owner", language="en")
                state.update({
                    "target_mode": "real-node",
                    "workflow_mode": "rpc_benchmark",
                    "chain_identity": {
                        "raw": "bsc",
                        "canonical": "bsc",
                        "adapter_family": "jsonrpc",
                        "status": "confirmed",
                        "case": "known",
                    },
                    "rpc_mode": "mixed",
                    "active_group": "endpoint_process",
                    "custom_rpc": {"job_local_override": True, "status": "needs_weights"},
                    "last_user_input": answer,
                })
                state["pending_question"] = question_for_chain_rpc(state, "endpoint_process") or {}

                self.assertEqual(state["pending_question"]["id"], "custom_rpc_weights")
                self.assertIs(state["pending_question"]["structured_input_owner"], True)
                result = process_turn(state)

                self.assertEqual(result["rpc_mode"], "mixed")
                self.assertTrue(result["workload"]["confirmed"])
                self.assertEqual(result["custom_rpc"]["status"], "validated")
                self.assertNotEqual(
                    (result.get("pending_question") or {}).get("id"),
                    "inferred_config_review",
                )

    def test_typed_evidence_owner_does_not_swallow_mixed_independent_demand(self) -> None:
        from agent.harness.domains.chain_rpc import question_for_chain_rpc
        from agent.harness.state import new_state
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn

        state = new_state("mixed-evidence-owner", language="en")
        state.update({
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "active_group": "endpoint_process",
            "chain_identity": {
                "raw": "flow-evm",
                "canonical": "flow-evm",
                "adapter_family": "jsonrpc",
                "status": "existing_family_needs_schema_evidence",
                "case": "case2",
                "candidate_method": "eth_blockNumber",
            },
        })
        state["pending_question"] = question_for_chain_rpc(state, "endpoint_process") or {}
        state["last_user_input"] = (
            "Also change the QPS profile.\n"
            '{"method":"eth_blockNumber","params":[]}'
        )
        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{
                "type": "clarify_unresolved",
                "clauses": [state["last_user_input"]],
            }]},
        ) as resolver, patch(
            "agent.harness.domains.rpc_endpoint.extract_rpc_schema_from_evidence",
        ) as extractor:
            result = process_turn(state)

        resolver.assert_called_once()
        extractor.assert_not_called()
        self.assertEqual(result["chain_identity"]["status"], "existing_family_needs_schema_evidence")
        self.assertEqual(result["pending_question"]["id"], "new_chain_schema_evidence")

    def test_numbered_answer_to_confirm_or_value_question_applies(self) -> None:
        """A numbered answer ("1"/"2") to a confirm_or_value question that renders

        numbered options must be applied directly, not handed to the LLM resolver.
        Regression for a live chaos failure: answering "1" to the MAINNET_RPC_URL
        sync-health confirm ("1. Y  2. N") was misrouted (read as a chain remark)
        and the question was re-asked.
        """

        from agent.harness.coordinator import _answer_fits_pending, _coerce_answer

        q = {
            "id": "MAINNET_RPC_URL_REVIEWED",
            "group": "chain_auxiliary_endpoints",
            "kind": "confirm_or_value",
            "field": "MAINNET_RPC_URL_REVIEWED",
            "options": [{"label": "Y", "value": "__default__"}, {"label": "N", "value": "__manual__"}],
            "manual_input_allowed": True,
        }
        self.assertTrue(_answer_fits_pending("1", q))
        self.assertEqual(_coerce_answer("1", q), "__default__")
        self.assertTrue(_answer_fits_pending("2", q))
        self.assertEqual(_coerce_answer("2", q), "__manual__")
        # A pasted custom URL still fits as a value.
        self.assertTrue(_answer_fits_pending("https://rpc.example.com", q))

    def test_health_probe_method_is_chain_specific_not_evm_only(self) -> None:
        """The LOCAL_RPC_URL liveness probe must use each chain's own health

        method, not eth_chainId for every jsonrpc-transport chain. Regression for
        a live chaos failure: a real solana endpoint was rejected ("Method not
        found") because the probe sent the EVM method eth_chainId; this also broke
        sui/near/starknet/tron/avalanche-x (jsonrpc transport, non-EVM).
        """

        from agent.validators.endpoint_probe import _validate_generic_jsonrpc_endpoint, health_probe_methods

        # EVM chains -> their declared eth_* health method; solana -> getHealth.
        self.assertEqual(health_probe_methods("solana", "jsonrpc"), (["getHealth"], {"getHealth": []}))
        self.assertEqual(health_probe_methods("ethereum", "jsonrpc"), (["eth_syncing"], {"eth_syncing": []}))
        # jsonrpc chains without a declared param-less health method fall back so
        # the probe uses the adapter's own health check, never an EVM method.
        self.assertEqual(health_probe_methods("sui", "jsonrpc"), (None, {}))
        # `bitcoin`'s own template declares `getblockchaininfo` as its health
        # method (matches every bitcoin_jsonrpc template's `_meta.health_probe`).
        self.assertEqual(health_probe_methods("bitcoin", "bitcoin_jsonrpc"), (["getblockchaininfo"], {"getblockchaininfo": []}))
        # Families whose transport isn't a plain POST JSON-RPC call (rest/
        # tendermint/hedera_dual) are left to the adapter health probe.
        self.assertEqual(health_probe_methods("algorand", "rest"), (None, {}))
        # A brand-new jsonrpc chain (no template yet) defaults to the EVM guess;
        # substrate/bitcoin_jsonrpc get their own family-generic default too
        # (`#59`: Case 2 endpoint validation could never pass for a template-less
        # chain in either family before this, since `_probe_health` requires a
        # `config/chains/<chain>.json` that a genuinely new chain never has).
        self.assertEqual(health_probe_methods("brand-new-l2", "jsonrpc"), (["eth_chainId"], {"eth_chainId": []}))
        self.assertEqual(health_probe_methods("brand-new-parachain", "substrate"), (["system_chain"], {"system_chain": []}))
        self.assertEqual(health_probe_methods("brand-new-fork", "bitcoin_jsonrpc"), (["getblockchaininfo"], {"getblockchaininfo": []}))

        # The generic jsonrpc path uses the first selected method as its health
        # probe, not a hardcoded eth_chainId.
        result = {"chain": "solana", "transport": "jsonrpc", "checks": [], "warnings": [], "blockers": [], "selected_methods": []}
        with patch("agent.validators.endpoint_probe._probe_generic_jsonrpc_method", return_value={"name": "endpoint_health_probe", "passed": True, "detail": "ok"}):
            out = _validate_generic_jsonrpc_endpoint(result, endpoint="https://x", methods=["getHealth"], method_params={}, timeout=1.0)
        self.assertEqual(out["safe_method"], "getHealth")

    def test_out_of_range_numbered_answer_keeps_question_without_llm(self) -> None:
        """A bare out-of-range digit at a choice menu ("0", "99", "4") must be

        rejected deterministically and keep the question, never handed to the LLM
        resolver. Regression for a live chaos failure: "0" at benchmark_mode was
        routed to `resolve_action_queue`, which read it as a chain change and
        dropped the pending question.
        """

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        def _menu_state(answer: str) -> dict:
            state = new_state("unit-thread", language="zh")
            state["active_group"] = "qps_profile"
            state["target_mode"] = "fake-node"
            state["workflow_mode"] = "rpc_benchmark"
            state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
            state["pending_question"] = {
                "id": "benchmark_mode",
                "group": "qps_profile",
                "kind": "numbered_choice",
                "field": "benchmark_mode",
                "options": [
                    {"label": "quick", "value": "quick"},
                    {"label": "standard", "value": "standard"},
                    {"label": "intensive", "value": "intensive"},
                ],
            }
            state["last_user_input"] = answer
            return state

        for bad in ("0", "99", "4"):
            with patch(
                "agent.harness.coordinator.resolve_action_queue",
                side_effect=AssertionError("out-of-range digit must not reach the LLM resolver"),
            ):
                result = process_turn(_menu_state(bad))
            self.assertEqual(result["pending_question"]["id"], "benchmark_mode", f"answer={bad!r}")
            self.assertFalse((result.get("qps_profile") or {}).get("mode"), f"answer={bad!r}")

    def test_bare_yes_no_does_not_guess_a_numbered_menu_choice(self) -> None:
        """A numbered menu and a Y/N confirmation are different contracts."""

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.chain_rpc_questions import _continue_question
        from agent.harness.state import new_state

        for answer in ("Y", "n"):
            state = new_state("unit-thread", language="en")
            state["active_group"] = "endpoint_process"
            state["target_mode"] = "fake-node"
            state["chain_identity"] = {
                "raw": "bsc",
                "canonical": "bsc",
                "status": "confirmed",
                "case": "known",
            }
            state["custom_rpc"] = {
                "status": "method_validated_next",
                "validated_methods": [{"method": "eth_blockNumber"}],
            }
            state["pending_question"] = _continue_question(state, "custom_rpc")
            state["last_user_input"] = answer
            with patch(
                "agent.harness.coordinator.resolve_action_queue",
                side_effect=AssertionError("bare Y/N at a numbered menu must not reach the LLM resolver"),
            ):
                result = process_turn(state)
            self.assertEqual(result["pending_question"]["id"], "custom_rpc_continue")
            self.assertEqual(result["custom_rpc"]["status"], "method_validated_next")

    def test_target_mode_change_confirmation_names_preserved_chain(self) -> None:
        """A mode switch must not silently reuse a stale chain: the confirmation

        prompt explicitly names the carried-over chain so the user can keep or
        change it (baseline dual-AI chaos gate, step 3).
        """

        from agent.harness.contracts import ActionProposal
        from agent.harness.domains.chain_rpc import apply_chain_rpc_action
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "sync-observe"
        state["active_group"] = "provider_deployment"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}

        outcome = apply_chain_rpc_action(
            state,
            ActionProposal(
                "mode-change",
                "choose_target_mode",
                {"target_mode": "fake-node", "target_mode_explicit": True},
                "high",
            ),
        )
        state = _commit_result(state, outcome, owner="chain_rpc")
        prompt = str(state["pending_question"]["prompt"])
        self.assertIn("bsc", prompt)
        self.assertEqual(state["pending_question"]["id"], "target_mode_change_confirm")

    def test_combined_modes_and_preflight_smoke_question_answers_both(self) -> None:
        """A single turn asking about the modes AND preflight/smoke must answer

        both. Regression for a live failure: "三种模式是什么？preflight/smoke 又是
        什么？" got the modes explanation only — the second question was dropped
        because the resolver collapses the turn into one topic and preflight/smoke
        had no definition anywhere.
        """

        from agent.harness.domains.orientation import answer_consultation
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        combined = "fake-node、real-node 或 sync-observe 是什么？ preflight/smoke 又是什么？"

        # Structured resolution emits one consultation action per question;
        # composing both responses must preserve both answers.
        resp = "\n".join(
            answer_consultation(state, {"topic": topic})
            for topic in ("mode_comparison", "execution_preflight_smoke")
        )
        self.assertIn("sync-observe", resp)
        self.assertRegex(resp, r"(预检|冒烟)")

        # A standalone preflight/smoke question is defined, not just listed.
        standalone = answer_consultation(state, {"topic": "execution_preflight_smoke"})
        self.assertRegex(standalone, r"预检")
        self.assertRegex(standalone, r"冒烟")

        # A pure modes question is NOT bloated with preflight/smoke text.
        modes_only = answer_consultation(state, {"topic": "mode_comparison"})
        self.assertNotRegex(modes_only, r"(预检|冒烟)")

    def test_adapter_family_hint_ignores_negated_protocol_mentions(self) -> None:
        """A negated protocol mention ("没有 json-rpc", "not json-rpc", "非 REST")

        must not be extracted as a positive adapter-family choice. Regression for
        a live dual-AI chaos failure: confirming an unsupported-protocol chain
        with "...没有 JSON-RPC" was read as adapter_family=jsonrpc, which skipped
        the Case-3 official-docs handoff. Positive mentions still resolve.
        """

        from agent.harness.input_values import adapter_family_hint

        # Negated mentions -> no family hint.
        self.assertEqual(adapter_family_hint("自定义二进制协议，没有 JSON-RPC"), "")
        self.assertEqual(adapter_family_hint("not json-rpc, custom binary"), "")
        self.assertEqual(adapter_family_hint("非 REST"), "")
        # Negated ENUMERATION: "not A/B/substrate" must negate every family, not
        # just the first (regression: Mina described as GraphQL was forced to
        # substrate and mis-routed to Case 2 instead of the Case-3 handoff).
        self.assertEqual(adapter_family_hint("只有 GraphQL，没有 JSON-RPC，也不是 REST/cosmos/substrate/bitcoin"), "")
        self.assertEqual(adapter_family_hint("not json-rpc, not rest, not substrate"), "")
        self.assertEqual(adapter_family_hint("without evm support"), "")
        # Positive mentions still resolve to the family.
        self.assertEqual(adapter_family_hint("it is EVM compatible"), "jsonrpc")
        self.assertEqual(adapter_family_hint("this is a substrate chain"), "substrate")
        self.assertEqual(adapter_family_hint("uses a REST api"), "rest")

    def test_declined_confirmation_preserves_independent_remaining_actions(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "provider_deployment"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["last_user_input"] = "use real-node and quick"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "choose_target_mode", "target_mode": "real-node", "target_mode_explicit": True, "source_evidence": "real-node", "confidence": "high"},
                    {"type": "set_qps_mode", "qps_mode": "quick", "mutation_explicit": True, "source_evidence": "quick", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "target_mode_change_confirm")
        result["last_user_input"] = "N"
        result = process_turn(result)

        self.assertEqual(result["target_mode"], "fake-node")
        self.assertEqual(result.get("action_queue"), [])
        self.assertEqual(result.get("qps_profile", {}).get("mode"), "quick")

    def test_new_chain_endpoint_probe_failure_remains_blocking(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "endpoint_process"
        state["chain_identity"] = {
            "raw": "flow",
            "canonical": "flow",
            "adapter_family": "jsonrpc",
            "status": "existing_family_needs_endpoint",
            "case": "existing_family",
        }
        state["pending_question"] = {
            "id": "new_chain_endpoint",
            "group": "endpoint_process",
            "kind": "url",
            "prompt": "Provide endpoint",
            "field": "new_chain_endpoint",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "https://example.invalid/rpc"

        with patch(
            "agent.harness.domains.rpc_endpoint.validate_rpc_endpoint",
            return_value={"ready": False, "status": "failed", "error": "boom", "evidence_file": "probe.json"},
        ):
            result = process_turn(state)

        self.assertEqual(result["chain_identity"]["status"], "existing_family_needs_endpoint")
        self.assertFalse(result["endpoint_evidence"]["candidate_endpoint_ready"])
        self.assertEqual(result["pending_question"]["id"], "new_chain_endpoint")
        self.assertIn("endpoint validation failed", "\n".join(result.get("visible_response") or []))

    def test_case2_go_back_invalidates_endpoint_evidence_before_family_reentry(self) -> None:
        from agent.harness.domains.chain_rpc import apply_chain_rpc_action, cancel_chain_rpc_question
        from agent.harness.contracts import ActionProposal
        from agent.harness.invariants import apply_state_delta
        from agent.harness.state import new_state

        state = new_state("case2-back-endpoint", language="en")
        state["target_mode"] = "real-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "flow",
            "canonical": "flow",
            "adapter_family": "jsonrpc",
            "status": "existing_family_needs_method",
            "case": "case2",
            "identity_confirmed": True,
        }
        state["endpoint_evidence"] = {
            "candidate_endpoint": "https://flow.example/rpc",
            "candidate_endpoint_ready": True,
            "new_chain_endpoint_probe": {"ready": True},
        }
        question = {
            "id": "new_chain_method",
            "group": "endpoint_process",
            "kind": "text",
        }
        cancelled = cancel_chain_rpc_question(state, question)
        restored = apply_state_delta(state, cancelled.delta, owner="chain_rpc")

        self.assertEqual(restored["chain_identity"]["status"], "needs_protocol_confirmation")
        self.assertFalse(restored.get("endpoint_evidence"))

        confirmed = apply_chain_rpc_action(
            restored,
            ActionProposal(
                "family-1",
                "choose_adapter_family",
                {"adapter_family": "jsonrpc"},
                "high",
            ),
        )
        self.assertEqual((confirmed.pending_question or {}).get("id"), "new_chain_endpoint")

    def test_pending_endpoint_question_explains_requirements_instead_of_repeating_prompt(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "endpoint_process"
        state["chain_identity"] = {
            "raw": "sola",
            "canonical": "sola",
            "adapter_family": "jsonrpc",
            "status": "existing_family_needs_endpoint",
            "case": "existing_family",
        }
        state["pending_question"] = {
            "id": "new_chain_endpoint",
            "group": "endpoint_process",
            "kind": "url",
            "prompt": "请提供可访问的 RPC endpoint",
            "field": "new_chain_endpoint",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "我没有 endpoint，先告诉我需要准备什么"

        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{"type": "answer_opening_question", "topic": "config_explanation", "subject": "new_chain_endpoint", "confidence": "high"}]},
        ):
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertEqual(result["pending_question"]["id"], "new_chain_endpoint")
        self.assertIn("可访问", text)
        self.assertIn("method", text)
        self.assertIn("只作为", text)
        self.assertNotIn("这条回复不像当前问题的答案", text)

    def test_change_group_activates_requested_group_instead_of_default_path(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "provider_deployment"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["last_user_input"] = "I want to adjust QPS first"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "change_group", "group": "qps_profile", "navigation_explicit": True, "source_evidence": "adjust QPS", "confidence": "high"}]}
            result = process_turn(state)

        self.assertEqual(result["active_group"], "qps_profile")
        self.assertEqual(result["pending_question"]["id"], "benchmark_mode")
        self.assertEqual(result["group_history"][-1], "provider_deployment")
        admitted = result["turn_context"]["admitted_actions"]
        self.assertEqual(admitted[0]["type"], "change_group")
        self.assertEqual(admitted[0]["group"], "qps_profile")

    def test_change_group_crosses_inferred_config_review_barrier(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.action_registry import ACTION_BY_TYPE
        from agent.harness.state import new_state

        state = new_state("navigation-barrier", language="en")
        state.update({
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "active_group": "chain_identity",
            "chain_identity": {
                "raw": "ethereum",
                "canonical": "ethereum",
                "adapter_family": "jsonrpc",
                "status": "confirmed",
                "case": "known",
            },
            "confirmed_config": {"BLOCKCHAIN_NODE": "ethereum"},
            "inferred_config": {
                "pending_review": {
                    "config_values": {"CLOUD_REGION": "asia-east1"},
                    "unmapped_values": {},
                    "source_format": "yaml",
                    "reason": "deployment notes",
                }
            },
            "pending_question": {
                "id": "inferred_config_review",
                "group": "chain_identity",
                "kind": "yes_no",
                "field": "inferred_config_review",
                "prompt": "Apply the inferred values?",
                "options": [
                    {"id": "1", "label": "Y", "value": True},
                    {"id": "2", "label": "N", "value": False},
                ],
                "accepted_action_types": ["answer_pending"],
                "queue_barrier": True,
            },
            "last_user_input": "Configure QPS first, then return to these inferred values.",
        })

        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{
                "type": "change_group",
                "group": "qps_profile",
                "navigation_explicit": True,
                "source_evidence": "Configure QPS first",
                "confidence": "high",
            }]},
        ):
            result = process_turn(state)

        self.assertTrue(ACTION_BY_TYPE["change_group"].crosses_pending_barrier)
        self.assertEqual(result["active_group"], "qps_profile")
        self.assertEqual(result["pending_question"]["id"], "benchmark_mode")
        self.assertNotIn("Apply the inferred values?", "\n".join(result.get("visible_response") or []))

        result["last_user_input"] = "3"
        result = process_turn(result)
        self.assertEqual(result["pending_question"]["id"], "qps_profile_confirm")

        result["last_user_input"] = "Y"
        result = process_turn(result)
        self.assertEqual(result["qps_profile"]["mode"], "intensive")
        self.assertEqual(result["pending_question"]["id"], "inferred_config_review")
        self.assertIn("CLOUD_REGION", "\n".join(result.get("visible_response") or []))

    def test_change_group_with_future_resume_enters_detour_before_restoring_review(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("navigation-future-resume", language="en")
        state.update({
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "active_group": "chain_identity",
            "chain_identity": {
                "raw": "ethereum",
                "canonical": "ethereum",
                "adapter_family": "jsonrpc",
                "status": "confirmed",
                "case": "known",
            },
            "confirmed_config": {"BLOCKCHAIN_NODE": "ethereum"},
            "inferred_config": {
                "pending_review": {
                    "config_values": {"CLOUD_REGION": "asia-east1"},
                    "unmapped_values": {"machine.type": "c3-standard-8"},
                    "source_format": "yaml",
                    "reason": "deployment notes",
                }
            },
            "pending_question": {
                "id": "inferred_config_review",
                "group": "chain_identity",
                "kind": "yes_no",
                "field": "inferred_config_review",
                "prompt": "Apply the inferred values?",
                "options": [
                    {"id": "1", "label": "Y", "value": True},
                    {"id": "2", "label": "N", "value": False},
                ],
                "accepted_action_types": ["answer_pending"],
                "queue_barrier": True,
            },
            "last_user_input": "Configure QPS first, then return to this exact review.",
        })

        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [
                {
                    "type": "change_group",
                    "group": "qps_profile",
                    "navigation_explicit": True,
                    "source_evidence": "Configure QPS first",
                    "confidence": "high",
                },
                {
                    "type": "resume_current_flow",
                    "source_evidence": "return to this exact review",
                    "confidence": "high",
                },
            ]},
        ):
            result = process_turn(state)

        self.assertEqual(result["active_group"], "qps_profile")
        self.assertEqual(result["pending_question"]["id"], "benchmark_mode")
        self.assertNotIn("Apply the inferred values?", "\n".join(result.get("visible_response") or []))
        self.assertEqual(result["interruption_stack"][-1]["question_id"], "inferred_config_review")

        result["last_user_input"] = "3"
        result = process_turn(result)
        result["last_user_input"] = "Y"
        result = process_turn(result)

        self.assertEqual(result["qps_profile"]["mode"], "intensive")
        self.assertEqual(result["pending_question"]["id"], "inferred_config_review")
        self.assertEqual(result["inferred_config"]["pending_review"]["unmapped_values"], {
            "machine.type": "c3-standard-8",
        })

    def test_unselected_qps_customization_crosses_old_barrier_and_preserves_return(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("qps-customization-before-mode", language="en")
        state.update({
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "active_group": "chain_identity",
            "chain_identity": {
                "raw": "ethereum",
                "canonical": "ethereum",
                "adapter_family": "jsonrpc",
                "status": "confirmed",
                "case": "known",
            },
            "confirmed_config": {"BLOCKCHAIN_NODE": "ethereum"},
            "inferred_config": {
                "pending_review": {
                    "config_values": {"CLOUD_REGION": "us-central1"},
                    "unmapped_values": {"OWNER_TICKET": "OPS-42"},
                    "source_format": "yaml",
                }
            },
            "pending_question": {
                "id": "inferred_config_review",
                "group": "chain_identity",
                "kind": "yes_no",
                "field": "inferred_config_review",
                "prompt": "Apply the inferred values?",
                "options": [
                    {"id": "1", "label": "Y", "value": True},
                    {"id": "2", "label": "N", "value": False},
                ],
                "accepted_action_types": ["answer_pending"],
                "queue_barrier": True,
            },
            "last_user_input": "Configure the QPS profile first, then return to this review.",
        })

        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [
                {
                    "type": "request_qps_customization",
                    "source_evidence": "Configure the QPS profile first",
                    "confidence": "high",
                },
                {
                    "type": "resume_current_flow",
                    "source_evidence": "return to this review",
                    "confidence": "high",
                },
            ]},
        ):
            result = process_turn(state)

        self.assertEqual(result["active_group"], "qps_profile")
        self.assertEqual(result["pending_question"]["id"], "benchmark_mode")
        self.assertEqual(result["interruption_stack"][-1]["question_id"], "inferred_config_review")
        self.assertEqual(
            (result.get("turn_context") or {}).get("admitted_actions", [])[0].get("type"),
            "request_qps_customization",
        )

    def test_change_group_resumes_an_environment_question_after_detour(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("environment-navigation-barrier", language="en")
        state.update({
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "active_group": "provider_deployment",
            "chain_identity": {
                "raw": "ethereum",
                "canonical": "ethereum",
                "adapter_family": "jsonrpc",
                "status": "confirmed",
                "case": "known",
            },
            "confirmed_config": {
                "BLOCKCHAIN_NODE": "ethereum",
                "CLOUD_REGION": "asia-east1",
            },
            "pending_question": {
                "id": "CLOUD_ZONE",
                "group": "provider_deployment",
                "kind": "manual_value",
                "field": "CLOUD_ZONE",
                "prompt": "Enter CLOUD_ZONE.",
                "manual_input_allowed": True,
                "options": [],
                "validation": {"value_type": "scalar_token"},
            },
            "last_user_input": "Configure benchmark intensity first.",
        })

        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{
                "type": "change_group",
                "group": "qps_profile",
                "navigation_explicit": True,
                "source_evidence": "Configure benchmark intensity first",
                "confidence": "high",
            }]},
        ):
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "benchmark_mode")
        result["last_user_input"] = "1"
        result = process_turn(result)
        result["last_user_input"] = "Y"
        result = process_turn(result)

        self.assertEqual(result["qps_profile"]["mode"], "quick")
        self.assertEqual(result["active_group"], "provider_deployment")
        self.assertEqual(result["pending_question"]["id"], "CLOUD_ZONE")

    def test_completed_group_jump_resumes_default_missing_group(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "provider_deployment"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["last_user_input"] = "I want to adjust QPS first"

        with patch("agent.harness.coordinator.resolve_action_queue", return_value={"actions": [{"type": "change_group", "group": "qps_profile", "navigation_explicit": True, "source_evidence": "adjust QPS", "confidence": "high"}]}):
            state = process_turn(state)

        state["last_user_input"] = "1"
        state = process_turn(state)
        self.assertEqual(state["pending_question"]["id"], "qps_profile_confirm")

        state["last_user_input"] = "Y"
        state = process_turn(state)

        self.assertEqual(state["qps_profile"]["mode"], "quick")
        self.assertTrue(state["qps_profile"]["confirmed"])
        self.assertEqual(state["active_group"], "provider_deployment")
        self.assertEqual(state["pending_question"]["id"], "CLOUD_REGION")

    def test_go_back_out_of_stuck_custom_rpc_endpoint_abandons_it(self) -> None:
        """B.17 (known-issues.md): navigating away from a stuck

        `custom_rpc_endpoint` question via `go_back` used to bounce right
        back to the identical question. Root cause was not missing
        navigation handling -- `_pop_previous_group` already correctly skips
        `group_history`'s top entry when it equals the current group
        (`endpoint_process`, the group *containing* the pending question) and
        lands on the real previous group (`network`). But `network` has
        nothing left to ask (already confirmed), so
        `_ask_next_blocking_question` recomputes via the shared routing chain
        and finds `custom_rpc` still incomplete -- re-rendering the exact
        question the user tried to leave, since leaving it never abandoned
        the half-finished custom-method attempt. Fixed by clearing
        `custom_rpc`/`workload.choice` when navigating away from this
        specific pending question, so routing naturally lands back on
        `workload_confirm` instead.
        """

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "real-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {
            "BLOCKCHAIN_NODE": "bsc",
            "CLOUD_REGION": "us-1",
            "CLOUD_ZONE": "us-1-z",
            "MACHINE_TYPE": "n2",
            "LEDGER_DEVICE": "vda",
            "DATA_VOL_TYPE": "hyperdisk-balanced",
            "DATA_VOL_SIZE": "926",
            "DATA_VOL_MAX_IOPS": "20000",
            "DATA_VOL_MAX_THROUGHPUT": "1000",
            "has_accounts_device": False,
            "NETWORK_INTERFACE": "eth0",
            "NETWORK_MAX_BANDWIDTH_GBPS": "100",
            "LOCAL_RPC_URL": "https://bsc-dataseed.binance.org",
            "BLOCKCHAIN_PROCESS_NAMES": "bsc-node",
            "MAINNET_RPC_URL_REVIEWED": True,
        }
        state["endpoint_evidence"] = {"local_rpc_url_ready": True}
        state["rpc_mode"] = "mixed"
        state["workload"] = {"choice": "custom_rpc"}
        state["custom_rpc"] = {"status": "needs_endpoint"}
        state["active_group"] = "endpoint_process"
        state["group_history"] = ["provider_deployment", "ledger_disk", "accounts_disk", "network", "endpoint_process"]
        state["pending_question"] = {
            "id": "custom_rpc_endpoint",
            "group": "endpoint_process",
            "kind": "url",
            "field": "custom_rpc_endpoint",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "算了，不弄了，返回 workload 菜单"

        with patch("agent.harness.coordinator.resolve_action_queue", return_value={"actions": [{"type": "go_back", "confidence": "high"}]}):
            result = process_turn(state)

        self.assertEqual(result["custom_rpc"], {})
        self.assertEqual(result["pending_question"]["id"], "workload_confirm")
        self.assertEqual(result["active_group"], "workload_rpc")

    def test_go_back_out_of_stuck_sync_observe_rpc_url_abandons_it(self) -> None:
        """Same bug class as B.17, found live via dual-AI chaos testing of

        sync-observe: `go_back` while `SYNC_OBSERVE_RPC_URL` is the pending
        question (sync-observe's `endpoint_only`/`existing_local_node` real
        endpoint probe, asked from within `endpoint_process`) bounced right
        back to the identical question. `_pop_previous_group` correctly skips
        `group_history`'s top entry (`endpoint_process`, the group
        *containing* the pending question) and lands on the real previous
        group (`network`), but `network` has nothing left to ask, so
        `_ask_next_blocking_question` recomputes via the shared routing chain
        and finds `sync_observe`'s source (`endpoint_only`) still
        unvalidated -- re-rendering the exact question the user tried to
        leave. Fixed by clearing `sync_observe.source` (and related
        acknowledgement/evidence flags) when navigating away from this
        pending question, the same way `custom_rpc_endpoint` is abandoned,
        so routing naturally lands back on `sync_observe_source` instead of
        looping.
        """

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["target_mode"] = "sync-observe"
        state["workflow_mode"] = "sync_observe"
        state["chain_identity"] = {"raw": "ethereum", "canonical": "ethereum", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {
            "BLOCKCHAIN_NODE": "ethereum",
            "CLOUD_REGION": "us-1",
            "CLOUD_ZONE": "us-1-z",
            "MACHINE_TYPE": "n2",
            "LEDGER_DEVICE": "vda",
            "DATA_VOL_TYPE": "hyperdisk-balanced",
            "DATA_VOL_SIZE": "926",
            "DATA_VOL_MAX_IOPS": "20000",
            "DATA_VOL_MAX_THROUGHPUT": "1000",
            "has_accounts_device": False,
            "NETWORK_INTERFACE": "eth0",
            "NETWORK_MAX_BANDWIDTH_GBPS": "100",
        }
        state["sync_observe"] = {"source": "endpoint_only", "local_attribution_available": False}
        state["endpoint_evidence"] = {}
        state["active_group"] = "endpoint_process"
        state["group_history"] = ["provider_deployment", "ledger_disk", "accounts_disk", "network", "sync_observe", "endpoint_process"]
        state["pending_question"] = {
            "id": "SYNC_OBSERVE_RPC_URL",
            "group": "endpoint_process",
            "kind": "url",
            "field": "SYNC_OBSERVE_RPC_URL",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "actually I don't have a real endpoint, let's go back"

        with patch("agent.harness.coordinator.resolve_action_queue", return_value={"actions": [{"type": "go_back", "confidence": "high"}]}):
            result = process_turn(state)

        self.assertFalse(result["sync_observe"].get("source"))
        self.assertEqual(result["pending_question"]["id"], "sync_observe_source")
        self.assertEqual(result["active_group"], "sync_observe")

    def test_go_back_uses_group_history_without_phrase_matching(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "qps_profile"
        state["group_history"] = ["opening", "provider_deployment", "workload_rpc"]
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["rpc_mode"] = "single"
        state["last_user_input"] = "previous"

        with patch("agent.harness.coordinator.resolve_action_queue", return_value={"actions": [{"type": "go_back", "confidence": "high"}]}):
            result = process_turn(state)

        self.assertEqual(result["active_group"], "workload_rpc")
        self.assertEqual(result["pending_question"]["id"], "workload_confirm")
        self.assertEqual(result["group_history"], ["opening", "provider_deployment"])

    def test_custom_rpc_choice_transfers_control_to_endpoint_group(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "workload_rpc"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["rpc_mode"] = "mixed"
        state["pending_question"] = {
            "id": "workload_confirm",
            "group": "workload_rpc",
            "kind": "numbered_choice",
            "field": "workload_choice",
            "options": [
                {"label": "Use defaults", "value": "default"},
                {"label": "Add custom RPC method", "value": "custom_rpc"},
                {"label": "Adjust mixed weights", "value": "weights"},
            ],
            "manual_input_allowed": False,
        }
        state["last_user_input"] = "2"

        result = process_turn(state)

        self.assertEqual(result["custom_rpc"]["status"], "needs_endpoint")
        self.assertEqual(result["active_group"], "endpoint_process")
        self.assertEqual(result["pending_question"]["id"], "custom_rpc_endpoint")

    def test_custom_rpc_jump_takes_precedence_over_final_real_node_endpoint(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "real-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "endpoint_process"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["pending_question"] = {
            "id": "LOCAL_RPC_URL",
            "group": "endpoint_process",
            "kind": "url",
            "field": "LOCAL_RPC_URL",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "我先不想给最终被测 endpoint，我想先加一个自定义 RPC method，示例 endpoint 不要当成 LOCAL_RPC_URL"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "start_custom_rpc", "rpc_endpoint": "endpoint", "source_evidence": state["last_user_input"], "reason": "validate a custom method first", "confidence": "high"},
                ]
            }
            with patch("agent.harness.domains.rpc_endpoint.validate_rpc_endpoint") as probe:
                result = process_turn(state)

        probe.assert_not_called()
        self.assertEqual(result["active_group"], "endpoint_process")
        self.assertEqual(result["custom_rpc"]["status"], "needs_endpoint")
        self.assertEqual(result["pending_question"]["id"], "custom_rpc_endpoint")
        self.assertNotIn("LOCAL_RPC_URL", result.get("confirmed_config", {}))
        self.assertNotEqual(result.get("endpoint_evidence", {}).get("local_rpc_url_ready"), True)

    def test_sync_observe_group_jump_requires_target_mode_confirmation(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "workload_rpc"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["last_user_input"] = "observe node sync instead"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "change_group", "group": "sync_observe", "navigation_explicit": True, "source_evidence": "observe node sync", "confidence": "high"}]}
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "target_mode_select")
        self.assertEqual(result["pending_question"]["group"], "target_mode")
        self.assertEqual(result["target_mode"], "fake-node")
        self.assertEqual(result["workflow_mode"], "rpc_benchmark")
        self.assertEqual(result["control"]["deferred_group"], "sync_observe")

    def test_public_group_jump_defers_to_declared_dependency(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["last_user_input"] = "Take me to the RPC workload first."

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{
                "type": "change_group",
                "group": "workload_rpc",
                "navigation_explicit": True,
                "source_evidence": state["last_user_input"],
                "confidence": "high",
            }]}
            result = process_turn(state)

        self.assertEqual(result["active_group"], "chain_identity")
        self.assertEqual(result["pending_question"]["id"], "chain")
        self.assertEqual(result["control"]["deferred_group"], "workload_rpc")

    def test_completed_public_group_jump_reopens_group_without_mutating_value(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state.update({
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "qps_profile": {
                "mode": "quick",
                "default_decision_made": True,
                "confirmed": True,
                "overrides": {},
            },
            "last_user_input": "Show the current QPS profile without changing it.",
        })
        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{
                "type": "change_group",
                "group": "qps_profile",
                "navigation_explicit": True,
                "source_evidence": state["last_user_input"],
                "confidence": "high",
            }]}
            result = process_turn(state)

        self.assertEqual(result["active_group"], "qps_profile")
        self.assertEqual(result["pending_question"]["id"], "benchmark_mode")
        self.assertEqual(result["qps_profile"]["mode"], "quick")

    def test_unknown_chain_uses_identity_gate_not_partial_coercion(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["pending_question"] = {
            "id": "chain",
            "group": "chain_identity",
            "kind": "chain",
            "field": "chain",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "sola"

        with (
            patch(
                "agent.harness.domains.chain_rpc.extract_chain_mention",
                return_value={"found": True, "chain_text": "solana", "confidence": "high"},
            ),
            patch("agent.harness.domains.chain_identity.resolve_unknown_chain_identity") as resolver,
        ):
            resolver.return_value = {
                "chain_exists": None,
                "canonical_chain_name": "sola",
                "adapter_family": "unknown",
                "confidence": "low",
            }
            result = process_turn(state)

        self.assertNotEqual(result.get("chain_identity", {}).get("canonical"), "solana")
        self.assertEqual(result["chain_identity"]["status"], "needs_identity_confirmation")
        self.assertEqual(result["pending_question"]["id"], "unknown_chain_identity_confirm")

    def test_unknown_chain_model_known_proposal_never_overwrites_raw_identity(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["target_mode"] = "real-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["pending_question"] = {
            "id": "chain",
            "group": "chain_identity",
            "kind": "chain",
            "field": "chain",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "LocalEvmDemo"

        resolution = {
            "chain_exists": True,
            "canonical_chain_name": "ethereum",
            "adapter_family": "jsonrpc",
            "confidence": "high",
        }
        with patch(
            "agent.harness.domains.chain_identity.resolve_unknown_chain_identity",
            return_value=resolution,
        ):
            result = process_turn(state)

        identity = result["chain_identity"]
        self.assertEqual(identity["raw"], "LocalEvmDemo")
        self.assertEqual(identity["canonical"], "LocalEvmDemo")
        self.assertEqual(identity["proposed_known_chain"], "ethereum")
        self.assertEqual(identity["status"], "needs_known_chain_confirmation")
        self.assertNotIn("BLOCKCHAIN_NODE", result.get("confirmed_config", {}))
        self.assertIn("LocalEvmDemo", result["pending_question"]["prompt"])
        self.assertIn("ethereum", result["pending_question"]["prompt"])

    def test_workload_detour_waits_for_chain_confirmation_then_resumes(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "Run it against BNB Smart Chain.",
            "canonical": "Run it against BNB Smart Chain.",
            "proposed_known_chain": "bsc",
            "status": "needs_known_chain_confirmation",
            "case": "known_candidate",
        }
        state["pending_question"] = {
            "id": "unknown_chain_identity_confirm",
            "group": "chain_identity",
            "kind": "numbered_choice",
            "field": "unknown_chain_decision",
            "options": [
                {
                    "id": "confirm_known_chain",
                    "label": "Use bsc",
                    "value": "confirm_known_chain",
                    "action": {"type": "answer_pending", "answer": "confirm_known_chain"},
                    "expected_patch": {"chain_identity.status": "confirmed"},
                },
                {
                    "id": "reenter_chain",
                    "label": "Re-enter",
                    "value": "reenter_chain",
                    "action": {"type": "answer_pending", "answer": "reenter_chain"},
                    "expected_patch": {"pending_question.id": "chain"},
                },
            ],
            "queue_barrier": True,
        }
        state["active_group"] = "chain_identity"
        state["last_user_input"] = "Take me to the RPC workload settings first."

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{
                "type": "change_group",
                "group": "workload_rpc",
                "navigation_explicit": True,
                "source_evidence": "Take me to the RPC workload settings first.",
                "confidence": "high",
            }]}
            detoured = process_turn(state)

        self.assertEqual(detoured["pending_question"]["id"], "unknown_chain_identity_confirm")
        self.assertEqual(detoured["active_group"], "chain_identity")
        self.assertEqual(detoured["control"]["deferred_group"], "workload_rpc")
        self.assertFalse(detoured.get("rpc_mode"))

        detoured["last_user_input"] = "Yes, the configured BSC chain is what I meant."
        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.side_effect = _admitted_mock_resolver({"actions": [{
                "type": "answer_pending",
                "selected_value": "confirm_known_chain",
                "source_evidence": "Yes, the configured BSC chain is what I meant.",
                "pending_option_semantic_verified": True,
                "semantic_purpose_verified": True,
                "confidence": "high",
            }]})
            resumed = process_turn(detoured)

        self.assertEqual(resumed["chain_identity"]["canonical"], "bsc")
        self.assertEqual(resumed["chain_identity"]["status"], "confirmed")
        self.assertEqual(resumed["pending_question"]["id"], "rpc_mode")
        self.assertEqual(resumed["active_group"], "workload_rpc")
        self.assertNotIn("deferred_group", resumed.get("control") or {})

    def test_rpc_mode_action_is_deferred_behind_unresolved_chain(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "mystery",
            "canonical": "mystery",
            "status": "needs_identity_confirmation",
            "case": "unknown",
        }
        state["active_group"] = "chain_identity"
        state["last_user_input"] = "Use single mode."

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{
                "type": "set_rpc_mode",
                "rpc_mode": "single",
                "mutation_explicit": True,
                "source_evidence": "single",
                "confidence": "high",
            }]}
            result = process_turn(state)

        self.assertFalse(result.get("rpc_mode"))
        self.assertEqual(result["pending_question"]["group"], "chain_identity")
        self.assertEqual(result["control"]["deferred_group"], "workload_rpc")
        self.assertTrue(any(item.get("type") == "set_rpc_mode" for item in result.get("action_queue") or []))

    def test_unknown_chain_identity_grounds_with_google_search_unconditionally(self) -> None:
        """B.1 (known-issues.md): `needs_google_search` -- a flag the

        chain-identity resolver used to set in its JSON response -- was never
        read anywhere in the repo; nothing acted on it, even though
        `web_research.google_search_available` (a *different* flag) is
        correctly wired up (`#48`/`#49`). Confirmed framework design: adding a
        new chain always re-verifies with a fresh google_search once it is
        available, unconditionally -- not gated by the underlying LLM's own
        confidence (even a `confidence: high` resolution still gets searched).
        The `needs_google_search` field was removed from the resolver
        schema/prompt entirely (an unused self-judgment gate does not belong
        in the framework once the real trigger is "search whenever
        available") rather than left as a second, unused decision point.
        Fixed by calling the same `run_google_search_grounding()` `#48`/`#49`
        already wired into the sync-observe client-setup handoff a second
        time here (this is the "new chain identity" call site;
        `extract_rpc_schema_from_evidence`'s "adding a custom RPC method"
        call site gets the identical treatment) whenever search is actually
        available, surfacing the grounded evidence in the confirmation
        prompt shown to the user.
        """

        from unittest.mock import patch

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state
        from agent.llm.search_grounding import SearchGroundingResult

        def _mk() -> dict:
            state = new_state("unit-thread")
            state["target_mode"] = "fake-node"
            state["workflow_mode"] = "rpc_benchmark"
            state["web_research"] = {"google_search_available": True}
            state["pending_question"] = {
                "id": "chain",
                "group": "chain_identity",
                "kind": "chain",
                "field": "chain",
                "manual_input_allowed": True,
            }
            state["last_user_input"] = "newlychain"
            return state

        # High confidence, no `needs_google_search` field at all -- search
        # must still run, proving it is not gated by that removed field.
        resolution = {
            "chain_exists": True,
            "canonical_chain_name": "newlychain",
            "adapter_family": "jsonrpc",
            "confidence": "high",
        }

        # search available -> grounding is actually called and shown.
        with (
            patch("agent.harness.domains.chain_identity.resolve_unknown_chain_identity", return_value=dict(resolution)),
            patch(
                "agent.harness.domains.chain_identity.run_google_search_grounding",
                return_value=SearchGroundingResult(available=True, query="q", text_summary="NewlyChain is a real L1, jsonrpc-compatible."),
            ) as grounding,
        ):
            result = process_turn(_mk())

        grounding.assert_called_once()
        prompt = result["pending_question"]["prompt"]
        self.assertIn("NewlyChain is a real L1", prompt)

        # search unavailable -> grounding is never called, no crash, and the
        # confirmation prompt still renders normally without a search line.
        with (
            patch("agent.harness.domains.chain_identity.resolve_unknown_chain_identity", return_value=dict(resolution)),
            patch("agent.harness.domains.chain_identity.run_google_search_grounding") as grounding2,
        ):
            state2 = _mk()
            state2["web_research"] = {"google_search_available": False}
            result2 = process_turn(state2)

        grounding2.assert_not_called()
        self.assertEqual(result2["pending_question"]["id"], "unknown_chain_identity_confirm")

    def test_unknown_chain_candidate_accepts_yes_and_advances(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "sola",
            "canonical": "sola",
            "proposed_known_chain": "solana",
            "status": "needs_known_chain_confirmation",
            "case": "known_candidate",
        }
        state["pending_question"] = {
            "id": "unknown_chain_identity_confirm",
            "group": "chain_identity",
            "kind": "numbered_choice",
            "field": "unknown_chain_decision",
            "options": [
                {"label": "Use `solana`", "value": "confirm_known_chain"},
                {"label": "No, re-enter chain name", "value": "reenter_chain"},
                {"label": "This is another real chain; choose protocol", "value": "choose_protocol"},
            ],
            "manual_input_allowed": False,
        }
        state["last_user_input"] = "Y"
        result = process_turn(state)

        self.assertEqual(result["chain_identity"]["status"], "confirmed")
        self.assertEqual(result["confirmed_config"]["BLOCKCHAIN_NODE"], "solana")
        self.assertNotEqual(result["pending_question"]["id"], "chain")

    def test_unknown_chain_pending_requires_protocol_confirmation_on_next_turn(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "abcd",
            "canonical": "abcd",
            "adapter_family": "unknown",
            "status": "needs_identity_confirmation",
            "case": "unknown",
        }
        state["pending_question"] = {
            "id": "unknown_chain_identity_confirm",
            "group": "chain_identity",
            "kind": "numbered_choice",
            "field": "unknown_chain_decision",
            "options": [
                {"label": "真实链，继续确认协议", "value": "choose_protocol"},
                {"label": "我要重新输入链名", "value": "reenter_chain"},
            ],
            "manual_input_allowed": False,
        }
        state["last_user_input"] = "它应该是 EVM/jsonrpc，先按这个协议验证"

        with (
            patch(
                "agent.harness.coordinator.resolve_action_queue",
                side_effect=_admitted_mock_resolver({
                    "actions": [
                        {"type": "answer_pending", "answer": "这是真实链", "selected_value": "choose_protocol", "source_evidence": "它应该是 EVM/jsonrpc", "pending_option_semantic_verified": True, "semantic_purpose_verified": True, "confidence": "high"},
                        {"type": "choose_adapter_family", "adapter_family": "jsonrpc", "confidence": "high"},
                    ]
                }),
            ),
        ):
            result = process_turn(state)

        self.assertEqual(result["chain_identity"]["canonical"], "abcd")
        self.assertEqual(result["chain_identity"]["status"], "needs_protocol_confirmation")
        self.assertEqual(result["pending_question"]["id"], "adapter_family_confirm")
        self.assertNotIn("endpoint", "\n".join(result.get("visible_response") or []).lower())

    def test_unknown_chain_protocol_hint_with_requirements_question_keeps_endpoint_pending(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "sola",
            "canonical": "sola",
            "adapter_family": "unknown",
            "status": "needs_protocol_confirmation",
            "case": "unknown",
        }
        state["pending_question"] = {
            "id": "adapter_family_confirm",
            "group": "chain_identity",
            "kind": "numbered_choice",
            "field": "adapter_family",
            "options": [
                {"label": "jsonrpc / EVM", "value": "jsonrpc"},
                {"label": "substrate", "value": "substrate"},
                {"label": "rest", "value": "rest"},
                {"label": "tendermint", "value": "tendermint"},
                {"label": "bitcoin_jsonrpc", "value": "bitcoin_jsonrpc"},
                {"label": "hedera_dual", "value": "hedera_dual"},
                {"label": "不属于以上协议族 / 不确定", "value": "unsupported"},
            ],
            "manual_input_allowed": False,
        }
        state["last_user_input"] = "它应该是 EVM/jsonrpc，但我现在没有 endpoint，先告诉我需要准备什么"

        with (
            patch(
                "agent.harness.coordinator.resolve_action_queue",
                side_effect=_admitted_mock_resolver({"actions": [
                    {"type": "answer_pending", "answer": "jsonrpc", "selected_value": "jsonrpc", "source_evidence": "EVM/jsonrpc", "pending_option_semantic_verified": True, "semantic_purpose_verified": True, "confidence": "high"},
                    {"type": "answer_opening_question", "topic": "config_explanation", "subject": "new_chain_endpoint", "confidence": "high"},
                ]}),
            ),
        ):
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertEqual(result["chain_identity"]["adapter_family"], "jsonrpc")
        self.assertEqual(result["chain_identity"]["status"], "existing_family_needs_endpoint")
        self.assertEqual(result["pending_question"]["id"], "new_chain_endpoint")
        self.assertIn("可访问", text)
        self.assertIn("method", text)
        self.assertIn("只作为", text)
        self.assertEqual(text.count("请提供可访问的 RPC endpoint"), 1)

    def test_unknown_chain_candidate_accepts_explicit_candidate_name(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "sola",
            "canonical": "sola",
            "proposed_known_chain": "solana",
            "status": "needs_known_chain_confirmation",
            "case": "known_candidate",
        }
        state["pending_question"] = {
            "id": "unknown_chain_identity_confirm",
            "group": "chain_identity",
            "kind": "numbered_choice",
            "field": "unknown_chain_decision",
            "options": [
                {"label": "Use `solana`", "value": "confirm_known_chain"},
                {"label": "No, re-enter chain name", "value": "reenter_chain"},
                {"label": "This is another real chain; choose protocol", "value": "choose_protocol"},
            ],
            "manual_input_allowed": False,
        }
        state["last_user_input"] = "solana"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "choose_chain", "chain_text": "solana", "confidence": "high"}]}
            result = process_turn(state)

        self.assertEqual(result["chain_identity"]["status"], "confirmed")
        self.assertEqual(result["confirmed_config"]["BLOCKCHAIN_NODE"], "solana")
        self.assertNotEqual(result["pending_question"]["id"], "chain")

    def test_chain_selection_prompt_distinguishes_initial_and_replacement(self) -> None:
        from agent.harness.contracts import ActionProposal
        from agent.harness.domains.chain_rpc import apply_chain_rpc_action
        from agent.harness.state import new_state

        initial = new_state("initial-chain", language="en")
        action = ActionProposal("request-chain", "request_chain_selection", {}, "high")
        initial_outcome = apply_chain_rpc_action(initial, action)
        initial_result = _commit_result(initial, initial_outcome, owner="chain_rpc")
        self.assertEqual(initial_result["pending_question"]["id"], "chain_change_input")
        self.assertEqual(
            set(initial_result["pending_question"]["accepted_action_types"]),
            {"answer_pending", "choose_chain", "change_chain"},
        )
        self.assertEqual(
            initial_result["pending_question"]["manual_action"],
            {"type": "choose_chain", "value_argument": "chain_text"},
        )
        self.assertIn("chain name to test", initial_result["pending_question"]["prompt"])
        self.assertNotIn("replacement", initial_result["pending_question"]["prompt"])

        replacement = new_state("replacement-chain", language="en")
        replacement["chain_identity"] = {
            "raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"
        }
        replacement["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        replacement_outcome = apply_chain_rpc_action(replacement, action)
        replacement_result = _commit_result(replacement, replacement_outcome, owner="chain_rpc")
        self.assertIn("replacement chain", replacement_result["pending_question"]["prompt"])
        self.assertIn("`bsc`", replacement_result["pending_question"]["prompt"])
        self.assertEqual(
            set(replacement_result["pending_question"]["accepted_action_types"]),
            {"answer_pending", "choose_chain", "change_chain"},
        )
        self.assertEqual(
            replacement_result["pending_question"]["manual_action"],
            {"type": "change_chain", "value_argument": "chain_text"},
        )

    def test_free_form_question_factories_declare_one_typed_manual_owner(self) -> None:
        from agent.harness.questions import choice_question, manual_question

        manual = manual_question(
            "provider_deployment",
            "CLOUD_REGION",
            "Enter a region.",
            field="CLOUD_REGION",
        )
        self.assertEqual(
            manual["manual_action"],
            {"type": "answer_pending", "value_argument": "answer"},
        )

        choice_with_manual = choice_question(
            "network",
            "NETWORK_INTERFACE",
            "Choose an interface.",
            field="NETWORK_INTERFACE",
            manual_input_allowed=True,
            options=[{"label": "eth0", "value": "eth0"}],
        )
        self.assertEqual(
            choice_with_manual["manual_action"],
            {"type": "answer_pending", "value_argument": "answer"},
        )

        finite_choice = choice_question(
            "accounts_disk",
            "has_accounts_device",
            "Separate accounts disk?",
            field="has_accounts_device",
            options=[{"label": "Y", "value": True}, {"label": "N", "value": False}],
        )
        self.assertNotIn("manual_action", finite_choice)

    def test_free_form_question_factory_rejects_invalid_owner_contract(self) -> None:
        from agent.harness.questions import manual_question

        with self.assertRaisesRegex(ValueError, "unknown manual_action type"):
            manual_question(
                "provider_deployment",
                "CLOUD_REGION",
                "Enter a region.",
                field="CLOUD_REGION",
                manual_action={"type": "not_an_action", "value_argument": "answer"},
            )
        with self.assertRaisesRegex(ValueError, "writable value_argument"):
            manual_question(
                "provider_deployment",
                "CLOUD_REGION",
                "Enter a region.",
                field="CLOUD_REGION",
                manual_action={"type": "answer_pending", "value_argument": "missing"},
            )





    def test_case2_compound_rpc_evidence_preserves_endpoint_and_requests_schema_review(self) -> None:
        from agent.harness.domains.chain_rpc import apply_chain_rpc_answer
        from agent.harness.state import new_state

        state = new_state("compound-rpc", language="en")
        state["chain_identity"] = {
            "raw": "sola",
            "canonical": "sola",
            "adapter_family": "jsonrpc",
            "status": "existing_family_needs_method",
            "case": "case2",
        }
        state["endpoint_evidence"] = {
            "candidate_endpoint": "http://geth-dev:8545",
            "candidate_endpoint_ready": True,
        }
        evidence = (
            "The docs example URL is evidence only.\n"
            "curl https://docs-provider.invalid/example --data "
            "'{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"eth_chainId\",\"params\":[]}'\n"
            "Response: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":\"0x539\"}"
        )

        extracted = {
            "status": "draft",
            "evidence_kind": "mixed",
            "transport": "jsonrpc",
            "method": "wrong_model_method",
            "params": [],
            "params_json": ["wrong"],
            "response_summary": "hexadecimal chain id string",
            "response_fields": [{"name": "result", "type": "string", "meaning": "chain id"}],
            "endpoint_url": "https://docs-provider.invalid/example",
            "confidence": "high",
            "evidence_summary": "request and response supplied by the user",
        }
        with patch("agent.harness.domains.rpc_endpoint.extract_rpc_schema_from_evidence", return_value=extracted):
            outcome = apply_chain_rpc_answer(
                state,
                {
                    "id": "new_chain_method",
                    "group": "endpoint_process",
                    "field": "new_chain_method",
                    "kind": "evidence",
                    "manual_input_allowed": True,
                },
                evidence,
                evidence,
            )
            state = _commit_result(state, outcome, owner="chain_rpc")

        self.assertEqual(_catalog_draft(state)["method"], "eth_chainId")
        self.assertEqual(state["chain_identity"]["status"], "existing_family_schema_needs_confirmation")
        self.assertEqual(state["pending_question"]["id"], "new_chain_schema_confirm")
        self.assertEqual(state["endpoint_evidence"]["candidate_endpoint"], "http://geth-dev:8545")
        self.assertEqual(state["chain_identity"]["schema_evidence"], evidence)
        self.assertEqual(_catalog_draft(state)["params_json"], [])
        self.assertEqual(_catalog_draft(state)["response_summary"], "hexadecimal chain id string")
        self.assertIn("hexadecimal chain id string", state["pending_question"]["prompt"])
        self.assertIn("Evidence summary:", state["pending_question"]["prompt"])
        self.assertNotIn("google_search evidence:", state["pending_question"]["prompt"])

        searched = dict(_catalog_draft(state))
        searched["search_result"] = {"available": True}
        from agent.harness.domains.chain_rpc_support import _schema_confirmation_prompt
        searched_prompt = _schema_confirmation_prompt("en", searched)
        self.assertIn("google_search evidence:", searched_prompt)

    def test_unknown_chain_candidate_real_chain_choice_does_not_keep_possible_known_canonical(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "sola",
            "canonical": "sola",
            "proposed_known_chain": "solana",
            "status": "needs_known_chain_confirmation",
            "case": "known_candidate",
            "llm_resolution": {"possible_known_chain": "solana", "adapter_family": "unknown"},
        }
        state["pending_question"] = {
            "id": "unknown_chain_identity_confirm",
            "group": "chain_identity",
            "kind": "numbered_choice",
            "field": "unknown_chain_decision",
            "options": [
                {"label": "使用 `solana`", "value": "confirm_known_chain"},
                {"label": "不是，重新输入链名", "value": "reenter_chain"},
                {"label": "这是另一条真实链，继续确认协议", "value": "choose_protocol"},
            ],
            "manual_input_allowed": False,
        }
        state["last_user_input"] = "3"

        result = process_turn(state)

        self.assertEqual(result["chain_identity"]["raw"], "sola")
        self.assertEqual(result["chain_identity"]["canonical"], "sola")
        self.assertEqual(result["chain_identity"]["status"], "needs_protocol_confirmation")
        self.assertEqual(result["pending_question"]["id"], "adapter_family_confirm")

    def test_free_form_chain_change_does_not_fill_pending_region(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "solana", "canonical": "solana", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "solana"}
        state["pending_question"] = {
            "id": "CLOUD_REGION",
            "group": "provider_deployment",
            "kind": "manual_value",
            "field": "CLOUD_REGION",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "change to ethereum"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "change_chain", "chain_text": "ethereum", "source_evidence": "change to ethereum", "confidence": "high"}]}
            result = process_turn(state)

        self.assertNotIn("CLOUD_REGION", result.get("confirmed_config", {}))
        self.assertEqual(result["pending_question"]["id"], "chain_change_confirm")

        result["last_user_input"] = "y"
        result = process_turn(result)
        self.assertNotIn("CLOUD_REGION", result.get("confirmed_config", {}))
        self.assertEqual(result["chain_identity"]["canonical"], "ethereum")
        self.assertEqual(result["confirmed_config"]["BLOCKCHAIN_NODE"], "ethereum")

    def test_target_mode_change_interrupts_manual_value_and_requires_confirmation(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["pending_question"] = {
            "id": "CLOUD_REGION",
            "group": "provider_deployment",
            "kind": "manual_value",
            "field": "CLOUD_REGION",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "use real-node instead"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "choose_target_mode", "target_mode": "real-node", "target_mode_explicit": True, "source_evidence": "real-node", "confidence": "high"}]}
            result = process_turn(state)

        self.assertNotIn("CLOUD_REGION", result.get("confirmed_config", {}))
        self.assertEqual(result["pending_question"]["id"], "target_mode_change_confirm")

        result["last_user_input"] = "Y"
        result = process_turn(result)

        self.assertEqual(result["target_mode"], "real-node")
        self.assertNotIn("CLOUD_REGION", result.get("confirmed_config", {}))

    def test_target_mode_change_candidate_survives_langgraph_checkpoint(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime
        from agent.harness.state import new_state

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoints.sqlite"
            runtime = AnyChainGraphRuntime(thread_id="unit-thread", checkpoint_path=checkpoint_path)
            initial = new_state("unit-thread")
            initial["target_mode"] = "fake-node"
            initial["workflow_mode"] = "rpc_benchmark"
            initial["active_group"] = "provider_deployment"
            initial["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
            initial["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
            initial["pending_question"] = {
                "id": "CLOUD_REGION",
                "group": "provider_deployment",
                "kind": "manual_value",
                "field": "CLOUD_REGION",
                "manual_input_allowed": True,
            }
            runtime.graph.update_state({"configurable": {"thread_id": "unit-thread"}}, initial)

            with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
                resolver.return_value = {"actions": [{"type": "choose_target_mode", "target_mode": "real-node", "target_mode_explicit": True, "source_evidence": "real-node", "confidence": "high"}]}
                state = runtime.invoke("use real-node instead", language="en")

            self.assertEqual(state["pending_question"]["id"], "target_mode_change_confirm")
            self.assertEqual(state["target_mode_change_candidate"], "real-node")

            state = runtime.invoke("Y", language="en")

        self.assertEqual(state["target_mode"], "real-node")
        self.assertNotIn("CLOUD_REGION", state.get("confirmed_config", {}))

    def test_manual_value_rejects_bare_yes_without_default_value(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "solana", "canonical": "solana", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "solana"}
        state["pending_question"] = {
            "id": "CLOUD_REGION",
            "group": "provider_deployment",
            "kind": "manual_value",
            "field": "CLOUD_REGION",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "y"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "unknown", "confidence": "low"}]}
            result = process_turn(state)

        self.assertNotIn("CLOUD_REGION", result.get("confirmed_config", {}))
        self.assertEqual(result["pending_question"]["id"], "CLOUD_REGION")
        self.assertIn("does not look like", result["visible_response"][0])

    def test_free_form_pending_choice_can_be_resolved_by_llm_mapper(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {
            "BLOCKCHAIN_NODE": "bsc",
            "CLOUD_REGION": "us-1",
            "CLOUD_ZONE": "us-1-z",
            "MACHINE_TYPE": "n2",
            "LEDGER_DEVICE": "vda",
            "DATA_VOL_TYPE": "hyperdisk-balanced",
            "DATA_VOL_SIZE": "926",
            "DATA_VOL_MAX_IOPS": "20000",
            "DATA_VOL_MAX_THROUGHPUT": "1000",
            "has_accounts_device": False,
            "NETWORK_INTERFACE": "eth0",
            "NETWORK_MAX_BANDWIDTH_GBPS": "100",
        }
        state["rpc_mode"] = "mixed"
        state["custom_rpc"] = {"status": "needs_scope", "method": "eth_blockNumber"}
        state["pending_question"] = {
            "id": "custom_rpc_scope",
            "group": "workload_rpc",
            "kind": "numbered_choice",
            "field": "custom_rpc_scope",
            "options": [
                {"label": "Use this method as single workload only", "value": "single_replace"},
                {"label": "Use only my custom methods in mixed", "value": "mixed_replace"},
                {"label": "Keep template defaults in mixed and add this method", "value": "mixed_add"},
            ],
            "manual_input_allowed": False,
        }
        state["last_user_input"] = "replace defaults"

        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            side_effect=_admitted_mock_resolver({"actions": [{
                "type": "answer_pending",
                "answer": "replace defaults",
                "selected_value": "mixed_replace",
                "source_evidence": "replace defaults",
                "pending_option_semantic_verified": True,
                "semantic_purpose_verified": True,
                "confidence": "high",
            }]}),
        ):
            result = process_turn(state)

        self.assertEqual(result["custom_rpc"]["scope"], "mixed_replace")
        self.assertEqual(result["custom_rpc"]["status"], "needs_weights")
        self.assertEqual(result["pending_question"]["id"], "custom_rpc_weights")

    def test_assignment_text_does_not_satisfy_numbered_scope_choice(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {
            "BLOCKCHAIN_NODE": "bsc",
            "CLOUD_REGION": "us-1",
            "CLOUD_ZONE": "us-1-z",
            "MACHINE_TYPE": "n2",
            "LEDGER_DEVICE": "vda",
            "DATA_VOL_TYPE": "hyperdisk-balanced",
            "DATA_VOL_SIZE": "926",
            "DATA_VOL_MAX_IOPS": "20000",
            "DATA_VOL_MAX_THROUGHPUT": "1000",
            "has_accounts_device": False,
            "NETWORK_INTERFACE": "eth0",
            "NETWORK_MAX_BANDWIDTH_GBPS": "100",
        }
        state["rpc_mode"] = "mixed"
        state["custom_rpc"] = {"status": "needs_scope", "method": "eth_blockNumber"}
        state["pending_question"] = {
            "id": "custom_rpc_scope",
            "group": "workload_rpc",
            "kind": "numbered_choice",
            "field": "custom_rpc_scope",
            "options": [
                {"label": "Use this method as single workload only", "value": "single_replace"},
                {"label": "Use only my custom methods in mixed", "value": "mixed_replace"},
                {"label": "Keep template defaults in mixed and add this method", "value": "mixed_add"},
            ],
            "manual_input_allowed": False,
        }
        state["last_user_input"] = "eth_blockNumber=100"

        with patch("agent.harness.coordinator.resolve_action_queue") as queue:
            queue.return_value = {"actions": [{"type": "unknown", "confidence": "low"}]}
            result = process_turn(state)

        self.assertEqual(result["custom_rpc"]["status"], "needs_scope")
        self.assertEqual(result["pending_question"]["id"], "custom_rpc_scope")

    def test_declined_chain_change_keeps_original_chain_and_region_pending(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["pending_question"] = {
            "id": "CLOUD_REGION",
            "group": "provider_deployment",
            "kind": "manual_value",
            "field": "CLOUD_REGION",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "change to ethereum"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "change_chain", "chain_text": "ethereum", "source_evidence": "change to ethereum", "confidence": "high"}]}
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "chain_change_confirm")
        result["last_user_input"] = "n"
        result = process_turn(result)

        self.assertEqual(result["chain_identity"]["canonical"], "bsc")
        self.assertEqual(result["confirmed_config"]["BLOCKCHAIN_NODE"], "bsc")
        self.assertNotIn("CLOUD_REGION", result.get("confirmed_config", {}))
        self.assertEqual(result["pending_question"]["id"], "CLOUD_REGION")

    def test_declined_chain_change_preserves_unrelated_queued_actions(self) -> None:
        from agent.harness.coordinator import _apply_pending_answer
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["active_group"] = "chain_identity"
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "canonical": "bsc",
            "status": "confirmed",
            "change_candidate": {"raw": "ethereum", "interrupted_group": "qps_profile"},
        }
        state["action_queue"] = [{"action_id": "qps-1", "type": "set_qps_mode", "qps_mode": "quick"}]
        question = {
            "id": "chain_change_confirm",
            "group": "chain_identity",
            "kind": "yes_no",
            "field": "chain_change_confirmed",
            "options": [{"label": "Y", "value": True}, {"label": "N", "value": False}],
            "interrupted_group": "qps_profile",
        }

        result = _apply_pending_answer(state, "n", question)

        self.assertEqual(result["chain_identity"]["canonical"], "bsc")
        self.assertEqual([item["action_id"] for item in result["action_queue"]], ["qps-1"])
        self.assertEqual(result["active_group"], "qps_profile")

    def test_keep_current_chain_does_not_invalidate_chain_dependent_state(self) -> None:
        from copy import deepcopy
        from agent.harness.coordinator import _apply_pending_answer
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["active_group"] = "chain_identity"
        state["target_mode"] = "real-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "canonical": "solana",
            "status": "confirmed",
            "change_candidate": {
                "raw": "sola",
                "interrupted_group": "qps_profile",
                "resolution": {"possible_known_chain": "solana"},
            },
        }
        state["confirmed_config"] = {
            "BLOCKCHAIN_NODE": "solana",
            "LOCAL_RPC_URL": "http://node:8899",
            "BLOCKCHAIN_PROCESS_NAMES": "solana-validator",
        }
        state["endpoint_evidence"] = {"local_rpc_url_ready": True}
        state["workload"] = {"confirmed": True, "methods": ["getBalance"]}
        state["preflight"] = {"approved": True}
        preserved = deepcopy({key: state[key] for key in ("confirmed_config", "endpoint_evidence", "workload", "preflight")})
        question = {
            "id": "chain_change_confirm",
            "group": "chain_identity",
            "kind": "numbered_choice",
            "field": "chain_change_confirmed",
            "options": [{"label": "Keep solana", "value": "confirm_known_chain"}],
        }

        result = _apply_pending_answer(state, "1", question)

        self.assertEqual(result["chain_identity"]["canonical"], "solana")
        for key, value in preserved.items():
            self.assertEqual(result[key], value)
        self.assertEqual(result["active_group"], "qps_profile")

    def test_workload_menu_options_are_executable_typed_transitions(self) -> None:
        from copy import deepcopy
        from agent.harness.coordinator import _apply_pending_answer, _question_for_group
        from agent.harness.state import new_state

        base = new_state("unit-thread")
        base["turn_index"] = 1
        base["active_group"] = "workload_rpc"
        base["target_mode"] = "fake-node"
        base["workflow_mode"] = "rpc_benchmark"
        base["chain_identity"] = {"canonical": "solana", "status": "confirmed", "case": "known"}
        base["confirmed_config"] = {"BLOCKCHAIN_NODE": "solana"}

        rpc_question = _question_for_group(base, "workload_rpc")
        self.assertEqual(rpc_question["contract_version"], 1)
        state = _apply_pending_answer(deepcopy(base), "1", rpc_question)
        self.assertEqual(state["rpc_mode"], "single")
        self.assertEqual(state["pending_question"]["id"], "workload_confirm")

        default_question = state["pending_question"]
        state["turn_index"] = 2
        default_result = _apply_pending_answer(deepcopy(state), "1", default_question)
        self.assertTrue(default_result["workload"]["confirmed"])
        self.assertEqual(default_result["workload"]["methods"], ["getAccountInfo"])

        state["turn_index"] = 3
        custom_result = _apply_pending_answer(deepcopy(state), "2", default_question)
        self.assertEqual(custom_result["custom_rpc"]["status"], "needs_endpoint")
        self.assertEqual(custom_result["active_group"], "endpoint_process")

        mixed = deepcopy(base)
        mixed["rpc_mode"] = "mixed"
        mixed_question = _question_for_group(mixed, "workload_rpc")
        self.assertEqual([item["value"] for item in mixed_question["options"]], ["default", "custom_rpc", "weights", "change_target"])
        mixed["turn_index"] = 4
        weights_result = _apply_pending_answer(mixed, "3", mixed_question)
        self.assertEqual(weights_result["custom_rpc"]["status"], "needs_weights")

        change_state = deepcopy(state)
        change_state["turn_index"] = 5
        change_result = _apply_pending_answer(change_state, "3", default_question)
        self.assertEqual(change_result["pending_question"]["id"], "target_change_scope")
        self.assertEqual(change_result["chain_identity"]["canonical"], "solana")

    def test_choose_chain_action_still_confirms_when_chain_already_exists(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["pending_question"] = {
            "id": "CLOUD_REGION",
            "group": "provider_deployment",
            "kind": "manual_value",
            "field": "CLOUD_REGION",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "ethereum"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "choose_chain", "chain_text": "ethereum", "source_evidence": "ethereum", "confidence": "high"}]}
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "chain_change_confirm")
        self.assertEqual(result["chain_identity"]["canonical"], "bsc")

    def test_qps_rejecting_defaults_enters_adjustment_subflow(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "solana", "canonical": "solana", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {
            "BLOCKCHAIN_NODE": "solana",
            "CLOUD_REGION": "us-1",
            "CLOUD_ZONE": "us-1-z",
            "MACHINE_TYPE": "n2",
            "LEDGER_DEVICE": "vda",
            "DATA_VOL_TYPE": "hyperdisk-balanced",
            "DATA_VOL_SIZE": "926",
            "DATA_VOL_MAX_IOPS": "20000",
            "DATA_VOL_MAX_THROUGHPUT": "1000",
            "has_accounts_device": False,
            "NETWORK_INTERFACE": "eth0",
            "NETWORK_MAX_BANDWIDTH_GBPS": "100",
        }
        state["rpc_mode"] = "single"
        state["workload"] = {"confirmed": True, "choice": "default"}
        state["pending_question"] = {
            "id": "benchmark_mode",
            "group": "qps_profile",
            "kind": "numbered_choice",
            "field": "benchmark_mode",
            "options": [{"label": "quick", "value": "quick"}, {"label": "standard", "value": "standard"}],
            "manual_input_allowed": False,
        }
        state["last_user_input"] = "1"
        state = process_turn(state)
        self.assertEqual(state["pending_question"]["id"], "qps_profile_confirm")

        state["last_user_input"] = "n"
        state = process_turn(state)
        self.assertEqual(state["pending_question"]["id"], "qps_adjust_field")

        state["last_user_input"] = "1"
        state = process_turn(state)
        self.assertEqual(state["pending_question"]["id"], "qps_adjust_value")

    def test_preflight_yes_calls_execution_application_service(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["pending_question"] = {
            "id": "preflight_smoke_confirm",
            "group": "preflight_smoke_execution",
            "kind": "yes_no",
            "field": "preflight_smoke_confirmed",
            "options": [{"label": "Y", "value": True}, {"label": "N", "value": False}],
            "manual_input_allowed": False,
        }
        state["last_user_input"] = "y"

        prepared = {
            "data": {
                "plan": {},
                "plan_file": "/tmp/unit-plan.json",
                "preflight": {"passed": True},
            },
            "evidence_paths": [],
        }
        service_result = unittest.mock.MagicMock()
        service_result.to_dict.return_value = {
            "status": "ok",
            "data": {"job": {"job_id": "job-unit", "status": "submitted"}},
            "warnings": [],
        }
        with (
            patch(
                "agent.harness.domains.execution_runtime._prepare_benchmark_with_runtime_contract",
                return_value=prepared,
            ),
            patch(
                "agent.harness.domains.execution_runtime.execution_service.execute",
                return_value=service_result,
            ) as execute,
        ):
            result = process_turn(state)

        execute.assert_called_once()
        self.assertTrue(any("submitted" in item.lower() for item in result["visible_response"]))

    def test_existing_chain_custom_rpc_validates_endpoint_method_and_weights(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {
            "BLOCKCHAIN_NODE": "bsc",
            "CLOUD_REGION": "us-1",
            "CLOUD_ZONE": "us-1-z",
            "MACHINE_TYPE": "n2",
            "LEDGER_DEVICE": "vda",
            "DATA_VOL_TYPE": "hyperdisk-balanced",
            "DATA_VOL_SIZE": "926",
            "DATA_VOL_MAX_IOPS": "20000",
            "DATA_VOL_MAX_THROUGHPUT": "1000",
            "has_accounts_device": False,
            "NETWORK_INTERFACE": "eth0",
            "NETWORK_MAX_BANDWIDTH_GBPS": "100",
        }
        state["rpc_mode"] = "mixed"
        state["pending_question"] = {
            "id": "workload_confirm",
            "group": "workload_rpc",
            "kind": "numbered_choice",
            "field": "workload_choice",
            "options": [
                {"label": "Use defaults", "value": "default"},
                {"label": "Add custom RPC method", "value": "custom_rpc"},
            ],
            "manual_input_allowed": False,
        }
        state["last_user_input"] = "2"
        state = process_turn(state)
        self.assertEqual(state["custom_rpc"]["status"], "needs_endpoint")

        ok_probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}
        with patch("agent.harness.domains.rpc_endpoint.validate_rpc_endpoint", return_value=ok_probe):
            state["last_user_input"] = "https://example.invalid/rpc"
            state = process_turn(state)
            self.assertEqual(state["custom_rpc"]["status"], "needs_method")

            state["last_user_input"] = "eth_blockNumber"
            state = process_turn(state)
            self.assertEqual(state["custom_rpc"]["status"], "needs_schema_evidence")

            state["last_user_input"] = "[]"
            with patch(
                "agent.harness.coordinator.resolve_action_queue",
                return_value={"actions": [{
                    "type": "rpc_catalog_command",
                    "catalog_command": "append_evidence",
                    "rpc_schema_evidence": state["last_user_input"],
                    "source_evidence": state["last_user_input"],
                    "confidence": "high",
                }]},
            ), patch(
                "agent.harness.domains.rpc_endpoint.extract_rpc_schema_from_evidence",
                return_value={
                    "status": "draft",
                    "evidence_kind": "request_schema",
                    "method": "eth_blockNumber",
                    "params": [],
                    "params_json": [],
                    "response_summary": "unknown",
                    "confidence": "high",
                },
            ):
                state = process_turn(state)
            self.assertEqual(state["custom_rpc"]["status"], "schema_needs_confirmation")
            self.assertEqual(state["pending_question"]["id"], "custom_rpc_schema_confirm")

            state = self._confirm_catalog_through_graph(state, process_turn)
            self.assertEqual(state["custom_rpc"]["status"], "method_validated_next")
            self.assertEqual(state["pending_question"]["id"], "custom_rpc_continue")

        state["last_user_input"] = "2"
        state = process_turn(state)
        self.assertEqual(state["custom_rpc"]["status"], "needs_scope")

        state["last_user_input"] = "2"
        state = process_turn(state)
        self.assertEqual(state["custom_rpc"]["status"], "needs_weights")

        state["last_user_input"] = "eth_blockNumber=70,eth_getBalance=20"
        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            side_effect=_admitted_mock_resolver({"actions": [{
                "type": "answer_pending",
                "answer": state["last_user_input"],
                "source_evidence": state["last_user_input"],
                "semantic_purpose_verified": True,
                "confidence": "high",
            }]}),
        ):
            state = process_turn(state)
        self.assertEqual(state["custom_rpc"]["status"], "needs_weights")
        self.assertEqual(state["pending_question"]["id"], "custom_rpc_weights")

        state["last_user_input"] = "eth_blockNumber=70,eth_getBalance=30"
        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            side_effect=_admitted_mock_resolver({"actions": [{
                "type": "answer_pending",
                "answer": state["last_user_input"],
                "source_evidence": state["last_user_input"],
                "semantic_purpose_verified": True,
                "confidence": "high",
            }]}),
        ):
            state = process_turn(state)
        self.assertEqual(state["custom_rpc"]["status"], "validated")
        self.assertTrue(state["workload"]["confirmed"])
        self.assertEqual(state["workload"]["mixed_weights"], {"eth_blockNumber": 70, "eth_getBalance": 30})

    def test_custom_rpc_scope_single_replace_with_multiple_methods_asks_disambiguation(self) -> None:
        """Phase 4 authorized change: Case 1 now asks which validated method

        to use as single (mirroring Case 2's `new_chain_single_method`
        behavior, per
        `test_new_chain_workload_scope_single_replace_with_multiple_methods_asks_new_chain_single_method`)
        instead of silently taking the first validated method.
        """

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
        state["custom_rpc"] = {
            "status": "needs_scope",
            "validated_methods": [
                {"method": "eth_blockNumber", "params": [], "evidence_file": ".agent/evidence/one.json"},
                {"method": "eth_getBalance", "params": [], "evidence_file": ".agent/evidence/two.json"},
            ],
        }
        state["pending_question"] = {
            "id": "custom_rpc_scope",
            "group": "endpoint_process",
            "kind": "numbered_choice",
            "field": "custom_rpc_scope",
            "options": [
                {"label": "Use this method as single workload only", "value": "single_replace"},
                {"label": "Use only my custom methods in mixed", "value": "mixed_replace"},
                {"label": "Keep template defaults in mixed and add this method", "value": "mixed_add"},
            ],
        }

        state["last_user_input"] = "1"
        state = process_turn(state)
        self.assertEqual(state["custom_rpc"]["status"], "needs_single_method")
        self.assertEqual(state["pending_question"]["id"], "custom_rpc_single_method")
        option_values = {option["value"] for option in state["pending_question"]["options"]}
        self.assertEqual(option_values, {"eth_blockNumber", "eth_getBalance"})

        state["last_user_input"] = "2"
        state = process_turn(state)
        self.assertEqual(state["custom_rpc"]["status"], "validated")
        self.assertEqual(state["rpc_mode"], "single")
        self.assertEqual(state["workload"]["methods"], ["eth_getBalance"])

    def test_typed_single_replace_request_with_multiple_methods_asks_disambiguation(self) -> None:
        """A typed LLM workload request must never select an arbitrary method."""

        from agent.harness.domains.chain_rpc import apply_chain_rpc_answer, question_for_chain_rpc
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "endpoint_process"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
        state["custom_rpc"] = {
            "status": "schema_needs_confirmation",
            "method": "eth_getBalance",
            "endpoint": "https://example.invalid/rpc",
            "endpoint_ready": True,
            "schema_draft": {"method": "eth_getBalance", "params": [], "params_json": [], "response_summary": "hex quantity"},
            "validated_methods": [
                {"method": "eth_blockNumber", "params": [], "evidence_file": ".agent/evidence/one.json"},
            ],
            "requested_workload": {"scope": "single_replace", "finish_methods": True},
        }
        question = {"id": "custom_rpc_schema_confirm", "group": "endpoint_process", "field": "custom_rpc_schema_confirm"}
        probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/two.json"}

        with patch("agent.harness.domains.rpc_endpoint.validate_rpc_endpoint", return_value=probe):
            outcome = apply_chain_rpc_answer(state, question, True, "Y")
            result = _commit_result(state, outcome, owner="chain_rpc")
            result = self._confirm_catalog_through_domain(result)
        self.assertEqual(result["custom_rpc"]["status"], "needs_single_method")
        self.assertNotIn("requested_workload", result["custom_rpc"])
        # It must NOT have silently committed a single arbitrary method.
        self.assertNotEqual(result.get("rpc_mode"), "single")
        self.assertFalse((result.get("workload") or {}).get("confirmed"))

        # The needs_single_method status must surface the disambiguation
        # question with both validated methods as options.
        question = question_for_chain_rpc(result, "endpoint_process")
        self.assertEqual(question["id"], "custom_rpc_single_method")
        option_values = {option["value"] for option in question["options"]}
        self.assertEqual(option_values, {"eth_blockNumber", "eth_getBalance"})

    def test_typed_single_replace_request_with_one_method_commits_directly(self) -> None:
        from agent.harness.domains.chain_rpc import apply_chain_rpc_answer
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["custom_rpc"] = {
            "status": "schema_needs_confirmation",
            "method": "eth_blockNumber",
            "endpoint": "https://example.invalid/rpc",
            "endpoint_ready": True,
            "schema_draft": {"method": "eth_blockNumber", "params": [], "params_json": [], "response_summary": "hex block number"},
            "requested_workload": {"scope": "single_replace", "finish_methods": True},
        }
        question = {"id": "custom_rpc_schema_confirm", "group": "endpoint_process", "field": "custom_rpc_schema_confirm"}
        probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/one.json"}

        with patch("agent.harness.domains.rpc_endpoint.validate_rpc_endpoint", return_value=probe):
            outcome = apply_chain_rpc_answer(state, question, True, "Y")
            result = _commit_result(state, outcome, owner="chain_rpc")
            result = self._confirm_catalog_through_domain(result)
        self.assertEqual(result["custom_rpc"]["status"], "validated")
        self.assertEqual(result["rpc_mode"], "single")
        self.assertEqual(result["workload"]["methods"], ["eth_blockNumber"])

    def test_custom_rpc_scope_single_replace_with_one_method_skips_disambiguation(self) -> None:
        """Negative-case companion: with exactly one validated method, Case 1

        must go straight to `validated` without asking anything, protecting
        against the new branch over-triggering.
        """

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
        state["custom_rpc"] = {
            "status": "needs_scope",
            "validated_methods": [
                {"method": "eth_blockNumber", "params": [], "evidence_file": ".agent/evidence/one.json"},
            ],
        }
        state["pending_question"] = {
            "id": "custom_rpc_scope",
            "group": "endpoint_process",
            "kind": "numbered_choice",
            "field": "custom_rpc_scope",
            "options": [
                {"label": "Use this method as single workload only", "value": "single_replace"},
                {"label": "Use only my custom methods in mixed", "value": "mixed_replace"},
                {"label": "Keep template defaults in mixed and add this method", "value": "mixed_add"},
            ],
        }

        state["last_user_input"] = "1"
        state = process_turn(state)
        self.assertEqual(state["custom_rpc"]["status"], "validated")
        self.assertEqual(state["rpc_mode"], "single")
        self.assertEqual(state["workload"]["methods"], ["eth_blockNumber"])

    def test_existing_chain_custom_rpc_extracts_schema_evidence_before_probe(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["custom_rpc"] = {
            "status": "needs_method",
            "endpoint": "https://example.invalid/rpc",
            "endpoint_ready": True,
        }
        from agent.harness.domains.chain_rpc import question_for_chain_rpc

        state["pending_question"] = question_for_chain_rpc(state, "endpoint_process") or {}
        draft = {
            "status": "draft",
            "method": "eth_getBalance",
            "params": [
                {"index": 0, "name": "address", "json_type": "string", "semantic_type": "account_address", "encoding": "20-byte hex", "meaning": "account address", "example": "0x0000000000000000000000000000000000000000", "required": True},
                {"index": 1, "name": "block", "json_type": "string", "semantic_type": "block_reference", "encoding": "tag", "meaning": "block tag", "example": "latest", "required": True},
            ],
            "response_summary": "hex balance",
            "confidence": "high",
        }

        with patch("agent.harness.domains.rpc_endpoint.extract_rpc_schema_from_evidence", return_value=draft):
            state["last_user_input"] = "curl --data '{\"method\":\"eth_getBalance\",\"params\":[\"0x0000000000000000000000000000000000000000\",\"latest\"]}'"
            state = _invoke_with_rpc_evidence(process_turn, state, state["last_user_input"])

        self.assertEqual(state["custom_rpc"]["status"], "schema_needs_confirmation")
        self.assertEqual(state["pending_question"]["id"], "custom_rpc_parameter_confirm")

        ok_probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}
        with patch("agent.harness.domains.rpc_endpoint.validate_rpc_endpoint", return_value=ok_probe) as probe:
            state = self._confirm_catalog_through_graph(state, process_turn)

        probe.assert_called_once()
        self.assertEqual(_catalog_draft(state)["params_json"], ["0x0000000000000000000000000000000000000000", "latest"])
        self.assertEqual(state["custom_rpc"]["status"], "method_validated_next")
        self.assertEqual(state["pending_question"]["id"], "custom_rpc_continue")

    def test_existing_chain_custom_rpc_accepts_json_rpc_request_as_direct_schema(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.chain_rpc import question_for_chain_rpc
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["custom_rpc"] = {
            "status": "needs_method",
            "endpoint": "https://example.invalid/rpc",
            "endpoint_ready": True,
        }
        state["pending_question"] = question_for_chain_rpc(state, "endpoint_process") or {}
        extracted = {
            "status": "draft",
            "method": "eth_blockNumber",
            "params": [],
            "params_json": [],
            "response_summary": "hex quantity block height",
            "response_fields": [],
            "confidence": "high",
        }
        with patch("agent.harness.domains.rpc_endpoint.extract_rpc_schema_from_evidence", return_value=extracted) as extractor, patch("agent.harness.domains.rpc_endpoint.validate_rpc_endpoint") as probe:
            state["last_user_input"] = '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}'
            state = _invoke_with_rpc_evidence(process_turn, state, state["last_user_input"])

        extractor.assert_called_once()
        probe.assert_not_called()
        self.assertEqual(_catalog_draft(state)["method"], "eth_blockNumber")
        self.assertEqual(_catalog_draft(state)["params_json"], [])
        self.assertEqual(_catalog_draft(state)["params"], [])
        self.assertEqual(state["custom_rpc"]["status"], "schema_needs_confirmation")
        self.assertEqual(state["pending_question"]["id"], "custom_rpc_schema_confirm")
        self.assertIn("response: unknown", state["pending_question"]["prompt"])

    def test_direct_multi_param_request_reviews_wire_and_semantic_types_before_probe(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("direct-multi-param", language="en")
        state["target_mode"] = "real-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "ethereum", "canonical": "ethereum", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["custom_rpc"] = {"status": "needs_schema_evidence", "endpoint": "https://example.invalid/rpc", "endpoint_ready": True, "method": "eth_getBalance"}
        state["pending_question"] = {"id": "custom_rpc_schema_evidence", "group": "endpoint_process", "kind": "evidence", "field": "custom_rpc_schema_evidence", "manual_input_allowed": True}
        extracted = {
            "status": "draft",
            "method": "eth_getBalance",
            "params": [
                {"index": 0, "name": "address", "json_type": "string", "semantic_type": "account_address", "encoding": "20-byte hex", "meaning": "account to query", "example": "model-must-not-replace-observed-value", "required": True},
                {"index": 1, "name": "block", "json_type": "string", "semantic_type": "block_tag", "encoding": "tag or hex quantity", "meaning": "state version", "example": "latest", "required": True},
            ],
            "response_summary": "hex quantity balance",
            "confidence": "high",
        }
        request = '{"jsonrpc":"2.0","method":"eth_getBalance","params":["0x0000000000000000000000000000000000000000","latest"],"id":1}'
        with patch("agent.harness.domains.rpc_endpoint.extract_rpc_schema_from_evidence", return_value=extracted), patch("agent.harness.domains.rpc_endpoint.validate_rpc_endpoint") as probe:
            state["last_user_input"] = request
            state = _invoke_with_rpc_evidence(process_turn, state, request)

        probe.assert_not_called()
        draft = _catalog_draft(state)
        self.assertEqual(draft["params_json"], ["0x0000000000000000000000000000000000000000", "latest"])
        self.assertEqual([row["name"] for row in draft["params"]], ["address", "block"])
        self.assertEqual([row["semantic_type"] for row in draft["params"]], ["account_address", "block_tag"])
        prompt = state["pending_question"]["prompt"]
        self.assertIn("JSON wire type=string", prompt)
        self.assertIn("blockchain semantic type=account_address", prompt)
        self.assertIn("encoding=20-byte hex", prompt)
        self.assertEqual(state["pending_question"]["id"], "custom_rpc_parameter_confirm")

    def test_yaml_rpc_exchange_uses_the_declared_evidence_contract(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.chain_rpc import question_for_chain_rpc
        from agent.harness.state import new_state

        state = new_state("yaml-rpc-evidence", language="en")
        state["target_mode"] = "real-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "endpoint_process"
        state["chain_identity"] = {
            "raw": "ethereum",
            "canonical": "ethereum",
            "adapter_family": "jsonrpc",
            "status": "confirmed",
            "case": "known",
        }
        state["custom_rpc"] = {
            "status": "needs_schema_evidence",
            "endpoint": "https://example.invalid/rpc",
            "endpoint_ready": True,
            "method": "eth_blockNumber",
        }
        state["pending_question"] = question_for_chain_rpc(state, "endpoint_process")
        evidence = """Verify this wire exchange:
request:
  jsonrpc: "2.0"
  id: 1
  method: eth_blockNumber
  params: []
response:
  jsonrpc: "2.0"
  id: 1
  result: '0x10'
"""
        extracted = {
            "status": "draft",
            "method": "eth_blockNumber",
            "params": [],
            "params_json": [],
            "response_summary": "hex block number",
            "response_fields": [{"name": "result", "json_type": "string", "meaning": "block number"}],
            "confidence": "high",
        }
        with patch("agent.harness.domains.rpc_endpoint.extract_rpc_schema_from_evidence", return_value=extracted):
            state["last_user_input"] = evidence
            result = process_turn(state)

        self.assertEqual((result["custom_rpc"]["catalog"]["last_transition"])["command"], "correct_draft")
        self.assertIn(
            result["pending_question"]["id"],
            {"custom_rpc_parameter_confirm", "custom_rpc_schema_confirm", "custom_rpc_response_confirm"},
        )
        self.assertNotIn("clarify_unresolved", result.get("turn_context", {}).get("admitted_action_types", []))

    def test_rpc_wire_syntax_authority_rejects_unrelated_yaml_and_logs(self) -> None:
        from agent.harness.input_values import (
            extract_rpc_params_or_request,
            has_rpc_response_evidence,
            has_rpc_wire_evidence,
        )

        exchange = """request:
  jsonrpc: "2.0"
  id: 1
  method: eth_getBalance
  params:
    - "0x0000000000000000000000000000000000000000"
    - latest
response:
  jsonrpc: "2.0"
  id: 1
  result: '0x0'
"""
        method, params = extract_rpc_params_or_request(exchange)
        self.assertEqual(method, "eth_getBalance")
        self.assertEqual(params, ["0x0000000000000000000000000000000000000000", "latest"])
        self.assertTrue(has_rpc_wire_evidence(exchange))
        self.assertTrue(has_rpc_response_evidence(exchange))
        self.assertFalse(has_rpc_wire_evidence("CLOUD_REGION: us-east1\nMACHINE_TYPE: n2-standard-8"))
        self.assertFalse(has_rpc_wire_evidence("worker log: method failed and result was unavailable"))
        self.assertFalse(has_rpc_wire_evidence("method: eth_blockNumber"))

    def test_split_request_and_response_evidence_are_merged_before_confirmation(self) -> None:
        from agent.harness.domains.chain_rpc import apply_chain_rpc_answer
        from agent.harness.state import new_state

        state = new_state("split-rpc-evidence", language="en")
        state["chain_identity"] = {"raw": "ethereum", "canonical": "ethereum", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["custom_rpc"] = {"status": "needs_schema_evidence", "endpoint": "https://example.invalid/rpc", "endpoint_ready": True, "method": "eth_chainId"}
        request_draft = {"status": "draft", "method": "eth_chainId", "params": [], "params_json": [], "response_summary": "", "confidence": "high"}
        response_draft = {"status": "draft", "method": "eth_chainId", "params": [], "params_json": [], "response_summary": "hex quantity chain id", "response_fields": [{"name": "result", "type": "string", "meaning": "chain id"}], "confidence": "high"}
        question = {
            "id": "custom_rpc_schema_evidence",
            "group": "endpoint_process",
            "field": "custom_rpc_schema_evidence",
            "kind": "evidence",
            "manual_input_allowed": True,
        }
        with patch("agent.harness.domains.rpc_endpoint.extract_rpc_schema_from_evidence", side_effect=[request_draft, response_draft]), patch("agent.harness.domains.rpc_endpoint.validate_rpc_endpoint") as probe:
            request_text = '{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}'
            outcome = apply_chain_rpc_answer(state, question, request_text, request_text)
            state = _commit_result(state, outcome, owner="chain_rpc")
            self.assertEqual(state["custom_rpc"]["status"], "schema_needs_confirmation")
            response_text = '{"jsonrpc":"2.0","id":1,"result":"0x1"}'
            outcome = apply_chain_rpc_answer(state, question, response_text, response_text)
            state = _commit_result(state, outcome, owner="chain_rpc")

        probe.assert_not_called()
        self.assertEqual(len(_catalog_draft(state)["evidence"]), 2)
        self.assertIn('"result":"0x1"', "\n".join(item["content"] for item in _catalog_draft(state)["evidence"]))
        self.assertEqual(_catalog_draft(state)["response_summary"], "hex quantity chain id")
        self.assertIn("response fields: result", state["pending_question"]["prompt"])

    def test_custom_rpc_endpoint_turn_consumes_inline_jsonrpc_request(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["custom_rpc"] = {"status": "needs_endpoint"}
        state["pending_question"] = {
            "id": "custom_rpc_endpoint",
            "group": "endpoint_process",
            "kind": "url",
            "field": "custom_rpc_endpoint",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = 'endpoint 是 http://fake-node:19000，method 是 eth_chainId，请求 {"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}'

        ok_probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}
        action_plan = {
            "actions": [
                {
                    "type": "rpc_catalog_command",
                    "catalog_command": "set_endpoint",
                    "rpc_endpoint": "http://fake-node:19000",
                    "source_evidence": state["last_user_input"],
                    "confidence": "high",
                },
                {
                    "type": "rpc_catalog_command",
                    "catalog_command": "set_method",
                    "rpc_method": "eth_chainId",
                    "source_evidence": state["last_user_input"],
                    "confidence": "high",
                },
                {
                    "type": "rpc_catalog_command",
                    "catalog_command": "append_evidence",
                    "rpc_schema_evidence": '{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}',
                    "source_evidence": state["last_user_input"],
                    "confidence": "high",
                },
            ]
        }
        extracted = {"status": "draft", "method": "eth_chainId", "params": [], "params_json": [], "response_summary": "hex quantity chain id", "confidence": "high"}
        with (
            patch("agent.harness.coordinator.resolve_action_queue", return_value=action_plan),
            patch("agent.harness.domains.rpc_endpoint.extract_rpc_schema_from_evidence", return_value=extracted),
            patch("agent.harness.domains.rpc_endpoint.validate_rpc_endpoint", return_value=ok_probe),
        ):
            result = process_turn(state)

        self.assertEqual(result["custom_rpc"]["status"], "schema_needs_confirmation")
        self.assertEqual(_catalog_draft(result)["method"], "eth_chainId")
        self.assertEqual(result["pending_question"]["id"], "custom_rpc_schema_confirm")

    def test_real_node_requires_local_rpc_probe_before_workload(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "real-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "ethereum", "canonical": "ethereum", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {
            "BLOCKCHAIN_NODE": "ethereum",
            "CLOUD_REGION": "us-1",
            "CLOUD_ZONE": "us-1-z",
            "MACHINE_TYPE": "n2",
            "LEDGER_DEVICE": "vda",
            "DATA_VOL_TYPE": "hyperdisk-balanced",
            "DATA_VOL_SIZE": "926",
            "DATA_VOL_MAX_IOPS": "20000",
            "DATA_VOL_MAX_THROUGHPUT": "1000",
            "has_accounts_device": False,
            "NETWORK_INTERFACE": "eth0",
            "NETWORK_MAX_BANDWIDTH_GBPS": "100",
        }
        state["last_user_input"] = "continue"

        with patch("agent.harness.coordinator.resolve_action_queue", return_value={"actions": [{"type": "unknown", "confidence": "low"}]}):
            state = process_turn(state)

        self.assertEqual(state["pending_question"]["id"], "LOCAL_RPC_URL")

        ok_probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}
        schema = {"status": "draft", "method": "eth_blockNumber", "params": [], "params_json": [], "response_summary": "hex block height", "confidence": "high"}
        with patch("agent.harness.domains.rpc_endpoint.extract_rpc_schema_from_evidence", return_value=schema), patch("agent.harness.domains.rpc_endpoint.validate_rpc_endpoint", return_value=ok_probe):
            state["last_user_input"] = "https://example.invalid/rpc"
            state = process_turn(state)

        self.assertEqual(state["confirmed_config"]["LOCAL_RPC_URL"], "https://example.invalid/rpc")
        self.assertTrue(state["endpoint_evidence"]["local_rpc_url_ready"])
        self.assertEqual(state["pending_question"]["id"], "BLOCKCHAIN_PROCESS_NAMES")

    def test_case2_mainnet_review_never_offers_nonexistent_template_default(self) -> None:
        from agent.harness.domains.chain_rpc import question_for_chain_rpc
        from agent.harness.state import new_state

        state = new_state("case2-mainnet-review", language="en")
        state["target_mode"] = "real-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "LocalEvmDemo",
            "canonical": "LocalEvmDemo",
            "adapter_family": "jsonrpc",
            "status": "confirmed",
            "case": "case2",
        }
        state["endpoint_evidence"] = {"local_rpc_url_ready": True}
        state["confirmed_config"] = {
            "LOCAL_RPC_URL": "http://geth-dev:8545",
            "BLOCKCHAIN_PROCESS_NAMES": "geth",
        }

        question = question_for_chain_rpc(state, "endpoint_process")

        self.assertIsNotNone(question)
        assert question is not None
        self.assertEqual(question["id"], "MAINNET_RPC_URL_REVIEWED")
        self.assertIn("no configured chain-template mainnet endpoint", question["prompt"])
        self.assertNotIn("Use the current chain template", question["prompt"])
        self.assertEqual([option["value"] for option in question["options"]], [False])
        self.assertTrue(question["manual_input_allowed"])

    def test_new_chain_existing_family_validates_endpoint_method_workload_then_runtime_choice(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "flow-evm",
            "canonical": "flow-evm",
            "adapter_family": "jsonrpc",
            "status": "existing_family_needs_endpoint",
            "case": "case2",
        }
        state["pending_question"] = {
            "id": "new_chain_endpoint",
            "group": "endpoint_process",
            "kind": "url",
            "field": "new_chain_endpoint",
            "manual_input_allowed": True,
        }

        ok_probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}
        unknown_response_schema = {
            "status": "draft",
            "method": "eth_blockNumber",
            "params": [],
            "params_json": [],
            "response_summary": "unknown",
            "response_fields": [],
            "confidence": "low",
        }
        with (
            patch("agent.harness.domains.rpc_endpoint.validate_rpc_endpoint", return_value=ok_probe),
            patch(
                "agent.harness.domains.rpc_endpoint.extract_rpc_schema_from_evidence",
                return_value=unknown_response_schema,
            ),
        ):
            state["last_user_input"] = "https://example.invalid/rpc"
            state = process_turn(state)
            self.assertEqual(state["chain_identity"]["status"], "existing_family_needs_method")
            self.assertEqual(state["pending_question"]["id"], "new_chain_method")

            state["last_user_input"] = "eth_blockNumber"
            state = process_turn(state)
            self.assertEqual(state["chain_identity"]["status"], "existing_family_needs_schema_evidence")
            self.assertEqual(state["pending_question"]["id"], "new_chain_schema_evidence")
            self.assertIn("`eth_blockNumber`", state["pending_question"]["prompt"])

            state["last_user_input"] = "[]"
            state = process_turn(state)

            self.assertEqual(state["chain_identity"]["status"], "existing_family_schema_needs_confirmation")
            self.assertEqual(state["pending_question"]["id"], "new_chain_schema_confirm")
            state["last_user_input"] = "y"
            state = process_turn(state)
            self.assertEqual(state["pending_question"]["id"], "new_chain_probe_confirm")
            state["last_user_input"] = "y"
            state = process_turn(state)
            self.assertEqual(state["chain_identity"]["status"], "existing_family_response_needs_confirmation")
            self.assertEqual(state["pending_question"]["id"], "new_chain_response_confirm")
            state["last_user_input"] = "y"
            state = process_turn(state)

        self.assertEqual(state["chain_identity"]["status"], "existing_family_method_validated_next")
        self.assertEqual(state["pending_question"]["id"], "new_chain_method_continue")
        self.assertIn("new_chain_method_probe", state["endpoint_evidence"])
        self.assertNotIn("LOCAL_RPC_URL", state.get("confirmed_config", {}))
        self.assertEqual(state["active_group"], "endpoint_process")

        state["last_user_input"] = "2"
        state = process_turn(state)
        self.assertEqual(state["chain_identity"]["status"], "existing_family_needs_workload_scope")
        self.assertEqual(state["active_group"], "endpoint_process")
        self.assertEqual(state["pending_question"]["id"], "new_chain_workload_scope")

        state["last_user_input"] = "1"
        state = process_turn(state)
        self.assertEqual(state["chain_identity"]["status"], "existing_family_runtime_choice")
        self.assertEqual(state["active_group"], "target_samples_fixtures")
        self.assertEqual(state["pending_question"]["id"], "new_chain_runtime_choice")
        self.assertNotIn("LOCAL_RPC_URL", state.get("confirmed_config", {}))

        state["last_user_input"] = "1"
        state = process_turn(state)

        self.assertEqual(state["target_mode"], "real-node")
        self.assertEqual(state["chain_identity"]["status"], "confirmed")
        self.assertEqual(state["chain_identity"]["case"], "case2_runtime_override")
        self.assertEqual(state["confirmed_config"]["BLOCKCHAIN_NODE"], "flow-evm")
        self.assertEqual(state["confirmed_config"]["LOCAL_RPC_URL"], "https://example.invalid/rpc")
        self.assertEqual(state["workload"]["methods"], ["eth_blockNumber"])
        self.assertTrue(state["workload"]["job_local_override"])
        self.assertEqual(state["pending_question"]["id"], "CLOUD_REGION")

    def test_new_chain_existing_family_mixed_requires_validated_method_weights(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "flow-evm",
            "canonical": "flow-evm",
            "adapter_family": "jsonrpc",
            "status": "existing_family_method_validated_next",
            "case": "case2",
            "validated_methods": [
                {"method": "eth_blockNumber", "params": [], "evidence_file": ".agent/evidence/one.json"},
                {"method": "eth_chainId", "params": [], "evidence_file": ".agent/evidence/two.json"},
            ],
        }
        state["pending_question"] = {
            "id": "new_chain_method_continue",
            "group": "endpoint_process",
            "kind": "numbered_choice",
            "field": "new_chain_method_continue",
            "options": [
                {"label": "Add another RPC method", "value": "add_another"},
                {"label": "This is enough", "value": "finish"},
            ],
        }

        state["last_user_input"] = "2"
        state = process_turn(state)
        self.assertEqual(state["pending_question"]["id"], "new_chain_workload_scope")

        state["last_user_input"] = "2"
        state = process_turn(state)
        self.assertEqual(state["chain_identity"]["status"], "existing_family_needs_weights")
        self.assertEqual(state["pending_question"]["id"], "new_chain_custom_weights")

        state["last_user_input"] = "eth_blockNumber=100"
        state = process_turn(state)
        self.assertEqual(state["chain_identity"]["status"], "existing_family_needs_weights")
        self.assertIn("缺少已验证 method", "\n".join(state["visible_response"]))

        state["last_user_input"] = "那改成 eth_blockNumber=70,eth_chainId=30"
        state = process_turn(state)
        self.assertEqual(state["chain_identity"]["status"], "existing_family_runtime_choice")
        self.assertEqual(state["rpc_mode"], "mixed")
        self.assertEqual(state["workload"]["mixed_weights"], {"eth_blockNumber": 70, "eth_chainId": 30})
        self.assertTrue(state["workload"]["job_local_override"])
        self.assertEqual(state["pending_question"]["id"], "new_chain_runtime_choice")

    def test_new_chain_real_node_workload_skips_fake_node_fixture_gate(self) -> None:
        from agent.harness.domains.chain_rpc import apply_chain_rpc_answer
        from agent.harness.domains.rpc_catalog import migrate_legacy_catalog
        from agent.harness.state import new_state

        state = new_state("real-node-case2", language="en")
        state["target_mode"] = "real-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "LocalEvmDemo",
            "canonical": "LocalEvmDemo",
            "adapter_family": "jsonrpc",
            "status": "existing_family_needs_weights",
            "case": "case2",
            "workload_scope": "mixed_replace",
            "validated_methods": [
                {"method": "eth_chainId", "params": [], "evidence_file": ".agent/evidence/one.json"},
                {"method": "eth_getBalance", "params": ["0x0", "latest"], "evidence_file": ".agent/evidence/two.json"},
            ],
        }
        state["endpoint_evidence"] = {
            "candidate_endpoint": "http://geth-dev:8545",
            "candidate_endpoint_ready": True,
        }
        state["pending_question"] = {
            "id": "new_chain_custom_weights",
            "group": "endpoint_process",
            "kind": "manual",
            "field": "new_chain_custom_weights",
            "manual_input_allowed": True,
        }
        state["active_group"] = "endpoint_process"
        state["last_user_input"] = "eth_chainId=30,eth_getBalance=70"
        migrate_legacy_catalog(state)

        outcome = apply_chain_rpc_answer(
            state,
            state["pending_question"],
            state["last_user_input"],
            state["last_user_input"],
        )
        result = _commit_result(state, outcome, owner="chain_rpc")

        self.assertEqual(result["chain_identity"]["status"], "confirmed")
        self.assertEqual(result["confirmed_config"]["BLOCKCHAIN_NODE"], "LocalEvmDemo")
        self.assertNotIn("LOCAL_RPC_URL", result["confirmed_config"])
        self.assertEqual(result["workload"]["mixed_weights"], {"eth_chainId": 30, "eth_getBalance": 70})
        self.assertNotEqual(result["pending_question"].get("id"), "new_chain_runtime_choice")
        self.assertNotIn("fake-node has no fixture", "\n".join(result["visible_response"]))

    def test_new_chain_workload_scope_single_replace_with_multiple_methods_asks_new_chain_single_method(self) -> None:
        """Phase 4 backfill: `existing_family_needs_single_method` had zero

        prior test coverage. This pins Case 2's own behavior (asking which
        validated method to use as single, when more than one is validated)
        before Case 1 is unified to match it.
        """

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "flow-evm",
            "canonical": "flow-evm",
            "adapter_family": "jsonrpc",
            "status": "existing_family_needs_workload_scope",
            "case": "case2",
            "validated_methods": [
                {"method": "eth_blockNumber", "params": [], "evidence_file": ".agent/evidence/one.json"},
                {"method": "eth_chainId", "params": [], "evidence_file": ".agent/evidence/two.json"},
            ],
        }
        state["pending_question"] = {
            "id": "new_chain_workload_scope",
            "group": "endpoint_process",
            "kind": "numbered_choice",
            "field": "new_chain_workload_scope",
            "options": [
                {"label": "Use one validated method as single", "value": "single_replace"},
                {"label": "Use only these validated methods in mixed and configure weights", "value": "mixed_replace"},
            ],
        }

        state["last_user_input"] = "1"
        state = process_turn(state)
        self.assertEqual(state["chain_identity"]["status"], "existing_family_needs_single_method")
        self.assertEqual(state["pending_question"]["id"], "new_chain_single_method")
        option_values = {option["value"] for option in state["pending_question"]["options"]}
        self.assertEqual(option_values, {"eth_blockNumber", "eth_chainId"})

        state["last_user_input"] = "1"
        state = process_turn(state)
        self.assertEqual(state["chain_identity"]["status"], "existing_family_runtime_choice")
        self.assertEqual(state["rpc_mode"], "single")
        self.assertEqual(state["workload"]["methods"], ["eth_blockNumber"])
        self.assertEqual(state["pending_question"]["id"], "new_chain_runtime_choice")

    def test_unknown_chain_with_user_protocol_hint_enters_supported_family_confirmation(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["active_group"] = "opening"
        state["last_user_input"] = "我想测 Flow，它应该是 EVM/jsonrpc"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {
                        "type": "choose_chain",
                        "chain_text": "Flow",
                        "chain_exists": True,
                        "canonical_chain_name": "flow",
                        "adapter_family": "jsonrpc",
                        "source_evidence": "Flow",
                        "confidence": "high",
                    }
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["chain_identity"]["canonical"], "Flow")
        self.assertEqual(result["chain_identity"]["proposed_canonical_name"], "flow")
        self.assertEqual(result["chain_identity"]["adapter_family"], "jsonrpc")
        self.assertEqual(result["pending_question"]["id"], "unknown_chain_identity_confirm")
        self.assertIn("协议族为 `jsonrpc`", result["visible_response"][0])

    def test_device_choice_single_default_accepts_yes(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["pending_question"] = {
            "id": "network_interface",
            "group": "network",
            "kind": "device",
            "field": "NETWORK_INTERFACE",
            "options": [{"label": "eth0 (default)", "value": "eth0"}],
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "y"

        result = process_turn(state)

        self.assertEqual(result.get("confirmed_config", {}).get("NETWORK_INTERFACE"), "eth0")
        self.assertEqual(result.get("pending_question", {}).get("id"), "NETWORK_MAX_BANDWIDTH_GBPS")

    def test_device_choice_multiple_candidates_rejects_bare_yes_without_llm_routing(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["pending_question"] = {
            "id": "LEDGER_DEVICE",
            "group": "ledger_disk",
            "kind": "device",
            "field": "LEDGER_DEVICE",
            "options": [{"label": "vda", "value": "vda"}, {"label": "vdb", "value": "vdb"}],
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "y"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            result = process_turn(state)

        resolver.assert_not_called()
        self.assertNotIn("LEDGER_DEVICE", result.get("confirmed_config", {}))
        self.assertEqual(result["pending_question"]["id"], "LEDGER_DEVICE")
        self.assertIn("Y/N", result["visible_response"][0])

    def test_new_chain_existing_family_extracts_schema_evidence_before_runtime_choice(self) -> None:
        from agent.harness.domains.chain_rpc import question_for_chain_rpc
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "flow-evm",
            "canonical": "flow-evm",
            "adapter_family": "jsonrpc",
            "status": "existing_family_needs_schema_evidence",
            "case": "case2",
            "candidate_method": "eth_blockNumber",
        }
        state["endpoint_evidence"] = {"candidate_endpoint": "https://example.invalid/rpc"}
        state["pending_question"] = question_for_chain_rpc(state, "endpoint_process") or {}
        self.assertEqual(state["pending_question"]["id"], "new_chain_schema_evidence")
        self.assertIs(state["pending_question"]["structured_input_owner"], True)
        draft = {
            "status": "draft",
            "method": "eth_blockNumber",
            "params": [],
            "response_summary": "latest block number as hex quantity",
            "confidence": "high",
        }

        with patch("agent.harness.domains.rpc_endpoint.extract_rpc_schema_from_evidence", return_value=draft):
            state["last_user_input"] = "curl --data '{\"method\":\"eth_blockNumber\",\"params\":[]}'"
            state = process_turn(state)

        self.assertEqual(state["chain_identity"]["status"], "existing_family_schema_needs_confirmation")
        self.assertEqual(state["pending_question"]["id"], "new_chain_schema_confirm")

        ok_probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}
        with patch("agent.harness.domains.rpc_endpoint.validate_rpc_endpoint", return_value=ok_probe):
            state = self._confirm_catalog_through_graph(state, process_turn)

        self.assertEqual(_catalog_draft(state)["params_json"], [])
        self.assertEqual(state["chain_identity"]["status"], "existing_family_method_validated_next")
        self.assertEqual(state["pending_question"]["id"], "new_chain_method_continue")
        self.assertIn("new_chain_method_probe", state["endpoint_evidence"])

    def test_pending_answer_clears_stale_action_queue_before_new_endpoint_question(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "flow",
            "canonical": "flow",
            "status": "needs_protocol_confirmation",
            "case": "unknown",
        }
        state["pending_question"] = {
            "id": "adapter_family_confirm",
            "group": "chain_identity",
            "kind": "numbered_choice",
            "field": "adapter_family",
            "options": [{"label": "jsonrpc / EVM", "value": "jsonrpc"}],
            "manual_input_allowed": False,
        }
        state["action_queue"] = [
            {
                "type": "propose_config_values",
                "config_values": {"LOCAL_RPC_URL": ""},
                "unmapped_values": {"endpoint": "provided by user"},
                "confidence": "high",
            }
        ]

        state["last_user_input"] = "1"
        state = process_turn(state)

        self.assertEqual(state["pending_question"]["id"], "new_chain_endpoint")
        self.assertEqual(state.get("action_queue"), [])

        ok_probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}
        with patch("agent.harness.domains.rpc_endpoint.validate_rpc_endpoint", return_value=ok_probe):
            state["last_user_input"] = "https://example.invalid/rpc"
            state = process_turn(state)

        self.assertEqual(state["chain_identity"]["status"], "existing_family_needs_method")
        self.assertEqual(state["endpoint_evidence"]["candidate_endpoint"], "https://example.invalid/rpc")
        self.assertEqual(state["pending_question"]["id"], "new_chain_method")

    def test_unsupported_family_stops_with_development_handoff_message(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "new-protocol-chain",
            "canonical": "new-protocol-chain",
            "status": "needs_protocol_confirmation",
            "case": "unknown",
        }
        state["pending_question"] = {
            "id": "adapter_family_confirm",
            "group": "chain_identity",
            "kind": "numbered_choice",
            "field": "adapter_family",
            "options": [{"label": "unsupported", "value": "unsupported"}],
            "manual_input_allowed": False,
        }
        state["last_user_input"] = "1"

        state = process_turn(state)

        self.assertEqual(state["chain_identity"]["status"], "unsupported_family_handoff")
        self.assertEqual(state["chain_identity"]["case"], "case3")
        self.assertIn("outside the supported adapter families", state["visible_response"][0])

    def test_sync_observe_requires_real_data_source_before_process_or_stop_condition(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "sync-observe"
        state["workflow_mode"] = "sync_observe"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {
            "BLOCKCHAIN_NODE": "bsc",
            "CLOUD_REGION": "us-1",
            "CLOUD_ZONE": "us-1-z",
            "MACHINE_TYPE": "n2",
            "LEDGER_DEVICE": "vda",
            "DATA_VOL_TYPE": "hyperdisk-balanced",
            "DATA_VOL_SIZE": "926",
            "DATA_VOL_MAX_IOPS": "20000",
            "DATA_VOL_MAX_THROUGHPUT": "1000",
            "has_accounts_device": False,
            "NETWORK_INTERFACE": "eth0",
            "NETWORK_MAX_BANDWIDTH_GBPS": "100",
        }
        state["last_user_input"] = "continue"

        with patch("agent.harness.coordinator.resolve_action_queue", return_value={"actions": [{"type": "unknown", "confidence": "low"}]}):
            state = process_turn(state)

        self.assertEqual(state["pending_question"]["id"], "sync_observe_source")
        self.assertIn("fake-node", state["visible_response"][0])

    def test_sync_observe_endpoint_only_probes_real_endpoint_and_skips_process_name(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "sync-observe"
        state["workflow_mode"] = "sync_observe"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {
            "BLOCKCHAIN_NODE": "bsc",
            "CLOUD_REGION": "us-1",
            "CLOUD_ZONE": "us-1-z",
            "MACHINE_TYPE": "n2",
            "LEDGER_DEVICE": "vda",
            "DATA_VOL_TYPE": "hyperdisk-balanced",
            "DATA_VOL_SIZE": "926",
            "DATA_VOL_MAX_IOPS": "20000",
            "DATA_VOL_MAX_THROUGHPUT": "1000",
            "has_accounts_device": False,
            "NETWORK_INTERFACE": "eth0",
            "NETWORK_MAX_BANDWIDTH_GBPS": "100",
        }
        state["pending_question"] = {
            "id": "sync_observe_source",
            "group": "sync_observe",
            "kind": "numbered_choice",
            "field": "sync_observe_source",
            "options": [{"label": "endpoint", "value": "endpoint_only"}],
            "manual_input_allowed": False,
        }
        state["last_user_input"] = "1"
        state = process_turn(state)
        self.assertEqual(state["pending_question"]["id"], "SYNC_OBSERVE_RPC_URL")

        ok_probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}
        with patch("agent.harness.domains.rpc_endpoint.validate_rpc_endpoint", return_value=ok_probe):
            state["last_user_input"] = "https://example.invalid/rpc"
            state = process_turn(state)

        self.assertTrue(state["endpoint_evidence"]["sync_rpc_url_ready"])
        self.assertEqual(state["confirmed_config"]["SYNC_OBSERVE_RPC_URL"], "https://example.invalid/rpc")
        self.assertNotIn("LOCAL_RPC_URL", state["confirmed_config"])
        self.assertNotEqual(state.get("pending_question", {}).get("id"), "BLOCKCHAIN_PROCESS_NAMES")
        self.assertEqual(state["pending_question"]["id"], "MAINNET_RPC_URL_REVIEWED")

    def test_sync_observe_client_setup_without_google_search_stops_with_handoff(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "sync-observe"
        state["workflow_mode"] = "sync_observe"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["web_research"] = {"google_search_available": False}
        state["confirmed_config"] = {
            "BLOCKCHAIN_NODE": "bsc",
            "CLOUD_REGION": "us-1",
            "CLOUD_ZONE": "us-1-z",
            "MACHINE_TYPE": "n2",
            "LEDGER_DEVICE": "vda",
            "DATA_VOL_TYPE": "hyperdisk-balanced",
            "DATA_VOL_SIZE": "926",
            "DATA_VOL_MAX_IOPS": "20000",
            "DATA_VOL_MAX_THROUGHPUT": "1000",
            "has_accounts_device": False,
            "NETWORK_INTERFACE": "eth0",
            "NETWORK_MAX_BANDWIDTH_GBPS": "100",
        }
        state["pending_question"] = {
            "id": "sync_observe_source",
            "group": "sync_observe",
            "kind": "numbered_choice",
            "field": "sync_observe_source",
            "options": [{"label": "client setup", "value": "client_setup"}],
            "manual_input_allowed": False,
        }
        state["last_user_input"] = "1"
        state = process_turn(state)

        self.assertIn("google_search", state["visible_response"][0])
        self.assertEqual(state.get("pending_question"), {})
        self.assertEqual(state["sync_observe"]["source"], "client_setup")
        self.assertNotEqual(state.get("active_group"), "preflight_smoke_execution")

    def test_sync_observe_client_setup_with_google_search_grounds_the_handoff_message(self) -> None:
        from unittest.mock import patch

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state
        from agent.llm.search_grounding import SearchGroundingResult

        state = new_state("unit-thread")
        state["target_mode"] = "sync-observe"
        state["workflow_mode"] = "sync_observe"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["web_research"] = {"google_search_available": True}
        state["confirmed_config"] = {
            "BLOCKCHAIN_NODE": "bsc",
            "CLOUD_REGION": "us-1",
            "CLOUD_ZONE": "us-1-z",
            "MACHINE_TYPE": "n2",
            "LEDGER_DEVICE": "vda",
            "DATA_VOL_TYPE": "hyperdisk-balanced",
            "DATA_VOL_SIZE": "926",
            "DATA_VOL_MAX_IOPS": "20000",
            "DATA_VOL_MAX_THROUGHPUT": "1000",
            "has_accounts_device": False,
            "NETWORK_INTERFACE": "eth0",
            "NETWORK_MAX_BANDWIDTH_GBPS": "100",
        }
        state["pending_question"] = {
            "id": "sync_observe_source",
            "group": "sync_observe",
            "kind": "numbered_choice",
            "field": "sync_observe_source",
            "options": [{"label": "client setup", "value": "client_setup"}],
            "manual_input_allowed": False,
        }
        fake_result = SearchGroundingResult(
            available=True,
            query="bsc official blockchain node client",
            text_summary="Use the official bsc-erigon Docker image.",
            citations=["https://example.invalid/bsc-docs"],
        )
        with patch("agent.harness.domains.sync_observe.run_google_search_grounding", return_value=fake_result) as mocked:
            state["last_user_input"] = "1"
            state = process_turn(state)

        mocked.assert_called_once()
        self.assertIn("Use the official bsc-erigon Docker image.", state["visible_response"][0])
        self.assertIn("https://example.invalid/bsc-docs", state["visible_response"][0])
        self.assertEqual(state["sync_observe"]["source"], "client_setup")

    def test_custom_rpc_success_asks_continue_instead_of_looping_schema(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["custom_rpc"] = {
            "status": "schema_needs_confirmation",
            "endpoint": "http://fake-node:19000",
            "endpoint_ready": True,
            "method": "eth_blockNumber",
            "schema_draft": {
                "status": "draft",
                "evidence_kind": "jsonrpc_request",
                "transport": "jsonrpc",
                "method": "eth_blockNumber",
                "params": [],
                "params_json": [],
                "response_summary": "hex block number",
                "confidence": "high",
            },
        }
        state["active_group"] = "endpoint_process"
        state["pending_question"] = {
            "id": "custom_rpc_schema_confirm",
            "group": "endpoint_process",
            "kind": "yes_no",
            "field": "custom_rpc_schema_confirm",
            "options": [{"label": "Y", "value": True}, {"label": "N", "value": False}],
        }
        state["last_user_input"] = "Y"

        with patch("agent.harness.domains.rpc_endpoint.validate_rpc_endpoint") as probe:
            probe.return_value = {"ready": True, "evidence_file": ".agent/evidence/endpoint-probes/demo.json"}
            result = self._confirm_catalog_through_graph(state, process_turn)

        self.assertEqual(result["custom_rpc"]["status"], "method_validated_next")
        self.assertEqual(result["pending_question"]["id"], "custom_rpc_continue")
        self.assertIn("What should happen next", "\n".join(result["visible_response"]))

    def test_custom_rpc_missing_response_requires_observed_response_review(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state.update({
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "active_group": "endpoint_process",
            "chain_identity": {"canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed"},
            "custom_rpc": {
                "status": "schema_needs_confirmation",
                "endpoint": "http://fake-node:19000",
                "endpoint_ready": True,
                "method": "eth_chainId",
                "schema_draft": {"method": "eth_chainId", "params": [], "params_json": [], "response_summary": "unknown"},
            },
            "pending_question": {
                "id": "custom_rpc_schema_confirm",
                "group": "endpoint_process",
                "kind": "yes_no",
                "field": "custom_rpc_schema_confirm",
                "options": [{"label": "Y", "value": True}, {"label": "N", "value": False}],
            },
            "last_user_input": "Y",
        })
        probe = {
            "ready": True,
            "status": "ok",
            "response_shape_hash": "shape-1",
            "evidence_file": ".agent/evidence/probe.json",
            "checks": [{"name": "method_probe:eth_chainId", "response_shape_hash": "shape-1", "response_sample": '{"jsonrpc":"2.0","result":"0x38"}', "http_status": 200}],
        }
        with patch("agent.harness.domains.rpc_endpoint.validate_rpc_endpoint", return_value=probe) as mocked_probe:
            result = process_turn(state)
            mocked_probe.assert_not_called()
            self.assertEqual(result["pending_question"]["id"], "custom_rpc_probe_confirm")
            probe_prompt = "\n".join(result["visible_response"])
            self.assertIn("no response contract is confirmed yet", probe_prompt)
            self.assertNotIn("request and response contracts are confirmed separately", probe_prompt)
            result["last_user_input"] = "Y"
            result = process_turn(result)

        self.assertEqual(result["custom_rpc"]["status"], "response_needs_confirmation")
        self.assertEqual(result["pending_question"]["id"], "custom_rpc_response_confirm")
        self.assertIn("shape-1", "\n".join(result["visible_response"]))

        result["last_user_input"] = "Y"
        result = process_turn(result)
        self.assertTrue(_catalog_draft(result)["response_confirmed"])
        self.assertEqual(result["custom_rpc"]["status"], "method_validated_next")
        self.assertEqual(result["pending_question"]["id"], "custom_rpc_continue")
        self.assertEqual(_catalog_methods(result)[0]["observed_response"]["shape_hash"], "shape-1")

    def test_rejected_observed_response_preserves_request_evidence(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn

        state = self._state_for_custom_rpc_response_confirmation()
        state["last_user_input"] = "N"
        result = process_turn(state)

        self.assertEqual(result["custom_rpc"]["status"], "needs_schema_evidence")
        self.assertFalse(_catalog_draft(result)["response_confirmed"])
        self.assertEqual(_catalog_draft(result)["method"], "eth_chainId")

    def test_semantically_admitted_response_rejection_preserves_typed_false(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn

        state = self._state_for_custom_rpc_response_confirmation()
        state["last_user_input"] = (
            "I do not accept that response contract; it is incomplete, "
            "and I will supply corrected evidence next."
        )
        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            side_effect=_admitted_mock_resolver({"actions": [{
                "type": "answer_pending",
                "answer": False,
                "selected_value": False,
                "source_evidence": "I do not accept that response contract",
                "pending_option_semantic_verified": True,
                "semantic_purpose_verified": True,
                "confidence": "high",
            }]}),
        ):
            result = process_turn(state)

        self.assertEqual(result["custom_rpc"]["status"], "needs_schema_evidence")
        self.assertFalse(_catalog_draft(result)["response_confirmed"])
        self.assertEqual(result["pending_question"]["id"], "custom_rpc_schema_evidence")

    def test_semantically_admitted_option_preserves_typed_numeric_zero(self) -> None:
        from unittest.mock import patch

        from agent.harness import coordinator
        from agent.harness.contracts import HandlerResult, StateDelta
        from agent.harness.domains.runtime import DomainRuntime
        from agent.harness.state import new_state

        state = new_state("typed-zero", language="en")
        state["active_group"] = "provider_deployment"
        state["pending_question"] = {
            "contract_version": 1,
            "id": "typed_zero_choice",
            "group": "provider_deployment",
            "kind": "numbered_choice",
            "field": "typed_zero_choice",
            "manual_input_allowed": False,
            "options": [{
                "id": "zero",
                "label": "Zero",
                "value": 0,
                "expected_patch": {"confirmed_config.CLOUD_REGION": 0},
            }],
            "accepted_action_types": ["answer_pending"],
            "validation": {},
        }
        existing = coordinator.DOMAIN_RUNTIME["environment"]

        def apply_answer(_state, _question, value, _text):
            return HandlerResult(
                delta=StateDelta.set_values({"confirmed_config": {"CLOUD_REGION": value}}),
                clear_pending=True,
                completion="completed",
            )

        runtime = DomainRuntime(
            apply_action=existing.apply_action,
            question_factory=existing.question_factory,
            apply_answer=apply_answer,
            cancel_question=existing.cancel_question,
        )
        with patch.dict(coordinator.DOMAIN_RUNTIME, {"environment": runtime}):
            result = coordinator._dispatch_pending_action(
                state,
                {"type": "answer_pending", "answer": 0, "selected_value": 0},
            )

        self.assertEqual(result["confirmed_config"]["CLOUD_REGION"], 0)
        self.assertEqual(result.get("pending_question"), {})

    def test_new_chain_semantic_response_rejection_satisfies_declared_postcondition(self) -> None:
        from agent.harness.domains.chain_rpc_questions import _response_confirmation_question
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn

        state = self._state_for_custom_rpc_response_confirmation()
        state["chain_identity"] = {
            "raw": "flow-evm",
            "canonical": "flow-evm",
            "adapter_family": "jsonrpc",
            "status": "existing_family_response_needs_confirmation",
            "case": "case2",
        }
        state["custom_rpc"].pop("status", None)
        state["pending_question"] = _response_confirmation_question(state, "new_chain")
        state["last_user_input"] = "Reject this observed response; I will correct the evidence."
        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            side_effect=_admitted_mock_resolver({"actions": [{
                "type": "answer_pending",
                "answer": False,
                "selected_value": False,
                "source_evidence": "Reject this observed response",
                "pending_option_semantic_verified": True,
                "semantic_purpose_verified": True,
                "confidence": "high",
            }]}),
        ):
            result = process_turn(state)

        self.assertEqual(
            result["chain_identity"]["status"],
            "existing_family_needs_schema_evidence",
        )
        self.assertFalse(_catalog_draft(result)["response_confirmed"])
        self.assertEqual(result["pending_question"]["id"], "new_chain_schema_evidence")

    @staticmethod
    def _state_for_custom_rpc_response_confirmation() -> dict:
        from agent.harness.domains.chain_rpc_questions import _response_confirmation_question
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state.update({
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "active_group": "endpoint_process",
            "chain_identity": {"canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed"},
            "custom_rpc": {
                "status": "response_needs_confirmation",
                "method": "eth_chainId",
                "params": [],
                "schema_draft": {"method": "eth_chainId", "params": [], "params_json": [], "response_summary": "unknown"},
                "observed_response": {"shape_hash": "shape-1", "sample": '{"result":"0x38"}'},
            },
        })
        state["pending_question"] = _response_confirmation_question(state, "custom_rpc")
        return state

    def test_new_chain_rest_evidence_does_not_probe_as_jsonrpc_method(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "abcd",
            "canonical": "abcd",
            "adapter_family": "jsonrpc",
            "status": "existing_family_needs_schema_evidence",
            "case": "case2",
            "candidate_method": "GetBlockByID",
        }
        state["endpoint_evidence"] = {"candidate_endpoint": "https://example.invalid", "candidate_endpoint_ready": True}
        state["active_group"] = "endpoint_process"
        state["pending_question"] = {
            "id": "new_chain_schema_evidence",
            "group": "endpoint_process",
            "kind": "evidence",
            "field": "new_chain_schema_evidence",
            "manual_input_allowed": True,
            "created_turn_index": 0,
        }
        state["last_user_input"] = "Get Blocks by ID\npath Parameters\nid required\nquery Parameters"

        with (
            patch(
                "agent.harness.coordinator.resolve_action_queue",
                return_value={"actions": [{
                    "type": "rpc_catalog_command",
                    "catalog_command": "append_evidence",
                    "rpc_schema_evidence": state["last_user_input"],
                    "source_evidence": state["last_user_input"],
                    "confidence": "high",
                }]},
            ),
            patch("agent.harness.domains.rpc_endpoint.extract_rpc_schema_from_evidence") as extract,
            patch("agent.harness.domains.rpc_endpoint.validate_rpc_endpoint") as probe,
        ):
            extract.return_value = {
                "status": "draft",
                "evidence_kind": "docs_excerpt",
                "transport": "rest",
                "method": "GET /v1/blocks/{id}",
                "rest_path": "/v1/blocks/{id}",
                "params": [{"index": 0, "name": "id", "type": "string", "example": "0" * 64, "required": True}],
                "confidence": "medium",
            }
            result = process_turn(state)

        probe.assert_not_called()
        self.assertEqual(result["pending_question"]["id"], "adapter_family_confirm")
        self.assertIn("REST API", "\n".join(result["visible_response"]))

    def test_custom_rpc_schema_conflict_rest_evidence_reasks_adapter_family(self) -> None:
        """Phase 4 authorized change: Case 1's schema-conflict recovery now

        re-asks the adapter family (mirroring Case 2's existing behavior,
        `test_new_chain_rest_evidence_does_not_probe_as_jsonrpc_method`)
        instead of only printing an explanatory message with no active
        re-ask.
        """

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "abcd", "canonical": "abcd", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["custom_rpc"] = {
            "method": "GetBlockByID",
            "endpoint": "https://example.invalid",
            "endpoint_ready": True,
            "status": "needs_schema_evidence",
        }
        state["active_group"] = "endpoint_process"
        state["pending_question"] = {
            "id": "custom_rpc_schema_evidence",
            "group": "endpoint_process",
            "kind": "evidence",
            "field": "custom_rpc_schema_evidence",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "Get Blocks by ID\npath Parameters\nid required\nquery Parameters"

        with (
            patch(
                "agent.harness.coordinator.resolve_action_queue",
                return_value={"actions": [{
                    "type": "rpc_catalog_command",
                    "catalog_command": "append_evidence",
                    "rpc_schema_evidence": state["last_user_input"],
                    "source_evidence": state["last_user_input"],
                    "confidence": "high",
                }]},
            ),
            patch("agent.harness.domains.rpc_endpoint.extract_rpc_schema_from_evidence") as extract,
            patch("agent.harness.domains.rpc_endpoint.validate_rpc_endpoint") as probe,
        ):
            extract.return_value = {
                "status": "draft",
                "evidence_kind": "docs_excerpt",
                "transport": "rest",
                "method": "GET /v1/blocks/{id}",
                "rest_path": "/v1/blocks/{id}",
                "params": [{"index": 0, "name": "id", "type": "string", "example": "0" * 64, "required": True}],
                "confidence": "medium",
            }
            result = process_turn(state)

        probe.assert_not_called()
        self.assertEqual(result["custom_rpc"]["status"], "needs_adapter_family_confirmation")
        self.assertFalse(result["custom_rpc"]["endpoint_ready"])
        self.assertEqual(result["pending_question"]["id"], "custom_rpc_adapter_family_confirm")
        self.assertIn("REST API", "\n".join(result["visible_response"]))

        result["last_user_input"] = "rest"
        result = process_turn(result)
        self.assertEqual(result["chain_identity"]["adapter_family"], "rest")
        self.assertEqual(result["custom_rpc"]["status"], "needs_endpoint")
        self.assertEqual(result["pending_question"]["id"], "custom_rpc_endpoint")

    def test_custom_rpc_schema_conflict_jsonrpc_evidence_from_rest_reasks_adapter_family(self) -> None:
        """Companion direction: rest chain given jsonrpc-shaped evidence.

        Confirms symmetric `endpoint_ready` clearing for Case 1 (a deliberate
        choice, since this is brand-new code with no prior asymmetric
        behavior to preserve — unlike Case 2, which only clears it one way).
        """

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "abcd", "canonical": "abcd", "adapter_family": "rest", "status": "confirmed", "case": "known"}
        state["custom_rpc"] = {
            "method": "eth_blockNumber",
            "endpoint": "https://example.invalid",
            "endpoint_ready": True,
            "status": "needs_schema_evidence",
        }
        state["active_group"] = "endpoint_process"
        state["pending_question"] = {
            "id": "custom_rpc_schema_evidence",
            "group": "endpoint_process",
            "kind": "evidence",
            "field": "custom_rpc_schema_evidence",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "Client docs say: call eth_blockNumber over JSON-RPC to get the latest block height."

        with (
            patch(
                "agent.harness.coordinator.resolve_action_queue",
                return_value={"actions": [{
                    "type": "rpc_catalog_command",
                    "catalog_command": "append_evidence",
                    "rpc_schema_evidence": state["last_user_input"],
                    "source_evidence": state["last_user_input"],
                    "confidence": "high",
                }]},
            ),
            patch("agent.harness.domains.rpc_endpoint.extract_rpc_schema_from_evidence") as extract,
            patch("agent.harness.domains.rpc_endpoint.validate_rpc_endpoint") as probe,
        ):
            extract.return_value = {
                "status": "draft",
                "evidence_kind": "json_rpc_request",
                "transport": "jsonrpc",
                "method": "eth_blockNumber",
                "params": [],
                "confidence": "high",
            }
            result = process_turn(state)

        probe.assert_not_called()
        self.assertEqual(result["custom_rpc"]["status"], "needs_adapter_family_confirmation")
        self.assertFalse(result["custom_rpc"]["endpoint_ready"])
        self.assertEqual(result["pending_question"]["id"], "custom_rpc_adapter_family_confirm")
        self.assertIn("JSON-RPC", "\n".join(result["visible_response"]))

    def test_url_is_not_accepted_as_new_chain_rpc_method_name(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["chain_identity"] = {
            "raw": "abcd",
            "canonical": "abcd",
            "adapter_family": "jsonrpc",
            "status": "existing_family_needs_method",
            "case": "case2",
        }
        state["active_group"] = "endpoint_process"
        state["pending_question"] = {
            "id": "new_chain_method",
            "group": "endpoint_process",
            "kind": "manual_value",
            "field": "new_chain_method",
            "manual_input_allowed": True,
            "validation": {"input_mode": "rpc_method_or_schema_evidence"},
        }
        state["last_user_input"] = "https://rest-testnet.onflow.org/v1/blocks"

        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{
                "type": "clarify_unresolved",
                "clauses": [state["last_user_input"]],
                "confidence": "high",
            }]},
        ):
            result = process_turn(state)

        self.assertEqual(result["chain_identity"]["status"], "existing_family_needs_method")
        self.assertNotEqual(result["chain_identity"].get("candidate_method"), "https://rest-testnet.onflow.org/v1/blocks")
        self.assertIn("尚未安全映射", "\n".join(result["visible_response"]))
        self.assertEqual(result["pending_question"]["id"], "new_chain_method")

    def test_new_chain_method_question_reviews_inline_jsonrpc_request_before_probe(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["chain_identity"] = {
            "raw": "abcd",
            "canonical": "abcd",
            "adapter_family": "jsonrpc",
            "status": "existing_family_needs_method",
            "case": "case2",
        }
        state["endpoint_evidence"] = {"candidate_endpoint": "http://fake-node:19000", "candidate_endpoint_ready": True}
        state["active_group"] = "endpoint_process"
        state["pending_question"] = {
            "id": "new_chain_method",
            "group": "endpoint_process",
            "kind": "manual_value",
            "field": "new_chain_method",
            "manual_input_allowed": True,
            "validation": {"input_mode": "rpc_method_or_schema_evidence"},
        }
        state["last_user_input"] = 'method 是 eth_blockNumber，请求是 {"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}'

        extracted = {
            "status": "draft",
            "evidence_kind": "jsonrpc_request",
            "transport": "jsonrpc",
            "method": "eth_blockNumber",
            "params": [],
            "params_json": [],
            "response_summary": "not supplied",
            "confidence": "high",
        }
        with patch("agent.harness.domains.rpc_endpoint.extract_rpc_schema_from_evidence", return_value=extracted):
            review = process_turn(state)

        self.assertEqual(review["chain_identity"]["status"], "existing_family_schema_needs_confirmation")
        self.assertEqual(_catalog_draft(review)["method"], "eth_blockNumber")
        self.assertEqual(review["pending_question"]["id"], "new_chain_schema_confirm")

        review["last_user_input"] = "Y"
        ok_probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}
        with patch("agent.harness.domains.rpc_endpoint.validate_rpc_endpoint", return_value=ok_probe):
            result = self._confirm_catalog_through_graph(review, process_turn)
        self.assertEqual(result["chain_identity"]["status"], "existing_family_method_validated_next")
        self.assertEqual(_catalog_draft(result)["method"], "eth_blockNumber")
        self.assertEqual(result["pending_question"]["id"], "new_chain_method_continue")

    def test_docs_excerpt_at_new_chain_method_question_stays_in_current_group(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["chain_identity"] = {
            "raw": "abcd",
            "canonical": "abcd",
            "adapter_family": "jsonrpc",
            "status": "existing_family_needs_method",
            "case": "case2",
        }
        state["active_group"] = "endpoint_process"
        state["pending_question"] = {
            "id": "new_chain_method",
            "group": "endpoint_process",
            "kind": "manual_value",
            "field": "new_chain_method",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "Get Blocks by ID. path Parameters id required query Parameters expand select"

        with patch("agent.harness.coordinator.resolve_action_queue", return_value={"actions": [{"type": "unknown", "confidence": "low"}]}) as router:
            result = process_turn(state)

        router.assert_called_once()
        self.assertEqual(result["chain_identity"]["status"], "existing_family_needs_method")
        self.assertNotIn("candidate_method", result["chain_identity"])
        self.assertIn("不像当前问题的答案", "\n".join(result["visible_response"]))

    def test_partial_chain_change_requires_identity_gate_not_silent_alias(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "solana", "canonical": "solana", "status": "confirmed", "case": "known"}
        state["active_group"] = "qps_profile"
        state["pending_question"] = {
            "id": "qps_profile_confirmed",
            "group": "qps_profile",
            "kind": "yes_no",
            "field": "qps_profile_confirmed",
            "options": [{"label": "Y", "value": True}, {"label": "N", "value": False}],
        }
        state["last_user_input"] = "我要切换链到 Sola"

        with (
            patch("agent.harness.coordinator.resolve_action_queue") as queue,
            patch("agent.harness.domains.chain_identity.resolve_unknown_chain_identity") as identify,
        ):
            queue.return_value = {"actions": [{"type": "change_chain", "chain_text": "Sola", "source_evidence": "切换链到 Sola", "confidence": "high"}]}
            identify.return_value = {
                "chain_exists": None,
                "canonical_chain_name": "Sola",
                "adapter_family": "unknown",
                "possible_known_chain": "solana",
                "confidence": "medium",
            }
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "chain_change_confirm")
        self.assertEqual(result["pending_question"]["kind"], "numbered_choice")
        self.assertIn("Sola", result["pending_question"]["prompt"])
        self.assertIn("solana", result["pending_question"]["prompt"])
        self.assertEqual(result["chain_identity"]["canonical"], "solana")

    def test_chain_domain_separates_candidate_identity_from_full_source_sentence(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["target_mode"] = "real-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["pending_question"] = {
            "id": "chain",
            "group": "chain_identity",
            "kind": "chain",
            "field": "chain",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "Run it against the product network AcmeLedger."

        with (
            patch("agent.harness.coordinator.resolve_action_queue") as queue,
            patch("agent.harness.domains.chain_rpc.extract_chain_mention") as mention,
        ):
            queue.return_value = {
                "actions": [
                    {
                        "type": "choose_chain",
                        "chain_text": "Run it against the product network AcmeLedger.",
                        "source_evidence": "Run it against the product network AcmeLedger.",
                        "chain_exists": None,
                        "canonical_chain_name": "AcmeLedger",
                        "adapter_family": "unknown",
                        "confidence": "medium",
                    }
                ]
            }
            mention.return_value = {
                "found": True,
                "chain_text": "AcmeLedger",
                "confidence": "high",
            }
            result = process_turn(state)

        self.assertEqual(result["chain_identity"]["raw"], "AcmeLedger")
        self.assertNotIn("Run it against", result["pending_question"]["prompt"])
        self.assertIn("AcmeLedger", result["pending_question"]["prompt"])

    def test_ambiguous_chain_turn_asks_user_to_choose_candidate(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = "我想用 fake-node 测一下，链名可能是 sola 或 solana，QPS 用 quick"

        with patch("agent.harness.coordinator.resolve_action_queue") as queue:
            queue.return_value = {
                "actions": [
                    {"type": "choose_target_mode", "target_mode": "fake-node", "target_mode_explicit": True, "source_evidence": "fake-node", "confidence": "high"},
                    {
                        "type": "choose_chain",
                        "chain_text": "sola",
                        "chain_candidates": ["sola", "solana"],
                        "source_evidence": "sola 或 solana",
                        "confidence": "high",
                    },
                    {
                        "type": "set_qps_mode",
                        "qps_mode": "quick",
                        "mutation_explicit": True,
                        "source_evidence": "QPS 用 quick",
                        "confidence": "high",
                    },
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "chain_ambiguity_confirm")
        rendered = "\n".join(result["visible_response"])
        self.assertIn("solana", rendered)
        self.assertIn("sola", rendered)

        result["last_user_input"] = "2"
        result = process_turn(result)

        self.assertEqual(result["chain_identity"]["canonical"], "solana")
        self.assertEqual(result["qps_profile"]["mode"], "quick")

    def test_chain_ambiguity_confirmation_drops_resolved_chain_actions_but_keeps_followups(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = "用 fake-node，solana 还是 bnb 都行，QPS quick"

        with patch("agent.harness.coordinator.resolve_action_queue") as queue:
            queue.return_value = {
                "actions": [
                    {"type": "choose_target_mode", "target_mode": "fake-node", "target_mode_explicit": True, "source_evidence": "fake-node", "confidence": "high"},
                    {"type": "choose_chain", "chain_text": "solana", "chain_candidates": ["solana", "bsc"], "source_evidence": "solana 还是 bnb", "confidence": "high"},
                    {"type": "set_qps_mode", "qps_mode": "quick", "mutation_explicit": True, "source_evidence": "QPS quick", "confidence": "high"},
                ]
            }
            first = process_turn(state)

        self.assertEqual(first["pending_question"]["id"], "chain_ambiguity_confirm")
        first["last_user_input"] = "1"
        second = process_turn(first)

        self.assertEqual(second["chain_identity"]["canonical"], "solana")
        self.assertEqual(second["qps_profile"]["mode"], "quick")
        self.assertNotEqual(second.get("pending_question", {}).get("id"), "chain_ambiguity_confirm")

    def test_ambiguous_chain_turn_still_prompts_when_llm_selects_known_candidate(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = "我想用 fake-node 测一下，链名可能是 sola 或 solana，QPS 用 quick"

        with patch("agent.harness.coordinator.resolve_action_queue") as queue:
            queue.return_value = {
                "actions": [
                    {"type": "choose_target_mode", "target_mode": "fake-node", "target_mode_explicit": True, "source_evidence": "fake-node", "confidence": "high"},
                    {"type": "choose_chain", "chain_text": "solana", "chain_candidates": ["sola", "solana"], "source_evidence": "sola 或 solana", "confidence": "high"},
                    {"type": "set_qps_mode", "qps_mode": "quick", "mutation_explicit": True, "source_evidence": "QPS 用 quick", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "chain_ambiguity_confirm")
        rendered = "\n".join(result["visible_response"])
        self.assertIn("solana", rendered)
        self.assertIn("sola", rendered)

    def test_pending_question_chain_detour_uses_typed_change_chain_action(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "solana", "canonical": "solana", "status": "confirmed", "case": "known"}
        state["active_group"] = "qps_profile"
        state["pending_question"] = {
            "id": "qps_profile_confirmed",
            "group": "qps_profile",
            "kind": "yes_no",
            "field": "qps_profile_confirmed",
            "options": [{"label": "Y", "value": True}, {"label": "N", "value": False}],
        }
        state["last_user_input"] = "先等一下，如果我说的其实是 Sola，不是 Solana，你应该怎么处理？"

        with (
            patch("agent.harness.coordinator.resolve_action_queue", return_value={"actions": [{"type": "change_chain", "chain_text": "Sola", "source_evidence": "Sola", "confidence": "high"}]}),
            patch("agent.harness.domains.chain_identity.resolve_unknown_chain_identity") as identify,
        ):
            identify.return_value = {
                "chain_exists": None,
                "canonical_chain_name": "Sola",
                "adapter_family": "unknown",
                "possible_known_chain": "solana",
                "confidence": "medium",
            }
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "chain_change_confirm")
        self.assertIn("Sola", result["pending_question"]["prompt"])
        self.assertIn("保持当前链", result["pending_question"]["prompt"])
        self.assertNotIn("切换到 `solana`", result["pending_question"]["prompt"])

    def test_chain_change_preserves_user_protocol_hint_for_case2(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "provider_deployment"
        state["chain_identity"] = {"raw": "solana", "canonical": "solana", "status": "confirmed", "case": "known"}
        state["pending_question"] = {
            "id": "CLOUD_REGION",
            "group": "provider_deployment",
            "kind": "manual_value",
            "field": "CLOUD_REGION",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "我需要换成 abcd，它是 EVM/jsonrpc 链"

        with patch("agent.harness.coordinator.resolve_action_queue") as queue:
            queue.return_value = {
                "actions": [
                    {
                        "type": "change_chain",
                        "chain_text": "abcd",
                        "adapter_family": "jsonrpc",
                        "chain_exists": True,
                        "source_evidence": "abcd，它是 EVM/jsonrpc 链",
                        "confidence": "high",
                    }
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "chain_change_confirm")
        self.assertIn("jsonrpc", result["pending_question"]["prompt"])
        self.assertIn("endpoint/RPC", result["pending_question"]["prompt"])

    def test_single_custom_rpc_method_accepts_inline_numeric_weight(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "endpoint_process"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["custom_rpc"] = {
            "status": "method_validated_next",
            "method": "eth_chainId",
            "params": [],
            "validated_methods": [{"method": "eth_chainId", "params": []}],
        }
        state["pending_question"] = {
            "id": "custom_rpc_continue",
            "group": "endpoint_process",
            "kind": "numbered_choice",
            "field": "custom_rpc_continue",
            "options": [
                {"label": "继续添加另一个自定义 RPC method", "value": "add_another"},
                {"label": "当前 method 已够，继续配置 workload", "value": "finish"},
            ],
            "manual_input_allowed": False,
        }
        state["last_user_input"] = "够了，只用这个 method 做 mixed，权重 100"

        with (
            patch(
                "agent.harness.coordinator.resolve_action_queue",
                return_value={"actions": [
                    {"type": "answer_pending", "answer": "够了", "selected_value": "finish", "source_evidence": "够了", "confidence": "high"},
                    {"type": "start_custom_rpc", "workload_scope": "mixed_replace", "rpc_weights": {"eth_chainId": 100}, "finish_methods": True, "confidence": "high"},
                ]},
            ),
        ):
            result = process_turn(state)

        self.assertEqual(result["rpc_mode"], "mixed")
        self.assertEqual(result["workload"]["mixed_weights"], {"eth_chainId": 100})
        self.assertTrue(result["workload"]["replace_defaults"])

    def test_compound_custom_rpc_answer_resumes_remaining_action_queue(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "endpoint_process"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["custom_rpc"] = {
            "status": "method_validated_next",
            "method": "eth_chainId",
            "params": [],
            "validated_methods": [{"method": "eth_chainId", "params": []}],
        }
        state["action_queue"] = [
            {"type": "set_qps_mode", "qps_mode": "quick", "mutation_explicit": True, "source_evidence": "quick", "confidence": "high", "_origin_text": "quick Grafana 不开"},
            {"type": "set_observability", "observability_mode": "disabled", "mutation_explicit": True, "source_evidence": "Grafana 不开", "confidence": "high", "_origin_text": "quick Grafana 不开"},
        ]
        state["pending_question"] = {
            "id": "custom_rpc_continue",
            "group": "endpoint_process",
            "kind": "numbered_choice",
            "field": "custom_rpc_continue",
            "options": [
                {"label": "继续添加另一个自定义 RPC method", "value": "add_another"},
                {"label": "当前 method 已够，继续配置 workload", "value": "finish"},
            ],
            "manual_input_allowed": False,
            "resume_action_queue": True,
        }
        state["last_user_input"] = "够了，只用这个 method 做 mixed，权重 100"

        with (
            patch(
                "agent.harness.coordinator.resolve_action_queue",
                side_effect=_admitted_mock_resolver({"actions": [
                    {"type": "answer_pending", "answer": "够了", "selected_value": "finish", "source_evidence": "够了", "pending_option_semantic_verified": True, "semantic_purpose_verified": True, "confidence": "high"},
                    {"type": "start_custom_rpc", "workload_scope": "mixed_replace", "rpc_weights": {"eth_chainId": 100}, "finish_methods": True, "confidence": "high"},
                    {"type": "set_qps_mode", "qps_mode": "quick", "mutation_explicit": True, "source_evidence": "quick", "confidence": "high"},
                    {"type": "set_observability", "observability_mode": "disabled", "mutation_explicit": True, "source_evidence": "Grafana 不开", "confidence": "high"},
                ]}),
            ),
        ):
            result = process_turn(state)

        self.assertEqual(result["workload"]["mixed_weights"], {"eth_chainId": 100})
        self.assertEqual(result["qps_profile"]["mode"], "quick")
        self.assertEqual(result["pending_question"]["id"], "qps_profile_confirm")
        self.assertEqual(result["action_queue"][0]["type"], "set_observability")

        # A compound turn may propose multiple groups, but each blocking
        # group still completes its own confirmation protocol before the
        # queue advances. The remaining action is not lost while paused.
        result["last_user_input"] = "Y"
        result = process_turn(result)

        self.assertTrue(result["qps_profile"]["confirmed"])
        self.assertEqual(result["observability"]["mode"], "disabled")
        self.assertEqual(result["pending_question"]["id"], "CLOUD_REGION")

    def test_generated_schema_question_preserves_remaining_action_queue(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "endpoint_process"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["custom_rpc"] = {
            "status": "needs_schema_evidence",
            "method": "eth_chainId",
            "endpoint_ready": True,
        }
        state["action_queue"] = [
            {"type": "set_qps_mode", "qps_mode": "quick", "confidence": "high", "_origin_text": "quick Grafana 不开"},
            {"type": "set_observability", "observability_mode": "disabled", "confidence": "high", "_origin_text": "quick Grafana 不开"},
        ]
        state["last_user_input"] = ""

        result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "custom_rpc_schema_evidence")
        self.assertTrue(result["pending_question"].get("resume_action_queue"))

    def test_pending_custom_rpc_method_finishes_subflow_before_deferred_qps(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "endpoint_process"
        state["chain_identity"] = {
            "raw": "bsc",
            "canonical": "bsc",
            "adapter_family": "jsonrpc",
            "status": "confirmed",
            "case": "known",
        }
        state["custom_rpc"] = {
            "status": "needs_method",
            "endpoint": "https://example.invalid/rpc",
            "endpoint_ready": True,
        }
        state["pending_question"] = {
            "id": "custom_rpc_method",
            "group": "endpoint_process",
            "kind": "manual_value",
            "field": "custom_rpc_method",
            "prompt": "Enter the custom RPC method name to validate.",
            "manual_input_allowed": True,
            "queue_barrier": True,
            "resume_action_queue": True,
            "validation": {"input_mode": "rpc_method_or_schema_evidence"},
        }
        state["action_queue"] = [
            {"type": "set_qps_mode", "qps_mode": "quick", "confidence": "high", "_origin_text": "quick"}
        ]
        state["last_user_input"] = "eth_chainId"

        result = process_turn(state)

        self.assertEqual(result["custom_rpc"]["status"], "needs_schema_evidence")
        self.assertEqual(result["pending_question"]["id"], "custom_rpc_schema_evidence")
        self.assertTrue(result["pending_question"].get("resume_action_queue"))
        self.assertEqual(result["action_queue"][0]["type"], "set_qps_mode")
        self.assertEqual(result.get("qps_profile"), {})
        self.assertIn("Provide schema evidence", "\n".join(result.get("visible_response") or []))

    def test_custom_rpc_catalog_intake_precedes_unresolved_target_mode(self) -> None:
        from agent.harness.coordinator import _process_action_queue
        from agent.harness.state import new_state

        state = new_state("custom-rpc-before-target-mode", language="en")
        state["active_group"] = "endpoint_process"
        state["chain_identity"] = {
            "raw": "bsc",
            "canonical": "bsc",
            "adapter_family": "jsonrpc",
            "status": "confirmed",
            "case": "known",
        }
        state["custom_rpc"] = {
            "status": "needs_method",
            "endpoint": "https://example.invalid/rpc",
            "endpoint_ready": True,
        }
        state["pending_question"] = {
            "id": "custom_rpc_method",
            "group": "endpoint_process",
            "kind": "manual_value",
            "field": "custom_rpc_method",
            "prompt": "Enter the custom RPC method name to validate.",
            "manual_input_allowed": True,
            "queue_barrier": True,
            "validation": {"input_mode": "rpc_method_or_schema_evidence"},
        }

        result = _process_action_queue(
            state,
            [
                {
                    "type": "rpc_catalog_command",
                    "catalog_command": "set_method",
                    "rpc_method": "eth_accounts",
                    "source_evidence": "Use eth_accounts for this one.",
                    "_plan_index": 0,
                },
                {
                    "type": "request_target_mode_selection",
                    "source_evidence": "I have not selected a target mode yet.",
                    "_plan_index": 1,
                },
            ],
            "Use eth_accounts for this one.",
        )

        self.assertEqual((result.get("custom_rpc") or {}).get("status"), "needs_schema_evidence")
        self.assertEqual(
            ((result.get("custom_rpc") or {}).get("catalog") or {}).get("draft", {}).get("method"),
            "eth_accounts",
        )
        self.assertEqual((result.get("pending_question") or {}).get("id"), "custom_rpc_schema_evidence")
        self.assertFalse(result.get("target_mode"))
        self.assertEqual(
            [row.get("type") for row in result.get("action_queue") or []],
            ["request_target_mode_selection"],
        )

    def test_new_chain_catalog_intake_precedes_unresolved_target_mode(self) -> None:
        from agent.harness.coordinator import _process_action_queue
        from agent.harness.state import new_state

        state = new_state("new-chain-rpc-before-target-mode", language="en")
        state["active_group"] = "endpoint_process"
        state["chain_identity"] = {
            "raw": "flow",
            "canonical": "flow",
            "adapter_family": "jsonrpc",
            "status": "existing_family_needs_method",
        }
        state["endpoint_evidence"] = {
            "candidate_endpoint": "https://example.invalid/rpc",
            "candidate_endpoint_ready": True,
        }
        state["pending_question"] = {
            "id": "new_chain_method",
            "group": "endpoint_process",
            "kind": "manual_value",
            "field": "new_chain_method",
            "prompt": "Enter the RPC method name to validate.",
            "manual_input_allowed": True,
            "queue_barrier": True,
            "validation": {"input_mode": "rpc_method_or_schema_evidence"},
        }

        result = _process_action_queue(
            state,
            [
                {
                    "type": "rpc_catalog_command",
                    "catalog_command": "set_method",
                    "rpc_method": "eth_blockNumber",
                    "source_evidence": "Use eth_blockNumber.",
                    "_plan_index": 0,
                },
                {
                    "type": "request_target_mode_selection",
                    "source_evidence": "Target mode is still undecided.",
                    "_plan_index": 1,
                },
            ],
            "Use eth_blockNumber.",
        )

        self.assertEqual(result["chain_identity"]["status"], "existing_family_needs_schema_evidence")
        self.assertEqual(
            ((result.get("custom_rpc") or {}).get("catalog") or {}).get("draft", {}).get("method"),
            "eth_blockNumber",
        )
        self.assertEqual((result.get("pending_question") or {}).get("id"), "new_chain_schema_evidence")
        self.assertNotIn("change_group", [row.get("type") for row in result.get("completed_actions") or []])
        self.assertFalse(result.get("target_mode"))
        self.assertEqual(
            [row.get("type") for row in result.get("action_queue") or []],
            ["request_target_mode_selection"],
        )

    def test_custom_rpc_workload_selection_still_waits_for_target_mode(self) -> None:
        from agent.harness.action_registry import ACTION_BY_TYPE
        from agent.harness.coordinator import _process_action_queue
        from agent.harness.state import new_state

        state = new_state("custom-rpc-workload-before-target-mode", language="en")
        state["active_group"] = "endpoint_process"
        state["chain_identity"] = {
            "raw": "bsc",
            "canonical": "bsc",
            "adapter_family": "jsonrpc",
            "status": "confirmed",
            "case": "known",
        }

        result = _process_action_queue(
            state,
            [
                {
                    "type": "rpc_workload_command",
                    "workload_scope": "single_replace",
                    "finish_methods": True,
                    "_plan_index": 0,
                },
                {
                    "type": "request_target_mode_selection",
                    "source_evidence": "Target mode is still undecided.",
                    "_plan_index": 1,
                },
            ],
            "Use only the validated method.",
        )

        self.assertIn("target_mode", ACTION_BY_TYPE["rpc_workload_command"].requires_capabilities)
        self.assertEqual((result.get("pending_question") or {}).get("id"), "target_mode_select")
        self.assertEqual([row.get("type") for row in result.get("action_queue") or []], ["rpc_workload_command"])
        self.assertNotIn("rpc_workload_command", [row.get("type") for row in result.get("completed_actions") or []])
        self.assertFalse(result.get("target_mode"))

    def test_start_custom_rpc_reentry_resumes_from_existing_endpoint_and_method(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "qps_profile"
        state["chain_identity"] = {
            "raw": "bsc",
            "canonical": "bsc",
            "adapter_family": "jsonrpc",
            "status": "confirmed",
            "case": "known",
        }
        state["rpc_mode"] = "single"
        state["workload"] = {"choice": "custom_rpc"}
        state["custom_rpc"] = {
            "status": "needs_endpoint",
            "endpoint": "https://example.invalid/rpc",
            "endpoint_ready": True,
            "method": "eth_chainId",
        }
        state["pending_question"] = {
            "id": "qps_profile_confirm",
            "group": "qps_profile",
            "kind": "yes_no",
            "field": "qps_profile_confirmed",
            "prompt": "Use defaults?",
            "manual_input_allowed": False,
            "options": [{"label": "Y", "value": True}, {"label": "N", "value": False}],
            "queue_barrier": True,
        }
        state["last_user_input"] = "Return to the unfinished custom RPC setup before QPS."

        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [
                {"type": "start_custom_rpc", "source_evidence": state["last_user_input"], "confidence": "high"},
                {"type": "change_group", "group": "endpoint_process", "navigation_explicit": True, "source_evidence": state["last_user_input"], "confidence": "high"},
            ]},
        ):
            result = process_turn(state)

        self.assertEqual(result["custom_rpc"]["status"], "needs_schema_evidence")
        self.assertEqual(result["pending_question"]["id"], "custom_rpc_schema_evidence")
        self.assertEqual(result["active_group"], "endpoint_process")
        self.assertEqual(result["custom_rpc"]["endpoint"], "https://example.invalid/rpc")
        self.assertEqual(_catalog_draft(result)["method"], "eth_chainId")

    def test_custom_rpc_without_chain_stops_at_typed_chain_prerequisite(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["target_mode"] = "real-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["last_user_input"] = "Use one custom method at http://geth-dev:8545."

        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{
                "type": "start_custom_rpc",
                "rpc_endpoint": "http://geth-dev:8545",
                "source_evidence": state["last_user_input"],
                "confidence": "high",
            }]},
        ):
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "chain")
        self.assertTrue(result["pending_question"]["queue_barrier"])
        self.assertEqual(result["action_queue"][0]["type"], "rpc_catalog_command")
        self.assertNotIn("endpoint_probe", result.get("custom_rpc") or {})

    def test_finishing_custom_rpc_method_displays_scope_question(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.chain_rpc import question_for_chain_rpc
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "endpoint_process"
        state["chain_identity"] = {
            "canonical": "bsc",
            "adapter_family": "jsonrpc",
            "status": "confirmed",
            "case": "known",
        }
        state["custom_rpc"] = {
            "status": "method_validated_next",
            "endpoint_ready": True,
            "method": "eth_chainId",
            "params": [],
            "validated_methods": [{"method": "eth_chainId", "params": []}],
        }
        state["pending_question"] = question_for_chain_rpc(state, "endpoint_process") or {}
        state["pending_question"]["resume_action_queue"] = True
        state["action_queue"] = [
            {"type": "set_observability", "observability_mode": "disabled", "mutation_explicit": True, "source_evidence": "disable observability", "confidence": "high"}
        ]
        state["last_user_input"] = "2"

        result = process_turn(state)

        self.assertEqual(result["custom_rpc"]["status"], "needs_scope")
        self.assertEqual(result["pending_question"]["id"], "custom_rpc_scope")
        self.assertIn("Choose how to apply", "\n".join(result.get("visible_response") or []))
        self.assertEqual(result["action_queue"][0]["type"], "set_observability")

    def test_completed_custom_rpc_scope_keeps_interrupted_qps_after_deferred_observability(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.chain_rpc import question_for_chain_rpc
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "endpoint_process"
        state["chain_identity"] = {
            "canonical": "bsc",
            "adapter_family": "jsonrpc",
            "status": "confirmed",
            "case": "known",
        }
        state["rpc_mode"] = "single"
        state["qps_profile"] = {"mode": "quick", "confirmed": False, "default_decision_made": False}
        state["custom_rpc"] = {
            "status": "needs_scope",
            "endpoint_ready": True,
            "method": "eth_chainId",
            "params": [],
            "validated_methods": [{"method": "eth_chainId", "params": []}],
        }
        state["pending_question"] = question_for_chain_rpc(state, "endpoint_process") or {}
        state["pending_question"]["resume_action_queue"] = True
        state["interruption_stack"] = [
            {"group": "qps_profile", "question_id": "qps_profile_confirm", "reason": "user_detour"}
        ]
        state["action_queue"] = [
            {"type": "set_observability", "observability_mode": "disabled", "mutation_explicit": True, "source_evidence": "disable observability", "confidence": "high"}
        ]
        state["last_user_input"] = "1"

        result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "qps_profile_confirm")
        self.assertEqual(result["observability"]["mode"], "disabled")
        self.assertEqual(result.get("action_queue"), [])

    def test_current_config_absorbs_duplicate_next_action_consultation(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"canonical": "bsc", "status": "confirmed"}
        state["custom_rpc"] = {"status": "validated", "job_local_override": True}
        state["last_user_input"] = "Show the current config and exact next action."

        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [
                {"type": "answer_opening_question", "topic": "current_config", "confidence": "high"},
                {"type": "answer_opening_question", "topic": "next_action", "confidence": "high"},
            ]},
        ):
            result = process_turn(state)

        response = "\n".join(result.get("visible_response") or [])
        self.assertEqual(response.count("Current state:"), 1)
        self.assertEqual(len(result.get("visible_response") or []), 1)
        self.assertIn("original config/chains template: unchanged", response)

    def test_current_config_and_job_history_remain_distinct_consultation_results(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.graph import _next_result
        from agent.harness.state import new_state

        state = new_state("compound-consultation", language="en")
        state["last_user_input"] = (
            "Do I have any saved configuration or previous jobs? "
            "Show me the concrete current status."
        )
        state["active_group"] = "opening"
        state["pending_question"] = {
            "id": "opening_next_action",
            "group": "opening",
            "kind": "numbered_choice",
            "options": [],
        }

        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [
                {
                    "type": "answer_opening_question",
                    "topic": "current_config",
                    "confidence": "high",
                },
                {
                    "type": "answer_opening_question",
                    "topic": "current_job",
                    "confidence": "high",
                },
            ]},
        ), patch("agent.harness.domains.orientation.list_jobs", return_value=[]):
            result = process_turn(state)

        response = "\n".join(result.get("visible_response") or [])
        self.assertIn("Current state:", response)
        self.assertIn("No historical job", response)
        self.assertEqual((result.get("pending_question") or {}).get("id"), "opening_next_action")
        next_result = _next_result(result)
        self.assertEqual(next_result.get("kind"), "result")
        self.assertTrue(next_result.get("pending_overlay"))
        self.assertEqual(next_result.get("question_id"), "opening_next_action")

    def test_specific_workload_consultation_absorbs_generic_config_responses(self) -> None:
        from agent.harness.coordinator import _drop_conflicting_answer_actions
        from agent.harness.state import new_state

        state = new_state("workload-consultation", language="en")
        actions = [
            {"type": "answer_opening_question", "topic": "config_explanation"},
            {"type": "answer_opening_question", "topic": "current_config"},
            {"type": "answer_opening_question", "topic": "workload_config"},
            {"type": "set_qps_mode", "qps_mode": "quick"},
        ]

        pruned = _drop_conflicting_answer_actions(state, actions)

        self.assertEqual(
            [item.get("topic") for item in pruned if item.get("type") == "answer_opening_question"],
            ["workload_config"],
        )
        self.assertTrue(any(item.get("type") == "set_qps_mode" for item in pruned))

    def test_rpc_mutation_uses_its_actionable_workload_question_as_the_consultation_answer(self) -> None:
        from agent.harness.coordinator import _drop_conflicting_answer_actions
        from agent.harness.state import new_state

        state = new_state("workload-mutation-consultation", language="en")
        actions = [
            {"type": "set_rpc_mode", "rpc_mode": "mixed"},
            {"type": "answer_opening_question", "topic": "workload_config"},
            {"type": "set_qps_mode", "qps_mode": "quick"},
        ]

        pruned = _drop_conflicting_answer_actions(state, actions)

        self.assertFalse(any(item.get("topic") == "workload_config" for item in pruned))
        self.assertEqual(
            [item.get("type") for item in pruned],
            ["set_rpc_mode", "set_qps_mode"],
        )

    def test_schema_answer_propagates_resume_to_next_custom_rpc_question(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "endpoint_process"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["custom_rpc"] = {
            "status": "needs_schema_evidence",
            "method": "eth_chainId",
            "endpoint": "https://example.invalid/rpc",
            "endpoint_ready": True,
        }
        state["action_queue"] = [{
            "type": "set_qps_mode",
            "qps_mode": "quick",
            "mutation_explicit": True,
            "source_evidence": "quick",
            "confidence": "high",
            "_origin_text": "quick",
        }]
        state["pending_question"] = {
            "id": "custom_rpc_schema_evidence",
            "group": "endpoint_process",
            "kind": "evidence",
            "field": "custom_rpc_schema_evidence",
            "manual_input_allowed": True,
            "accepted_action_types": ["rpc_catalog_command"],
            "manual_action": {
                "type": "rpc_catalog_command",
                "catalog_command": "append_evidence",
                "value_argument": "rpc_schema_evidence",
                "use_complete_turn": True,
            },
            "queue_barrier": True,
            "resume_action_queue": True,
        }
        state["last_user_input"] = "[]"
        probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}

        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{
                "type": "rpc_catalog_command",
                "catalog_command": "append_evidence",
                "rpc_schema_evidence": state["last_user_input"],
                "source_evidence": state["last_user_input"],
                "confidence": "high",
            }]},
        ), patch("agent.harness.domains.rpc_endpoint.extract_rpc_schema_from_evidence", return_value={"status": "draft", "method": "eth_chainId", "params": [], "params_json": [], "response_summary": "hex chain id", "confidence": "high"}), patch("agent.harness.domains.rpc_endpoint.validate_rpc_endpoint", return_value=probe):
            result = process_turn(state)
            self.assertEqual(result["pending_question"]["id"], "custom_rpc_schema_confirm")
            self.assertTrue(result["pending_question"].get("resume_action_queue"))
            result = self._confirm_catalog_through_graph(result, process_turn)

        self.assertEqual(result["pending_question"]["id"], "custom_rpc_continue")
        self.assertTrue(result["pending_question"].get("resume_action_queue"))

    def test_custom_rpc_schema_accepts_no_parameters_natural_language(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "endpoint_process"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["custom_rpc"] = {
            "status": "needs_schema_evidence",
            "method": "eth_chainId",
            "endpoint": "https://example.invalid/rpc",
            "endpoint_ready": True,
        }
        state["pending_question"] = {
            "id": "custom_rpc_schema_evidence",
            "group": "endpoint_process",
            "kind": "evidence",
            "field": "custom_rpc_schema_evidence",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "没有参数"
        probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}

        with (
            patch(
                "agent.harness.coordinator.resolve_action_queue",
                return_value={"actions": [{
                    "type": "rpc_catalog_command",
                    "catalog_command": "append_evidence",
                    "rpc_schema_evidence": state["last_user_input"],
                    "source_evidence": state["last_user_input"],
                    "confidence": "high",
                }]},
            ),
            patch(
                "agent.harness.domains.rpc_endpoint.extract_rpc_schema_from_evidence",
                return_value={
                    "status": "draft",
                    "evidence_kind": "parameter_schema",
                    "method": "eth_chainId",
                    "params": [],
                    "params_json": [],
                    "response_summary": "unknown",
                    "confidence": "high",
                },
            ),
            patch("agent.harness.domains.rpc_endpoint.validate_rpc_endpoint", return_value=probe),
        ):
            result = process_turn(state)

        self.assertEqual(_catalog_draft(result)["params"], [])
        self.assertEqual(result["pending_question"]["id"], "custom_rpc_schema_confirm")

    def test_rpc_evidence_domain_rejects_fact_free_consultation_transactionally(self) -> None:
        from agent.harness.contracts import ActionProposal
        from agent.harness.domains.chain_rpc import apply_chain_rpc_action
        from agent.harness.state import new_state

        state = new_state("fact-free-rpc-evidence", language="en")
        state["target_mode"] = "real-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "LocalEvmDemo",
            "canonical": "LocalEvmDemo",
            "adapter_family": "jsonrpc",
            "status": "existing_family_needs_schema_evidence",
            "case": "case2",
        }
        state["custom_rpc"] = {
            "catalog": {
                "contract_version": 1,
                "revision": 2,
                "methods": [],
                "draft": {
                    "method": "eth_chainId",
                    "phase": "evidence",
                    "revision": 2,
                    "evidence": [],
                },
                "finished": False,
            },
        }
        state["pending_question"] = {
            "id": "new_chain_schema_evidence",
            "group": "endpoint_process",
            "kind": "evidence",
            "field": "new_chain_schema_evidence",
            "manual_input_allowed": True,
            "created_turn_index": 0,
        }
        before_catalog = deepcopy(state["custom_rpc"]["catalog"])
        before_question = deepcopy(state["pending_question"])

        with patch(
            "agent.harness.domains.rpc_endpoint.extract_rpc_schema_from_evidence",
            return_value={
                "status": "draft",
                "evidence_kind": "unknown",
                "method": "",
                "params": [],
                "response_summary": "unknown",
            },
        ):
            outcome = apply_chain_rpc_action(
                state,
                ActionProposal(
                    "fact-free",
                    "rpc_catalog_command",
                    {
                        "catalog_command": "append_evidence",
                        "rpc_schema_evidence": "Summarize the chain, endpoint, and current method you retained.",
                    },
                    "high",
                ),
            )
            result = _commit_result(state, outcome, owner="chain_rpc")

        self.assertEqual(result["custom_rpc"]["catalog"], before_catalog)
        self.assertEqual(result["pending_question"], before_question)
        self.assertIn("nothing was written", result["visible_response"][0])

    def test_rpc_evidence_domain_commits_source_grounded_zero_parameter_docs(self) -> None:
        from agent.harness.contracts import ActionProposal
        from agent.harness.domains.chain_rpc import apply_chain_rpc_action
        from agent.harness.state import new_state

        state = new_state("grounded-rpc-evidence", language="en")
        state["target_mode"] = "real-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "bsc",
            "canonical": "bsc",
            "adapter_family": "jsonrpc",
            "status": "confirmed",
            "case": "known",
        }
        state["custom_rpc"] = {
            "status": "needs_schema_evidence",
            "endpoint": "http://geth-dev:8545",
            "endpoint_ready": True,
            "catalog": {
                "contract_version": 1,
                "revision": 1,
                "methods": [],
                "draft": {
                    "method": "eth_chainId",
                    "phase": "evidence",
                    "revision": 1,
                    "evidence": [],
                },
                "finished": False,
            },
        }
        extracted = {
            "status": "draft",
            "evidence_kind": "docs_excerpt",
            "method": "eth_chainId",
            "params": [],
            "params_json": [],
            "response_summary": "hex quantity chain identifier",
            "response_fields": [],
        }

        with patch(
            "agent.harness.domains.rpc_endpoint.extract_rpc_schema_from_evidence",
            return_value=extracted,
        ):
            outcome = apply_chain_rpc_action(
                state,
                ActionProposal(
                    "grounded",
                    "rpc_catalog_command",
                    {
                        "catalog_command": "append_evidence",
                        "rpc_schema_evidence": "Official docs: eth_chainId takes no parameters and returns a hex quantity chain identifier.",
                    },
                    "high",
                ),
            )
            result = _commit_result(state, outcome, owner="chain_rpc")

        self.assertGreater(result["custom_rpc"]["catalog"]["revision"], 1)
        self.assertEqual(result["pending_question"]["id"], "custom_rpc_schema_confirm")

    def test_legacy_custom_rpc_compiles_explicit_schema_evidence_action(self) -> None:
        from agent.harness.coordinator import _apply_queue_action
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["last_user_input"] = (
            "我要用 fake-node 测 BSC，添加自定义 RPC method eth_chainId，"
            "endpoint 是 https://example.invalid/rpc，没有参数，mixed 只跑这个 method，权重 100"
        )
        state["pending_question"] = {}
        action = {
            "type": "start_custom_rpc",
            "rpc_method": "eth_chainId",
            "rpc_endpoint": "https://example.invalid/rpc",
            "rpc_schema_evidence": "没有参数",
            "workload_scope": "mixed_replace",
            "rpc_weights": {"eth_chainId": 100},
            "finish_methods": True,
            "source_evidence": state["last_user_input"],
            "confidence": "high",
        }
        probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}

        with patch("agent.harness.domains.rpc_endpoint.extract_rpc_schema_from_evidence", return_value={"status": "draft", "method": "eth_chainId", "params": [], "params_json": [], "response_summary": "hex chain id", "confidence": "high"}), patch("agent.harness.domains.rpc_endpoint.validate_rpc_endpoint", return_value=probe):
            from agent.harness.action_registry import compile_legacy_custom_rpc_action
            result = state
            for compiled in compile_legacy_custom_rpc_action(action):
                result = _apply_queue_action(result, compiled, state["last_user_input"])
            self.assertEqual(result["custom_rpc"]["status"], "schema_needs_confirmation")
            result = self._confirm_catalog_through_domain(result)

        self.assertNotIn("LOCAL_RPC_URL", result.get("confirmed_config", {}))
        self.assertEqual(_catalog_draft(result)["params_json"], [])
        self.assertEqual(result["custom_rpc"]["status"], "validated")
        self.assertEqual(result["rpc_mode"], "mixed")
        self.assertEqual(result["workload"]["mixed_weights"], {"eth_chainId": 100})
        self.assertTrue(result["workload"]["replace_defaults"])
        self.assertIn("自定义 RPC mixed workload 已确认", "\n".join(result["visible_response"]))

    def test_catalog_method_action_does_not_steal_schema_from_turn_text(self) -> None:
        from agent.harness.contracts import ActionProposal
        from agent.harness.domains.chain_rpc import apply_chain_rpc_action
        from agent.harness.state import new_state

        state = new_state("single-catalog-transition", language="en")
        state.update({
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "chain_identity": {
                "raw": "bsc",
                "canonical": "bsc",
                "adapter_family": "jsonrpc",
                "status": "confirmed",
            },
            "custom_rpc": {
                "status": "needs_method",
                "endpoint": "http://geth-dev:8545",
                "endpoint_ready": True,
            },
            "last_user_input": "Use eth_chainId; it has no parameters and returns a hex quantity.",
        })

        with patch("agent.harness.domains.rpc_endpoint.extract_rpc_schema_from_evidence") as extractor:
            outcome = apply_chain_rpc_action(
                state,
                ActionProposal(
                    "set-method-only",
                    "rpc_catalog_command",
                    {"catalog_command": "set_method", "rpc_method": "eth_chainId"},
                    "high",
                ),
            )
            result = _commit_result(state, outcome, owner="chain_rpc")

        extractor.assert_not_called()
        self.assertEqual(result["custom_rpc"]["status"], "needs_schema_evidence")
        self.assertEqual(_catalog_draft(result)["method"], "eth_chainId")
        self.assertFalse(_catalog_draft(result).get("params_json"))

    def test_custom_rpc_validation_recovers_inline_weight_hint_after_method_is_known(self) -> None:
        from agent.harness.domains.chain_rpc import apply_chain_rpc_answer
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["custom_rpc"] = {
            "method": "eth_chainId",
            "endpoint": "https://example.invalid/rpc",
            "endpoint_ready": True,
            "requested_workload": {
                "scope": "mixed_replace",
                "weights": {"eth_chainId": 100},
                "finish_methods": True,
            },
        }
        state["last_user_input"] = "没有参数，mixed 只跑这个 method，权重 100，quick，Grafana 不开"
        probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}

        with patch("agent.harness.domains.rpc_endpoint.extract_rpc_schema_from_evidence", return_value={"status": "draft", "method": "eth_chainId", "params": [], "params_json": [], "response_summary": "hex chain id", "confidence": "high"}), patch("agent.harness.domains.rpc_endpoint.validate_rpc_endpoint", return_value=probe):
            outcome = apply_chain_rpc_answer(
                state,
                {
                    "id": "custom_rpc_schema_evidence",
                    "group": "endpoint_process",
                    "field": "custom_rpc_schema_evidence",
                    "kind": "evidence",
                },
                "[]",
                "[]",
            )
            result = _commit_result(state, outcome, owner="chain_rpc")
            self.assertEqual(result["custom_rpc"]["status"], "schema_needs_confirmation")
            before_confirmation = result
            outcome = apply_chain_rpc_answer(
                result,
                {
                    "id": "custom_rpc_schema_confirm",
                    "group": "endpoint_process",
                    "field": "custom_rpc_schema_confirm",
                    "kind": "yes_no",
                },
                True,
                "y",
            )
            result = _commit_result(before_confirmation, outcome, owner="chain_rpc")
            result = self._confirm_catalog_through_domain(result)

        self.assertEqual(result["custom_rpc"]["status"], "validated")
        self.assertEqual(result["workload"]["mixed_weights"], {"eth_chainId": 100})
        self.assertTrue(result["workload"]["replace_defaults"])

    def test_inline_weight_ignores_endpoint_version_numbers(self) -> None:
        from agent.harness.input_values import parse_weight_spec_for_methods

        text = "endpoint 是 https://example.invalid/v1/token，没有参数，mixed 只跑这个 method，权重 100"
        weights = parse_weight_spec_for_methods(text, ["eth_chainId"])

        self.assertEqual(weights, {"eth_chainId": 100})

    def test_custom_rpc_schema_confirmation_uses_original_turn_for_weights(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["custom_rpc"] = {
            "method": "eth_chainId",
            "endpoint": "https://example.invalid/rpc",
            "endpoint_ready": True,
            "source_turn_text": "没有参数，mixed 只跑这个 method，权重 100，quick，Grafana 不开",
            "requested_workload": {"scope": "mixed_replace", "weights": {"eth_chainId": 100}, "finish_methods": True},
            "schema_draft": {
                "status": "draft",
                "method": "eth_chainId",
                "params": [],
                "params_json": [],
                "evidence_kind": "jsonrpc_request",
                "transport": "jsonrpc",
                "response_summary": "hex chain id",
            },
            "status": "schema_needs_confirmation",
        }
        state["pending_question"] = {
            "id": "custom_rpc_schema_confirm",
            "group": "endpoint_process",
            "kind": "yes_no",
            "field": "custom_rpc_schema_confirm",
            "options": [{"label": "Y", "value": True}, {"label": "N", "value": False}],
        }
        state["last_user_input"] = "Y"
        probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}

        with patch("agent.harness.domains.rpc_endpoint.validate_rpc_endpoint", return_value=probe):
            result = self._confirm_catalog_through_graph(state, process_turn)

        self.assertEqual(result["custom_rpc"]["status"], "validated")
        self.assertEqual(result["workload"]["mixed_weights"], {"eth_chainId": 100})
        self.assertIn("自定义 RPC mixed workload 已确认", "\n".join(result["visible_response"]))

    def test_case2_protocol_confirm_consumes_queued_endpoint_and_method(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        source = (
            "我想改测 Flow，它是 EVM/jsonrpc 链，endpoint 是 https://example.invalid/rpc，"
            "method 用 eth_blockNumber，没有参数"
        )
        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "flow",
            "canonical": "flow",
            "adapter_family": "jsonrpc",
            "status": "needs_identity_confirmation",
            "case": "unknown",
            "llm_resolution": {"canonical_chain_name": "flow", "adapter_family": "jsonrpc", "chain_exists": True},
        }
        state["pending_question"] = {
            "id": "unknown_chain_identity_confirm",
            "group": "chain_identity",
            "kind": "numbered_choice",
            "field": "unknown_chain_decision",
            "options": [{"label": "确认", "value": "confirm_proposed_protocol"}],
            "queue_barrier": True,
            "resume_action_queue": True,
        }
        state["action_queue"] = [
            {
                "type": "rpc_catalog_command",
                "catalog_command": "set_endpoint",
                "rpc_endpoint": "https://example.invalid/rpc",
                "source_evidence": source,
                "_origin_text": source,
                "confidence": "medium",
            },
            {
                "type": "rpc_catalog_command",
                "catalog_command": "set_method",
                "rpc_method": "eth_blockNumber",
                "source_evidence": source,
                "_origin_text": source,
                "confidence": "medium",
            },
        ]
        state["last_user_input"] = "1"
        probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}

        with patch("agent.harness.domains.rpc_endpoint.validate_rpc_endpoint", return_value=probe):
            result = process_turn(state)

        self.assertEqual(result["endpoint_evidence"]["candidate_endpoint"], "https://example.invalid/rpc")
        self.assertIn(result["chain_identity"]["status"], {"existing_family_needs_schema_evidence", "existing_family_schema_needs_confirmation", "existing_family_method_validated_next", "existing_family_runtime_choice"})
        self.assertNotEqual(result["pending_question"].get("id"), "new_chain_endpoint")

    def test_case3_handoff_stops_instead_of_falling_back_to_chain_prompt(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "sola",
            "canonical": "sola",
            "adapter_family": "unknown",
            "status": "needs_protocol_confirmation",
            "case": "unknown",
        }
        state["pending_question"] = {
            "id": "adapter_family_confirm",
            "group": "chain_identity",
            "kind": "numbered_choice",
            "field": "adapter_family",
            "options": [{"label": "不属于以上协议族", "value": "unsupported"}],
        }
        state["last_user_input"] = "unsupported"

        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{
                "type": "answer_pending",
                "answer": state["last_user_input"],
                "selected_value": state["last_user_input"],
                "source_evidence": state["last_user_input"],
                "confidence": "high",
            }]},
        ):
            result = process_turn(state)

        self.assertEqual(result["chain_identity"]["status"], "unsupported_family_handoff")
        self.assertEqual(result.get("pending_question"), {})
        self.assertIn("二次开发", "\n".join(result["visible_response"]))
        self.assertNotIn("你想测试哪条链", "\n".join(result["visible_response"]))

    def test_case3_handoff_collects_followup_evidence_without_fallback(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "WeirdP2PChain",
            "canonical": "WeirdP2PChain",
            "adapter_family": "unsupported",
            "status": "unsupported_family_handoff",
            "case": "case3",
        }
        state["secondary_handoff"] = {
            "status": "collecting_evidence",
            "kind": "case3_protocol_adapter_implementation",
            "evidence": [],
        }
        state["active_group"] = "chain_identity"
        state["pending_question"] = {
            "id": "case3_protocol_evidence",
            "group": "chain_identity",
            "kind": "evidence",
            "field": "case3_protocol_evidence",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "protocol: weird-p2p"

        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{
                "type": "secondary_handoff_command",
                "handoff_command": "append_evidence",
                "handoff_evidence": state["last_user_input"],
                "confidence": "high",
            }]},
        ):
            result = process_turn(state)

        self.assertEqual(result["chain_identity"]["status"], "case3_collecting_evidence")
        self.assertEqual(result["secondary_handoff"]["evidence"], ["protocol: weird-p2p"])
        self.assertEqual(result.get("pending_question", {}).get("id"), "case3_evidence_next")
        self.assertIn("已记录", "\n".join(result.get("visible_response") or []))
        self.assertNotIn("你想测试哪条链", "\n".join(result.get("visible_response") or []))

    def test_case3_handoff_evidence_shaped_navigation_routes_through_resolver(self) -> None:
        """Phase 6 item 2: the keyword blocklist `_looks_like_handoff_navigation`

        was removed. A turn that is evidence-shaped (contains an evidence token
        such as `http`) but is actually a navigation/command ("analyze my latest
        report") must be classified by the action-queue resolver and routed
        away, not captured as secondary-development evidence.
        """

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["job"] = {"job_id": "job_demo", "status": "completed"}
        state["chain_identity"] = {
            "raw": "WeirdP2PChain",
            "canonical": "WeirdP2PChain",
            "adapter_family": "unsupported",
            "status": "case3_collecting_evidence",
            "case": "case3",
        }
        state["secondary_handoff"] = {
            "status": "collecting_evidence",
            "kind": "case3_protocol_adapter_implementation",
            "evidence": ["protocol: weird-p2p"],
        }
        state["last_user_input"] = "先别管这些，帮我分析最近一次 job 的 http 报告"

        with patch("agent.harness.coordinator.resolve_action_queue", return_value={"actions": [{"type": "analyze_report", "confidence": "high"}]}):
            result = process_turn(state)

        # Evidence must NOT grow — the resolver classified this as navigation.
        self.assertEqual(result["secondary_handoff"]["evidence"], ["protocol: weird-p2p"])
        self.assertNotIn("已记录第", "\n".join(result.get("visible_response") or []))

    def test_case3_handoff_generation_request_uses_collected_evidence(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["chain_identity"] = {
            "raw": "WeirdP2PChain",
            "canonical": "WeirdP2PChain",
            "adapter_family": "unsupported",
            "status": "unsupported_family_handoff",
            "case": "case3",
        }
        state["secondary_handoff"] = {
            "status": "collecting_evidence",
            "kind": "case3_protocol_adapter_implementation",
            "evidence": ["protocol: weird-p2p"],
        }
        state["active_group"] = "chain_identity"
        state["pending_question"] = {
            "id": "case3_evidence_next",
            "group": "chain_identity",
            "kind": "numbered_choice",
            "field": "case3_evidence_next",
            "options": [
                {"label": "继续补充证据", "value": "add_more"},
                {"label": "生成二次开发交接", "value": "generate_handoff"},
            ],
        }
        state["last_user_input"] = "2"

        result = process_turn(state)

        self.assertEqual(len(result["secondary_handoff"]["evidence"]), 1)
        self.assertEqual(result.get("pending_question"), {})
        self.assertIn("adapter", "\n".join(result.get("visible_response") or []))
        self.assertIn("二次开发交接草案", "\n".join(result.get("visible_response") or []))
        self.assertIn("protocol: weird-p2p", "\n".join(result.get("visible_response") or []))

    def test_case3_handoff_does_not_capture_report_analysis_request(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["job"] = {"job_id": "job_demo", "status": "completed"}
        state["chain_identity"] = {
            "raw": "WeirdP2PChain",
            "canonical": "WeirdP2PChain",
            "adapter_family": "unsupported",
            "status": "unsupported_family_handoff",
            "case": "case3",
        }
        state["secondary_handoff"] = {
            "status": "collecting_evidence",
            "kind": "case3_protocol_adapter_implementation",
            "evidence": ["protocol: weird-p2p"],
        }
        state["last_user_input"] = "先别管之前配置，帮我分析最近一次 job 的报告和日志"

        with (
            patch("agent.harness.coordinator.resolve_action_queue") as resolver,
            patch("agent.harness.domains.analysis.list_jobs", return_value=[]),
            patch("agent.harness.domains.analysis.resume_job") as resume,
        ):
            resolver.return_value = {"actions": [{"type": "analyze_report", "confidence": "high"}]}
            resume.return_value = {
                "job_id": "job_demo",
                "status": "completed",
                "run_dir": ".agent/jobs/job_demo",
                "runtime_env_file": ".agent/jobs/job_demo/runtime.env",
                "artifact_index": ".agent/jobs/job_demo/artifact_index.json",
                "next_actions": ["status", "analyze"],
            }
            result = process_turn(state)

        self.assertEqual(result["secondary_handoff"]["evidence"], ["protocol: weird-p2p"])
        self.assertIn("job_demo", "\n".join(result.get("visible_response") or []))

    def test_case3_handoff_does_not_capture_generic_test_help_request(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["chain_identity"] = {
            "raw": "WeirdP2PChain",
            "canonical": "WeirdP2PChain",
            "adapter_family": "unsupported",
            "status": "unsupported_family_handoff",
            "case": "case3",
        }
        state["secondary_handoff"] = {
            "status": "collecting_evidence",
            "kind": "case3_protocol_adapter_implementation",
            "evidence": ["protocol: weird-p2p"],
        }
        state["last_user_input"] = "我需要测试，但是我不知道可以测试什么"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "greeting", "confidence": "medium"}]}
            result = process_turn(state)

        self.assertEqual(result["secondary_handoff"]["evidence"], ["protocol: weird-p2p"])
        self.assertNotIn("二次开发证据", "\n".join(result.get("visible_response") or []))

    def test_case3_handoff_does_not_capture_confusion_question(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["chain_identity"] = {
            "raw": "WeirdP2PChain",
            "canonical": "WeirdP2PChain",
            "adapter_family": "unsupported",
            "status": "unsupported_family_handoff",
            "case": "case3",
        }
        state["secondary_handoff"] = {
            "status": "collecting_evidence",
            "kind": "case3_protocol_adapter_implementation",
            "evidence": ["protocol: weird-p2p"],
        }
        state["last_user_input"] = "什么意思？你在讲什么"

        result = _invoke_with_admitted_actions(
            process_turn,
            state,
            [{
                "type": "answer_opening_question",
                "topic": "current_context",
                "source_evidence": state["last_user_input"],
                "confidence": "high",
            }],
        )

        self.assertEqual(result["secondary_handoff"]["evidence"], ["protocol: weird-p2p"])
        self.assertNotIn("已记录第", "\n".join(result.get("visible_response") or []))

    def test_pending_config_review_merges_additional_yaml_lines(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["inferred_config"] = {
            "pending_review": {
                "config_values": {"CLOUD_REGION": "asia-east1"},
                "unmapped_values": {},
                "source_format": "mixed",
                "reason": "initial line",
            }
        }
        state["pending_question"] = {
            "id": "inferred_config_review",
            "group": "provider_deployment",
            "kind": "yes_no",
            "field": "inferred_config_review",
            "options": [{"label": "Y", "value": True}, {"label": "N", "value": False}],
            "accepted_action_types": ["answer_pending", "propose_config_values"],
        }
        state["last_user_input"] = "CLOUD_ZONE: asia-east1-c"

        result = _invoke_with_admitted_actions(
            process_turn,
            state,
            [{
                "type": "propose_config_values",
                "source_format": "yaml",
                "config_values": {"CLOUD_ZONE": "asia-east1-c"},
                "unmapped_values": {},
                "source_evidence": state["last_user_input"],
                "confidence": "high",
            }],
        )

        proposal = result["inferred_config"]["pending_review"]["config_values"]
        self.assertEqual(proposal["CLOUD_REGION"], "asia-east1")
        self.assertEqual(proposal["CLOUD_ZONE"], "asia-east1-c")
        self.assertEqual(result["pending_question"]["id"], "inferred_config_review")
        self.assertIn("CLOUD_ZONE", "\n".join(result["visible_response"]))

        result["last_user_input"] = "MACHINE_TYPE=n2-standard-16"
        merged = _invoke_with_admitted_actions(
            process_turn,
            result,
            [{
                "type": "propose_config_values",
                "source_format": "env",
                "config_values": {"MACHINE_TYPE": "n2-standard-16"},
                "unmapped_values": {},
                "source_evidence": result["last_user_input"],
                "confidence": "high",
            }],
        )

        proposal = merged["inferred_config"]["pending_review"]["config_values"]
        self.assertEqual(proposal["CLOUD_REGION"], "asia-east1")
        self.assertEqual(proposal["CLOUD_ZONE"], "asia-east1-c")
        self.assertEqual(proposal["MACHINE_TYPE"], "n2-standard-16")
        self.assertEqual(merged["pending_question"]["id"], "inferred_config_review")
        self.assertIn("MACHINE_TYPE", "\n".join(merged["visible_response"]))

    def test_pending_config_fragment_and_consultation_share_one_admitted_plan(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("pending-review-compound-turn", language="en")
        state["active_group"] = "provider_deployment"
        state["inferred_config"] = {
            "pending_review": {
                "config_values": {"CLOUD_ZONE": "us-east1-b"},
                "unmapped_values": {},
                "source_format": "mixed",
                "reason": "initial fragment",
            }
        }
        state["pending_question"] = {
            "id": "inferred_config_review",
            "group": "provider_deployment",
            "kind": "yes_no",
            "field": "inferred_config_review",
            "options": [{"label": "Y", "value": True}, {"label": "N", "value": False}],
            "accepted_action_types": ["answer_pending", "propose_config_values"],
        }
        state["last_user_input"] = (
            "CLOUD_REGION=us-east1\n"
            "Also explain what this Agent can do."
        )

        result = _invoke_with_admitted_actions(
            process_turn,
            state,
            [
                {
                    "type": "propose_config_values",
                    "source_format": "env",
                    "config_values": {"CLOUD_REGION": "us-east1"},
                    "unmapped_values": {},
                    "source_evidence": "CLOUD_REGION=us-east1",
                    "confidence": "high",
                },
                {
                    "type": "answer_opening_question",
                    "topic": "agent_capabilities",
                    "source_evidence": "Also explain what this Agent can do.",
                    "confidence": "high",
                },
            ],
        )

        proposal = result["inferred_config"]["pending_review"]["config_values"]
        self.assertEqual(proposal["CLOUD_ZONE"], "us-east1-b")
        self.assertEqual(proposal["CLOUD_REGION"], "us-east1")
        self.assertEqual(result["pending_question"]["id"], "inferred_config_review")
        self.assertEqual(
            {
                item.get("type")
                for item in (result.get("turn_context") or {}).get("admitted_actions") or []
            },
            {"propose_config_values", "answer_opening_question"},
        )
        rendered = "\n".join(result.get("visible_response") or [])
        self.assertIn("AnyChain Benchmark Agent", rendered)
        self.assertIn("CLOUD_REGION", rendered)

    def test_terminal_style_fragmented_config_paste_maps_common_short_fields(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["inferred_config"] = {
            "pending_review": {
                "config_values": {"CLOUD_REGION": "asia-east1"},
                "unmapped_values": {},
                "source_format": "mixed",
                "reason": "initial line",
            }
        }
        state["pending_question"] = {
            "id": "inferred_config_review",
            "group": "provider_deployment",
            "kind": "yes_no",
            "field": "inferred_config_review",
            "options": [{"label": "Y", "value": True}, {"label": "N", "value": False}],
            "accepted_action_types": ["answer_pending", "propose_config_values"],
        }
        fragments = [
            ("zone: asia-east1-c", "CLOUD_ZONE", "asia-east1-c"),
            ("machine_type: n2-standard-16", "MACHINE_TYPE", "n2-standard-16"),
            ("ledger_device: vda", "LEDGER_DEVICE", "vda"),
            ("data_vol_type: hyperdisk-balanced,", "DATA_VOL_TYPE", "hyperdisk-balanced"),
            ("data_vol_size: 926GiB", "DATA_VOL_SIZE", "926"),
            ("iops: 20000 IOPS", "DATA_VOL_MAX_IOPS", "20000"),
            ("throughput: 1000 MiB/s", "DATA_VOL_MAX_THROUGHPUT", "1000"),
            ("interface: eth0", "NETWORK_INTERFACE", "eth0"),
            ("bandwidth: 100Gbps", "NETWORK_MAX_BANDWIDTH_GBPS", "100"),
        ]
        for line, field, value in fragments:
            state["last_user_input"] = line
            state = _invoke_with_admitted_actions(
                process_turn,
                state,
                [{
                    "type": "propose_config_values",
                    "source_format": "mixed",
                    "config_values": {field: value},
                    "unmapped_values": {},
                    "source_evidence": line,
                    "confidence": "high",
                }],
            )

        proposal = state["inferred_config"]["pending_review"]["config_values"]
        self.assertEqual(proposal["CLOUD_REGION"], "asia-east1")
        self.assertEqual(proposal["CLOUD_ZONE"], "asia-east1-c")
        self.assertEqual(proposal["MACHINE_TYPE"], "n2-standard-16")
        self.assertEqual(proposal["LEDGER_DEVICE"], "vda")
        self.assertEqual(proposal["DATA_VOL_TYPE"], "hyperdisk-balanced")
        self.assertEqual(proposal["DATA_VOL_SIZE"], "926")
        self.assertEqual(proposal["DATA_VOL_MAX_IOPS"], "20000")
        self.assertEqual(proposal["DATA_VOL_MAX_THROUGHPUT"], "1000")
        self.assertEqual(proposal["NETWORK_INTERFACE"], "eth0")
        self.assertEqual(proposal["NETWORK_MAX_BANDWIDTH_GBPS"], "100")
        self.assertIn("NETWORK_MAX_BANDWIDTH_GBPS", "\n".join(state["visible_response"]))

    def test_accepting_merged_config_review_drops_stale_config_proposals_but_resumes_followups(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["inferred_config"] = {
            "pending_review": {
                "config_values": {
                    "CLOUD_REGION": "asia-east1",
                    "CLOUD_ZONE": "asia-east1-c",
                    "MACHINE_TYPE": "n2-standard-16",
                    "LEDGER_DEVICE": "vda",
                    "DATA_VOL_TYPE": "hyperdisk-balanced",
                    "DATA_VOL_SIZE": "926",
                    "DATA_VOL_MAX_IOPS": "20000",
                    "DATA_VOL_MAX_THROUGHPUT": "1000",
                    "NETWORK_INTERFACE": "eth0",
                    "NETWORK_MAX_BANDWIDTH_GBPS": "100",
                },
                "unmapped_values": {},
                "source_format": "mixed",
                "reason": "fragmented terminal paste",
            }
        }
        state["pending_question"] = {
            "id": "inferred_config_review",
            "group": "provider_deployment",
            "kind": "yes_no",
            "field": "inferred_config_review",
            "options": [{"label": "Y", "value": True}, {"label": "N", "value": False}],
            "resume_action_queue": True,
        }
        state["action_queue"] = [
            {
                "type": "propose_config_values",
                "config_values": {"CLOUD_REGION": "asia-east1"},
                "unmapped_values": {"unmapped_1": "storage:"},
                "_origin_text": "stale early paste fragment",
            },
            {"type": "set_accounts_presence", "has_accounts_device": False, "_origin_text": "我没有 accounts 盘"},
            {
                "type": "set_qps_mode",
                "qps_mode": "quick",
                "mutation_explicit": True,
                "source_evidence": "QPS 用 quick",
                "_origin_text": "QPS 用 quick",
            },
        ]
        state["last_user_input"] = "Y"

        result = process_turn(state)

        self.assertEqual(result["confirmed_config"]["DATA_VOL_MAX_IOPS"], "20000")
        self.assertEqual(result["confirmed_config"]["NETWORK_MAX_BANDWIDTH_GBPS"], "100")
        self.assertFalse(result["confirmed_config"]["has_accounts_device"])
        self.assertEqual(result["qps_profile"]["mode"], "quick")
        self.assertNotEqual(result.get("pending_question", {}).get("id"), "inferred_config_review")
        self.assertFalse(any(item.get("type") == "propose_config_values" for item in result.get("action_queue") or []))
        self.assertNotIn("unmapped_1", "\n".join(result.get("visible_response") or []))

    def test_pending_config_review_preserves_exact_prompt_across_language_change(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["inferred_config"] = {
            "pending_review": {
                "config_values": {"CLOUD_REGION": "asia-east1"},
                "unmapped_values": {},
                "source_format": "mixed",
                "reason": "additional config fragments",
            }
        }
        state["pending_question"] = {
            "id": "inferred_config_review",
            "group": "provider_deployment",
            "kind": "yes_no",
            "field": "inferred_config_review",
            "prompt": "I inferred these candidate config values from your pasted content:",
            "options": [{"label": "Y", "value": True}, {"label": "N", "value": False}],
        }
        state["last_user_input"] = "我还想用 fake-node 测 solana"

        result = _invoke_with_admitted_actions(
            process_turn,
            state,
            [
                {
                    "type": "choose_target_mode",
                    "target_mode": "fake-node",
                    "target_mode_explicit": True,
                    "source_evidence": "fake-node",
                    "confidence": "high",
                },
                {
                    "type": "choose_chain",
                    "chain_text": "solana",
                    "source_evidence": "solana",
                    "confidence": "high",
                },
            ],
        )
        text = "\n".join(result["visible_response"])

        self.assertIn("I inferred these candidate config values", text)
        self.assertNotIn("我从你粘贴的内容中推断出这些配置候选值", text)

    def test_complete_traceback_is_analyzed_as_one_semantic_turn(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = (
            "Traceback (most recent call last):\n"
            '  File "agent/terminal/repl.py", line 123, in run\n'
            "RuntimeError: endpoint probe failed\n"
            "这是什么意思？"
        )
        with (
            patch("agent.harness.coordinator.resolve_action_queue") as resolver,
            patch("agent.harness.domains.analysis.analyze_evidence_with_model", return_value="endpoint 检查失败"),
        ):
            resolver.return_value = {
                "actions": [{
                    "type": "analyze_evidence",
                    "evidence": state["last_user_input"],
                    "confidence": "high",
                }]
            }
            result = process_turn(state)

        self.assertEqual(result.get("evidence_collection"), {})
        self.assertIn("Traceback", result["evidence_buffer"][-1]["text"])
        self.assertIn("endpoint 检查失败", "\n".join(result["visible_response"]))

    def test_log_analysis_request_without_log_asks_for_evidence(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = "这是日志，你可以帮我分析么？"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "analyze_evidence", "confidence": "high"}]}
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertEqual(result.get("evidence_collection"), {})
        self.assertFalse(result.get("evidence_buffer"))
        self.assertIn("请粘贴真实日志", text)
        self.assertNotIn("已开始接收多行错误/日志证据", text)

    def test_saved_log_evidence_followup_analyzes_buffer_not_opening_context(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["evidence_buffer"] = [{
            "text": "Traceback (most recent call last):\n  File \"/workspace/agent/terminal/repl.py\", line 1, in <module>\nRuntimeError: endpoint probe failed"
        }]
        state["last_user_input"] = "这是什么意思？下一步怎么修？"

        with (
            patch("agent.harness.coordinator.resolve_action_queue", return_value={"actions": [{
                "type": "analyze_evidence",
                "evidence": state["evidence_buffer"][-1]["text"],
                "confidence": "high",
            }]}),
            patch(
                "agent.harness.domains.analysis.analyze_evidence_with_model",
                return_value=(
                    "证据显示 endpoint probe failed。观察事实：RuntimeError。"
                    "下一步应验证 endpoint、协议族与 request/params。"
                ),
            ),
        ):
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        # Evidence analysis is a turn-local consultation. It must not steal
        # workflow ownership from the group that will handle the next turn.
        self.assertEqual(result["active_group"], "opening")
        self.assertIn("endpoint probe failed", text)
        self.assertIn("下一步", text)
        self.assertNotIn("测试前需要准备什么", text)

    def test_multiline_log_paste_is_stored_as_one_complete_evidence_record(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = (
            "Traceback (most recent call last):\n"
            "  File \"agent/terminal/repl.py\", line 123, in run\n"
            "RuntimeError: boom"
        )

        with (
            patch("agent.harness.coordinator.resolve_action_queue") as resolver,
            patch("agent.harness.domains.analysis.analyze_evidence_with_model", return_value="boom analysis"),
        ):
            resolver.return_value = {
                "actions": [{
                    "type": "analyze_evidence",
                    "evidence": state["last_user_input"],
                    "confidence": "high",
                }]
            }
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertEqual(len(result["evidence_buffer"]), 1)
        self.assertEqual(len(result["evidence_buffer"][0]["text"].splitlines()), 3)
        self.assertIn("boom analysis", text)

    def test_blank_line_during_log_evidence_collection_does_not_resume_config(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["evidence_collection"] = {
            "question": {"id": "freeform_evidence", "kind": "log_evidence"},
            "lines": ["Traceback (most recent call last):"],
            "language": "zh",
        }
        state["last_user_input"] = ""

        result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertEqual(result["evidence_collection"]["lines"], ["Traceback (most recent call last):"])
        self.assertIn("还没有收到新的日志内容", text)
        self.assertNotIn("检测到最近 job", text)

    def test_question_during_evidence_collection_is_analyzed_without_becoming_evidence(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("evidence-question", language="zh")
        original_lines = ["Traceback (most recent call last):", "RuntimeError: endpoint probe failed"]
        state["evidence_collection"] = {
            "question": {"id": "freeform_evidence", "kind": "log_evidence"},
            "lines": list(original_lines),
            "language": "zh",
            "status": "active",
        }
        state["last_user_input"] = "这是什么意思，应该怎么修？"
        with (
            patch("agent.harness.coordinator.resolve_action_queue", return_value={"actions": [{
                "type": "analyze_evidence",
                "evidence": "\n".join(original_lines),
                "question": state["last_user_input"],
                "confidence": "high",
            }]}),
            patch(
                "agent.harness.domains.analysis.analyze_evidence_with_model",
                return_value="endpoint probe failed，需要校验 endpoint。",
            ),
        ):
            result = process_turn(state)

        self.assertEqual(result["evidence_collection"]["lines"], original_lines)
        self.assertNotIn(state["last_user_input"], result["evidence_collection"]["lines"])
        self.assertIn("endpoint probe failed", "\n".join(result.get("visible_response") or []))

    def test_current_context_during_evidence_collection_reports_redacted_buffer(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("evidence-current-context", language="en")
        original_lines = ["curl https://rpc.example/v1/abcdefghijklmnopqrstuvwxyz123456"]
        state["evidence_collection"] = {
            "question": {"id": "new_chain_schema_evidence", "kind": "evidence"},
            "lines": list(original_lines),
            "language": "en",
        }
        state["last_user_input"] = "Before I continue, what have you collected so far?"
        with patch("agent.harness.coordinator.resolve_action_queue", return_value={"actions": [{
            "type": "answer_opening_question",
            "topic": "current_context",
            "source_evidence": "what have you collected so far?",
            "confidence": "medium",
        }]}):
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertEqual(result["evidence_collection"]["lines"], original_lines)
        self.assertIn("1 saved line", text)
        self.assertIn("***REDACTED***", text)
        self.assertNotIn("abcdefghijklmnopqrstuvwxyz123456", text)

    def test_navigation_during_evidence_collection_preserves_collected_lines(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("evidence-navigation", language="zh")
        original_lines = ["request:", '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[]}']
        state["evidence_collection"] = {
            "question": {"id": "freeform_evidence", "kind": "log_evidence"},
            "lines": list(original_lines),
            "language": "zh",
            "status": "active",
        }
        state["last_user_input"] = "先去配置 QPS"
        with patch("agent.harness.coordinator.resolve_action_queue", return_value={"actions": [
            {
                "type": "pause_evidence_collection",
                "source_evidence": state["last_user_input"],
                "confidence": "high",
            },
            {
                "type": "change_group",
                "group": "qps_profile",
                "navigation_explicit": True,
                "source_evidence": state["last_user_input"],
                "confidence": "high",
            },
        ]}):
            result = process_turn(state)

        self.assertEqual(result["evidence_collection"]["lines"], original_lines)
        self.assertEqual(result["evidence_collection"]["status"], "paused")
        self.assertEqual(result["active_group"], "target_mode")
        self.assertEqual((result.get("pending_question") or {}).get("group"), "target_mode")
        self.assertEqual((result.get("control") or {}).get("deferred_group"), "qps_profile")

    def test_execution_rejects_append_after_prior_action_pauses_collection(self) -> None:
        from agent.harness.coordinator import _process_action_queue
        from agent.harness.state import new_state

        state = new_state("stale-evidence-action", language="en")
        state["evidence_collection"] = {
            "question": {"id": "freeform_evidence", "kind": "log_evidence"},
            "lines": ["Traceback"],
            "language": "en",
            "status": "active",
        }
        source = "Pause this paste; the next line belongs elsewhere."
        result = _process_action_queue(state, [
            {
                "type": "pause_evidence_collection",
                "source_evidence": source,
                "confidence": "high",
            },
            {
                "type": "append_evidence_collection",
                "evidence": "do not append",
                "source_evidence": source,
                "confidence": "high",
            },
        ], source)

        self.assertEqual(result["evidence_collection"]["status"], "paused")
        self.assertEqual(result["evidence_collection"]["lines"], ["Traceback"])
        self.assertIn("lifecycle_inapplicable", [row.get("error") for row in result.get("action_errors") or []])

    def test_paused_evidence_collection_resumes_only_through_typed_action(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("evidence-resume", language="en")
        original_lines = ["request:", '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[]}']
        state["evidence_collection"] = {
            "question": {"id": "new_chain_schema_evidence", "kind": "evidence"},
            "lines": list(original_lines),
            "language": "en",
            "status": "paused",
        }
        state["last_user_input"] = "Resume the evidence paste now."
        with patch("agent.harness.coordinator.resolve_action_queue", return_value={"actions": [{
            "type": "resume_evidence_collection",
            "source_evidence": state["last_user_input"],
            "confidence": "high",
        }]}):
            result = process_turn(state)

        self.assertEqual(result["evidence_collection"]["lines"], original_lines)
        self.assertEqual(result["evidence_collection"]["status"], "active")
        self.assertIn("Resumed evidence collection", "\n".join(result.get("visible_response") or []))

    def test_evidence_detour_commits_qps_value_without_analysis_overlay(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        source = "Pause this paste and let me configure quick QPS first."
        state = new_state("evidence-qps-detour", language="en")
        state.update({
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "active_group": "workload_rpc",
            "last_user_input": source,
            "evidence_collection": {
                "question": {"id": "new_chain_schema_evidence", "kind": "evidence"},
                "lines": ["curl --location http://fake-node:19000 \\"],
                "language": "en",
                "status": "active",
            },
        })
        with patch("agent.harness.coordinator.resolve_action_queue", return_value={"actions": [
            {
                "type": "pause_evidence_collection",
                "source_evidence": "Pause this paste",
                "confidence": "high",
            },
            {
                "type": "change_group",
                "group": "qps_profile",
                "navigation_explicit": True,
                "source_evidence": "configure quick QPS first",
                "confidence": "high",
            },
            {
                "type": "set_qps_mode",
                "qps_mode": "quick",
                "mutation_explicit": True,
                "source_evidence": "quick QPS",
                "confidence": "high",
            },
        ]}):
            result = process_turn(state)

        self.assertEqual(result["qps_profile"]["mode"], "quick")
        self.assertEqual(result["active_group"], "qps_profile")
        self.assertEqual(result["evidence_collection"]["status"], "paused")
        self.assertEqual(result["evidence_collection"]["lines"], state["evidence_collection"]["lines"])
        self.assertNotIn("Observed Facts", "\n".join(result.get("visible_response") or []))

    def test_blank_turn_does_not_append_or_resume_a_paused_collection(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("evidence-paused-blank", language="en")
        state["evidence_collection"] = {
            "question": {"id": "freeform_evidence", "kind": "log_evidence"},
            "lines": ["Traceback"],
            "language": "en",
            "status": "paused",
        }
        state["last_user_input"] = ""

        result = process_turn(state)

        self.assertEqual(result["evidence_collection"]["lines"], ["Traceback"])
        self.assertEqual(result["evidence_collection"]["status"], "paused")
        self.assertNotIn("Continue pasting", "\n".join(result.get("visible_response") or []))

    def test_explicit_evidence_collection_cancel_discards_only_collection(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("evidence-cancel", language="en")
        state["confirmed_config"] = {"CLOUD_REGION": "us-east1"}
        state["evidence_collection"] = {
            "question": {"id": "freeform_evidence", "kind": "log_evidence"},
            "lines": ["Traceback (most recent call last):"],
            "language": "en",
            "status": "active",
        }
        state["last_user_input"] = "Cancel this evidence collection."
        with patch("agent.harness.coordinator.resolve_action_queue", return_value={"actions": [{
            "type": "cancel_evidence_collection",
            "source_evidence": state["last_user_input"],
            "confidence": "high",
        }]}):
            result = process_turn(state)

        self.assertEqual(result["evidence_collection"], {})
        self.assertEqual(result["confirmed_config"], {"CLOUD_REGION": "us-east1"})

    def test_unrelated_greeting_during_evidence_collection_is_not_appended(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("evidence-greeting", language="en")
        original_lines = ["Traceback (most recent call last):"]
        state["evidence_collection"] = {
            "question": {"id": "freeform_evidence", "kind": "log_evidence"},
            "lines": list(original_lines),
            "language": "en",
        }
        state["last_user_input"] = "Hello, who are you?"
        with patch("agent.harness.coordinator.resolve_action_queue", return_value={"actions": [{
            "type": "greeting",
            "source_evidence": state["last_user_input"],
            "confidence": "high",
        }]}):
            result = process_turn(state)

        self.assertEqual(result["evidence_collection"]["lines"], original_lines)
        self.assertNotIn(state["last_user_input"], result["evidence_collection"]["lines"])

    def test_intent_provider_failure_does_not_pollute_stale_log_evidence_collection(self) -> None:
        from unittest.mock import patch

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["evidence_collection"] = {
            "question": {"id": "freeform_evidence", "kind": "log_evidence"},
            "lines": ["Traceback (most recent call last):", '  File "x.py", line 1'],
            "language": "zh",
        }
        state["last_user_input"] = "你是谁？"

        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{"type": "unknown", "confidence": "low"}], "reason": "resolver failed"},
        ):
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertEqual(result["evidence_collection"]["lines"], ["Traceback (most recent call last):", '  File "x.py", line 1'])
        self.assertIn("模型服务", text)
        self.assertNotIn("已记录第", text)

    def test_multiline_rpc_evidence_is_collected_before_protocol_conflict(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "abcd",
            "canonical": "abcd",
            "adapter_family": "jsonrpc",
            "status": "existing_family_needs_schema_evidence",
            "case": "case2",
            "candidate_method": "eth_blockNumber",
        }
        state["endpoint_evidence"] = {"candidate_endpoint": "http://fake-node:19000", "candidate_endpoint_ready": True}
        state["active_group"] = "endpoint_process"
        state["pending_question"] = {
            "id": "new_chain_schema_evidence",
            "group": "endpoint_process",
            "kind": "evidence",
            "field": "new_chain_schema_evidence",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "curl --location 'http://fake-node:19000' \\"

        first = process_turn(state)
        self.assertEqual(first["evidence_collection"]["question"]["id"], "new_chain_schema_evidence")
        self.assertEqual(first.get("pending_question"), {})
        self.assertIn("多行 RPC 证据", "\n".join(first["visible_response"]))

        first["last_user_input"] = '--data \'{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}\''
        with patch("agent.harness.coordinator.resolve_action_queue", return_value={"actions": [{
            "type": "append_evidence_collection",
            "evidence": first["last_user_input"],
            "source_evidence": first["last_user_input"],
            "confidence": "high",
        }]}):
            second = process_turn(first)
        self.assertTrue(second["evidence_collection"]["lines"])

        second["last_user_input"] = 'response: {"jsonrpc":"2.0","id":1,"result":"0x10"}'
        with (
            patch("agent.harness.coordinator.resolve_action_queue", return_value={"actions": [{
                "type": "append_evidence_collection",
                "evidence": second["last_user_input"],
                "source_evidence": second["last_user_input"],
                "confidence": "high",
            }]}),
            patch("agent.harness.domains.rpc_endpoint.extract_rpc_schema_from_evidence") as extract,
        ):
            extract.return_value = {
                "status": "draft",
                "evidence_kind": "jsonrpc_request",
                "transport": "jsonrpc",
                "method": "eth_blockNumber",
                "params": [],
                "params_json": [],
                "response_summary": "hex block number",
                "confidence": "high",
            }
            third = process_turn(second)

        self.assertEqual(third["chain_identity"]["status"], "existing_family_schema_needs_confirmation")
        self.assertEqual(third["pending_question"]["id"], "new_chain_schema_confirm")

    def test_opening_requirements_question_returns_preparation_checklist(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = "如果我需要测试，我都需要做什么，提供什么"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "answer_opening_question", "topic": "requirements", "confidence": "high"}]}
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("测试类型", text)
        self.assertIn("CLOUD_REGION", text)
        self.assertIn("Ledger/data", text)
        self.assertIn("preflight/smoke", text)
        self.assertNotIn("已知链：acala", text)

    def test_opening_correction_question_does_not_repeat_capabilities(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = "你是否理解我的问题？"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "answer_opening_question", "topic": "correction", "confidence": "high"}]}
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("我理解", text)
        self.assertIn("继续原问题", text)
        self.assertNotIn("协议族分布", text)

    def test_opening_identity_question_answers_agent_identity_not_chain_list(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = "你是谁"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "answer_opening_question", "topic": "identity", "confidence": "high"}]}
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("AnyChain Benchmark Agent", text)
        self.assertIn("配置、校验、执行和报告分析", text)
        self.assertNotIn("已知链：", text)

    def test_identity_destination_question_uses_llm_identity_topic(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = "你要去哪里？"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "answer_opening_question", "topic": "identity", "confidence": "high"}]}
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("AnyChain Benchmark Agent", text)
        self.assertIn("配置、校验、执行和报告分析", text)
        self.assertNotIn("当前没有待确认问题", text)

    def test_opening_agent_capabilities_are_product_summary_not_raw_chain_list(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = "你能做什么"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "answer_opening_question", "topic": "agent_capabilities", "confidence": "high"}]}
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("fake-node", text)
        self.assertIn("real-node", text)
        self.assertIn("sync-observe", text)
        self.assertIn("preflight/smoke", text)
        self.assertNotIn("已知链：acala", text)

    def test_bare_capabilities_topic_maps_to_agent_capabilities_not_chain_list(self) -> None:
        """Phase 6 item 4: `intent.py`'s topic enum documents a bare

        `capabilities` value with no dedicated prompt rule. `coordinator.py` used
        to normalize it to `supported_chains` (the raw chain list), which
        contradicted the sibling `capability`/`what_can_you_do` normalization
        (agent capabilities). It must map to the agent-capabilities product
        summary, consistent with what the schema text implies.
        """

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = "capabilities?"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "answer_opening_question", "topic": "capabilities", "confidence": "high"}]}
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("fake-node", text)
        self.assertIn("real-node", text)
        self.assertIn("sync-observe", text)
        self.assertIn("preflight/smoke", text)

    def test_startup_discovery_topic_explains_inference_separately_from_confirmed_config(self) -> None:
        """Matrix 1 from the 2026-07-10 handoff document.

        After clearing config, asking whether startup inference still
        exists must return the discovery summary, not fallback-only text,
        and must not conflate empty confirmed_config with missing discovery.
        """

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["discovery"] = {
            "cloud": {"provider": "gcp", "platform": "gce"},
            "host": {"cpu_count": 16, "memory_gib": 64},
            "network": {"default_interface": "eth0"},
            "disks": {"proposed_ledger_device": "/dev/nvme0n1"},
            "dependencies": {"missing_required": []},
        }
        state["confirmed_config"] = {}
        state["last_user_input"] = "我现在关于最初推断的信息还有么"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "answer_opening_question", "topic": "startup_discovery", "confidence": "high"}]}
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("gcp", text)
        self.assertIn("eth0", text)
        self.assertIn("nvme0n1", text)
        self.assertNotIn("当前没有待确认问题", text)

    def test_colloquial_question_without_question_mark_does_not_answer_manual_value_pending(self) -> None:
        """A live DeepSeek run surfaced this: a pending manual_value question

        (e.g. CLOUD_ZONE) silently swallowed a colloquial Chinese question
        ending in a sentence-final particle (了么/呢) instead of a
        ？, because `_is_plain_scalar_answer` alone does not detect it as a
        question. `confirm_or_value` already guarded against this; manual_value
        and device did not.
        """

        from agent.harness.coordinator import _answer_fits_pending

        question = {
            "id": "CLOUD_ZONE",
            "group": "provider_deployment",
            "kind": "manual_value",
            "field": "CLOUD_ZONE",
            "manual_input_allowed": True,
        }
        colloquial_question = "你有一个环境依赖的检测脚本，这个脚本会帮我推断一些变量，这些变量推断了么"
        self.assertFalse(_answer_fits_pending(colloquial_question, question))

    def test_chain_pending_keeps_only_atomic_or_known_answers_on_local_path(self) -> None:
        from agent.harness.coordinator import _answer_fits_pending

        question = {
            "id": "chain",
            "group": "chain_identity",
            "kind": "chain",
            "field": "chain",
            "manual_input_allowed": True,
        }
        self.assertTrue(_answer_fits_pending("solana", question))
        self.assertTrue(_answer_fits_pending("sola", question))
        self.assertFalse(_answer_fits_pending("Run this benchmark against ETH for me.", question))
        self.assertFalse(_answer_fits_pending("BNB Greenfield", question))

    def test_evidence_pending_admits_wire_syntax_but_routes_natural_language(self) -> None:
        from agent.harness.coordinator import _answer_fits_pending

        evidence_question = {
            "id": "new_chain_schema_evidence",
            "group": "endpoint_process",
            "kind": "evidence",
            "field": "new_chain_schema_evidence",
            "manual_input_allowed": True,
        }
        method_question = {
            "id": "new_chain_method",
            "group": "endpoint_process",
            "kind": "manual_value",
            "field": "new_chain_method",
            "manual_input_allowed": True,
            "validation": {"input_mode": "rpc_method_or_schema_evidence"},
        }

        self.assertTrue(_answer_fits_pending("[]", evidence_question))
        self.assertTrue(_answer_fits_pending(
            '{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}',
            evidence_question,
        ))
        self.assertFalse(_answer_fits_pending(
            "Before I paste evidence, summarize the state you retained.",
            evidence_question,
        ))
        self.assertTrue(_answer_fits_pending("eth_chainId", method_question))
        self.assertFalse(_answer_fits_pending(
            "What method are you waiting for, and what comes next?",
            method_question,
        ))

    def test_scalar_value_containing_current_as_substring_is_not_treated_as_a_question(self) -> None:
        """Code review (Phase 1/2 diff) found `_looks_like_user_question`'s

        "current" marker was a plain substring check, newly wired into the
        manual_value/device/url branches. A legitimate single-token answer
        like an API key or device path containing "current" as a substring
        (e.g. "concurrent-tier-01") must not be rejected as a question.
        """

        from agent.harness.coordinator import _answer_fits_pending

        manual_value_question = {
            "id": "RPC_API_KEY",
            "group": "chain_auxiliary_endpoints",
            "kind": "manual_value",
            "field": "RPC_API_KEY",
            "manual_input_allowed": True,
            "validation": {"value_type": "scalar_token"},
        }
        self.assertTrue(_answer_fits_pending("concurrent-tier-01", manual_value_question))

        device_question = {
            "id": "LEDGER_DEVICE",
            "group": "ledger_disk",
            "kind": "device",
            "field": "LEDGER_DEVICE",
            "manual_input_allowed": True,
            "validation": {"value_type": "scalar_token"},
        }
        self.assertTrue(_answer_fits_pending("/dev/current-disk", device_question))

        real_question = {
            "id": "CLOUD_ZONE",
            "group": "provider_deployment",
            "kind": "manual_value",
            "field": "CLOUD_ZONE",
            "manual_input_allowed": True,
        }
        self.assertFalse(_answer_fits_pending("what is the current zone?", real_question))

    def test_opening_mode_comparison_has_sync_observe_without_vegeta(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = "fake-node real-node sync-observe 有什么区别"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "answer_opening_question", "topic": "mode_comparison", "confidence": "high"}]}
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("fake-node", text)
        self.assertIn("real-node", text)
        self.assertIn("sync-observe", text)
        self.assertIn("不走 vegeta", text.lower())
        self.assertIn("fake-node 的作用", text)
        self.assertIn("支持多少 QPS", text)
        self.assertIn("real-node benchmark", text)

    def test_opening_performance_goal_guidance_prefers_real_node(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = "我需要观察能支持多少 qps，性能瓶颈在哪里"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "answer_opening_question", "topic": "performance_benchmark_guidance", "confidence": "high"}]}
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("real-node benchmark", text)
        self.assertIn("LOCAL_RPC_URL", text)
        self.assertIn("验证框架闭环", text)
        self.assertIn("sync-observe", text)

    def test_performance_goal_keeps_guidance_when_llm_extracts_topic_and_chain(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["last_user_input"] = "I want to benchmark BSC max throughput and find bottlenecks, not just test the tool itself."

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "answer_opening_question", "topic": "performance_benchmark_guidance", "confidence": "high"},
                    {"type": "choose_chain", "chain_text": "BSC", "source_evidence": "BSC", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("real-node benchmark", text)
        self.assertIn("fake-node only validates", text)
        self.assertIn("Confirmed chain: `bsc`", text)
        self.assertEqual(result["chain_identity"]["canonical"], "bsc")
        self.assertEqual(result["pending_question"]["id"], "target_mode_select")

    def test_performance_goal_rejects_conflicting_fake_node_target_mode_action(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = "我需要测试 solana 能扛多少 qps，fake-node 可以测这个吗"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "answer_opening_question", "topic": "performance_benchmark_guidance", "confidence": "high"},
                    {"type": "choose_target_mode", "target_mode": "fake-node", "target_mode_explicit": True, "confidence": "high"},
                    {"type": "choose_chain", "chain_text": "solana", "source_evidence": "solana", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertEqual(result.get("target_mode"), "")
        self.assertEqual(result["chain_identity"]["canonical"], "solana")
        self.assertEqual(result["pending_question"]["id"], "target_mode_select")
        self.assertIn("real-node benchmark", text)

    def test_opening_option_three_enters_sync_observe_without_llm_recommendation(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.orientation import opening_question
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["active_group"] = "opening"
        state["chain_identity"] = {"raw": "solana", "canonical": "solana", "adapter_family": "jsonrpc", "status": "confirmed"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "solana"}
        state["pending_question"] = opening_question(state)
        state["last_user_input"] = "3"

        result = process_turn(state)

        self.assertEqual(result["target_mode"], "sync-observe")
        self.assertEqual(result["workflow_mode"], "sync_observe")
        self.assertNotIn("建议先用 fake-node", "\n".join(result.get("visible_response") or []))

    def test_opening_option_answer_discards_stale_consultation_queue(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.orientation import opening_question
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["active_group"] = "opening"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["pending_question"] = opening_question(state)
        state["pending_question"]["resume_action_queue"] = True
        state["action_queue"] = [{"type": "answer_opening_question", "topic": "requirements", "confidence": "high"}]
        state["last_user_input"] = "3"

        result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertEqual(result["target_mode"], "sync-observe")
        self.assertEqual(result["workflow_mode"], "sync_observe")
        self.assertEqual(result.get("action_queue"), [])
        self.assertNotIn("如果你要开始一次测试", text)
        self.assertNotEqual(result.get("pending_question", {}).get("id"), "")

    def test_mode_consultation_does_not_select_mentioned_mode(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["chain_identity"] = {"raw": "BSC", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["last_user_input"] = "Why can't fake-node tell me the real bottleneck?"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "choose_target_mode", "target_mode": "fake-node", "target_mode_explicit": False, "confidence": "high"},
                    {"type": "answer_opening_question", "topic": "mode_comparison", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertEqual(result.get("target_mode"), "")
        self.assertIn("fake-node only validates", text)
        self.assertIn("real-node benchmark", text)

    def test_mode_consultation_prunes_conflicting_fake_node_recommendation(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["chain_identity"] = {"raw": "BSC", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["last_user_input"] = "Why can't fake-node tell me the real bottleneck? If I care about block sync speed, which mode should I use?"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "answer_opening_question", "topic": "performance_benchmark_guidance", "confidence": "high"},
                    {"type": "answer_opening_question", "topic": "mode_comparison", "confidence": "high"},
                    {"type": "answer_opening_question", "topic": "recommendation", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("real-node benchmark", text)
        self.assertIn("sync-observe", text)
        self.assertNotIn("I recommend a fake-node smoke first", text)

    def test_chinese_mode_consultation_does_not_select_mentioned_fake_node(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["chain_identity"] = {"raw": "BNB", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["last_user_input"] = "那假节点还有什么意义？它能告诉我真实性能吗？"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "choose_target_mode", "target_mode": "fake-node", "target_mode_explicit": False, "confidence": "high"},
                    {"type": "answer_opening_question", "topic": "mode_comparison", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertEqual(result.get("target_mode"), "")
        self.assertIn("fake-node", text)
        self.assertIn("不代表真实节点性能", text)

    def test_explicit_sync_observe_selection_survives_consultation_guard(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["chain_identity"] = {"raw": "BSC", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["last_user_input"] = "Then let's observe sync behavior for BSC, but don't run vegeta."

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "choose_target_mode", "target_mode": "sync-observe", "target_mode_explicit": True, "source_evidence": "observe sync", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        self.assertEqual(result.get("target_mode"), "sync-observe")
        self.assertEqual(result.get("workflow_mode"), "sync_observe")

    def test_chinese_block_catchup_phrase_selects_sync_observe(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["chain_identity"] = {"raw": "BNB", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["last_user_input"] = "那我先观察 bsc 追块，不要跑 vegeta 压测"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "choose_target_mode", "target_mode": "sync-observe", "target_mode_explicit": True, "source_evidence": "观察 bsc 追块", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        self.assertEqual(result.get("target_mode"), "sync-observe")
        self.assertEqual(result.get("workflow_mode"), "sync_observe")

    def test_pending_accounts_answer_with_extra_question_applies_answer_then_routes_question(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "real-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["pending_question"] = {
            "id": "has_accounts_device",
            "group": "accounts_disk",
            "kind": "yes_no",
            "field": "has_accounts_device",
            "options": [{"label": "Y", "value": True}, {"label": "N", "value": False}],
        }
        state["last_user_input"] = "没有 accounts 盘，顺便告诉我 real-node 需要提供什么 endpoint"

        with (
            patch("agent.harness.coordinator.resolve_action_queue") as resolver,
        ):
            resolver.side_effect = _admitted_mock_resolver({"actions": [
                {"type": "answer_pending", "answer": "没有 accounts 盘", "selected_value": False, "source_evidence": "没有 accounts 盘", "pending_option_semantic_verified": True, "semantic_purpose_verified": True, "confidence": "high"},
                {"type": "answer_opening_question", "topic": "requirements", "confidence": "high"},
            ]})
            result = process_turn(state)

        self.assertIs(result["confirmed_config"]["has_accounts_device"], False)
        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("LOCAL_RPC_URL", text)
        self.assertIn("endpoint", text)
        self.assertNotIn("这个节点是否有独立的 accounts/state 磁盘", text)

    def test_pending_accounts_plain_natural_answer_only_continues_fallback(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"canonical": "solana", "adapter_family": "jsonrpc", "status": "confirmed"}
        state["confirmed_config"] = {
            "CLOUD_REGION": "asia-east1",
            "CLOUD_ZONE": "asia-east1-c",
            "MACHINE_TYPE": "n2-standard-16",
            "LEDGER_DEVICE": "vda",
            "DATA_VOL_TYPE": "hyperdisk-balanced",
            "DATA_VOL_SIZE": "926",
            "DATA_VOL_MAX_IOPS": "20000",
            "DATA_VOL_MAX_THROUGHPUT": "1000",
        }
        state["pending_question"] = {
            "id": "has_accounts_device",
            "group": "accounts_disk",
            "kind": "yes_no",
            "field": "has_accounts_device",
            "options": [{"label": "Y", "value": True}, {"label": "N", "value": False}],
        }
        state["last_user_input"] = "没有 accounts 盘"

        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            side_effect=_admitted_mock_resolver({"actions": [{"type": "answer_pending", "answer": "没有 accounts 盘", "selected_value": False, "source_evidence": "没有 accounts 盘", "pending_option_semantic_verified": True, "semantic_purpose_verified": True, "confidence": "high"}]}),
        ):
            result = process_turn(state)

        self.assertIs(result["confirmed_config"]["has_accounts_device"], False)
        self.assertNotEqual(result.get("pending_question", {}).get("id"), "has_accounts_device")

    def test_mixed_goal_and_config_turn_preserves_complete_typed_action_plan(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = (
            "我要测试 BNB real-node，region asia-east1，zone asia-east1-c，机器 n2-standard-16，"
            "ledger vda，磁盘 hyperdisk-balanced，IOPS 20000，吞吐 1000"
        )

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "choose_target_mode", "target_mode": "real-node", "target_mode_explicit": True, "source_evidence": "BNB real-node", "confidence": "high"},
                    {"type": "choose_chain", "chain_text": "BNB", "source_evidence": "BNB", "confidence": "high"},
                    {
                        "type": "propose_config_values",
                        "config_values": {
                            "CLOUD_REGION": "asia-east1",
                            "CLOUD_ZONE": "asia-east1-c",
                            "MACHINE_TYPE": "n2-standard-16",
                            "LEDGER_DEVICE": "vda",
                        },
                        "confidence": "high",
                    }
                ]
            }
            resolver.side_effect = _admitted_mock_resolver(resolver.return_value)
            result = process_turn(state)

        self.assertEqual(result["target_mode"], "real-node")
        self.assertEqual(result["chain_identity"]["canonical"], "bsc")
        self.assertEqual(result["pending_question"]["id"], "inferred_config_review")
        self.assertIn("CLOUD_REGION", "\n".join(result.get("visible_response") or []))

    def test_opening_current_config_question_reports_state(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "real-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed"}
        state["pending_question"] = {"id": "DATA_VOL_SIZE", "group": "ledger_disk", "kind": "confirm_or_value"}
        state["last_user_input"] = "当前链和模式是什么？"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "answer_opening_question", "topic": "current_config", "confidence": "high"}]}
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("chain: `bsc`", text)
        self.assertIn("target_mode: `real-node`", text)
        self.assertIn("DATA_VOL_SIZE", text)
        self.assertNotIn("已知链：", text)

    def test_current_context_question_explains_pending_qps_confirmation(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["qps_profile"] = {"mode": "quick", "confirmed": False}
        state["pending_question"] = {
            "id": "qps_profile_confirm",
            "group": "qps_profile",
            "kind": "yes_no",
            "field": "qps_profile_confirmed",
            "options": [{"label": "Y", "value": True}, {"label": "N", "value": False}],
            "prompt": "是否使用 quick 默认 QPS 配置？",
        }
        state["last_user_input"] = "这个是做什么的？"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "answer_opening_question", "topic": "current_context", "confidence": "high"}]}
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("QPS profile", text)
        self.assertIn("INITIAL_QPS", text)
        self.assertIn("fake-node smoke", text)

    def test_current_context_without_pending_does_not_leak_internal_next_action(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed"}
        state["confirmed_config"] = {
            "CLOUD_REGION": "us-1",
            "CLOUD_ZONE": "us-1-z",
            "MACHINE_TYPE": "n2",
            "LEDGER_DEVICE": "vda",
            "DATA_VOL_TYPE": "hyperdisk-balanced",
            "DATA_VOL_SIZE": "926",
            "DATA_VOL_MAX_IOPS": "20000",
            "DATA_VOL_MAX_THROUGHPUT": "1000",
            "has_accounts_device": False,
            "NETWORK_INTERFACE": "eth0",
            "NETWORK_MAX_BANDWIDTH_GBPS": "100",
        }
        state["rpc_mode"] = "single"
        state["workload"] = {"confirmed": True}
        state["qps_profile"] = {"mode": "quick", "confirmed": True}
        state["observability"] = {"mode": "disabled"}
        state["advanced_tuning"] = {"default_decision_made": True, "confirmed": True}
        state["last_user_input"] = "那我们应该从哪里开始"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "answer_opening_question", "topic": "current_context", "confidence": "high"}]}
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("preflight/smoke", text)
        self.assertIn("回复 `Y`", text)
        self.assertNotIn("Continue with", text)
        self.assertNotIn("preflight_smoke_execution", text)

    def _fully_configured_state_before_advanced_tuning(self):
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed"}
        state["confirmed_config"] = {
            "CLOUD_REGION": "us-1",
            "CLOUD_ZONE": "us-1-z",
            "MACHINE_TYPE": "n2",
            "LEDGER_DEVICE": "vda",
            "DATA_VOL_TYPE": "hyperdisk-balanced",
            "DATA_VOL_SIZE": "926",
            "DATA_VOL_MAX_IOPS": "20000",
            "DATA_VOL_MAX_THROUGHPUT": "1000",
            "has_accounts_device": False,
            "NETWORK_INTERFACE": "eth0",
            "NETWORK_MAX_BANDWIDTH_GBPS": "100",
        }
        state["rpc_mode"] = "single"
        state["workload"] = {"confirmed": True}
        state["qps_profile"] = {"mode": "quick", "confirmed": True}
        state["observability"] = {"mode": "disabled"}
        return state

    def test_group_registry_order_matches_state_default_group_order(self) -> None:
        """Architecture audit Finding A: three independent group-order lists

        used to disagree (`state.DEFAULT_GROUP_ORDER`,
        `intent.ALLOWED_GROUPS`, `workflows.group_registry.GROUP_ORDER`).
        All three must now derive from or exactly match one canonical list.
        """

        from agent.harness.intent import ALLOWED_GROUPS
        from agent.harness.state import DEFAULT_GROUP_ORDER
        from agent.workflows.group_registry import GROUP_ORDER, USER_NAVIGABLE_GROUPS

        self.assertEqual(list(GROUP_ORDER), list(DEFAULT_GROUP_ORDER))
        self.assertEqual(list(ALLOWED_GROUPS), list(USER_NAVIGABLE_GROUPS))
        self.assertLess(set(ALLOWED_GROUPS), set(DEFAULT_GROUP_ORDER))
        self.assertNotIn("hardware_discovery", DEFAULT_GROUP_ORDER)

    def test_adapter_family_lists_derive_from_single_source(self) -> None:
        """Architecture audit Finding C3: the supported adapter-family set was

        retyped in six places (one with real value drift). They must all now
        derive from `agent.onboarding.families.SUPPORTED_FAMILIES`.
        """

        from agent.harness.domains.chain_rpc import SUPPORTED_ADAPTER_FAMILIES, question_for_chain_rpc
        from agent.harness.state import new_state
        from agent.harness.intent import ADAPTER_FAMILIES
        from agent.onboarding.families import SUPPORTED_FAMILIES
        from agent.onboarding.template_drafter import (
            JSONRPC_TRANSPORT_FAMILIES,
            REST_TRANSPORT_FAMILIES,
        )
        from agent.validators.endpoint_probe import GENERIC_JSONRPC_PROBE_FAMILIES

        canonical = set(SUPPORTED_FAMILIES)
        self.assertEqual(SUPPORTED_ADAPTER_FAMILIES, canonical)
        self.assertEqual(list(ADAPTER_FAMILIES), list(SUPPORTED_FAMILIES))
        # The protocol-family question offers every family plus an "unsupported".
        state = new_state("unit-thread", language="en")
        state["chain_identity"] = {"status": "needs_protocol_confirmation", "case": "unknown"}
        option_values = {opt["value"] for opt in question_for_chain_rpc(state, "chain_identity")["options"]}
        self.assertEqual(option_values - {"unsupported"}, canonical)
        # template_drafter partitions the canonical families into two transports.
        self.assertEqual(REST_TRANSPORT_FAMILIES | JSONRPC_TRANSPORT_FAMILIES, canonical)
        self.assertEqual(REST_TRANSPORT_FAMILIES & JSONRPC_TRANSPORT_FAMILIES, set())
        # endpoint_probe's generic-JSON-RPC set must only contain canonical
        # family values (no non-canonical evm/ethereum aliases). It covers every
        # family whose transport is a plain POST JSON-RPC call -- `jsonrpc`,
        # `substrate`, and `bitcoin_jsonrpc` -- not just `jsonrpc`; the
        # GET/REST-shaped families (`rest`/`tendermint`/`hedera_dual`) are
        # deliberately excluded (see `#59` in known-issues.md).
        self.assertTrue(GENERIC_JSONRPC_PROBE_FAMILIES <= canonical)
        self.assertEqual(GENERIC_JSONRPC_PROBE_FAMILIES, {"jsonrpc", "substrate", "bitcoin_jsonrpc"})

    def test_generic_jsonrpc_probe_uses_canonical_family_not_evm_alias(self) -> None:
        """Finding C3 value drift: `_should_use_generic_jsonrpc_probe` used to

        match non-canonical aliases (`evm`/`ethereum`/`ethereum_jsonrpc`) that
        the adapter-family taxonomy never produces. For a chain without a
        template, the canonical `jsonrpc` family must trigger the generic probe
        while a stale alias must not (it falls back to method-shape inspection).
        """

        from agent.validators.endpoint_probe import _should_use_generic_jsonrpc_probe

        # Unknown chain (no template on disk) + canonical family => generic probe.
        self.assertTrue(
            _should_use_generic_jsonrpc_probe("chain-with-no-template-xyz", "jsonrpc", [], {})
        )
        # A non-canonical alias no longer short-circuits to the generic probe;
        # with no methods to inspect it degrades to False rather than True.
        self.assertFalse(
            _should_use_generic_jsonrpc_probe("chain-with-no-template-xyz", "evm", [], {})
        )

    def test_chain_needing_rpc_api_key_is_routed_to_auxiliary_endpoint_group(self) -> None:
        """`starknet`'s template substitutes `${RPC_API_KEY}` at runtime, so

        this group must actually ask for it instead of skipping straight to
        workload configuration.
        """

        from agent.harness.coordinator import _ask_next_blocking_question
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "starknet", "canonical": "starknet", "status": "confirmed"}
        state["confirmed_config"] = {
            "CLOUD_REGION": "us-1",
            "CLOUD_ZONE": "us-1-z",
            "MACHINE_TYPE": "n2",
            "LEDGER_DEVICE": "vda",
            "DATA_VOL_TYPE": "hyperdisk-balanced",
            "DATA_VOL_SIZE": "926",
            "DATA_VOL_MAX_IOPS": "20000",
            "DATA_VOL_MAX_THROUGHPUT": "1000",
            "has_accounts_device": False,
            "NETWORK_INTERFACE": "eth0",
            "NETWORK_MAX_BANDWIDTH_GBPS": "100",
        }
        result = _ask_next_blocking_question(state)

        self.assertEqual(result["pending_question"]["group"], "chain_auxiliary_endpoints")
        self.assertEqual(result["pending_question"]["id"], "RPC_API_KEY")

    def test_chain_not_needing_auxiliary_fields_skips_the_group_entirely(self) -> None:
        """Most chains (e.g. `bsc`) do not substitute any of the seven

        chain_auxiliary_endpoints fields at runtime; the group must not ask
        a question nobody's chain template will ever use.
        """

        from agent.harness.coordinator import _ask_next_blocking_question
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed"}
        state["confirmed_config"] = {
            "CLOUD_REGION": "us-1",
            "CLOUD_ZONE": "us-1-z",
            "MACHINE_TYPE": "n2",
            "LEDGER_DEVICE": "vda",
            "DATA_VOL_TYPE": "hyperdisk-balanced",
            "DATA_VOL_SIZE": "926",
            "DATA_VOL_MAX_IOPS": "20000",
            "DATA_VOL_MAX_THROUGHPUT": "1000",
            "has_accounts_device": False,
            "NETWORK_INTERFACE": "eth0",
            "NETWORK_MAX_BANDWIDTH_GBPS": "100",
        }
        result = _ask_next_blocking_question(state)

        self.assertNotEqual(result["pending_question"].get("group"), "chain_auxiliary_endpoints")

    def test_proactively_pasted_chain_auxiliary_field_is_applied_not_discarded(self) -> None:
        """Code review found `CONFIRMABLE_CONFIG_FIELDS` was not extended for

        the 7 new chain_auxiliary_endpoints fields, so a proactively-pasted
        value like RPC_API_KEY fell into unmapped_values instead of
        confirmed_config, and the harness would re-ask for it.
        """

        from agent.harness.domains.environment import build_config_proposal

        proposal = build_config_proposal({"config_values": {"RPC_API_KEY": "abc123"}})

        self.assertEqual(proposal["config_values"].get("RPC_API_KEY"), "abc123")
        self.assertNotIn("RPC_API_KEY", proposal["unmapped_values"])

    def test_advanced_tuning_is_reachable_and_not_a_dead_end(self) -> None:
        """Architecture audit Finding A: `advanced_tuning` was declared in

        `DEFAULT_GROUP_ORDER` but never implemented in `coordinator.py`, so a
        user routed there hit a dead end. This is the fixed behavior: the
        group asks a real, documented question grounded in
        `config/user_config.sh` / `config/internal_config.sh` field names.
        """

        from agent.harness.coordinator import _ask_next_blocking_question

        state = self._fully_configured_state_before_advanced_tuning()
        result = _ask_next_blocking_question(state)

        self.assertEqual(result["pending_question"]["id"], "advanced_tuning_confirm")
        self.assertEqual(result["pending_question"]["group"], "advanced_tuning")
        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("MONITOR_INTERVAL", text)
        self.assertIn("BOTTLENECK_CPU_THRESHOLD", text)

    def test_advanced_tuning_accepting_defaults_advances_to_preflight(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn

        state = self._fully_configured_state_before_advanced_tuning()
        state["last_user_input"] = "Y"
        state["pending_question"] = {
            "id": "advanced_tuning_confirm",
            "group": "advanced_tuning",
            "kind": "yes_no",
            "field": "advanced_tuning_confirmed",
            "options": [{"label": "Y", "value": True}, {"label": "N", "value": False}],
            "manual_input_allowed": False,
        }
        result = process_turn(state)

        self.assertTrue(result["advanced_tuning"]["confirmed"])
        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("preflight", text.lower())

    def test_advanced_tuning_adjustment_loop_records_override_then_finishes(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn

        state = self._fully_configured_state_before_advanced_tuning()
        state["advanced_tuning"] = {"default_decision_made": True, "confirmed": False}
        state["last_user_input"] = "BOTTLENECK_CPU_THRESHOLD"
        state["pending_question"] = {
            "id": "advanced_tuning_adjust_field",
            "group": "advanced_tuning",
            "kind": "numbered_choice",
            "field": "advanced_tuning_adjust_field",
            "options": [{"label": "BOTTLENECK_CPU_THRESHOLD", "value": "BOTTLENECK_CPU_THRESHOLD"}, {"label": "Finish adjustments", "value": "done"}],
            "manual_input_allowed": False,
        }
        after_field_choice = process_turn(state)
        self.assertEqual(after_field_choice["advanced_tuning"]["adjust_field"], "BOTTLENECK_CPU_THRESHOLD")
        self.assertEqual(after_field_choice["pending_question"]["id"], "advanced_tuning_adjust_value")

        after_field_choice["last_user_input"] = "75"
        after_value = process_turn(after_field_choice)
        self.assertEqual(after_value["advanced_tuning"]["overrides"]["BOTTLENECK_CPU_THRESHOLD"], "75")
        self.assertFalse(after_value["advanced_tuning"].get("confirmed", False))

        after_value["last_user_input"] = "done"
        after_value["pending_question"] = {
            "id": "advanced_tuning_adjust_field",
            "group": "advanced_tuning",
            "kind": "numbered_choice",
            "field": "advanced_tuning_adjust_field",
            "options": [{"label": "Finish adjustments", "value": "done"}],
            "manual_input_allowed": False,
        }
        finished = process_turn(after_value)
        self.assertTrue(finished["advanced_tuning"]["confirmed"])

    def test_free_text_advanced_tuning_adjust_redirects_into_advanced_tuning_group(self) -> None:
        """Phase 6 item 7: a free-text "adjust the CPU bottleneck threshold"

        turn (with no active advanced_tuning pending question) must be
        redirected into the advanced_tuning group via change_group, mirroring
        the existing QPS redirect rule, instead of being answered as a
        config explanation. The intent prompt now emits
        change_group group=advanced_tuning for this phrasing; this test proves
        that action actually routes into the group's own Y/N gate rather than
        being dropped.
        """

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn

        state = self._fully_configured_state_before_advanced_tuning()
        # Park on an unrelated, already-satisfied group so activating
        # advanced_tuning is a genuine redirect, not natural progression.
        state["active_group"] = "workload_rpc"
        state["pending_question"] = {}
        state["last_user_input"] = "我想改一下 CPU 瓶颈阈值"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [{"type": "change_group", "group": "advanced_tuning", "confidence": "high", "selection_contract_verified": True}]
            }
            result = process_turn(state)

        self.assertEqual(result["active_group"], "advanced_tuning")
        self.assertEqual(result["pending_question"]["id"], "advanced_tuning_confirm")

    def test_oracle_and_groups_agree_on_next_group_for_advanced_tuning(self) -> None:
        """Architecture audit Finding B1: `oracle.py` and `coordinator.py` used to

        reimplement the same next-group precondition chain independently
        and could disagree. Both must now delegate to
        `agent.harness.routing.next_group_and_reason`.
        """

        from agent.harness.coordinator import _next_group
        from agent.harness.oracle import _next_group_and_reason

        state = self._fully_configured_state_before_advanced_tuning()
        self.assertEqual(_next_group(state), "advanced_tuning")
        self.assertEqual(_next_group_and_reason(state), ("advanced_tuning", "review advanced tuning settings"))

    def test_oracle_and_groups_agree_on_next_group_after_client_setup_ack(self) -> None:
        """After the client-setup handoff is acknowledged, `coordinator.py`'s

        `_question_for_group` asks `sync_observe_after_client_setup` (choose
        the real data source now that a client is being prepared) before it
        will ever ask for a stop condition. `routing.next_group_and_reason`
        (and therefore `oracle.py`'s advisory "next blocking item" preview)
        used to skip straight to "choose sync-observe stop condition"
        instead, because it had no branch for `client_setup_acknowledged`
        being true, only for it being false — a stale/misleading preview
        immediately after the handoff turn.
        """

        from agent.harness.coordinator import _question_for_group
        from agent.harness.oracle import _next_group_and_reason
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "sync-observe"
        state["workflow_mode"] = "sync_observe"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {
            "BLOCKCHAIN_NODE": "bsc",
            "CLOUD_REGION": "us-1",
            "CLOUD_ZONE": "us-1-z",
            "MACHINE_TYPE": "n2",
            "LEDGER_DEVICE": "vda",
            "DATA_VOL_TYPE": "hyperdisk-balanced",
            "DATA_VOL_SIZE": "926",
            "DATA_VOL_MAX_IOPS": "20000",
            "DATA_VOL_MAX_THROUGHPUT": "1000",
            "has_accounts_device": False,
            "NETWORK_INTERFACE": "eth0",
            "NETWORK_MAX_BANDWIDTH_GBPS": "100",
        }
        state["sync_observe"] = {"source": "client_setup", "client_setup_acknowledged": True}

        question = _question_for_group(state, "sync_observe")
        self.assertEqual(question["id"], "sync_observe_after_client_setup")

        group, reason = _next_group_and_reason(state)
        self.assertEqual(group, "sync_observe")
        self.assertEqual(reason, "choose a real sync-observe source after client setup guidance")
        self.assertNotEqual(reason, "choose sync-observe stop condition")

    def test_explicit_group_jump_is_not_replaced_by_an_invalidated_fallback_group(self) -> None:
        """Found via live dual-AI chaos (2026-07-13, tendermint-family coverage

        sweep): a single turn that both switches the real-node chain and asks
        to jump ahead (e.g. "switch to hedera, only have a real endpoint, ...")
        can resolve into a compound action queue: `change_chain` (which pauses
        on a `chain_change_confirm` interrupt) followed by a queued
        `change_group` targeting a later group (e.g. `workload_rpc`). Once the
        interrupt is confirmed, `_invalidate_for_chain_change` correctly clears
        `endpoint_evidence` for the new chain, but the queued `change_group`
        then ran via `_activate_group_question` with no check that the target
        group's prerequisites were still satisfied -- it jumped straight to
        `workload_rpc` and asked for the new chain's workload before its
        endpoint had ever been validated, while `confirmed_config`'s stale
        `LOCAL_RPC_URL` (still the OLD chain's endpoint) sat unchanged.
        Confirmed live via checkpoint inspection: `active_group` became
        `workload_rpc` immediately after the chain-change confirm, with
        `endpoint_evidence == {}` and `confirmed_config["LOCAL_RPC_URL"]`
        still pointing at the previous chain's endpoint.
        """

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "real-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "cosmos-hub", "canonical": "cosmos-hub", "adapter_family": "tendermint", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {
            "BLOCKCHAIN_NODE": "cosmos-hub",
            "CLOUD_REGION": "us-1",
            "CLOUD_ZONE": "us-1-z",
            "MACHINE_TYPE": "n2",
            "LEDGER_DEVICE": "vda",
            "DATA_VOL_TYPE": "hyperdisk-balanced",
            "DATA_VOL_SIZE": "926",
            "DATA_VOL_MAX_IOPS": "20000",
            "DATA_VOL_MAX_THROUGHPUT": "1000",
            "has_accounts_device": False,
            "NETWORK_INTERFACE": "eth0",
            "NETWORK_MAX_BANDWIDTH_GBPS": "100",
            "LOCAL_RPC_URL": "https://cosmos-rest.publicnode.com",
            "BLOCKCHAIN_PROCESS_NAMES": "cosmos-hub-node",
            "MAINNET_RPC_URL_REVIEWED": True,
        }
        state["endpoint_evidence"] = {"local_rpc_url_ready": True}
        state["rpc_mode"] = "mixed"
        state["workload"] = {"confirmed": True, "choice": "default"}
        state["active_group"] = "qps_profile"
        state["pending_question"] = {
            "id": "benchmark_mode",
            "group": "qps_profile",
            "kind": "numbered_choice",
            "field": "benchmark_mode",
            "options": [{"label": "quick", "value": "quick"}],
            "manual_input_allowed": False,
        }

        state["last_user_input"] = "先别定 QPS，我想先换成 hedera，然后回到 RPC workload 配置"
        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "change_chain", "chain_text": "hedera", "source_evidence": "换成 hedera", "confidence": "high"},
                    {"type": "change_group", "group": "workload_rpc", "navigation_explicit": True, "source_evidence": "回到 RPC workload 配置", "confidence": "high"},
                ]
            }
            state = process_turn(state)
        self.assertEqual(state["pending_question"]["id"], "chain_change_confirm")

        state["last_user_input"] = "1"
        state = process_turn(state)

        self.assertEqual(state["chain_identity"]["canonical"], "hedera")
        self.assertEqual(state["endpoint_evidence"], {})
        self.assertEqual(state["active_group"], "workload_rpc")
        self.assertEqual(state["pending_question"]["id"], "rpc_mode")
        self.assertIn("endpoint_process", state.get("invalidated_groups") or [])

    def test_routing_recognizes_phase4_custom_rpc_statuses(self) -> None:
        """Code review (Phase 4 diff) found `routing.next_group_and_reason`'s

        custom_rpc status allowlist was never extended for the two statuses
        Phase 4 introduced (`needs_single_method`, `needs_adapter_family_confirmation`),
        so the harness would silently skip past the pending question straight
        to `workload_rpc` while `custom_rpc.status` stayed stuck forever.
        """

        from agent.harness.routing import next_group_and_reason
        from agent.harness.state import new_state

        base = new_state("unit-thread")
        base["target_mode"] = "fake-node"
        base["workflow_mode"] = "rpc_benchmark"
        base["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
        base["confirmed_config"] = {
            "CLOUD_REGION": "us-1",
            "CLOUD_ZONE": "us-1-z",
            "MACHINE_TYPE": "n2",
            "LEDGER_DEVICE": "vda",
            "DATA_VOL_TYPE": "hyperdisk-balanced",
            "DATA_VOL_SIZE": "926",
            "DATA_VOL_MAX_IOPS": "20000",
            "DATA_VOL_MAX_THROUGHPUT": "1000",
            "has_accounts_device": False,
            "NETWORK_INTERFACE": "eth0",
            "NETWORK_MAX_BANDWIDTH_GBPS": "100",
        }
        for status in ("needs_single_method", "needs_adapter_family_confirmation"):
            state = dict(base)
            state["custom_rpc"] = {"status": status, "validated_methods": [{"method": "eth_blockNumber"}]}
            group, reason = next_group_and_reason(state)
            self.assertEqual(group, "endpoint_process", f"status={status}")
            self.assertIn(status, reason)

    def test_restart_help_does_not_fall_through_to_completed_group_status(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed"}
        state["observability"] = {"mode": "disabled"}
        state["last_user_input"] = "如何重新开始呢"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "answer_opening_question", "topic": "reset_help", "confidence": "high"}]}
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("完全重新开始", text)
        self.assertIn("完全重新开始", text)
        self.assertNotIn("可观测性模式", text)
        self.assertNotIn("preflight_smoke", text)

    def test_reset_session_action_clears_config_and_invocation_context(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["framework_summary"] = {"chain_count": 36}
        state["discovery"] = {"cloud": {"provider": "other"}}
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc", "CLOUD_REGION": "us-1"}
        state["observability"] = {"mode": "disabled"}
        state["last_user_input"] = "清空配置重新开始"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "reset_session", "confidence": "high"}]}
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("已清空之前的 Agent 配置", text)
        self.assertEqual(result.get("chain_identity"), {})
        self.assertEqual(result.get("confirmed_config"), {})
        self.assertEqual(result.get("target_mode"), "")
        self.assertEqual(result.get("framework_summary"), {})
        self.assertEqual(result.get("discovery"), {})

    def test_reset_session_keeps_later_actions_from_the_same_user_turn(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["target_mode"] = "real-node"
        state["chain_identity"] = {"raw": "solana", "canonical": "solana", "status": "confirmed"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "solana", "CLOUD_REGION": "old-region"}
        state["last_user_input"] = "Start fresh with BSC fake-node."

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "reset_session", "confidence": "high"},
                    {
                        "type": "choose_target_mode",
                        "target_mode": "fake-node",
                        "target_mode_explicit": True,
                        "source_evidence": "fake-node",
                        "confidence": "high",
                    },
                    {
                        "type": "choose_chain",
                        "chain_text": "BSC",
                        "source_evidence": "BSC",
                        "confidence": "high",
                    },
                ]
            }
            result = process_turn(state)

        self.assertEqual(result.get("target_mode"), "fake-node")
        self.assertEqual((result.get("chain_identity") or {}).get("canonical"), "bsc")
        self.assertNotEqual((result.get("confirmed_config") or {}).get("CLOUD_REGION"), "old-region")

    def test_change_to_another_chain_without_name_asks_chain_instead_of_preflight(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed"}
        state["confirmed_config"] = {
            "BLOCKCHAIN_NODE": "bsc",
            "CLOUD_REGION": "us-1",
            "CLOUD_ZONE": "us-1-z",
            "MACHINE_TYPE": "n2",
            "LEDGER_DEVICE": "vda",
            "DATA_VOL_TYPE": "hyperdisk-balanced",
            "DATA_VOL_SIZE": "926",
            "DATA_VOL_MAX_IOPS": "20000",
            "DATA_VOL_MAX_THROUGHPUT": "1000",
            "has_accounts_device": False,
            "NETWORK_INTERFACE": "eth0",
            "NETWORK_MAX_BANDWIDTH_GBPS": "100",
        }
        state["rpc_mode"] = "single"
        state["workload"] = {"confirmed": True}
        state["qps_profile"] = {"mode": "quick", "confirmed": True}
        state["observability"] = {"mode": "disabled"}
        state["last_user_input"] = "我需要重新测试别的链，可以么"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "change_group", "group": "chain_identity", "navigation_explicit": True, "source_evidence": "重新测试别的链", "confidence": "high"}]}
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertEqual(result.get("active_group"), "chain_identity")
        self.assertEqual((result.get("pending_question") or {}).get("id"), "chain_change_input")
        self.assertIn("输入要切换到的链名", text)
        self.assertNotIn("是否运行 preflight", text)

    def test_opening_recommendation_for_unsure_quick_validation(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = "我想先随便跑一下，但不知道 fake-node 和 real-node 选哪个"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "answer_opening_question", "topic": "recommendation", "confidence": "high"}]}
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("建议先用 fake-node smoke", text)
        self.assertIn("solana", text)
        self.assertEqual(result.get("target_mode"), "")

    def test_pending_jump_to_workload_preserves_followup_qps_action(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "solana", "canonical": "solana", "adapter_family": "jsonrpc", "status": "confirmed"}
        state["pending_question"] = {
            "id": "CLOUD_REGION",
            "group": "provider_deployment",
            "kind": "manual_value",
            "field": "CLOUD_REGION",
            "manual_input_allowed": True,
        }
        state["last_user_input"] = "先别问 region，我想先看看 solana 默认 workload 是什么，然后 QPS 用 quick"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "change_group", "group": "workload_rpc", "navigation_explicit": True, "source_evidence": "先看看 solana 默认 workload", "confidence": "high"},
                    {"type": "set_qps_mode", "qps_mode": "quick", "mutation_explicit": True, "source_evidence": "QPS 用 quick", "confidence": "high"},
                ]
            }
            first = process_turn(state)

        self.assertEqual(first["pending_question"]["id"], "rpc_mode")
        self.assertTrue(first["pending_question"].get("resume_action_queue"))
        self.assertTrue(first.get("action_queue"))

        first["last_user_input"] = "single"
        second = process_turn(first)

        self.assertEqual(second["rpc_mode"], "single")
        self.assertEqual(second["pending_question"]["id"], "workload_confirm")
        self.assertTrue(second["pending_question"].get("resume_action_queue"))
        self.assertIn("workload", "\n".join(second.get("visible_response") or []).lower())

        second["last_user_input"] = "1"
        third = process_turn(second)

        self.assertTrue(third["workload"]["confirmed"])
        self.assertEqual(third["qps_profile"]["mode"], "quick")
        self.assertEqual(third["pending_question"]["id"], "qps_profile_confirm")

    def test_pending_config_review_defers_new_actions_until_user_confirms(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["inferred_config"] = {
            "pending_review": {
                "config_values": {"CLOUD_REGION": "us-1"},
                "unmapped_values": {},
                "source_format": "mixed",
                "reason": "unit",
            }
        }
        state["pending_question"] = {
            "id": "inferred_config_review",
            "group": "provider_deployment",
            "kind": "yes_no",
            "field": "inferred_config_review",
            "options": [{"label": "Y", "value": True}, {"label": "N", "value": False}],
        }
        state["last_user_input"] = "先把 QPS 改成 quick"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "set_qps_mode", "qps_mode": "quick", "mutation_explicit": True, "source_evidence": "QPS 改成 quick", "confidence": "high"}]}
            first = process_turn(state)

        self.assertEqual(first["pending_question"]["id"], "inferred_config_review")
        self.assertNotIn("resume_action_queue", first["pending_question"])
        self.assertEqual(
            [item["type"] for item in first["action_queue"]],
            ["request_target_mode_selection", "set_qps_mode"],
        )

        first["last_user_input"] = "Y"
        second = process_turn(first)

        self.assertEqual(second["confirmed_config"]["CLOUD_REGION"], "us-1")
        self.assertEqual(second["pending_question"]["id"], "target_mode_select")
        self.assertEqual(second["action_queue"][0]["type"], "set_qps_mode")

        second["last_user_input"] = "1"
        third = process_turn(second)

        self.assertEqual(third["target_mode"], "fake-node")
        self.assertEqual(third["qps_profile"]["mode"], "quick")
        self.assertEqual(third["pending_question"]["id"], "qps_profile_confirm")

    def test_custom_rpc_capability_question_does_not_start_endpoint_workflow(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "adapter_family": "jsonrpc"}
        state["pending_question"] = {
            "id": "rpc_mode",
            "group": "workload_rpc",
            "kind": "numbered_choice",
            "field": "rpc_mode",
            "options": [{"label": "single", "value": "single"}, {"label": "mixed", "value": "mixed"}],
        }
        state["last_user_input"] = "默认 workload 是什么？我可以加自定义 rpc 吗？"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "answer_opening_question", "topic": "extension", "subject": "workload_rpc", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertNotEqual(result.get("pending_question", {}).get("id"), "custom_rpc_endpoint")
        self.assertNotEqual((result.get("custom_rpc") or {}).get("status"), "needs_endpoint")
        self.assertIn("扩展分三类", text)

    def test_workload_pending_context_uses_user_facing_explanation(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["pending_question"] = {
            "id": "workload_confirm",
            "group": "workload_rpc",
            "kind": "numbered_choice",
            "field": "workload_choice",
            "options": [{"label": "使用默认值", "value": "default"}],
        }
        state["last_user_input"] = "这个是做什么的？"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {
                        "type": "answer_opening_question",
                        "topic": "config_explanation",
                        "subject": "workload_choice",
                        "confidence": "high",
                    }
                ]
            }
            result = process_turn(state)
        text = "\n".join(result.get("visible_response") or [])

        self.assertIn("默认 RPC workload", text)
        self.assertNotIn("workload_rpc 用于生成", text)

    def test_model_cannot_store_navigation_prose_as_pending_scalar(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "solana", "canonical": "solana", "status": "confirmed"}
        state["active_group"] = "provider_deployment"
        state["pending_question"] = {
            "id": "CLOUD_REGION",
            "group": "provider_deployment",
            "kind": "manual_value",
            "field": "CLOUD_REGION",
            "manual_input_allowed": True,
            "validation": {"input_mode": "scalar"},
        }
        state["last_user_input"] = "先别配置 region，我要回到 RPC workload"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "change_group", "group": "workload_rpc", "navigation_explicit": True, "source_evidence": "回到 RPC workload", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        self.assertNotIn("CLOUD_REGION", result.get("confirmed_config") or {})
        self.assertEqual(result.get("active_group"), "workload_rpc")
        self.assertEqual((result.get("pending_question") or {}).get("id"), "rpc_mode")

    def test_same_group_navigation_preserves_active_typed_question(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.chain_rpc import question_for_chain_rpc
        from agent.harness.state import new_state

        state = new_state("same-group-navigation", language="en")
        state.update({
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "chain_identity": {"raw": "bsc", "canonical": "bsc", "status": "confirmed"},
            "rpc_mode": "single",
            "workload": {"confirmed": False},
            "active_group": "workload_rpc",
            "last_user_input": "go back to RPC config",
        })
        state["pending_question"] = question_for_chain_rpc(state, "workload_rpc") or {}
        original_question = dict(state["pending_question"])

        with patch("agent.harness.coordinator.resolve_action_queue", return_value={
            "actions": [{
                "type": "change_group",
                "group": "workload_rpc",
                "navigation_explicit": True,
                "source_evidence": "go back to RPC config",
                "semantic_purpose_verified": True,
                "confidence": "high",
            }],
        }):
            result = process_turn(state)

        self.assertEqual(result.get("pending_question"), original_question)
        self.assertEqual(result.get("active_group"), "workload_rpc")
        self.assertIn("Current chain template workload", "\n".join(result.get("visible_response") or []))

    def test_model_selected_choice_executes_without_redundant_answer_text(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["active_group"] = "opening"
        state["pending_question"] = {
            "id": "opening_next_action",
            "group": "opening",
            "kind": "numbered_choice",
            "field": "opening_next_action",
            "options": [
                {
                    "id": "fake-node",
                    "label": "Start fake-node",
                    "value": "fake-node",
                    "action": {
                        "type": "choose_target_mode",
                        "target_mode": "fake-node",
                        "target_mode_explicit": True,
                    },
                    "expected_patch": {"target_mode": "fake-node"},
                }
            ],
        }
        state["last_user_input"] = "Use the fake-node option and set Solana mixed quick"

        with patch("agent.harness.coordinator.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {
                        "type": "answer_pending",
                        "selected_value": "fake-node",
                        "source_evidence": "fake-node",
                        "pending_option_semantic_verified": True,
                        "semantic_purpose_verified": True,
                        "_admission_action_id": "opening-choice-1",
                        "_transaction_action_ids": ["opening-choice-1"],
                        "confidence": "high",
                    },
                    {
                        "type": "choose_chain",
                        "chain_text": "solana",
                        "source_evidence": "Solana",
                        "confidence": "high",
                    },
                    {
                        "type": "set_rpc_mode",
                        "rpc_mode": "mixed",
                        "mutation_explicit": True,
                        "source_evidence": "mixed",
                        "confidence": "high",
                    },
                    {
                        "type": "set_qps_mode",
                        "qps_mode": "quick",
                        "mutation_explicit": True,
                        "source_evidence": "quick",
                        "confidence": "high",
                    },
                ],
                "pending_choice_contracts": [{
                    "action_index": 0,
                    "admission_action_id": "opening-choice-1",
                    "question": {"id": "opening_next_action", "group": "opening"},
                    "option": {"id": "fake-node", "selected_value": "fake-node"},
                    "semantic_units": [{
                        "unit_id": "unit-1",
                        "clause_id": "clause-1",
                        "source_text": "Use the fake-node option and set Solana mixed quick",
                    }],
                }],
            }
            result = process_turn(state)

        self.assertEqual(result.get("target_mode"), "fake-node")
        self.assertEqual((result.get("chain_identity") or {}).get("canonical"), "solana")
        self.assertNotIn("the pending answer is empty", "\n".join(result.get("visible_response") or []))

    def test_declared_option_owner_cannot_bypass_canonical_pending_contract(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.orientation import opening_question
        from agent.harness.state import new_state

        state = new_state("semantic-option", language="en")
        state["pending_question"] = opening_question(state)
        state["last_user_input"] = "I just want a low-risk dry run with the simulated node first."
        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{
                "type": "choose_target_mode",
                "target_mode": "fake-node",
                "target_mode_explicit": True,
                "source_evidence": "low-risk dry run with the simulated node first",
                "semantic_purpose_verified": True,
                "target_mode_semantic_verified": True,
                "confidence": "medium",
            }]},
        ):
            result = process_turn(state)

        self.assertEqual(result.get("target_mode"), "")
        self.assertEqual((result.get("pending_question") or {}).get("id"), "opening_next_action")

    def test_declared_option_binding_rejects_source_evidence_outside_current_turn(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.orientation import opening_question
        from agent.harness.state import new_state

        state = new_state("semantic-option-forged", language="en")
        state["pending_question"] = opening_question(state)
        state["last_user_input"] = "Please recommend the safest place to begin."
        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{
                "type": "choose_target_mode",
                "target_mode": "fake-node",
                "target_mode_explicit": True,
                "source_evidence": "fake-node",
                "confidence": "high",
            }]},
        ):
            result = process_turn(state)

        self.assertEqual(result.get("target_mode"), "")
        self.assertEqual((result.get("pending_question") or {}).get("id"), "opening_next_action")

    def test_semantic_target_mode_selection_uses_the_declared_pending_option(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("semantic-target-mode", language="en")
        state["active_group"] = "target_mode"
        state["pending_question"] = {
            "id": "target_mode_select",
            "group": "target_mode",
            "kind": "numbered_choice",
            "field": "target_mode",
            "options": [
                {
                    "id": mode,
                    "label": mode,
                    "value": mode,
                    "action": {
                        "type": "choose_target_mode",
                        "target_mode": mode,
                        "target_mode_explicit": True,
                    },
                }
                for mode in ("fake-node", "real-node", "sync-observe")
            ],
            "accepted_action_types": ["answer_pending", "choose_target_mode"],
            "queue_barrier": True,
        }
        state["last_user_input"] = (
            "Plans changed: do not generate traffic. Watch the running node catch up to chain head."
        )
        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{
                "type": "answer_pending",
                "answer": "sync-observe",
                "selected_value": "sync-observe",
                "source_evidence": "Watch the running node catch up to chain head.",
                "pending_option_semantic_verified": True,
                "semantic_purpose_verified": True,
                "_admission_action_id": "target-mode-choice-1",
                "_transaction_action_ids": ["target-mode-choice-1"],
                "confidence": "medium",
            }], "pending_choice_contracts": [{
                "action_index": 0,
                "admission_action_id": "target-mode-choice-1",
                "question": {"id": "target_mode_select", "group": "target_mode"},
                "option": {"id": "sync-observe", "selected_value": "sync-observe"},
                "semantic_units": [{
                    "unit_id": "unit-1",
                    "clause_id": "clause-1",
                    "source_text": "Watch the running node catch up to chain head.",
                }],
            }]},
        ):
            result = process_turn(state)

        self.assertEqual(result.get("target_mode"), "sync-observe")
        admitted = (result.get("turn_context") or {}).get("admitted_actions") or []
        self.assertIn("choose_target_mode", {item.get("type") for item in admitted})

    def test_semantic_target_mode_selection_cannot_escape_the_pending_contract(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("semantic-target-mode-invalid", language="en")
        state["active_group"] = "target_mode"
        state["pending_question"] = {
            "id": "target_mode_select",
            "group": "target_mode",
            "kind": "numbered_choice",
            "field": "target_mode",
            "options": [{
                "id": "fake-node",
                "label": "fake-node",
                "value": "fake-node",
                "action": {
                    "type": "choose_target_mode",
                    "target_mode": "fake-node",
                    "target_mode_explicit": True,
                },
            }],
            "accepted_action_types": ["answer_pending", "choose_target_mode"],
            "queue_barrier": True,
        }
        state["last_user_input"] = "Watch a running node catch up."
        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{
                "type": "choose_target_mode",
                "target_mode": "sync-observe",
                "target_mode_explicit": True,
                "source_evidence": "Watch a running node catch up.",
                "pending_option_semantic_verified": True,
                "semantic_purpose_verified": True,
                "confidence": "medium",
            }]},
        ):
            result = process_turn(state)

        self.assertEqual(result.get("target_mode"), "")
        self.assertEqual((result.get("pending_question") or {}).get("id"), "target_mode_select")

    def test_model_blank_selected_value_does_not_hide_manual_answer(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
        state["custom_rpc"] = {"status": "needs_endpoint", "method": "eth_chainId"}
        state["active_group"] = "endpoint_process"
        state["pending_question"] = {
            "contract_version": 1,
            "id": "custom_rpc_endpoint",
            "group": "endpoint_process",
            "kind": "url",
            "field": "custom_rpc_endpoint",
            "manual_input_allowed": True,
            "options": [],
            "accepted_action_types": ["answer_pending", "start_custom_rpc"],
            "validation": {"value_type": "url"},
        }
        state["last_user_input"] = (
            "My selected validation endpoint is http://fake-node:19000. "
            "The documentation example https://example.invalid/rpc is not selected."
        )
        plan = {
            "actions": [{
                "type": "answer_pending",
                "answer": "http://fake-node:19000",
                "selected_value": "",
                "source_evidence": "My selected validation endpoint is http://fake-node:19000.",
                "confidence": "high",
            }]
        }

        with patch("agent.harness.coordinator.resolve_action_queue", return_value=plan), patch(
            "agent.harness.domains.rpc_endpoint.validate_rpc_endpoint",
            return_value={"ready": True, "status": "ready", "evidence_file": "probe.json"},
        ):
            result = process_turn(state)

        self.assertEqual((result.get("custom_rpc") or {}).get("endpoint"), "http://fake-node:19000")
        self.assertNotIn("example.invalid", str(result.get("confirmed_config") or {}))
        self.assertNotIn("example.invalid", str(result.get("endpoint_evidence") or {}))

    def test_new_pending_question_replaces_the_previous_rendered_question(self) -> None:
        from agent.harness.contracts import HandlerResult
        from agent.harness.coordinator import _apply_handler_result, _render_question
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        previous = {
            "id": "first_question",
            "group": "provider_deployment",
            "kind": "manual_value",
            "field": "CLOUD_REGION",
            "prompt": "Enter the first value.",
            "manual_input_allowed": True,
            "options": [],
            "accepted_action_types": ["answer_pending"],
            "validation": {"value_type": "scalar_token"},
        }
        replacement = {
            "id": "second_question",
            "group": "target_mode",
            "kind": "numbered_choice",
            "field": "target_mode",
            "prompt": "Choose the replacement value.",
            "manual_input_allowed": False,
            "options": [
                {
                    "id": "real-node",
                    "label": "real-node",
                    "value": "real-node",
                    "action": {"type": "choose_target_mode", "target_mode": "real-node"},
                    "expected_patch": {"target_mode": "real-node"},
                    "return_policy": "fallback",
                }
            ],
            "accepted_action_types": ["answer_pending", "choose_target_mode"],
            "validation": {},
        }
        state["pending_question"] = previous
        state["visible_response"] = ["Endpoint validation passed.", _render_question(previous, "en")]

        result = _apply_handler_result(
            state,
            HandlerResult(pending_question=replacement),
            owner="coordinator",
        )

        self.assertEqual((result.get("pending_question") or {}).get("id"), "second_question")
        self.assertIn(_render_question(replacement, "en"), result.get("visible_response") or [])
        self.assertNotIn(_render_question(previous, "en"), result.get("visible_response") or [])
        self.assertIn("Endpoint validation passed.", result.get("visible_response") or [])

    def test_model_chain_summary_cannot_claim_google_search_provenance(self) -> None:
        from agent.harness.domains.chain_identity import _verified_search_summary

        model_only = {
            "evidence_summary": "Model training context says this is a REST chain.",
            "confidence": "high",
        }
        grounded = {
            **model_only,
            "search_result": {
                "available": True,
                "text_summary": "Official documentation confirms the protocol.",
            },
        }

        self.assertEqual(_verified_search_summary(model_only), "")
        self.assertEqual(
            _verified_search_summary(grounded),
            "Official documentation confirms the protocol.",
        )

    def test_english_custom_weight_error_does_not_mix_chinese_details(self) -> None:
        from agent.harness.domains.chain_rpc import apply_chain_rpc_answer
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["chain_identity"] = {"canonical": "bsc", "status": "confirmed", "case": "known"}
        state["custom_rpc"] = {
            "scope": "mixed_replace",
            "validated_methods": [{"method": "eth_chainId", "params": []}],
        }

        outcome = apply_chain_rpc_answer(
            state,
            {
                "id": "custom_rpc_weights",
                "group": "endpoint_process",
                "field": "custom_rpc_weights",
                "kind": "manual_value",
                "manual_input_allowed": True,
            },
            "eth_chainId=70",
            "eth_chainId=70",
        )
        state = _commit_result(state, outcome, owner="chain_rpc")

        text = "\n".join(state.get("visible_response") or [])
        self.assertIn("Weight total is 70", text)
        self.assertNotIn("权重", text)

    def test_invalidated_workload_enters_reconfiguring_lifecycle(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed"}
        state["confirmed_config"] = {
            "BLOCKCHAIN_NODE": "bsc",
            "CLOUD_REGION": "us-1",
            "CLOUD_ZONE": "us-1-z",
            "MACHINE_TYPE": "n2",
            "LEDGER_DEVICE": "vda",
            "DATA_VOL_TYPE": "hyperdisk-balanced",
            "DATA_VOL_SIZE": "926",
            "DATA_VOL_MAX_IOPS": "20000",
            "DATA_VOL_MAX_THROUGHPUT": "1000",
            "has_accounts_device": False,
            "NETWORK_INTERFACE": "eth0",
        }
        state["active_group"] = "network"
        state["pending_question"] = {
            "contract_version": 1,
            "id": "NETWORK_MAX_BANDWIDTH_GBPS",
            "group": "network",
            "kind": "manual_value",
            "field": "NETWORK_MAX_BANDWIDTH_GBPS",
            "manual_input_allowed": True,
            "options": [],
            "accepted_action_types": ["answer_pending"],
            "validation": {"value_type": "positive_number"},
        }
        state["invalidated_groups"] = ["workload_rpc"]
        state["group_states"] = {"workload_rpc": {"status": "invalidated"}}
        state["last_user_input"] = "100"

        result = process_turn(state)

        self.assertEqual(result.get("active_group"), "workload_rpc")
        self.assertEqual((result.get("pending_question") or {}).get("id"), "rpc_mode")
        self.assertEqual((result.get("group_states") or {}).get("workload_rpc", {}).get("status"), "reconfiguring")
        self.assertIn("workload_rpc", result.get("invalidated_groups") or [])


    def test_ordered_compound_plan_mutates_detour_then_returns_to_pending_group(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("ordered-detour", language="zh")
        state.update({
            "target_mode": "sync-observe",
            "workflow_mode": "sync_observe",
            "chain_identity": {
                "raw": "bsc",
                "canonical": "bsc",
                "adapter_family": "jsonrpc",
                "status": "confirmed",
            },
            "confirmed_config": {"BLOCKCHAIN_NODE": "bsc", "CLOUD_REGION": "asia-east1"},
            "active_group": "provider_deployment",
            "pending_question": {
                "id": "CLOUD_ZONE",
                "group": "provider_deployment",
                "kind": "manual_value",
                "field": "CLOUD_ZONE",
                "prompt": "请输入 CLOUD_ZONE。",
                "validation": {"value_type": "scalar_token", "max_length": 180},
            },
            "last_user_input": "先跳到可观测性，设成 exporter-only；然后回来继续填 zone。",
        })
        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [
                {
                    "type": "set_observability",
                    "observability_mode": "exporter",
                    "mutation_explicit": True,
                    "source_evidence": "设成 exporter-only",
                    "confidence": "high",
                },
                {
                    "type": "change_group",
                    "group": "provider_deployment",
                    "navigation_explicit": True,
                    "source_evidence": "回来继续填 zone",
                    "confidence": "high",
                },
            ]},
        ):
            result = process_turn(state)

        self.assertEqual(result["observability"]["mode"], "exporter")
        self.assertEqual(result["pending_question"]["id"], "CLOUD_ZONE")
        self.assertEqual(result["active_group"], "provider_deployment")
        self.assertEqual(
            [item["type"] for item in result["completed_actions"]],
            ["set_observability", "change_group"],
        )

    def test_compound_mutually_exclusive_workflows_preserve_later_goal(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("ordered-workflow-goals", language="en")
        state["last_user_input"] = "Observe BSC sync now, then benchmark its real RPC capacity later."
        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [
                {
                    "type": "choose_target_mode",
                    "target_mode": "sync-observe",
                    "target_mode_explicit": True,
                    "source_evidence": "Observe BSC sync now",
                    "confidence": "high",
                },
                {
                    "type": "choose_chain",
                    "chain_text": "bsc",
                    "chain_candidates": ["bsc"],
                    "source_evidence": "BSC",
                    "confidence": "high",
                },
                {
                    "type": "queue_workflow_goal",
                    "target_mode": "real-node",
                    "goal": "benchmark its real RPC capacity later",
                    "source_evidence": "benchmark its real RPC capacity later",
                    "confidence": "high",
                },
            ]},
        ):
            result = process_turn(state)

        self.assertEqual(result["target_mode"], "sync-observe")
        self.assertEqual((result.get("chain_identity") or {}).get("canonical"), "bsc")
        self.assertEqual(result["workflow_goals"], [{
            "target_mode": "real-node",
            "goal": "benchmark its real RPC capacity later",
            "source_evidence": "benchmark its real RPC capacity later",
        }])
        self.assertIn("Saved a later goal", "\n".join(result.get("visible_response") or []))

    def test_activating_saved_workflow_goal_uses_normal_mode_change_contract(self) -> None:
        from agent.harness.coordinator import _process_action_queue
        from agent.harness.state import new_state

        state = new_state("activate-workflow-goal", language="en")
        state.update({
            "target_mode": "sync-observe",
            "workflow_mode": "sync_observe",
            "chain_identity": {"canonical": "bsc", "status": "confirmed", "adapter_family": "jsonrpc"},
            "workflow_goals": [{
                "target_mode": "real-node",
                "goal": "benchmark real RPC capacity",
                "source_evidence": "then benchmark real RPC capacity",
            }],
        })

        result = _process_action_queue(
            state,
            [{
                "type": "activate_next_workflow_goal",
                "source_evidence": "start the saved later goal",
                "confidence": "high",
            }],
            "start the saved later goal",
        )

        self.assertEqual(result["target_mode"], "sync-observe")
        self.assertEqual(result["workflow_goals"], [])
        self.assertEqual((result.get("pending_question") or {}).get("id"), "target_mode_change_confirm")
        self.assertEqual((result.get("target_mode_change_candidate") or ""), "real-node")

    def test_resume_summary_exposes_saved_workflow_goal(self) -> None:
        from agent.harness.domains.orientation import resume_summary
        from agent.harness.state import new_state

        state = new_state("saved-goal-resume-summary", language="en")
        state["workflow_goals"] = [{
            "target_mode": "sync-observe",
            "goal": "observe synchronization after the benchmark",
            "source_evidence": "then observe synchronization",
        }]

        summary = resume_summary(state)

        self.assertIn(
            "deferred requests: sync-observe: observe synchronization after the benchmark",
            summary,
        )

    def test_discarding_saved_workflow_goal_removes_only_the_oldest(self) -> None:
        from agent.harness.coordinator import _process_action_queue
        from agent.harness.state import new_state

        state = new_state("discard-workflow-goal", language="en")
        state["workflow_goals"] = [
            {
                "target_mode": "sync-observe",
                "goal": "observe synchronization",
                "source_evidence": "then observe synchronization",
            },
            {
                "target_mode": "real-node",
                "goal": "benchmark real RPC capacity",
                "source_evidence": "then benchmark real RPC capacity",
            },
        ]

        result = _process_action_queue(
            state,
            [{
                "type": "discard_next_workflow_goal",
                "source_evidence": "remove the saved synchronization follow-up",
                "confidence": "high",
            }],
            "remove the saved synchronization follow-up",
        )

        self.assertEqual(result["workflow_goals"], [state["workflow_goals"][1]])
        self.assertIn("Removed the oldest saved workflow goal", "\n".join(result.get("visible_response") or []))

    def test_saved_workflow_goal_transitions_require_current_turn_evidence(self) -> None:
        from agent.harness.action_registry import validate_action_contract

        for action_type in (
            "activate_next_workflow_goal",
            "discard_next_workflow_goal",
        ):
            with self.subTest(action_type=action_type):
                with self.assertRaisesRegex(ValueError, "source_evidence"):
                    validate_action_contract({"type": action_type})
                validated = validate_action_contract({
                    "type": action_type,
                    "source_evidence": "the saved follow-up",
                })
                self.assertEqual(validated["source_evidence"], "the saved follow-up")




    def test_product_graph_executes_registry_recovered_back_navigation(self) -> None:
        import json
        from types import SimpleNamespace

        from agent.harness.state import new_state
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn

        explanatory = "I changed my mind."
        operation = "Please take me back to the previous workflow step."
        unresolved = {
            "actions": [],
            "semantic_units": [
                {
                    "unit_id": "unit-1",
                    "clause_id": "clause-1",
                    "source_text": explanatory,
                    "disposition": "unresolved",
                    "action_indexes": [],
                    "reason": "unresolved explanatory context",
                },
                {
                    "unit_id": "unit-2",
                    "clause_id": "clause-2",
                    "source_text": operation,
                    "disposition": "unresolved",
                    "action_indexes": [],
                    "reason": "unresolved navigation request",
                },
            ],
        }

        class Provider:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def complete(self, request: object) -> object:
                system = str(request.messages[0].content)
                self.calls.append(system)
                if "typed intent planner" in system:
                    payload = {
                        "actions": [{
                            "type": "go_back",
                            "source_evidence": operation,
                            "confidence": "high",
                        }],
                        "semantic_units": [
                            {
                                "unit_id": "unit-1",
                                "clause_id": "clause-1",
                                "source_text": explanatory,
                                "disposition": "action",
                                "action_indexes": [0],
                                "reason": "explanatory support for the same navigation",
                            },
                            {
                                "unit_id": "unit-2",
                                "clause_id": "clause-2",
                                "source_text": operation,
                                "disposition": "action",
                                "action_indexes": [0],
                                "reason": "direct backward navigation",
                            },
                        ],
                        "reason": "one immutable backward-navigation plan",
                    }
                elif "independent admission authority" in system:
                    review = json.loads(request.messages[1].content)
                    action = review["actions"][0]
                    units = {row["unit_id"]: row for row in review["semantic_units"]}
                    evidence = []
                    for unit_id in action["unit_ids"]:
                        source = units[unit_id]["source_text"]
                        direct = source == operation
                        evidence.append({
                            "unit_id": unit_id,
                            "quote": source,
                            "relation": "direct" if direct else "support",
                            "support_relation": "" if direct else "explanatory_context",
                        })
                    payload = {
                        "plan_hash": review["plan_hash"],
                        "action_verdicts": [{
                            "action_id": action["action_id"],
                            "verdict": "admit",
                            "unit_ids": list(action["unit_ids"]),
                            "evidence": evidence,
                            "grounded_arguments": [
                                {
                                    "argument_name": argument,
                                    "evidence_quote": operation,
                                }
                                for argument in action.get("required_value_grounding_arguments") or []
                            ],
                            "pending_answer_argument": "",
                            "reason": "the immutable action preserves both source units",
                        }],
                        "unit_verdicts": [{
                            "unit_id": row["unit_id"],
                            "verdict": "complete" if row["source_text"] == operation else "support",
                            "owner_action_ids": list(row["owner_action_ids"]),
                            "evidence_quote": row["source_text"],
                            "omitted_action_type": "",
                            "reason": "the navigation demand is represented",
                        } for row in review["semantic_units"]],
                        "reason": "the complete immutable plan is admitted",
                    }
                else:
                    raise AssertionError(f"unexpected model contract: {system[:120]}")
                return SimpleNamespace(text=json.dumps(payload))

        provider = Provider()
        state = new_state("back-product-graph", language="en")
        state.update({
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "active_group": "qps_profile",
            "group_history": ["workload_rpc"],
            "chain_identity": {
                "canonical": "bsc",
                "status": "confirmed",
                "adapter_family": "jsonrpc",
            },
            "confirmed_config": {"BLOCKCHAIN_NODE": "bsc"},
            "rpc_mode": "single",
            "last_user_input": f"{explanatory} {operation}",
        })

        with patch("agent.harness.intent.provider_from_config", return_value=provider):
            result = process_turn(state, allow_semantic_resolver=True)

        self.assertEqual(
            result["active_group"],
            "workload_rpc",
            {
                "visible_response": result.get("visible_response"),
                "failure_recovery": result.get("failure_recovery"),
                "provider_call_count": len(provider.calls),
            },
        )
        self.assertEqual(result["group_history"], [])
        self.assertEqual([item["type"] for item in result["completed_actions"]], ["go_back"])
        self.assertEqual(len(provider.calls), 2)
        self.assertTrue(any("independent admission authority" in call for call in provider.calls))


    def test_repeated_back_navigation_uses_history_then_reports_no_destination(self) -> None:
        from agent.harness.coordinator import _process_action_queue
        from agent.harness.state import new_state

        state = new_state("repeated-registered-back", language="en")
        state.update({
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "active_group": "qps_profile",
            "group_history": ["workload_rpc"],
            "chain_identity": {
                "canonical": "bsc",
                "status": "confirmed",
                "adapter_family": "jsonrpc",
            },
            "confirmed_config": {"BLOCKCHAIN_NODE": "bsc"},
            "rpc_mode": "single",
        })
        action = {"type": "go_back", "source_evidence": "return one step"}

        first = _process_action_queue(state, [action], "return one step")
        second = _process_action_queue(first, [action], "return one step")

        self.assertEqual(first["active_group"], "workload_rpc")
        self.assertEqual(first["group_history"], [])
        self.assertEqual(first["pending_question"]["id"], "workload_confirm")
        self.assertEqual(second["active_group"], "workload_rpc")
        self.assertEqual(second["group_history"], [])
        self.assertFalse(second.get("pending_question"))
        self.assertIn(
            "There is no previous configuration group",
            "\n".join(second.get("visible_response") or []),
        )



    def test_explicit_group_destination_suppresses_conflicting_back_action(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("conflicting-navigation", language="en")
        state.update({
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "active_group": "qps_profile",
            "group_history": ["workload_rpc"],
            "chain_identity": {
                "canonical": "bsc",
                "status": "confirmed",
                "adapter_family": "jsonrpc",
            },
            "confirmed_config": {"BLOCKCHAIN_NODE": "bsc"},
            "last_user_input": "go back to disk settings",
        })
        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [
                {"type": "go_back", "source_evidence": "go back"},
                {
                    "type": "change_group",
                    "group": "ledger_disk",
                    "navigation_explicit": True,
                    "source_evidence": "disk settings",
                },
            ]},
        ):
            result = process_turn(state)

        self.assertEqual(result["active_group"], "ledger_disk")
        self.assertNotEqual(result["active_group"], "workload_rpc")
        self.assertEqual(
            [item["type"] for item in result.get("completed_actions") or []],
            ["change_group"],
        )

    def test_saved_workflow_goal_is_resumable_and_visible_in_status(self) -> None:
        from agent.harness.domains.orientation import has_resumable_configuration
        from agent.harness.oracle import format_current_state
        from agent.harness.state import new_state

        state = new_state("saved-goal-status", language="en")
        state["workflow_goals"] = [{
            "target_mode": "real-node",
            "goal": "benchmark real RPC capacity",
            "source_evidence": "later benchmark real RPC capacity",
        }]

        self.assertTrue(has_resumable_configuration(state))
        self.assertIn("real-node: benchmark real RPC capacity", format_current_state(state, "en"))

    def test_workflow_goal_without_exact_user_evidence_is_rejected(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("invalid-workflow-goal", language="en")
        state["last_user_input"] = "observe sync now"
        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{
                "type": "queue_workflow_goal",
                "target_mode": "real-node",
                "goal": "benchmark later",
                "source_evidence": "invented evidence",
                "confidence": "high",
            }]},
        ):
            result = process_turn(state)

        self.assertEqual(result.get("workflow_goals"), [])

    def test_partial_qps_customization_suspends_current_question_and_asks_for_value(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("partial-qps-jump", language="en")
        state.update({
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "chain_identity": {"canonical": "bsc", "status": "confirmed", "adapter_family": "jsonrpc"},
            "active_group": "provider_deployment",
            "pending_question": {
                "id": "CLOUD_ZONE",
                "group": "provider_deployment",
                "kind": "manual_value",
                "field": "CLOUD_ZONE",
                "prompt": "Enter CLOUD_ZONE.",
                "manual_input_allowed": True,
                "options": [],
                "validation": {"value_type": "scalar_token"},
            },
            "last_user_input": "Leave zone for later. Use quick, but I want to set max QPS myself.",
        })
        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [
                {
                    "type": "set_qps_mode",
                    "qps_mode": "quick",
                    "mutation_explicit": True,
                    "source_evidence": "Use quick",
                    "confidence": "high",
                },
                {
                    "type": "request_qps_customization",
                    "qps_fields": ["MAX_QPS"],
                    "source_evidence": "I want to set max QPS myself",
                    "confidence": "high",
                },
            ]},
        ):
            result = process_turn(state)

        self.assertEqual((result.get("qps_profile") or {}).get("mode"), "quick")
        self.assertEqual((result.get("qps_profile") or {}).get("adjust_field"), "MAX_QPS")
        self.assertEqual((result.get("pending_question") or {}).get("id"), "qps_adjust_value")
        self.assertTrue(any(
            str(frame.get("question_id") or "") == "CLOUD_ZONE"
            for frame in result.get("interruption_stack") or []
        ))
        self.assertNotIn("qps_overrides is required", "\n".join(result.get("visible_response") or []))

    def test_sync_fixed_duration_option_remains_in_progress_until_duration_is_entered(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.sync_observe import question_for_sync_observe
        from agent.harness.state import new_state

        state = new_state("sync-duration-contract", language="en")
        state.update({
            "target_mode": "sync-observe",
            "workflow_mode": "sync_observe",
            "chain_identity": {"canonical": "bsc", "status": "confirmed", "adapter_family": "jsonrpc"},
            "sync_observe": {"source": "endpoint_only"},
            "endpoint_evidence": {"sync_rpc_url_ready": True},
            "confirmed_config": {
                "SYNC_OBSERVE_RPC_URL": "http://node:8545",
                "MAINNET_RPC_URL_REVIEWED": True,
            },
            "active_group": "sync_observe",
            "last_user_input": "2",
        })
        state["pending_question"] = question_for_sync_observe(state) or {}

        result = process_turn(state)

        self.assertEqual((result.get("sync_observe") or {}).get("stop_condition"), "duration")
        self.assertEqual((result.get("pending_question") or {}).get("id"), "sync_observe_duration_seconds")
        self.assertNotEqual(result.get("active_group"), "failure_recovery")

    def test_detected_disk_size_contract_accepts_direct_custom_value(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.environment import question_for_environment
        from agent.harness.state import new_state

        state = new_state("disk-size-direct-value", language="en")
        state.update({
            "active_group": "ledger_disk",
            "confirmed_config": {"LEDGER_DEVICE": "vda", "DATA_VOL_TYPE": "hyperdisk-balanced"},
            "discovery": {"disks": {"candidates": [{"name": "vda", "size": "926.3G", "type": "disk"}]}},
            "last_user_input": "1024",
        })
        state["pending_question"] = question_for_environment(state, "ledger_disk") or {}

        result = process_turn(state)

        self.assertEqual((result.get("confirmed_config") or {}).get("DATA_VOL_SIZE"), "1024")
        self.assertEqual((result.get("pending_question") or {}).get("id"), "DATA_VOL_MAX_IOPS")

    def test_observability_alias_is_normalized_at_product_boundary(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("observability-alias", language="en")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["last_user_input"] = "use exporter-only"
        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{
                "type": "set_observability",
                "observability_mode": "exporter-only",
                "mutation_explicit": True,
                "source_evidence": "exporter-only",
                "confidence": "high",
            }]},
        ):
            result = process_turn(state)

        self.assertEqual((result.get("observability") or {}).get("mode"), "exporter")
        self.assertNotIn("observability_mode must be", "\n".join(result.get("visible_response") or []))

    def test_registry_declares_consultations_turn_local_and_commands_durable(self) -> None:
        from agent.harness.action_registry import ACTION_BY_TYPE

        for action_type in (
            "greeting",
            "ask_capabilities",
            "answer_opening_question",
            "analyze_report",
        ):
            self.assertEqual(ACTION_BY_TYPE[action_type].lifetime, "turn_local")
        for action_type in ("choose_target_mode", "set_qps_mode", "propose_config_values", "inspect_failure"):
            self.assertEqual(ACTION_BY_TYPE[action_type].lifetime, "durable")
        self.assertEqual(ACTION_BY_TYPE["analyze_evidence"].lifetime, "turn_local")
        self.assertTrue(ACTION_BY_TYPE["analyze_evidence"].crosses_pending_barrier)

    def test_inferred_review_consultation_is_same_turn_and_preserves_exact_question(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.environment import config_proposal_review_question
        from agent.harness.state import new_state

        state = new_state("turn-local-inferred-review", language="en")
        proposal = {
            "config_values": {"CLOUD_REGION": "asia-east1"},
            "unmapped_values": {},
            "source_format": "yaml",
        }
        state["inferred_config"] = {"pending_review": proposal}
        state["active_group"] = "provider_deployment"
        state["pending_question"] = config_proposal_review_question(
            "provider_deployment", proposal, language="en",
        )
        state["pending_question"]["test_contract_marker"] = "byte-equivalent"
        original_question = dict(state["pending_question"])
        state["last_user_input"] = "Explain the current configuration before I confirm it."

        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{
                "type": "answer_opening_question",
                "topic": "current_config",
                "confidence": "high",
            }]},
        ):
            result = process_turn(state)

        self.assertEqual(result["pending_question"], original_question)
        self.assertEqual(result.get("action_queue"), [])
        self.assertFalse(any(
            item.get("type") == "answer_opening_question"
            for item in result.get("completed_actions") or []
        ))
        self.assertIn("Current state:", "\n".join(result.get("visible_response") or []))
        prompt = str(original_question.get("prompt") or "")
        self.assertEqual(sum(prompt in item for item in result.get("visible_response") or []), 1)

    def test_turn_local_consultation_preserves_non_review_queue_barrier(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.questions import manual_question
        from agent.harness.state import new_state

        state = new_state("turn-local-manual-barrier", language="en")
        state["active_group"] = "provider_deployment"
        state["pending_question"] = manual_question(
            "provider_deployment",
            "CLOUD_ZONE",
            "Enter CLOUD_ZONE exactly.",
            field="CLOUD_ZONE",
        )
        state["pending_question"]["queue_barrier"] = True
        state["pending_question"]["test_contract_marker"] = "unchanged"
        original_question = dict(state["pending_question"])
        state["last_user_input"] = "Why is this value needed?"

        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{
                "type": "answer_opening_question",
                "topic": "config_explanation",
                "subject": "CLOUD_ZONE",
                "confidence": "high",
            }]},
        ):
            result = process_turn(state)

        self.assertEqual(result["pending_question"], original_question)
        self.assertEqual(result.get("action_queue"), [])
        self.assertIn("Purpose:", "\n".join(result.get("visible_response") or []))

    def test_inferred_review_consultation_never_replays_after_confirmation(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.environment import config_proposal_review_question
        from agent.harness.state import new_state

        state = new_state("stale-consultation-pruned", language="en")
        proposal = {
            "config_values": {"CLOUD_REGION": "asia-east1"},
            "unmapped_values": {},
            "source_format": "yaml",
        }
        state["inferred_config"] = {"pending_review": proposal}
        state["active_group"] = "provider_deployment"
        state["pending_question"] = config_proposal_review_question(
            "provider_deployment", proposal, language="en",
        )
        state["action_queue"] = [{
            "type": "answer_opening_question",
            "topic": "capabilities",
            "confidence": "high",
            "action_id": "legacy-stale-consultation",
        }]
        state["last_user_input"] = "Y"

        result = process_turn(state)

        self.assertEqual((result.get("confirmed_config") or {}).get("CLOUD_REGION"), "asia-east1")
        self.assertEqual(result.get("action_queue"), [])
        self.assertNotEqual((result.get("pending_question") or {}).get("id"), "inferred_config_review")
        self.assertNotIn("Current framework facts", "\n".join(result.get("visible_response") or []))

    def test_inferred_review_evidence_analysis_is_same_turn_and_never_replays(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.environment import config_proposal_review_question
        from agent.harness.state import new_state

        state = new_state("turn-local-evidence-review", language="en")
        proposal = {
            "config_values": {"CLOUD_REGION": "asia-east1"},
            "unmapped_values": {},
            "source_format": "yaml",
        }
        state["inferred_config"] = {"pending_review": proposal}
        state["active_group"] = "provider_deployment"
        state["pending_question"] = config_proposal_review_question(
            "provider_deployment", proposal, language="en",
        )
        state["pending_question"]["test_contract_marker"] = "unchanged"
        original_question = dict(state["pending_question"])
        state["last_user_input"] = "Analyze this now: RuntimeError: endpoint probe failed"

        with (
            patch(
                "agent.harness.coordinator.resolve_action_queue",
                return_value={"actions": [{
                    "type": "analyze_evidence",
                    "evidence": "RuntimeError: endpoint probe failed",
                    "confidence": "high",
                }]},
            ),
            patch(
                "agent.harness.domains.analysis.analyze_evidence_with_model",
                return_value="endpoint analysis completed",
            ),
        ):
            first = process_turn(state)

        self.assertEqual(first["pending_question"], original_question)
        self.assertEqual(first.get("action_queue"), [])
        self.assertIn("endpoint analysis completed", "\n".join(first.get("visible_response") or []))
        self.assertNotIn(
            str(original_question.get("prompt") or ""),
            "\n".join(first.get("visible_response") or []),
        )

        first["last_user_input"] = "Y"
        confirmed = process_turn(first)
        self.assertEqual((confirmed.get("confirmed_config") or {}).get("CLOUD_REGION"), "asia-east1")
        self.assertNotIn("endpoint analysis completed", "\n".join(confirmed.get("visible_response") or []))

    def test_all_read_only_analysis_actions_cross_review_barrier_only_in_current_turn(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.contracts import HandlerResult
        from agent.harness.domains.environment import config_proposal_review_question
        from agent.harness.domains.runtime import DOMAIN_RUNTIME, DomainRuntime
        from agent.harness.state import new_state

        for action_type, owner in (("analyze_report", "analysis"),):
            with self.subTest(action_type=action_type):
                state = new_state(f"turn-local-{action_type}", language="en")
                proposal = {
                    "config_values": {"CLOUD_REGION": "asia-east1"},
                    "unmapped_values": {},
                    "source_format": "yaml",
                }
                state["inferred_config"] = {"pending_review": proposal}
                state["active_group"] = "provider_deployment"
                state["pending_question"] = config_proposal_review_question(
                    "provider_deployment", proposal, language="en",
                )
                state["pending_question"]["test_contract_marker"] = "unchanged"
                original_question = dict(state["pending_question"])
                state["last_user_input"] = f"Run {action_type} now"
                marker = f"{action_type} completed in this turn"

                def apply_read_only(_state, action, *, visible=marker):
                    return HandlerResult(
                        consumed_action_ids=(action.action_id,),
                        visible_result=visible,
                        completion="completed",
                        stop_after_response=True,
                    )

                runtime = DomainRuntime(apply_action=apply_read_only)
                with (
                    patch(
                        "agent.harness.coordinator.resolve_action_queue",
                        return_value={"actions": [{"type": action_type, "confidence": "high"}]},
                    ),
                    patch.dict(DOMAIN_RUNTIME, {owner: runtime}),
                ):
                    first = process_turn(state)

                self.assertEqual(first["pending_question"], original_question)
                self.assertEqual(first.get("action_queue"), [])
                self.assertIn(marker, "\n".join(first.get("visible_response") or []))

                first["last_user_input"] = "Y"
                confirmed = process_turn(first)
                self.assertEqual((confirmed.get("confirmed_config") or {}).get("CLOUD_REGION"), "asia-east1")
                self.assertNotIn(marker, "\n".join(confirmed.get("visible_response") or []))

    def test_failure_inspection_is_turn_local_at_its_domain_barrier(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.recovery import question_for_recovery
        from agent.harness.failures import build_failure_record
        from agent.harness.state import new_state

        state = new_state("turn-local-failure-inspection", language="en")
        state["active_group"] = "failure_recovery"
        state["failure_recovery"] = {
            "status": "pending",
            "record": build_failure_record(
                "WORKLOAD_PROCESS_FAILED",
                source="workload",
                severity="blocking",
                facts=[{"detail": "exit 2"}],
            ),
        }
        state["pending_question"] = question_for_recovery(state, "failure_recovery") or {}
        state["pending_question"]["test_contract_marker"] = "unchanged"
        original_question = dict(state["pending_question"])
        state["last_user_input"] = "2"

        with patch(
            "agent.harness.domains.recovery.analyze_evidence_with_model",
            return_value="failure advisory completed",
        ):
            first = process_turn(state)

        self.assertEqual(first["pending_question"], original_question)
        self.assertEqual(first.get("action_queue"), [])
        self.assertIn("failure advisory completed", "\n".join(first.get("visible_response") or []))

        first["last_user_input"] = "3"
        completed = process_turn(first)
        self.assertNotIn("failure advisory completed", "\n".join(completed.get("visible_response") or []))

    def test_inferred_review_consultation_keeps_only_durable_command_queued(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.environment import config_proposal_review_question
        from agent.harness.state import new_state

        state = new_state("local-and-durable-review", language="en")
        proposal = {
            "config_values": {"CLOUD_REGION": "asia-east1"},
            "unmapped_values": {},
            "source_format": "yaml",
        }
        state.update({
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "active_group": "provider_deployment",
            "inferred_config": {"pending_review": proposal},
            "last_user_input": "Show status, and use quick QPS after I confirm this review.",
        })
        state["pending_question"] = config_proposal_review_question(
            "provider_deployment", proposal, language="en",
        )
        original_question = dict(state["pending_question"])

        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [
                {"type": "answer_opening_question", "topic": "current_config", "confidence": "high"},
                {
                    "type": "set_qps_mode",
                    "qps_mode": "quick",
                    "mutation_explicit": True,
                    "source_evidence": "use quick QPS",
                    "confidence": "high",
                },
            ]},
        ):
            result = process_turn(state)

        self.assertEqual(result["pending_question"], original_question)
        self.assertEqual([item.get("type") for item in result.get("action_queue") or []], ["set_qps_mode"])
        self.assertIn("Current state:", "\n".join(result.get("visible_response") or []))

    def test_qps_customization_request_before_mode_enters_qps_subflow(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.state import new_state

        state = new_state("structured-domain-blocker", language="en")
        state.update({
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "confirmed_config": {"CLOUD_REGION": "asia-east1"},
            "last_user_input": "Adjust the QPS profile before choosing a mode.",
        })
        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{
                "type": "request_qps_customization",
                "source_evidence": "Adjust the QPS profile",
                "confidence": "high",
            }]},
        ):
            result = process_turn(state)

        self.assertFalse(result.get("failure_recovery"))
        self.assertEqual(result.get("confirmed_config"), {"CLOUD_REGION": "asia-east1"})
        self.assertEqual(result.get("action_queue"), [])
        self.assertTrue(any(item.get("type") == "request_qps_customization" for item in result.get("completed_actions") or []))
        self.assertEqual((result.get("pending_question") or {}).get("id"), "benchmark_mode")

        result["last_user_input"] = "1"
        result = process_turn(result)
        self.assertEqual((result.get("qps_profile") or {}).get("mode"), "quick")
        self.assertEqual((result.get("pending_question") or {}).get("id"), "qps_adjust_field")

    def test_incomplete_detour_group_keeps_control_before_resuming_interrupted_question(self) -> None:
        from agent.harness.coordinator import _ask_next_blocking_question
        from agent.harness.questions import choice_question
        from agent.harness.state import new_state

        state = new_state("detour-control", language="en")
        state.update({
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "rpc_mode": "single",
            "chain_identity": {"canonical": "bsc", "status": "supported"},
            "workload": {"confirmed": False},
            "qps_profile": {"mode": "quick", "confirmed": False, "default_decision_made": True},
            "active_group": "qps_profile",
            "interruption_stack": [{
                "group": "workload_rpc",
                "question_id": "workload_confirm",
                "reason": "detour",
            }],
            "pending_question": {},
            "visible_response": [],
        })

        result = _ask_next_blocking_question(state)

        self.assertEqual(result.get("active_group"), "qps_profile")
        self.assertEqual((result.get("pending_question") or {}).get("id"), "qps_adjust_field")
        self.assertEqual(len(result.get("interruption_stack") or []), 1)

    def test_session_reset_preserves_current_turn_audit_context_only(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.orientation import resume_question
        from agent.harness.state import new_state

        state = new_state("reset-turn-audit", language="en")
        state.update({
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "confirmed_config": {"CLOUD_REGION": "asia-east1"},
            "last_user_input": "3",
        })
        state["pending_question"] = resume_question(state)

        result = process_turn(state)

        self.assertEqual(result.get("target_mode"), "")
        self.assertEqual(result.get("confirmed_config"), {})
        self.assertEqual(result.get("last_user_input"), "3")
        context = result.get("turn_context") or {}
        self.assertEqual(context.get("kind"), "pending")
        self.assertEqual(context.get("text"), "3")
        self.assertEqual(
            (context.get("pending_snapshot") or {}).get("id"),
            "resume_harness_session",
        )
        self.assertTrue(any(
            item.get("type") == "answer_pending"
            for item in context.get("admitted_actions") or []
        ))

    def test_resume_modify_installs_registry_owned_group_selector(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.orientation import resume_question
        from agent.harness.state import new_state
        from agent.workflows.group_registry import GROUP_SPEC_BY_NAME, is_user_navigable_group

        state = new_state("resume-modify-selector", language="en")
        state.update({
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "chain_identity": {"canonical": "bsc", "status": "supported"},
            "last_user_input": "2",
        })
        state["pending_question"] = resume_question(state)

        result = process_turn(state)
        pending = result.get("pending_question") or {}
        option_groups = [
            str((item.get("action") or {}).get("group") or "")
            for item in pending.get("options") or []
        ]

        self.assertEqual(pending.get("id"), "resume_modify_group")
        self.assertEqual(pending.get("group"), "opening")
        self.assertTrue(pending.get("queue_barrier"))
        self.assertIn("change_group", pending.get("accepted_action_types") or [])
        self.assertNotIn("opening", option_groups)
        self.assertIn("target_mode", option_groups)
        self.assertIn("accounts_disk", option_groups)
        self.assertIn("observability", option_groups)
        self.assertIn("qps_profile", option_groups)
        self.assertNotIn("endpoint_process", option_groups)
        self.assertNotIn("chain_auxiliary_endpoints", option_groups)
        self.assertNotIn("target_samples_fixtures", option_groups)
        self.assertNotIn("preflight_smoke_execution", option_groups)
        self.assertNotIn("sync_observe", option_groups)
        self.assertTrue(all(is_user_navigable_group(group) for group in option_groups))
        self.assertTrue(all(
            not GROUP_SPEC_BY_NAME[group].workflow_modes
            or "rpc_benchmark" in GROUP_SPEC_BY_NAME[group].workflow_modes
            for group in option_groups
        ))

    def test_resume_modify_exact_group_selection_executes_declared_navigation(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.orientation import resume_question
        from agent.harness.state import new_state

        state = new_state("resume-modify-observability", language="en")
        state.update({
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "chain_identity": {"canonical": "bsc", "status": "supported"},
            "last_user_input": "2",
        })
        state["pending_question"] = resume_question(state)
        result = process_turn(state)
        result["last_user_input"] = "observability"

        result = process_turn(result)

        self.assertEqual(result.get("active_group"), "observability")
        self.assertEqual((result.get("pending_question") or {}).get("id"), "observability_mode")
        self.assertTrue(any(
            item.get("type") == "change_group"
            and item.get("group") == "observability"
            for item in (result.get("turn_context") or {}).get("admitted_actions") or []
        ))

    def test_resume_modify_reopens_completed_observability_contract(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.orientation import resume_question
        from agent.harness.state import new_state

        state = new_state("resume-reconfigure-observability", language="en")
        state.update({
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "chain_identity": {"canonical": "bsc", "status": "supported"},
            "observability": {"mode": "disabled"},
            "last_user_input": "2",
        })
        state["pending_question"] = resume_question(state)
        result = process_turn(state)
        result["last_user_input"] = "observability"

        result = process_turn(result)

        self.assertEqual(result.get("observability"), {"mode": "disabled"})
        self.assertEqual(result.get("active_group"), "observability")
        self.assertEqual((result.get("pending_question") or {}).get("id"), "observability_mode")

    def test_resume_modify_reopens_completed_provider_without_mutating_confirmed_values(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.orientation import resume_question
        from agent.harness.state import new_state

        confirmed = {
            "CLOUD_PROVIDER": "gcp",
            "CLOUD_REGION": "asia-east1",
            "CLOUD_ZONE": "asia-east1-c",
            "MACHINE_TYPE": "n2-standard-16",
        }
        state = new_state("resume-reconfigure-provider", language="en")
        state.update({
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "chain_identity": {"canonical": "bsc", "status": "supported"},
            "confirmed_config": dict(confirmed),
            "last_user_input": "2",
        })
        state["pending_question"] = resume_question(state)
        result = process_turn(state)
        result["last_user_input"] = "provider_deployment"

        result = process_turn(result)

        self.assertEqual(result.get("confirmed_config"), confirmed)
        self.assertEqual(result.get("active_group"), "provider_deployment")
        self.assertEqual((result.get("pending_question") or {}).get("id"), "CLOUD_REGION")

    def test_resume_modify_reopens_confirmed_chain_at_typed_change_contract(self) -> None:
        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.orientation import resume_question
        from agent.harness.state import new_state

        state = new_state("resume-reconfigure-chain", language="en")
        state.update({
            "target_mode": "fake-node",
            "workflow_mode": "rpc_benchmark",
            "chain_identity": {"canonical": "bsc", "status": "supported"},
            "confirmed_config": {"BLOCKCHAIN_NODE": "bsc"},
            "last_user_input": "2",
        })
        state["pending_question"] = resume_question(state)
        result = process_turn(state)
        result["last_user_input"] = "chain_identity"

        result = process_turn(result)

        self.assertEqual((result.get("chain_identity") or {}).get("canonical"), "bsc")
        self.assertEqual(result.get("active_group"), "chain_identity")
        self.assertEqual((result.get("pending_question") or {}).get("id"), "chain_change_input")

    def test_semantic_answer_uses_declared_option_label_for_typed_value(self) -> None:
        from unittest.mock import patch

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from agent.harness.domains.performance import question_for_performance
        from agent.harness.state import new_state

        state = new_state("semantic-typed-option", language="en")
        state.update({
            "target_mode": "real-node",
            "workflow_mode": "rpc_benchmark",
            "active_group": "qps_profile",
            "qps_profile": {
                "mode": "quick",
                "confirmed": False,
                "default_decision_made": False,
            },
            "last_user_input": "Keep those values as shown.",
        })
        state["pending_question"] = question_for_performance(state, "qps_profile") or {}

        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            side_effect=_admitted_mock_resolver({"actions": [{
                "type": "answer_pending",
                "answer": True,
                "selected_value": True,
                "source_evidence": "Keep those values as shown.",
                "pending_option_semantic_verified": True,
                "semantic_purpose_verified": True,
                "confidence": "medium",
            }]}),
        ):
            result = process_turn(state)

        self.assertTrue((result.get("qps_profile") or {}).get("confirmed"))
        self.assertTrue(any(
            item.get("type") == "answer_pending"
            for item in (result.get("turn_context") or {}).get("admitted_actions") or []
        ))

    def test_semantic_false_option_is_not_rejected_as_an_empty_pending_value(self) -> None:
        from copy import deepcopy
        from unittest.mock import patch

        from tests.agent_live.graph_turn import invoke_product_graph_turn as process_turn
        from tests.agent_live.harness_contract_scenarios import question_scenarios

        scenario = next(
            item
            for item in question_scenarios("en")
            if item.scenario_id == "mainnet_review"
        )
        state = deepcopy(scenario.seed_state)
        state["pending_question"] = deepcopy(scenario.question)
        state["last_user_input"] = (
            "No, do not use the template comparison endpoint; I will provide my own later."
        )
        with patch(
            "agent.harness.coordinator.resolve_action_queue",
            return_value={"actions": [{
                "type": "answer_pending",
                "answer": False,
                "selected_value": False,
                "source_evidence": state["last_user_input"],
                "pending_option_semantic_verified": True,
            }]},
        ):
            result = process_turn(state)

        self.assertIs(result["confirmed_config"]["MAINNET_RPC_URL_REVIEWED"], True)
        self.assertNotEqual(
            (result.get("pending_question") or {}).get("id"),
            "MAINNET_RPC_URL_REVIEWED",
        )


if __name__ == "__main__":
    unittest.main()

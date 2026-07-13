"""Tests for the unexposed LangGraph Harness skeleton."""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


LANGGRAPH_AVAILABLE = importlib.util.find_spec("langgraph") is not None


@unittest.skipUnless(LANGGRAPH_AVAILABLE, "langgraph is not installed in this Python environment")
class LangGraphHarnessSkeletonTest(unittest.TestCase):
    def test_single_free_text_resolver_is_the_action_queue(self) -> None:
        """Architecture audit: the older single-action resolver generation

        (`resolve_intent_action`/`_system_prompt`/`_payload`, backed by
        `_route_single_action`) was a strict subset of the action-queue path
        and was deleted so free-text turns route through exactly one resolver.
        Guard against reintroducing a second resolver that could drift again.
        """

        from agent.harness import groups, intent

        self.assertTrue(hasattr(intent, "resolve_action_queue"))
        self.assertFalse(hasattr(intent, "resolve_intent_action"))
        self.assertFalse(hasattr(intent, "_system_prompt"))
        self.assertFalse(hasattr(intent, "_payload"))
        self.assertFalse(hasattr(groups, "_route_single_action"))

    def test_opening_turn_returns_typed_pending_question(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoints.sqlite"
            runtime = AnyChainGraphRuntime(thread_id="unit-thread", checkpoint_path=checkpoint_path)
            with patch("agent.harness.groups.resolve_action_queue") as resolver:
                resolver.return_value = {"actions": [{"type": "greeting", "confidence": "high"}]}
                state = runtime.invoke("Hi", language="en")

        self.assertEqual(state["active_group"], "opening")
        self.assertEqual(state["pending_question"]["id"], "opening_next_action")
        self.assertEqual(state["pending_question"]["kind"], "numbered_choice")
        self.assertFalse(state["pending_question"]["manual_input_allowed"])
        self.assertIn("What would you like", state["visible_response"][0])

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

    def test_graph_runtime_reset_preserves_startup_discovery(self) -> None:
        """AnyChainGraphRuntime.reset() must agree with groups._reset_workflow_state.

        The terminal's resume-session "clear" option calls this method
        directly. If it wipes `discovery`, the Agent cannot answer "is
        startup inference still there" after the user clears configuration.
        """

        from agent.harness.graph import AnyChainGraphRuntime

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AnyChainGraphRuntime(thread_id="reset-thread", checkpoint_path=Path(tmpdir) / "checkpoints.sqlite")
            runtime.update(
                {
                    "discovery": {"cloud": {"provider": "gcp"}},
                    "framework_summary": {"chain_count": 36},
                    "confirmed_config": {"BLOCKCHAIN_NODE": "bsc", "CLOUD_REGION": "us-1"},
                    "target_mode": "fake-node",
                    "chain_identity": {"raw": "bsc", "canonical": "bsc", "status": "confirmed"},
                }
            )
            state = runtime.reset(language="en")

        self.assertEqual(state.get("discovery", {}).get("cloud", {}).get("provider"), "gcp")
        self.assertEqual(state.get("framework_summary", {}).get("chain_count"), 36)
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
        `groups.process_turn` was called directly (as every other test in
        this file does) but silently reset to empty on every real turn
        through `AnyChainGraphRuntime`, because `advanced_tuning` was never
        added to the `AgentGraphState` class body. No amount of testing
        against `process_turn` directly can catch this class of bug — only
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
        """Real `AnyChainGraphRuntime` round-trip, not a direct `process_turn`

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
        self.assertEqual(status, 200)
        self.assertIn("ok", sample)

    def test_target_mode_choice_asks_chain_before_provider_values(self) -> None:
        from agent.harness.graph import AnyChainGraphRuntime

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoints.sqlite"
            runtime = AnyChainGraphRuntime(thread_id="unit-thread", checkpoint_path=checkpoint_path)
            with patch("agent.harness.groups.resolve_action_queue") as resolver:
                resolver.return_value = {"actions": [{"type": "greeting", "confidence": "high"}]}
                runtime.invoke("Hi", language="en")
            state = runtime.invoke("1", language="en")

        self.assertEqual(state["target_mode"], "fake-node")
        self.assertEqual(state["active_group"], "chain_identity")
        self.assertEqual(state["pending_question"]["id"], "chain")

    def test_chain_turn_can_set_target_mode_in_same_intent(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["active_group"] = "opening"
        state["last_user_input"] = "我想观察 BNB 节点同步，不想打 RPC 压测流量"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "choose_target_mode", "target_mode": "sync-observe", "target_mode_explicit": True, "confidence": "high"},
                    {"type": "choose_chain", "chain_text": "BNB", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["target_mode"], "sync-observe")
        self.assertEqual(result["workflow_mode"], "sync_observe")
        self.assertEqual(result["chain_identity"]["canonical"], "bsc")
        self.assertEqual(result["confirmed_config"]["BLOCKCHAIN_NODE"], "bsc")
        self.assertEqual(result["active_group"], "provider_deployment")

    def test_target_mode_turn_uses_chain_mention_extractor_when_primary_intent_drops_chain(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["active_group"] = "opening"
        state["last_user_input"] = "我想观察 BNB 节点同步，不想打 RPC 压测流量"

        with (
            patch(
                "agent.harness.groups.resolve_action_queue",
                return_value={
                    "actions": [
                        {"type": "choose_chain", "target_mode": "sync-observe", "target_mode_explicit": True, "confidence": "high"}
                    ]
                },
            ),
            patch(
                "agent.harness.groups.extract_chain_mention",
                return_value={"found": True, "chain_text": "BNB", "confidence": "high"},
            ),
        ):
            result = process_turn(state)

        self.assertEqual(result["target_mode"], "sync-observe")
        self.assertEqual(result["workflow_mode"], "sync_observe")
        self.assertEqual(result["chain_identity"]["canonical"], "bsc")
        self.assertEqual(result["pending_question"]["id"], "CLOUD_REGION")

    def test_multi_action_turn_sets_nonblocking_groups_then_falls_back(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["active_group"] = "opening"
        state["last_user_input"] = "我要用 fake-node 测试 BNB，用 mixed，quick QPS，并开启本地 Grafana"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "choose_target_mode", "target_mode": "fake-node", "target_mode_explicit": True, "confidence": "high"},
                    {"type": "choose_chain", "chain_text": "BNB", "confidence": "high"},
                    {"type": "set_rpc_mode", "rpc_mode": "mixed", "confidence": "high"},
                    {"type": "set_qps_mode", "qps_mode": "quick", "confidence": "high"},
                    {"type": "set_observability", "observability_mode": "local", "confidence": "high"},
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
        self.assertEqual(resumed["observability"]["mode"], "local")
        self.assertEqual(resumed["pending_question"]["id"], "qps_profile_confirm")

    def test_capability_action_does_not_swallow_benchmark_goal_in_queue(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["active_group"] = "opening"
        state["last_user_input"] = "我想先随便跑通一下框架，但不确定 fake-node 还是 real-node；可能测 BNB，也想看看支持哪些链"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "ask_capabilities", "confidence": "high"},
                    {"type": "choose_chain", "chain_text": "BNB", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        visible = "\n".join(result.get("visible_response") or [])
        self.assertIn("当前框架", visible)
        self.assertNotIn("{'chain':", visible)
        self.assertIn("已确认链为 `bsc`", visible)
        self.assertEqual(result["chain_identity"]["canonical"], "bsc")
        self.assertEqual(result.get("target_mode"), "")
        self.assertEqual(result["pending_question"]["id"], "opening_next_action")
        self.assertEqual(len(result["completed_actions"]), 2)

    def test_analyze_report_action_returns_visible_job_entrypoint(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["active_group"] = "opening"
        state["latest_job_id"] = "job_demo"
        state["last_user_input"] = "看最近任务报告"

        with (
            patch("agent.harness.groups.resolve_action_queue") as resolver,
            patch("agent.harness.groups.resume_job") as resume,
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

        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["active_group"] = "opening"
        state["evidence_buffer"] = [{"text": "Traceback (most recent call last):\nRuntimeError('endpoint probe failed: connection refused')"}]

        # Still routes to stale-evidence analysis when there's no job reference —
        # existing behavior for a genuine follow-up about the pasted evidence.
        state["last_user_input"] = "分析一下这个原因"
        no_job_result = process_turn(state)
        self.assertIn("connection refused", "\n".join(no_job_result.get("visible_response") or []))

        # A turn naming "the latest job" must fall through to normal routing
        # (the resolver's analyze_report action), not the stale evidence.
        job_state = new_state("unit-thread-2", language="zh")
        job_state["active_group"] = "opening"
        job_state["evidence_buffer"] = [{"text": "Traceback (most recent call last):\nRuntimeError('endpoint probe failed: connection refused')"}]
        job_state["latest_job_id"] = "job_20260712163828_76ac0a38"
        job_state["last_user_input"] = "分析最新的 job"
        with (
            patch("agent.harness.groups.resolve_action_queue") as resolver,
            patch("agent.harness.groups.resume_job") as resume,
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
            patch("agent.harness.groups.resolve_action_queue") as resolver,
            patch("agent.harness.groups.resume_job") as resume,
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
        from agent.harness.groups import process_turn
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

        with patch("agent.harness.groups.validate_rpc_endpoint") as probe:
            result = process_turn(state)

        probe.assert_not_called()
        self.assertIn("evidence_buffer", result)
        self.assertEqual(result["pending_question"], {})
        self.assertIn("证据", "\n".join(result.get("visible_response") or []))

    def test_final_endpoint_pending_requires_bare_endpoint_not_complex_intent(self) -> None:
        from agent.harness.groups import _answer_fits_pending

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
        from agent.harness.groups import process_turn
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

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
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
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "inferred_config_review")
        self.assertNotIn("CLOUD_REGION", result["confirmed_config"])
        self.assertNotIn("LEDGER_DEVICE", result["confirmed_config"])
        self.assertIn("CLOUD_REGION", result["visible_response"][0])
        self.assertIn("disk.ledger", result["visible_response"][0])

        result["last_user_input"] = "Y"
        result = process_turn(result)

        self.assertEqual(result["confirmed_config"]["CLOUD_REGION"], "asia-east1")
        self.assertEqual(result["confirmed_config"]["CLOUD_ZONE"], "asia-east1-c")
        self.assertEqual(result["confirmed_config"]["MACHINE_TYPE"], "n2-standard-16")
        self.assertEqual(result["confirmed_config"]["LEDGER_DEVICE"], "vda")
        self.assertEqual(result["confirmed_config"]["DATA_VOL_SIZE"], "926")
        self.assertEqual(result["pending_question"]["id"], "DATA_VOL_TYPE")
        self.assertIn("accepted_reviews", result["inferred_config"])

    def test_multiline_config_paste_is_preserved_when_action_queue_only_extracts_chain(self) -> None:
        from agent.harness.groups import process_turn
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

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "choose_target_mode", "target_mode": "fake-node", "target_mode_explicit": True, "confidence": "high"},
                    {"type": "choose_chain", "chain_text": "BNB", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["target_mode"], "fake-node")
        self.assertEqual(result["chain_identity"]["canonical"], "bsc")
        self.assertEqual(result["pending_question"]["id"], "inferred_config_review")
        self.assertNotIn("CLOUD_REGION", result["confirmed_config"])
        self.assertIn("CLOUD_REGION", result["visible_response"][0])
        self.assertIn("DATA_VOL_MAX_IOPS", result["visible_response"][0])

    def test_qps_override_and_accounts_presence_can_be_set_from_one_freeform_turn(self) -> None:
        from agent.harness.groups import process_turn
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

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "set_qps_override", "qps_overrides": {"INITIAL_QPS": 5}, "confidence": "high"},
                    {"type": "set_accounts_presence", "has_accounts_device": False, "confidence": "high"},
                    {"type": "change_group", "group": "ledger_disk", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["qps_profile"]["overrides"]["INITIAL_QPS"], "5")
        self.assertTrue(result["qps_profile"]["confirmed"])
        self.assertFalse(result["confirmed_config"]["has_accounts_device"])
        self.assertEqual(result["pending_question"]["id"], "DATA_VOL_SIZE")

    def test_jump_to_completed_group_continues_to_next_fallback_question(self) -> None:
        from agent.harness.groups import process_turn
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

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "change_group", "group": "accounts_disk", "confidence": "high"}]}
            result = process_turn(state)

        self.assertIn("没有阻塞项", "\n".join(result.get("visible_response") or []))
        self.assertEqual(result["pending_question"]["id"], "DATA_VOL_SIZE")

    def test_config_proposal_rejection_does_not_mutate_confirmed_config(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["active_group"] = "provider_deployment"
        state["last_user_input"] = "CLOUD_REGION=us-1\nCLOUD_ZONE=us-1-z\nunknown_flag=true"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
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
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "inferred_config_review")
        result["last_user_input"] = "N"
        result = process_turn(result)

        self.assertEqual(result["confirmed_config"], {})
        self.assertNotIn("pending_review", result.get("inferred_config", {}))
        self.assertIn("Discarded", result["visible_response"][0])

    def test_inferred_config_review_is_blocking_until_user_confirms(self) -> None:
        from agent.harness.groups import process_turn
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

        result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "inferred_config_review")
        self.assertNotIn("CLOUD_REGION", result["confirmed_config"])
        self.assertIn("请先确认", "\n".join(result["visible_response"]))

    def test_config_proposal_saves_endpoint_as_candidate_not_validated_config(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["target_mode"] = "real-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "ethereum", "canonical": "ethereum", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "ethereum"}
        state["active_group"] = "endpoint_process"
        state["last_user_input"] = '{"LOCAL_RPC_URL":"https://node.example","RPC_MODE":"mixed"}'

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {
                        "type": "propose_config_values",
                        "source_format": "json",
                        "config_values": {"LOCAL_RPC_URL": "https://node.example", "RPC_MODE": "mixed"},
                        "confidence": "high",
                    }
                ]
            }
            result = process_turn(state)

        result["last_user_input"] = "Y"
        result = process_turn(result)

        self.assertEqual(result["rpc_mode"], "mixed")
        self.assertEqual(result["endpoint_evidence"]["proposed_values"]["LOCAL_RPC_URL"], "https://node.example")
        self.assertNotIn("LOCAL_RPC_URL", result["confirmed_config"])
        self.assertNotEqual(result.get("endpoint_evidence", {}).get("local_rpc_url_ready"), True)

    def test_config_proposal_normalizes_units_and_accounts_absence(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["active_group"] = "provider_deployment"
        state["last_user_input"] = "pasted environment facts"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
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
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["active_group"] = "provider_deployment"
        state["last_user_input"] = "环境信息：没有 accounts，ledger=vda"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
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
            result = process_turn(state)

        prompt = "\n".join(result.get("visible_response") or [])
        self.assertIn("HAS_ACCOUNTS_DEVICE: `False`", prompt)
        self.assertNotIn("ACCOUNTS_DEVICE: ``", prompt)

    def test_config_review_then_continues_to_custom_rpc_action(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["active_group"] = "provider_deployment"
        state["last_user_input"] = "我要加自定义 RPC method eth_chainId，endpoint http://fake-node:19000，region asia-east1"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
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
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "inferred_config_review")

        ok_probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}
        with patch("agent.harness.groups.validate_rpc_endpoint", return_value=ok_probe):
            result["last_user_input"] = "Y"
            result = process_turn(result)

        self.assertEqual(result["confirmed_config"]["CLOUD_REGION"], "asia-east1")
        self.assertNotIn("LOCAL_RPC_URL", result["confirmed_config"])
        self.assertEqual(result["custom_rpc"]["method"], "eth_chainId")
        self.assertTrue(result["custom_rpc"]["endpoint_ready"])
        self.assertEqual(result["custom_rpc"]["status"], "needs_schema_evidence")
        self.assertEqual(result["pending_question"]["id"], "custom_rpc_schema_evidence")
        self.assertIn("custom_rpc_endpoint_probe", result["endpoint_evidence"])

    def test_config_review_is_prioritized_before_custom_rpc_even_if_model_orders_it_later(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["active_group"] = "provider_deployment"
        state["last_user_input"] = "我要加自定义 RPC method eth_chainId，endpoint http://fake-node:19000，region asia-east1"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
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
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "inferred_config_review")
        self.assertEqual(result["action_queue"][0]["type"], "start_custom_rpc")
        self.assertNotIn("endpoint_ready", result.get("custom_rpc", {}))

    def test_custom_rpc_schema_answer_clears_stale_queue_control(self) -> None:
        from agent.harness.groups import process_turn
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
        with patch("agent.harness.groups.validate_rpc_endpoint", return_value=ok_probe):
            result = process_turn(state)

        self.assertEqual(result["custom_rpc"]["status"], "method_validated_next")
        self.assertEqual(result["pending_question"]["id"], "custom_rpc_continue")
        self.assertEqual(result.get("action_queue"), [])
        self.assertNotIn("CLOUD_REGION", result["confirmed_config"])

    def test_direct_config_assignment_goes_to_named_field_not_current_pending(self) -> None:
        from agent.harness.groups import process_turn
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

        result = process_turn(state)

        self.assertEqual(result["confirmed_config"].get("MACHINE_TYPE"), "n2-standard-16")
        self.assertNotEqual(result["confirmed_config"].get("CLOUD_ZONE"), "MACHINE_TYPE=n2-standard-16")
        self.assertEqual(result["pending_question"]["id"], "CLOUD_ZONE")

    def test_comma_separated_direct_config_assignments_apply_each_field(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
        state["active_group"] = "provider_deployment"
        state["last_user_input"] = "CLOUD_REGION=us-1, CLOUD_ZONE=us-1-z, MACHINE_TYPE=n2"

        result = process_turn(state)

        self.assertEqual(result["confirmed_config"].get("CLOUD_REGION"), "us-1")
        self.assertEqual(result["confirmed_config"].get("CLOUD_ZONE"), "us-1-z")
        self.assertEqual(result["confirmed_config"].get("MACHINE_TYPE"), "n2")
        self.assertEqual(result["pending_question"]["id"], "LEDGER_DEVICE")

    def test_declining_preflight_smoke_pauses_instead_of_looping(self) -> None:
        from agent.harness.groups import process_turn
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
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["active_group"] = "opening"
        state["last_user_input"] = "我要用 fake-node 测试 BNB，用 mixed，QPS quick，并开启本地 Grafana"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {
                        "type": "choose_target_mode",
                        "target_mode": "real-node",
                        "target_mode_explicit": True,
                        "confidence": "high",
                        "reason": "bad model default",
                    },
                    {"type": "choose_chain", "chain_text": "BNB", "confidence": "high"},
                    {"type": "set_rpc_mode", "rpc_mode": "mixed", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        self.assertNotEqual(result.get("target_mode"), "real-node")
        self.assertEqual(result["chain_identity"]["canonical"], "bsc")
        self.assertEqual(result["rpc_mode"], "")
        self.assertEqual(result["pending_question"]["id"], "opening_next_action")
        self.assertTrue(result.get("action_queue"))

    def test_return_to_origin_group_does_not_preempt_new_blocking_group(self) -> None:
        from agent.harness.groups import process_turn
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

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "change_group", "group": "qps_profile", "confidence": "high"},
                    {"type": "set_qps_mode", "qps_mode": "quick", "confidence": "high"},
                    {"type": "change_group", "group": "ledger_disk", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["qps_profile"]["mode"], "quick")
        self.assertEqual(result["pending_question"]["id"], "qps_profile_confirm")
        self.assertEqual(result["active_group"], "qps_profile")

    def test_pending_question_can_route_to_observability_without_being_swallowed(self) -> None:
        from agent.harness.groups import process_turn
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

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "change_group", "group": "observability", "confidence": "high"},
                    {"type": "set_observability", "observability_mode": "disabled", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["observability"]["mode"], "disabled")
        self.assertEqual(result["pending_question"]["id"], "CLOUD_REGION")
        self.assertIn("disabled", "\n".join(result["visible_response"]))

    def test_completed_observability_group_reports_status_when_revisited(self) -> None:
        from agent.harness.groups import process_turn
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
        state["last_user_input"] = "Grafana 现在是不是不开"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "change_group", "group": "observability", "confidence": "high"}]}
            result = process_turn(state)

        self.assertEqual(result["pending_question"], {})
        self.assertIn("disabled", "\n".join(result["visible_response"]))

    def test_sync_observe_multi_action_records_demo_source(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["active_group"] = "opening"
        state["last_user_input"] = "我想观察 BNB 节点同步，不打 RPC 压测，并只做流程 demo"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "choose_target_mode", "target_mode": "sync-observe", "target_mode_explicit": True, "confidence": "high"},
                    {"type": "choose_chain", "chain_text": "BNB", "confidence": "high"},
                    {"type": "set_sync_observe_source", "sync_observe_source": "demo_only", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["target_mode"], "sync-observe")
        self.assertEqual(result["workflow_mode"], "sync_observe")
        self.assertEqual(result["chain_identity"]["canonical"], "bsc")
        self.assertEqual(result["sync_observe"]["source"], "demo_only")
        self.assertEqual(result["pending_question"]["id"], "CLOUD_REGION")

    def test_paused_action_queue_resumes_after_confirmation(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "provider_deployment"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["last_user_input"] = "换成 eth，不使用 fake-node，QPS quick"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "choose_target_mode", "target_mode": "real-node", "target_mode_explicit": True, "confidence": "high"},
                    {"type": "change_chain", "chain_text": "ethereum", "confidence": "high"},
                    {"type": "set_qps_mode", "qps_mode": "quick", "confidence": "high"},
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

        from agent.harness.groups import process_turn
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
        with patch(
            "agent.harness.groups.resolve_pending_choice",
            side_effect=AssertionError("numbered yes/no answer must not reach the LLM resolver"),
        ), patch("agent.harness.groups.resolve_action_queue", return_value={"actions": []}):
            result = process_turn(state)
        self.assertEqual(result["target_mode"], "fake-node")
        self.assertEqual(result["workflow_mode"], "rpc_benchmark")

        # "2" == second option (N): declines, keeps the original mode.
        state = _confirm_state()
        state["last_user_input"] = "2"
        with patch(
            "agent.harness.groups.resolve_pending_choice",
            side_effect=AssertionError("numbered yes/no answer must not reach the LLM resolver"),
        ), patch("agent.harness.groups.resolve_action_queue", return_value={"actions": []}):
            result = process_turn(state)
        self.assertEqual(result["target_mode"], "sync-observe")

    def test_custom_rpc_weights_reject_unknown_method_but_allow_template_default(self) -> None:
        """Custom-RPC mixed weights must reject a method that is neither a

        validated custom method nor a chain template default (a typo/garbage such
        as "eth_fooBar"), while still allowing a genuine template-default method.
        Regression for a live chaos finding: "eth_fooBar=50" was accepted just
        because the weights summed to 100, creating a non-existent benchmark method.
        """

        from agent.harness.groups import process_turn
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
            }
            state["last_user_input"] = answer
            return state

        # Garbage method -> rejected (kept at needs_weights).
        rejected = process_turn(_weights_state("eth_getBlockByNumber=50,eth_fooBar=50"))
        self.assertEqual(rejected["custom_rpc"]["status"], "needs_weights")
        # A real bsc template default (eth_blockNumber) is allowed alongside a custom method.
        accepted = process_turn(_weights_state("eth_getBlockByNumber=50,eth_blockNumber=50"))
        self.assertEqual(accepted["custom_rpc"]["status"], "validated")

        # Empty allowed-set (no validated custom methods AND no template — e.g. a
        # job-local chain with no template): every named method is unknown and
        # must be rejected, not silently accepted because the weights sum to 100.
        def _empty_allowed_state(answer: str) -> dict:
            state = _weights_state(answer)
            state["chain_identity"] = {"raw": "zzz-unknown", "canonical": "zzz-unknown", "status": "confirmed", "case": "known"}
            state["custom_rpc"] = {"status": "needs_weights", "scope": "mixed_replace", "validated_methods": []}
            return state

        empty = process_turn(_empty_allowed_state("eth_fooBar=100"))
        self.assertEqual(empty["custom_rpc"]["status"], "needs_weights")

    def test_qps_override_rejects_invalid_values(self) -> None:
        """QPS override values must be validated: positive integers, MAX >= INITIAL.

        Regression for a live chaos failure: `set_qps_override` stored negative /
        zero / inverted / non-integer values silently (nothing validated them
        downstream), so an invalid QPS config could reach preflight/execution.
        """

        from agent.harness.groups import _invalid_qps_overrides, process_turn
        from agent.harness.state import new_state

        # Pure-function validation.
        self.assertTrue(_invalid_qps_overrides({"INITIAL_QPS": "-100"}))
        self.assertTrue(_invalid_qps_overrides({"MAX_QPS": "0"}))
        self.assertTrue(_invalid_qps_overrides({"INITIAL_QPS": "5000", "MAX_QPS": "100"}))
        self.assertTrue(_invalid_qps_overrides({"QPS_STEP": "2.5"}))
        self.assertTrue(_invalid_qps_overrides({"DURATION": "abc"}))
        self.assertFalse(_invalid_qps_overrides({"INITIAL_QPS": "1000", "MAX_QPS": "5000", "QPS_STEP": "500"}))
        self.assertFalse(_invalid_qps_overrides({"MAX_QPS": "50000"}))

        # End-to-end: an invalid override must not confirm the QPS profile.
        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "qps_profile"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
        state["qps_profile"] = {"mode": "standard"}
        state["last_user_input"] = "把 INITIAL_QPS 设成 -100"
        with patch("agent.harness.groups.resolve_action_queue") as resolver:
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
        with patch("agent.harness.groups.resolve_action_queue") as resolver:
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

        from agent.harness.groups import _manual_question, process_turn
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
            "agent.harness.groups.resolve_action_queue",
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
        adjust_state["pending_question"] = _manual_question(
            "qps_profile", "qps_adjust_value", "请输入 MAX_QPS 的值。", kind="manual_value"
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

        from agent.harness.groups import _environment_readiness_response, process_turn
        from agent.harness.state import new_state

        # Content builder answers from startup discovery (status + detected specs).
        state = new_state("unit-thread", language="zh")
        state["discovery"] = {
            "dependencies": {"missing_required": [], "missing_optional": ["docker"]},
            "cloud": {"provider": "gcp", "machine_type": "e2-standard-4"},
            "host": {"cpu_count": 4, "memory_gib": 15.62, "os": "linux"},
        }
        ready = _environment_readiness_response(state, "zh")
        self.assertIn("e2-standard-4", ready)
        self.assertIn("就绪", ready)

        state["discovery"]["dependencies"]["missing_required"] = ["vegeta"]
        not_ready = _environment_readiness_response(state, "zh")
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
            "agent.harness.groups.resolve_action_queue",
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

        from agent.harness.groups import _question_for_group, process_turn
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
            with patch("agent.harness.groups.resolve_action_queue", return_value={"actions": []}):
                r = process_turn(s)
            self.assertEqual((r.get("confirmed_config") or {}).get("CLOUD_REGION"), "us-central1")

        # Declining (N) asks for a custom value next turn, which is stored verbatim.
        s = _mk()
        s["pending_question"] = q
        s["last_user_input"] = "n"
        with patch("agent.harness.groups.resolve_action_queue", return_value={"actions": []}):
            r = process_turn(s)
        self.assertTrue((r.get("inferred_config") or {}).get("CLOUD_REGION_manual_required"))
        self.assertIsNone((r.get("confirmed_config") or {}).get("CLOUD_REGION"))
        q2 = _question_for_group(r, "provider_deployment")
        self.assertEqual(q2["kind"], "manual_value")
        r["pending_question"] = q2
        r["last_user_input"] = "asia-east1"
        with patch("agent.harness.groups.resolve_action_queue", return_value={"actions": []}):
            r2 = process_turn(r)
        self.assertEqual((r2.get("confirmed_config") or {}).get("CLOUD_REGION"), "asia-east1")

    def test_config_field_explanation_answers_from_runtime_contract(self) -> None:
        """Asking what a config field means / whether it affects results must be

        answered concretely from the runtime field contract (purpose, inference,
        applies-to), not with a generic "paste the field and I'll explain" reply.
        Regression for a live chaos turn: during the DATA_VOL_TYPE prompt, "这个
        磁盘类型会不会影响压测结果" returned the canned config placeholder.
        """

        from agent.harness.groups import _config_field_explanation, process_turn
        from agent.harness.state import new_state

        # Resolver names the field in subject.
        by_subject = _config_field_explanation(new_state("t", language="zh"), "DATA_VOL_TYPE", "zh")
        self.assertIsNotNone(by_subject)
        self.assertIn("DATA_VOL_TYPE", by_subject)
        self.assertIn("作用", by_subject)  # concrete purpose, not a placeholder

        # No subject -> fall back to the active pending question's field.
        pending_state = new_state("t2", language="zh")
        pending_state["pending_question"] = {"id": "DATA_VOL_TYPE", "group": "ledger_disk", "field": "DATA_VOL_TYPE"}
        by_pending = _config_field_explanation(pending_state, "", "zh")
        self.assertIsNotNone(by_pending)
        self.assertIn("DATA_VOL_TYPE", by_pending)

        # Unknown field -> None (handler keeps its generic fallback).
        self.assertIsNone(_config_field_explanation(new_state("t3", language="zh"), "NOT_A_FIELD", "zh"))

        # End-to-end via the typed config_explanation topic during a pending field.
        state = new_state("t4", language="zh")
        state["pending_question"] = {"id": "DATA_VOL_TYPE", "group": "ledger_disk", "field": "DATA_VOL_TYPE"}
        state["last_user_input"] = "这个磁盘类型会不会影响压测结果"
        with patch(
            "agent.harness.groups.resolve_action_queue",
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

        from agent.harness.groups import _config_field_explanation, process_turn
        from agent.harness.state import new_state

        by_subject = _config_field_explanation(new_state("t", language="zh"), "sync_observe_stop_condition", "zh")
        self.assertIsNotNone(by_subject)
        self.assertIn("sync_observe_stop_condition", by_subject)  # falls back to key, not `` ``
        self.assertIn("作用", by_subject)
        self.assertNotIn("``", by_subject)

        # The resolver's `subject` is free-text guessed from phrasing, not a
        # canonical key lookup, and produced the pluralized
        # "sync_observe_stop_conditions" for "系统里默认支持哪几种停止条件" live —
        # must still resolve to the singular registered field.
        by_pluralized_subject = _config_field_explanation(
            new_state("t1b", language="zh"), "sync_observe_stop_conditions", "zh"
        )
        self.assertIsNotNone(by_pluralized_subject)
        self.assertIn("sync_observe_stop_condition", by_pluralized_subject)

        # Unrelated free text must still miss (normalization isn't so loose it
        # matches anything).
        self.assertIsNone(_config_field_explanation(new_state("t1c", language="zh"), "totally_unrelated_thing", "zh"))

        state = new_state("t2", language="zh")
        state["pending_question"] = {"id": "sync_observe_stop_condition", "group": "sync_observe", "field": "sync_observe_stop_condition"}
        state["last_user_input"] = "系统里默认支持哪几种停止条件"
        with patch(
            "agent.harness.groups.resolve_action_queue",
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

        from agent.harness.groups import _config_field_explanation
        from agent.harness.state import new_state

        for field in ("QPS_STEP", "INITIAL_QPS", "MAX_QPS", "MAX_LATENCY_THRESHOLD", "BOTTLENECK_CPU_THRESHOLD"):
            explanation = _config_field_explanation(new_state(f"t-{field}", language="zh"), field, "zh")
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

        from agent.harness.groups import _pending_context_response
        from agent.harness.state import new_state

        state = new_state("t", language="zh")
        for question_id in ("LOCAL_RPC_URL", "SYNC_OBSERVE_RPC_URL"):
            question = {"id": question_id, "group": "endpoint_process", "field": question_id}
            response = _pending_context_response(state, question, "zh")
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

        from agent.harness.groups import _manual_question, process_turn
        from agent.harness.state import new_state

        def _make_state() -> dict:
            state = new_state("unit-thread-duration", language="zh")
            state["target_mode"] = "sync-observe"
            state["workflow_mode"] = "sync_observe"
            state["active_group"] = "sync_observe"
            state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
            state["sync_observe"] = {"source": "endpoint_only", "stop_condition": "duration"}
            state["pending_question"] = _manual_question(
                "sync_observe", "sync_observe_duration_seconds", "请输入 sync-observe 观察时长，单位秒。", kind="manual_value"
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
        self.assertEqual((good_result.get("sync_observe") or {}).get("duration_seconds"), "600")

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

        from agent.harness.groups import _manual_question, process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread-exporter", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "observability"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
        state["last_user_input"] = "第三个，只要exporter"
        with patch(
            "agent.harness.groups.resolve_action_queue",
            return_value={"actions": [{"type": "set_observability", "observability_mode": "exporter", "confidence": "high"}]},
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
        pending_state["pending_question"] = _manual_question(
            "observability", "observability_mode", "请选择可观测性模式。", kind="numbered_choice"
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

        from agent.harness.groups import process_turn
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
        self.assertEqual((good_result.get("sync_observe") or {}).get("duration_seconds"), "600")
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
        self.assertEqual((aliased_result.get("sync_observe") or {}).get("duration_seconds"), "600")

    def test_declined_yes_no_with_trailing_intent_stays_deterministic(self) -> None:
        """A leading Y/N answer with trailing free text must not reach the LLM resolver.

        Regression for a live chaos dead end: "N，我要调一下" to the
        `qps_profile_confirmed` yes/no confirm did not exact-match {"y","yes",
        "n","no"}, so it fell through to the LLM resolver, which had no
        functional action type for "still answering the pending question" and
        emitted `answer_pending` (permanently rejected in `_apply_queue_action`)
        followed by a `change_group` that no-opped because the target group
        equalled the origin group with a still-blocking question — producing a
        turn with a completely empty `visible_response` (the terminal's
        "ADK 没有返回可显示内容" fallback). The fix keeps a leading y/n token with
        trailing text on the deterministic pending-answer path so the harness's
        own decline handling (which correctly renders the next question) runs
        instead, without ever invoking the resolver.
        """

        from unittest.mock import patch as _patch

        from agent.harness.groups import process_turn
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

        with _patch("agent.harness.groups.resolve_action_queue") as resolver:
            result = process_turn(state)
        resolver.assert_not_called()  # must never reach the LLM resolver

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
        `resolve_pending_choice` to `selected_value=False`, but the deterministic
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

        from agent.harness.groups import _coerce_answer, process_turn
        from agent.harness.state import new_state

        preflight_question = {
            "id": "preflight_smoke_confirm",
            "kind": "yes_no",
            "field": "preflight_smoke_confirmed",
            "options": [{"label": "Y", "value": True}, {"label": "N", "value": False}],
            "manual_input_allowed": False,
        }
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
            patch("agent.harness.groups.resolve_action_queue", return_value={"actions": []}),
            patch("agent.harness.groups.resolve_pending_choice") as choice_resolver,
            patch("agent.harness.groups.run_approved_preflight_and_smoke") as run_preflight,
        ):
            choice_resolver.return_value = {"matched": True, "confidence": "high", "selected_value": False}
            result = process_turn(state)

        run_preflight.assert_not_called()  # must NOT execute on a decline
        self.assertFalse((result.get("preflight") or {}).get("approved"))

    def test_disk_and_network_numeric_fields_reject_negative_values(self) -> None:
        """DATA_VOL_SIZE/MAX_IOPS/MAX_THROUGHPUT, their ACCOUNTS_VOL_* twins, and

        NETWORK_MAX_BANDWIDTH_GBPS must reject negative/zero/non-numeric values
        via the direct manual-question answer path. Regression for a live chaos
        finding: typing "-50" to "Confirm DATA_VOL_SIZE in GiB" (and the same for
        MAX_IOPS, ACCOUNTS_VOL_SIZE, NETWORK_MAX_BANDWIDTH_GBPS) was accepted with
        zero validation and the flow advanced to the next question regardless.
        """

        from agent.harness.groups import _manual_question, process_turn
        from agent.harness.state import new_state

        def _answer(group: str, field: str, answer: str) -> dict:
            state = new_state(f"unit-thread-disk-neg-{field}-{answer}", language="zh")
            state["target_mode"] = "fake-node"
            state["workflow_mode"] = "rpc_benchmark"
            state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
            state["active_group"] = group
            state["pending_question"] = _manual_question(group, field, f"Confirm {field}.", kind="manual_value")
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

        from agent.harness.groups import process_turn
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

        from agent.harness.groups import _invalid_advanced_tuning_value, _manual_question, process_turn
        from agent.harness.state import new_state

        self.assertTrue(_invalid_advanced_tuning_value("MAX_LATENCY_THRESHOLD", "-500"))
        self.assertTrue(_invalid_advanced_tuning_value("BOTTLENECK_CPU_THRESHOLD", "150"))
        self.assertFalse(_invalid_advanced_tuning_value("BOTTLENECK_CPU_THRESHOLD", "85"))
        self.assertFalse(_invalid_advanced_tuning_value("MAX_LATENCY_THRESHOLD", "500"))

        def _adjust(adjust_field: str, answer: str) -> dict:
            state = new_state(f"unit-thread-tuning-{adjust_field}-{answer}", language="zh")
            state["target_mode"] = "fake-node"
            state["workflow_mode"] = "rpc_benchmark"
            state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
            state["active_group"] = "advanced_tuning"
            state["advanced_tuning"] = {"default_decision_made": True, "confirmed": False, "adjust_field": adjust_field}
            state["pending_question"] = _manual_question(
                "advanced_tuning", "advanced_tuning_adjust_value", f"Enter the value for {adjust_field}.", kind="manual_value"
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

        from agent.harness.groups import _manual_question, process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread-local-obs", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
        state["last_user_input"] = "second one, local Prometheus/Grafana"
        with patch(
            "agent.harness.groups.resolve_action_queue",
            return_value={"actions": [{"type": "set_observability", "observability_mode": "local", "confidence": "high"}]},
        ):
            result = process_turn(state)
        self.assertEqual((result.get("observability") or {}).get("mode"), "local")
        response = "\n".join(result.get("visible_response") or [])
        self.assertIn("9091", response)
        self.assertIn("3001", response)

        pending_state = new_state("unit-thread-local-obs-2", language="zh")
        pending_state["target_mode"] = "fake-node"
        pending_state["workflow_mode"] = "rpc_benchmark"
        pending_state["pending_question"] = _manual_question(
            "observability", "observability_mode", "Choose observability mode.", kind="numbered_choice"
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

        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        def _accept_state(answer: str) -> dict:
            state = new_state("unit-thread", language="zh")
            state["active_group"] = "opening"
            state["pending_question"] = {
                "id": "accept_recommendation",
                "group": "opening",
                "kind": "yes_no",
                "field": "accept_recommendation",
                "options": [{"label": "Y", "value": True}, {"label": "N", "value": False}],
                "recommended_setup": {"chain": "solana", "target_mode": "fake-node"},
            }
            state["last_user_input"] = answer
            return state

        accepted = process_turn(_accept_state("y"))
        self.assertEqual(accepted["target_mode"], "fake-node")
        self.assertEqual((accepted.get("chain_identity") or {}).get("canonical"), "solana")

        declined = process_turn(_accept_state("2"))
        self.assertNotEqual(declined.get("target_mode"), "fake-node")

    def test_custom_rpc_probe_endpoint_is_not_the_benchmark_endpoint(self) -> None:
        """A custom-RPC method's probe/pasted endpoint is evidence only. On

        real-node, the final benchmark LOCAL_RPC_URL must still be asked
        separately — no matter how many custom methods were added, and even
        though a candidate probe endpoint exists. (Users often paste method
        content whose URL is not the endpoint they want to test.)
        """

        from agent.harness.groups import _question_for_group
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

    def test_chain_workload_question_answered_not_selected(self) -> None:
        """"bsc 有哪些 rpc workload" is a question about a chain's default methods,

        not a chain selection. It must be answered with that chain's workload and
        must NOT select the chain — even when the resolver misfires to
        choose_chain (which it reliably does live). Regression for a real
        transcript where such questions either selected the chain or dumped the
        generic chain list.
        """

        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        # Conforming contract: the resolver types the question as
        # answer_opening_question topic=supported_chains with the chain in
        # `subject`. The harness must answer it from that chain's template and
        # must NOT select the chain. (Resolver prompt rule lives in intent.py; it
        # is exercised live by dual-AI chaos, not mocked here.)
        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = "bsc 有哪些 rpc workload"
        with patch(
            "agent.harness.groups.resolve_action_queue",
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
            "agent.harness.groups.resolve_action_queue",
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

    def test_multiline_paste_with_analysis_request_captured_as_evidence(self) -> None:
        """A multi-line paste with an explicit analysis request (a benchmark

        result dump, not a Python traceback) must be captured as analyzable
        evidence, not handed to the intent resolver. Regression for a live chaos
        failure: pasting a solana benchmark report with "帮我分析为什么成功率低"
        made the resolver pluck "solana" as a chain selection and discard the
        report + analysis request.
        """

        from agent.harness.groups import _is_multiline_paste, _route_free_text
        from agent.harness.state import new_state

        report = (
            "这是我压测 solana 的结果，帮我分析为什么成功率低：\n"
            "Success ratio: 62.30%\n"
            "37.7% => 429 Too Many Requests\n"
            "CPU: node process avg 780%"
        )
        state = new_state("unit-thread", language="zh")
        with patch(
            "agent.harness.groups.resolve_action_queue",
            side_effect=AssertionError("multiline analysis paste must not reach the chain/intent resolver"),
        ):
            result = _route_free_text(state, report)
        self.assertTrue((result or {}).get("evidence_collection", {}).get("lines"))
        # A one-line answer is not a paste and must not be captured as evidence.
        self.assertFalse(_is_multiline_paste("solana"))

    def test_noop_same_chain_change_preserves_active_question(self) -> None:
        """A no-op "change" to the already-selected chain must not drop an active

        pending question. Regression for a live chaos failure: meaningless input
        ("🚀🚀🚀") that the resolver misread as choose_chain(bsc) wiped the active
        benchmark_mode question and derailed the flow.
        """

        from agent.harness.groups import _request_chain_change_confirmation
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

        result = _request_chain_change_confirmation(state, "bsc", {"chain_text": "bsc"})
        self.assertEqual(result["pending_question"]["id"], "benchmark_mode")

    def test_preflight_group_not_offered_when_config_incomplete(self) -> None:
        """A jump to the preflight/execution group must not offer the run

        confirmation ("配置已收集，是否运行?") when the config is not actually
        ready. Regression for a live chaos failure: "run preflight/smoke now" on a
        real-node with no endpoint/workload/QPS reached preflight_smoke_confirm
        and falsely claimed config was collected.
        """

        from agent.harness.groups import _question_for_group
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

        from agent.harness.groups import _question_for_group
        from agent.harness.state import new_state

        def _sync_state(sync: dict) -> dict:
            state = new_state("unit-thread", language="en")
            state["workflow_mode"] = "sync_observe"
            state["endpoint_evidence"] = {"sync_rpc_url_ready": True}
            state["confirmed_config"] = {"MAINNET_RPC_URL_REVIEWED": True}
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

        from agent.harness.groups import _apply_endpoint_answer
        from agent.harness.state import new_state

        def _probe_ok(*args, **kwargs):
            return {"ready": True, "evidence_file": "x.json", "safe_method": "eth_getBlockByNumber", "checks": []}

        state = new_state("t", language="zh")
        state["chain_identity"] = {"canonical": "bsc", "status": "confirmed", "case": "known"}
        state["custom_rpc"] = {"status": "needs_method", "endpoint": "https://bsc-rpc.publicnode.com", "endpoint_ready": True}
        q = {"id": "custom_rpc_method", "group": "endpoint_process", "kind": "manual_value", "field": "custom_rpc_method", "manual_input_allowed": True}
        blob = 'curl https://some-other-node.example.com -X POST --data \'{"jsonrpc":"2.0","method":"eth_getBlockByNumber","params":["latest",false],"id":1}\''
        with patch("agent.harness.groups.validate_rpc_endpoint", side_effect=_probe_ok):
            result = _apply_endpoint_answer(state, blob, q)
        self.assertEqual((result.get("custom_rpc") or {}).get("method"), "eth_getBlockByNumber")
        response = "\n".join(result.get("visible_response") or [])
        self.assertNotIn("这看起来像 endpoint", response)  # not the rejection message

        # A bare URL with no JSON-RPC body is still rejected as not-a-method.
        state2 = new_state("t2", language="zh")
        state2["chain_identity"] = {"canonical": "bsc", "status": "confirmed", "case": "known"}
        state2["custom_rpc"] = {"status": "needs_method", "endpoint": "https://bsc-rpc.publicnode.com", "endpoint_ready": True}
        rejected = _apply_endpoint_answer(state2, "https://docs.example.com/api/eth_getLogs", q)
        self.assertIn("这看起来像 endpoint", "\n".join(rejected.get("visible_response") or []))

    def test_analyze_latest_job_uses_disk_latest_not_stale_hint(self) -> None:
        """"Analyze the latest job" must analyze the most recent job on disk, not a

        stale `latest_job_id` startup hint. Regression for a live chaos failure:
        after submitting a new smoke job, "分析最近 job" kept analyzing the older
        running job detected at startup (the injected hint was never refreshed).
        """

        from unittest.mock import patch

        from agent.harness.groups import _report_artifact_entry_response
        from agent.harness.state import new_state

        state = new_state("t", language="zh")
        state["latest_job_id"] = "job_STALE_running"  # stale startup hint
        newest = [{"job_id": "job_NEWEST_failed", "status": "failed"}]
        summary = {"status": "failed", "run_dir": "x", "artifact_index": "i.json", "runtime_env_file": "r.env", "next_actions": ["status", "analyze"]}
        with patch("agent.harness.groups.list_jobs", return_value=newest), patch("agent.harness.groups.resume_job", return_value=summary):
            out = _report_artifact_entry_response(state)
        self.assertIn("job_NEWEST_failed", out)
        self.assertNotIn("job_STALE_running", out)

        # Falls back to the hint only when there are no jobs on disk.
        with patch("agent.harness.groups.list_jobs", return_value=[]), patch("agent.harness.groups.resume_job", return_value=summary) as rj:
            _report_artifact_entry_response(state)
            rj.assert_called_with("job_STALE_running")

    def test_prepare_kwargs_are_all_accepted_by_prepare_benchmark_run(self) -> None:
        """Every key `_prepare_kwargs` produces must be a parameter of

        `prepare_benchmark_run`. Regression for a live crash: a new kwarg
        (`sync_observe_local_attribution`) was added to the harness side but not to
        the pipeline signature, so preflight raised TypeError — masked by the
        generic "model call failed" message.
        """

        import inspect

        from agent.harness.nodes.execution import _prepare_kwargs
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
        local = build_configuration_checklist({**base, "sync_observe_local_attribution": True}, plan)
        self.assertIn("node_process_identity", local["missing_blockers"])
        # Endpoint-only (no local process) -> waived.
        endpoint_only = build_configuration_checklist({**base, "sync_observe_local_attribution": False}, plan)
        self.assertNotIn("node_process_identity", endpoint_only["missing_blockers"])

    def test_mixed_default_workload_confirms_weights_for_preflight(self) -> None:
        """A confirmed mixed workload (default template weights or validated custom

        weights) must satisfy `mixed_weights_confirmed`. Regression for a live
        chaos failure: choosing "use defaults" for a mixed workload left the flag
        unset, so preflight was offered ("配置已收集") but then blocked on
        `mixed_weights_confirmed` with no way forward.
        """

        from agent.harness.nodes.execution import _prepare_kwargs
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

    def test_numbered_answer_to_confirm_or_value_question_applies(self) -> None:
        """A numbered answer ("1"/"2") to a confirm_or_value question that renders

        numbered options must be applied directly, not handed to the LLM resolver.
        Regression for a live chaos failure: answering "1" to the MAINNET_RPC_URL
        sync-health confirm ("1. Y  2. N") was misrouted (read as a chain remark)
        and the question was re-asked.
        """

        from agent.harness.groups import _answer_fits_pending, _coerce_answer

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
        # Non-jsonrpc families are left to the adapter health probe.
        self.assertEqual(health_probe_methods("bitcoin", "bitcoin_jsonrpc"), (None, {}))
        # A brand-new jsonrpc chain (no template yet) defaults to the EVM guess.
        self.assertEqual(health_probe_methods("brand-new-l2", "jsonrpc"), (["eth_chainId"], {"eth_chainId": []}))

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

        from agent.harness.groups import process_turn
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
                "agent.harness.groups.resolve_action_queue",
                side_effect=AssertionError("out-of-range digit must not reach the LLM resolver"),
            ), patch(
                "agent.harness.groups.resolve_pending_choice",
                side_effect=AssertionError("out-of-range digit must not reach the choice resolver"),
            ):
                result = process_turn(_menu_state(bad))
            self.assertEqual(result["pending_question"]["id"], "benchmark_mode", f"answer={bad!r}")
            self.assertFalse((result.get("qps_profile") or {}).get("mode"), f"answer={bad!r}")

    def test_target_mode_change_confirmation_names_preserved_chain(self) -> None:
        """A mode switch must not silently reuse a stale chain: the confirmation

        prompt explicitly names the carried-over chain so the user can keep or
        change it (baseline dual-AI chaos gate, step 3).
        """

        from agent.harness.groups import _request_target_mode_change_confirmation
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "sync-observe"
        state["active_group"] = "provider_deployment"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}

        state = _request_target_mode_change_confirmation(state, "fake-node")
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

        from agent.harness.groups import _opening_consultation_response
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        combined = "fake-node、real-node 或 sync-observe 是什么？ preflight/smoke 又是什么？"

        # Whichever of the two topics the resolver picks, both halves are answered.
        for topic in ("mode_comparison", "execution_preflight_smoke"):
            resp = _opening_consultation_response(state, {"topic": topic}, combined)
            self.assertIn("sync-observe", resp)  # modes half
            self.assertRegex(resp, r"(预检|冒烟)")  # actual preflight/smoke definition

        # A standalone preflight/smoke question is defined, not just listed.
        standalone = _opening_consultation_response(
            state, {"topic": "execution_preflight_smoke"}, "preflight 和 smoke 是什么？"
        )
        self.assertRegex(standalone, r"预检")
        self.assertRegex(standalone, r"冒烟")

        # A pure modes question is NOT bloated with preflight/smoke text.
        modes_only = _opening_consultation_response(
            state, {"topic": "mode_comparison"}, "fake-node、real-node、sync-observe 有什么区别？"
        )
        self.assertNotRegex(modes_only, r"(预检|冒烟)")

    def test_adapter_family_hint_ignores_negated_protocol_mentions(self) -> None:
        """A negated protocol mention ("没有 json-rpc", "not json-rpc", "非 REST")

        must not be extracted as a positive adapter-family choice. Regression for
        a live dual-AI chaos failure: confirming an unsupported-protocol chain
        with "...没有 JSON-RPC" was read as adapter_family=jsonrpc, which skipped
        the Case-3 official-docs handoff. Positive mentions still resolve.
        """

        from agent.harness.groups import _adapter_family_hint_from_text

        # Negated mentions -> no family hint.
        self.assertEqual(_adapter_family_hint_from_text("自定义二进制协议，没有 JSON-RPC"), "")
        self.assertEqual(_adapter_family_hint_from_text("not json-rpc, custom binary"), "")
        self.assertEqual(_adapter_family_hint_from_text("非 REST"), "")
        # Negated ENUMERATION: "not A/B/substrate" must negate every family, not
        # just the first (regression: Mina described as GraphQL was forced to
        # substrate and mis-routed to Case 2 instead of the Case-3 handoff).
        self.assertEqual(_adapter_family_hint_from_text("只有 GraphQL，没有 JSON-RPC，也不是 REST/cosmos/substrate/bitcoin"), "")
        self.assertEqual(_adapter_family_hint_from_text("not json-rpc, not rest, not substrate"), "")
        self.assertEqual(_adapter_family_hint_from_text("without evm support"), "")
        # Positive mentions still resolve to the family.
        self.assertEqual(_adapter_family_hint_from_text("it is EVM compatible"), "jsonrpc")
        self.assertEqual(_adapter_family_hint_from_text("this is a substrate chain"), "substrate")
        self.assertEqual(_adapter_family_hint_from_text("uses a REST api"), "rest")

    def test_declined_confirmation_discards_remaining_action_queue(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "provider_deployment"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["last_user_input"] = "use real-node and quick"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "choose_target_mode", "target_mode": "real-node", "target_mode_explicit": True, "confidence": "high"},
                    {"type": "set_qps_mode", "qps_mode": "quick", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "target_mode_change_confirm")
        result["last_user_input"] = "N"
        result = process_turn(result)

        self.assertEqual(result["target_mode"], "fake-node")
        self.assertEqual(result.get("action_queue"), [])
        self.assertFalse(result.get("qps_profile", {}).get("mode"))

    def test_new_chain_endpoint_probe_failure_remains_blocking(self) -> None:
        from agent.harness.groups import process_turn
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
            "agent.harness.groups.validate_rpc_endpoint",
            return_value={"ready": False, "status": "failed", "error": "boom", "evidence_file": "probe.json"},
        ):
            result = process_turn(state)

        self.assertEqual(result["chain_identity"]["status"], "existing_family_needs_endpoint")
        self.assertFalse(result["endpoint_evidence"]["candidate_endpoint_ready"])
        self.assertEqual(result["pending_question"]["id"], "new_chain_endpoint")
        self.assertIn("endpoint validation failed", "\n".join(result.get("visible_response") or []))

    def test_pending_endpoint_question_explains_requirements_instead_of_repeating_prompt(self) -> None:
        from agent.harness.groups import process_turn
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

        result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertEqual(result["pending_question"]["id"], "new_chain_endpoint")
        self.assertIn("可访问", text)
        self.assertIn("method", text)
        self.assertIn("只作为", text)
        self.assertNotIn("这条回复不像当前问题的答案", text)

    def test_change_group_activates_requested_group_instead_of_default_path(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "provider_deployment"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["last_user_input"] = "I want to adjust QPS first"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "change_group", "group": "qps_profile", "confidence": "high"}]}
            result = process_turn(state)

        self.assertEqual(result["active_group"], "qps_profile")
        self.assertEqual(result["pending_question"]["id"], "benchmark_mode")
        self.assertEqual(result["group_history"][-1], "provider_deployment")

    def test_completed_group_jump_resumes_default_missing_group(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "provider_deployment"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["last_user_input"] = "I want to adjust QPS first"

        with patch("agent.harness.groups.resolve_action_queue", return_value={"actions": [{"type": "change_group", "group": "qps_profile", "confidence": "high"}]}):
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

    def test_go_back_uses_group_history_without_phrase_matching(self) -> None:
        from agent.harness.groups import process_turn
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

        with patch("agent.harness.groups.resolve_action_queue", return_value={"actions": [{"type": "go_back", "confidence": "high"}]}):
            result = process_turn(state)

        self.assertEqual(result["active_group"], "workload_rpc")
        self.assertEqual(result["pending_question"]["id"], "workload_confirm")
        self.assertEqual(result["group_history"], ["opening", "provider_deployment"])

    def test_custom_rpc_choice_transfers_control_to_endpoint_group(self) -> None:
        from agent.harness.groups import process_turn
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
        from agent.harness.groups import process_turn
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

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "start_custom_rpc", "rpc_endpoint": "endpoint", "reason": "validate a custom method first", "confidence": "high"},
                ]
            }
            with patch("agent.harness.groups.validate_rpc_endpoint") as probe:
                result = process_turn(state)

        probe.assert_not_called()
        self.assertEqual(result["active_group"], "endpoint_process")
        self.assertEqual(result["custom_rpc"]["status"], "needs_endpoint")
        self.assertEqual(result["pending_question"]["id"], "custom_rpc_endpoint")
        self.assertNotIn("LOCAL_RPC_URL", result.get("confirmed_config", {}))
        self.assertNotEqual(result.get("endpoint_evidence", {}).get("local_rpc_url_ready"), True)

    def test_sync_observe_group_jump_requires_target_mode_confirmation(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "workload_rpc"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["last_user_input"] = "observe node sync instead"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "change_group", "group": "sync_observe", "confidence": "high"}]}
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "target_mode_change_confirm")
        self.assertEqual(result["target_mode_change_candidate"], "sync-observe")
        self.assertEqual(result["target_mode"], "fake-node")

    def test_unknown_chain_uses_identity_gate_not_partial_coercion(self) -> None:
        from agent.harness.groups import process_turn
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

        with patch("agent.harness.groups.resolve_unknown_chain_identity") as resolver:
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

    def test_unknown_chain_candidate_accepts_yes_and_advances(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "sola",
            "canonical": "solana",
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

    def test_unknown_chain_pending_accepts_protocol_hint_as_case2_entry(self) -> None:
        from agent.harness.groups import process_turn
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

        result = process_turn(state)

        self.assertEqual(result["chain_identity"]["adapter_family"], "jsonrpc")
        self.assertEqual(result["chain_identity"]["status"], "existing_family_needs_endpoint")
        self.assertEqual(result["pending_question"]["id"], "new_chain_endpoint")

    def test_unknown_chain_protocol_hint_with_requirements_question_keeps_endpoint_pending(self) -> None:
        from agent.harness.groups import process_turn
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

        result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertEqual(result["chain_identity"]["adapter_family"], "jsonrpc")
        self.assertEqual(result["chain_identity"]["status"], "existing_family_needs_endpoint")
        self.assertEqual(result["pending_question"]["id"], "new_chain_endpoint")
        self.assertIn("可访问", text)
        self.assertIn("method", text)
        self.assertIn("只作为", text)
        self.assertNotIn("请提供可访问的 RPC endpoint", text)

    def test_unknown_chain_candidate_accepts_explicit_candidate_name(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "sola",
            "canonical": "solana",
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

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "choose_chain", "chain_text": "solana", "confidence": "high"}]}
            result = process_turn(state)

        self.assertEqual(result["chain_identity"]["status"], "confirmed")
        self.assertEqual(result["confirmed_config"]["BLOCKCHAIN_NODE"], "solana")
        self.assertNotEqual(result["pending_question"]["id"], "chain")

    def test_unknown_chain_candidate_real_chain_choice_does_not_keep_possible_known_canonical(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {
            "raw": "sola",
            "canonical": "solana",
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
        from agent.harness.groups import process_turn
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

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "change_chain", "chain_text": "ethereum", "confidence": "high"}]}
            result = process_turn(state)

        self.assertNotIn("CLOUD_REGION", result.get("confirmed_config", {}))
        self.assertEqual(result["pending_question"]["id"], "chain_change_confirm")

        result["last_user_input"] = "y"
        result = process_turn(result)
        self.assertNotIn("CLOUD_REGION", result.get("confirmed_config", {}))
        self.assertEqual(result["chain_identity"]["canonical"], "ethereum")
        self.assertEqual(result["confirmed_config"]["BLOCKCHAIN_NODE"], "ethereum")

    def test_target_mode_change_interrupts_manual_value_and_requires_confirmation(self) -> None:
        from agent.harness.groups import process_turn
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

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "choose_target_mode", "target_mode": "real-node", "target_mode_explicit": True, "confidence": "high"}]}
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

            with patch("agent.harness.groups.resolve_action_queue") as resolver:
                resolver.return_value = {"actions": [{"type": "choose_target_mode", "target_mode": "real-node", "target_mode_explicit": True, "confidence": "high"}]}
                state = runtime.invoke("use real-node instead", language="en")

            self.assertEqual(state["pending_question"]["id"], "target_mode_change_confirm")
            self.assertEqual(state["target_mode_change_candidate"], "real-node")

            state = runtime.invoke("Y", language="en")

        self.assertEqual(state["target_mode"], "real-node")
        self.assertNotIn("CLOUD_REGION", state.get("confirmed_config", {}))

    def test_manual_value_rejects_bare_yes_without_default_value(self) -> None:
        from agent.harness.groups import process_turn
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

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "unknown", "confidence": "low"}]}
            result = process_turn(state)

        self.assertNotIn("CLOUD_REGION", result.get("confirmed_config", {}))
        self.assertEqual(result["pending_question"]["id"], "CLOUD_REGION")
        self.assertIn("does not look like", result["visible_response"][0])

    def test_free_form_pending_choice_can_be_resolved_by_llm_mapper(self) -> None:
        from agent.harness.groups import process_turn
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

        with (
            patch("agent.harness.groups.resolve_action_queue", return_value={"actions": [{"type": "unknown", "confidence": "low"}]}),
            patch("agent.harness.groups.resolve_pending_choice") as resolver,
        ):
            resolver.return_value = {
                "matched": True,
                "selected_value": "mixed_replace",
                "selected_label": "Use only my custom methods in mixed",
                "confidence": "high",
            }
            result = process_turn(state)

        self.assertEqual(result["custom_rpc"]["scope"], "mixed_replace")
        self.assertEqual(result["custom_rpc"]["status"], "needs_weights")
        self.assertEqual(result["pending_question"]["id"], "custom_rpc_weights")

    def test_assignment_text_does_not_satisfy_numbered_scope_choice(self) -> None:
        from agent.harness.groups import process_turn
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

        with patch("agent.harness.groups.resolve_pending_choice") as resolver, patch("agent.harness.groups.resolve_action_queue") as queue:
            queue.return_value = {"actions": [{"type": "unknown", "confidence": "low"}]}
            result = process_turn(state)

        resolver.assert_not_called()
        self.assertEqual(result["custom_rpc"]["status"], "needs_scope")
        self.assertEqual(result["pending_question"]["id"], "custom_rpc_scope")

    def test_declined_chain_change_keeps_original_chain_and_region_pending(self) -> None:
        from agent.harness.groups import process_turn
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

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "change_chain", "chain_text": "ethereum", "confidence": "high"}]}
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "chain_change_confirm")
        result["last_user_input"] = "n"
        result = process_turn(result)

        self.assertEqual(result["chain_identity"]["canonical"], "bsc")
        self.assertEqual(result["confirmed_config"]["BLOCKCHAIN_NODE"], "bsc")
        self.assertNotIn("CLOUD_REGION", result.get("confirmed_config", {}))
        self.assertEqual(result["pending_question"]["id"], "CLOUD_REGION")

    def test_choose_chain_action_still_confirms_when_chain_already_exists(self) -> None:
        from agent.harness.groups import process_turn
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

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "choose_chain", "chain_text": "ethereum", "confidence": "high"}]}
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "chain_change_confirm")
        self.assertEqual(result["chain_identity"]["canonical"], "bsc")

    def test_qps_rejecting_defaults_enters_adjustment_subflow(self) -> None:
        from agent.harness.groups import process_turn
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

    def test_preflight_yes_calls_harness_execution_helper(self) -> None:
        from agent.harness.groups import process_turn
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

        with patch("agent.harness.groups.run_approved_preflight_and_smoke") as execute:
            execute.return_value = {**state, "visible_response": ["executed"], "pending_question": {}, "_stop_after_response": True}
            result = process_turn(state)

        execute.assert_called_once()
        self.assertEqual(result["visible_response"], ["executed"])

    def test_existing_chain_custom_rpc_validates_endpoint_method_and_weights(self) -> None:
        from agent.harness.groups import process_turn
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
        with patch("agent.harness.groups.validate_rpc_endpoint", return_value=ok_probe):
            state["last_user_input"] = "https://example.invalid/rpc"
            state = process_turn(state)
            self.assertEqual(state["custom_rpc"]["status"], "needs_method")

            state["last_user_input"] = "eth_blockNumber"
            state = process_turn(state)
            self.assertEqual(state["custom_rpc"]["status"], "needs_schema_evidence")

            state["last_user_input"] = "[]"
            state = process_turn(state)
            self.assertEqual(state["custom_rpc"]["status"], "method_validated_next")
            self.assertEqual(state["pending_question"]["id"], "custom_rpc_continue")

        state["last_user_input"] = "2"
        state = process_turn(state)
        self.assertEqual(state["custom_rpc"]["status"], "needs_scope")

        state["last_user_input"] = "2"
        state = process_turn(state)
        self.assertEqual(state["custom_rpc"]["status"], "needs_weights")

        state["last_user_input"] = "eth_blockNumber=70,eth_getBalance=20"
        state = process_turn(state)
        self.assertEqual(state["custom_rpc"]["status"], "needs_weights")
        self.assertEqual(state["pending_question"]["id"], "custom_rpc_weights")

        state["last_user_input"] = "eth_blockNumber=70,eth_getBalance=30"
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

        from agent.harness.groups import process_turn
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

    def test_inline_single_replace_hint_with_multiple_methods_asks_disambiguation(self) -> None:
        """The inline `_custom_rpc_inline_workload_hint` fast-path must not

        silently pick `methods[0]` when several methods validated. It now
        mirrors the canonical `custom_rpc_scope` handler and routes to the
        `needs_single_method` disambiguation (regression for the deferred
        Phase 4 follow-up: the inline path bypassed that guard).
        """

        from agent.harness.groups import (
            _apply_custom_rpc_inline_workload_hint,
            _question_for_group,
        )
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["active_group"] = "endpoint_process"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed", "case": "known"}
        state["custom_rpc"] = {
            "status": "method_validated_next",
            "validated_methods": [
                {"method": "eth_blockNumber", "params": [], "evidence_file": ".agent/evidence/one.json"},
                {"method": "eth_getBalance", "params": [], "evidence_file": ".agent/evidence/two.json"},
            ],
            "inline_workload_hint": {"scope": "single_replace", "finish": True},
        }

        applied = _apply_custom_rpc_inline_workload_hint(state)
        self.assertTrue(applied)
        self.assertEqual(state["custom_rpc"]["status"], "needs_single_method")
        # The hint must be consumed so it cannot silently re-fire next turn.
        self.assertNotIn("inline_workload_hint", state["custom_rpc"])
        # It must NOT have silently committed a single arbitrary method.
        self.assertNotEqual(state.get("rpc_mode"), "single")
        self.assertFalse((state.get("workload") or {}).get("confirmed"))

        # The needs_single_method status must surface the disambiguation
        # question with both validated methods as options.
        question = _question_for_group(state, "endpoint_process")
        self.assertEqual(question["id"], "custom_rpc_single_method")
        option_values = {option["value"] for option in question["options"]}
        self.assertEqual(option_values, {"eth_blockNumber", "eth_getBalance"})

    def test_inline_single_replace_hint_with_one_method_commits_directly(self) -> None:
        """Negative companion: with exactly one validated method the inline

        single_replace hint still commits without a disambiguation detour.
        """

        from agent.harness.groups import _apply_custom_rpc_inline_workload_hint
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["custom_rpc"] = {
            "status": "method_validated_next",
            "validated_methods": [
                {"method": "eth_blockNumber", "params": [], "evidence_file": ".agent/evidence/one.json"},
            ],
            "inline_workload_hint": {"scope": "single_replace", "finish": True},
        }

        applied = _apply_custom_rpc_inline_workload_hint(state)
        self.assertTrue(applied)
        self.assertEqual(state["custom_rpc"]["status"], "validated")
        self.assertEqual(state["rpc_mode"], "single")
        self.assertEqual(state["workload"]["methods"], ["eth_blockNumber"])

    def test_custom_rpc_scope_single_replace_with_one_method_skips_disambiguation(self) -> None:
        """Negative-case companion: with exactly one validated method, Case 1

        must go straight to `validated` without asking anything, protecting
        against the new branch over-triggering.
        """

        from agent.harness.groups import process_turn
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
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["custom_rpc"] = {
            "status": "needs_schema_evidence",
            "endpoint": "https://example.invalid/rpc",
            "endpoint_ready": True,
            "method": "eth_getBalance",
        }
        state["pending_question"] = {
            "id": "custom_rpc_schema_evidence",
            "group": "endpoint_process",
            "kind": "evidence",
            "field": "custom_rpc_schema_evidence",
            "manual_input_allowed": True,
        }
        draft = {
            "status": "draft",
            "method": "eth_getBalance",
            "params": [
                {"index": 0, "name": "address", "type": "string", "meaning": "account address", "example": "0x0000000000000000000000000000000000000000", "required": True},
                {"index": 1, "name": "block", "type": "string", "meaning": "block tag", "example": "latest", "required": True},
            ],
            "response_summary": "hex balance",
            "confidence": "high",
        }

        with patch("agent.harness.groups.extract_rpc_schema_from_evidence", return_value=draft):
            state["last_user_input"] = "curl --data '{\"method\":\"eth_getBalance\",\"params\":[\"0x0000000000000000000000000000000000000000\",\"latest\"]}'"
            state = process_turn(state)

        self.assertEqual(state["custom_rpc"]["status"], "schema_needs_confirmation")
        self.assertEqual(state["pending_question"]["id"], "custom_rpc_schema_confirm")

        ok_probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}
        with patch("agent.harness.groups.validate_rpc_endpoint", return_value=ok_probe) as probe:
            state["last_user_input"] = "y"
            state = process_turn(state)

        probe.assert_called_once()
        self.assertEqual(state["custom_rpc"]["params"], ["0x0000000000000000000000000000000000000000", "latest"])
        self.assertEqual(state["custom_rpc"]["status"], "method_validated_next")
        self.assertEqual(state["pending_question"]["id"], "custom_rpc_continue")

    def test_existing_chain_custom_rpc_accepts_json_rpc_request_as_direct_schema(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["custom_rpc"] = {
            "status": "needs_schema_evidence",
            "endpoint": "https://example.invalid/rpc",
            "endpoint_ready": True,
            "method": "eth_getBalance",
        }
        state["pending_question"] = {
            "id": "custom_rpc_schema_evidence",
            "group": "endpoint_process",
            "kind": "evidence",
            "field": "custom_rpc_schema_evidence",
            "manual_input_allowed": True,
        }
        ok_probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}

        with patch("agent.harness.groups.extract_rpc_schema_from_evidence") as extractor, patch("agent.harness.groups.validate_rpc_endpoint", return_value=ok_probe) as probe:
            state["last_user_input"] = '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}'
            state = process_turn(state)

        extractor.assert_not_called()
        probe.assert_called_once()
        self.assertEqual(state["custom_rpc"]["method"], "eth_blockNumber")
        self.assertEqual(state["custom_rpc"]["params"], [])
        self.assertEqual(state["custom_rpc"]["status"], "method_validated_next")
        self.assertEqual(state["pending_question"]["id"], "custom_rpc_continue")

    def test_custom_rpc_endpoint_turn_consumes_inline_jsonrpc_request(self) -> None:
        from agent.harness.groups import process_turn
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
        with patch("agent.harness.groups.validate_rpc_endpoint", return_value=ok_probe):
            result = process_turn(state)

        self.assertEqual(result["custom_rpc"]["status"], "method_validated_next")
        self.assertEqual(result["custom_rpc"]["method"], "eth_chainId")
        self.assertEqual(result["pending_question"]["id"], "custom_rpc_continue")

    def test_real_node_requires_local_rpc_probe_before_workload(self) -> None:
        from agent.harness.groups import process_turn
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

        with patch("agent.harness.groups.resolve_action_queue", return_value={"actions": [{"type": "unknown", "confidence": "low"}]}):
            state = process_turn(state)

        self.assertEqual(state["pending_question"]["id"], "LOCAL_RPC_URL")

        ok_probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}
        with patch("agent.harness.groups.validate_rpc_endpoint", return_value=ok_probe):
            state["last_user_input"] = "https://example.invalid/rpc"
            state = process_turn(state)

        self.assertEqual(state["confirmed_config"]["LOCAL_RPC_URL"], "https://example.invalid/rpc")
        self.assertTrue(state["endpoint_evidence"]["local_rpc_url_ready"])
        self.assertEqual(state["pending_question"]["id"], "BLOCKCHAIN_PROCESS_NAMES")

    def test_new_chain_existing_family_validates_endpoint_method_workload_then_runtime_choice(self) -> None:
        from agent.harness.groups import process_turn
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
        with patch("agent.harness.groups.validate_rpc_endpoint", return_value=ok_probe):
            state["last_user_input"] = "https://example.invalid/rpc"
            state = process_turn(state)
            self.assertEqual(state["chain_identity"]["status"], "existing_family_needs_method")
            self.assertEqual(state["pending_question"]["id"], "new_chain_method")

            state["last_user_input"] = "eth_blockNumber"
            state = process_turn(state)
            self.assertEqual(state["chain_identity"]["status"], "existing_family_needs_schema_evidence")
            self.assertEqual(state["pending_question"]["id"], "new_chain_schema_evidence")

            state["last_user_input"] = "[]"
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
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread")
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

    def test_new_chain_workload_scope_single_replace_with_multiple_methods_asks_new_chain_single_method(self) -> None:
        """Phase 4 backfill: `existing_family_needs_single_method` had zero

        prior test coverage. This pins Case 2's own behavior (asking which
        validated method to use as single, when more than one is validated)
        before Case 1 is unified to match it.
        """

        from agent.harness.groups import process_turn
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
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["active_group"] = "opening"
        state["last_user_input"] = "我想测 Flow，它应该是 EVM/jsonrpc"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {
                        "type": "choose_chain",
                        "chain_text": "Flow",
                        "chain_exists": True,
                        "canonical_chain_name": "flow",
                        "adapter_family": "jsonrpc",
                        "confidence": "high",
                    }
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["chain_identity"]["canonical"], "flow")
        self.assertEqual(result["chain_identity"]["adapter_family"], "jsonrpc")
        self.assertEqual(result["pending_question"]["id"], "unknown_chain_identity_confirm")
        self.assertIn("协议族为 `jsonrpc`", result["visible_response"][0])

    def test_device_choice_single_default_accepts_yes(self) -> None:
        from agent.harness.groups import process_turn
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
        from agent.harness.groups import process_turn
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

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            result = process_turn(state)

        resolver.assert_not_called()
        self.assertNotIn("LEDGER_DEVICE", result.get("confirmed_config", {}))
        self.assertEqual(result["pending_question"]["id"], "LEDGER_DEVICE")
        self.assertIn("Y/N", result["visible_response"][0])

    def test_new_chain_existing_family_extracts_schema_evidence_before_runtime_choice(self) -> None:
        from agent.harness.groups import process_turn
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
        state["pending_question"] = {
            "id": "new_chain_schema_evidence",
            "group": "endpoint_process",
            "kind": "evidence",
            "field": "new_chain_schema_evidence",
            "manual_input_allowed": True,
        }
        draft = {
            "status": "draft",
            "method": "eth_blockNumber",
            "params": [],
            "response_summary": "latest block number as hex quantity",
            "confidence": "high",
        }

        with patch("agent.harness.groups.extract_rpc_schema_from_evidence", return_value=draft):
            state["last_user_input"] = "curl --data '{\"method\":\"eth_blockNumber\",\"params\":[]}'"
            state = process_turn(state)

        self.assertEqual(state["chain_identity"]["status"], "existing_family_schema_needs_confirmation")
        self.assertEqual(state["pending_question"]["id"], "new_chain_schema_confirm")

        ok_probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}
        with patch("agent.harness.groups.validate_rpc_endpoint", return_value=ok_probe):
            state["last_user_input"] = "y"
            state = process_turn(state)

        self.assertEqual(state["chain_identity"]["candidate_params"], [])
        self.assertEqual(state["chain_identity"]["status"], "existing_family_method_validated_next")
        self.assertEqual(state["pending_question"]["id"], "new_chain_method_continue")
        self.assertIn("new_chain_method_probe", state["endpoint_evidence"])

    def test_pending_answer_clears_stale_action_queue_before_new_endpoint_question(self) -> None:
        from agent.harness.groups import process_turn
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
        with patch("agent.harness.groups.validate_rpc_endpoint", return_value=ok_probe):
            state["last_user_input"] = "https://example.invalid/rpc"
            state = process_turn(state)

        self.assertEqual(state["chain_identity"]["status"], "existing_family_needs_method")
        self.assertEqual(state["endpoint_evidence"]["candidate_endpoint"], "https://example.invalid/rpc")
        self.assertEqual(state["pending_question"]["id"], "new_chain_method")

    def test_unsupported_family_stops_with_development_handoff_message(self) -> None:
        from agent.harness.groups import process_turn
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
        from agent.harness.groups import process_turn
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

        with patch("agent.harness.groups.resolve_action_queue", return_value={"actions": [{"type": "unknown", "confidence": "low"}]}):
            state = process_turn(state)

        self.assertEqual(state["pending_question"]["id"], "sync_observe_source")
        self.assertIn("fake-node", state["visible_response"][0])

    def test_sync_observe_endpoint_only_probes_real_endpoint_and_skips_process_name(self) -> None:
        from agent.harness.groups import process_turn
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
        with patch("agent.harness.groups.validate_rpc_endpoint", return_value=ok_probe):
            state["last_user_input"] = "https://example.invalid/rpc"
            state = process_turn(state)

        self.assertTrue(state["endpoint_evidence"]["sync_rpc_url_ready"])
        self.assertEqual(state["confirmed_config"]["LOCAL_RPC_URL"], "https://example.invalid/rpc")
        self.assertNotEqual(state.get("pending_question", {}).get("id"), "BLOCKCHAIN_PROCESS_NAMES")
        self.assertEqual(state["pending_question"]["id"], "MAINNET_RPC_URL_REVIEWED")

    def test_sync_observe_client_setup_without_google_search_stops_with_handoff(self) -> None:
        from agent.harness.groups import process_turn
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
        self.assertEqual(state["pending_question"]["id"], "sync_observe_client_setup_ack")

        state["last_user_input"] = "y"
        state = process_turn(state)

        self.assertIn("google_search is unavailable", state["visible_response"][0])
        self.assertEqual(state.get("pending_question"), {})
        self.assertNotEqual(state.get("active_group"), "preflight_smoke_execution")

    def test_retired_adk_runner_bridge_cannot_own_product_workflow(self) -> None:
        from agent.adk_app.runner_bridge import ADKRunnerBridge, run_text_once, runner_bridge_status, sanitize_adk_text

        bridge_file = Path(__file__).resolve().parents[1] / "agent" / "adk_app" / "runner_bridge.py"
        bridge_text = bridge_file.read_text(encoding="utf-8")

        self.assertNotIn("workflows.conversation_state", bridge_text)
        self.assertNotIn("build_terminal_turn_prompt", bridge_text)
        self.assertNotIn("build_root_agent", bridge_text)
        self.assertNotIn("google.genai", bridge_text)
        self.assertIsInstance(runner_bridge_status().as_dict(), dict)
        self.assertEqual(sanitize_adk_text("  ok  "), "ok")
        with self.assertRaisesRegex(RuntimeError, "retired"):
            ADKRunnerBridge()
        with self.assertRaisesRegex(RuntimeError, "retired"):
            run_text_once("Hi")

    def test_adk_root_and_registry_do_not_expose_retired_workflow_runtime(self) -> None:
        import sys

        agent_root = Path(__file__).resolve().parents[1] / "agent"
        if str(agent_root) not in sys.path:
            sys.path.insert(0, str(agent_root))

        from agent.adk_app.root_agent import ADK_MODEL_BRIDGE_INSTRUCTION
        from agent.adk_app.tools.registry import get_adk_tools

        repo = Path(__file__).resolve().parents[1]
        root_text = (repo / "agent" / "adk_app" / "root_agent.py").read_text(encoding="utf-8")
        registry_text = (repo / "agent" / "adk_app" / "tools" / "registry.py").read_text(encoding="utf-8")
        tool_names = {getattr(item, "__name__", str(item)) for item in get_adk_tools()}

        self.assertIn("LangGraph Harness", ADK_MODEL_BRIDGE_INSTRUCTION)
        self.assertNotIn("before_tool_callback", root_text)
        self.assertNotIn("after_model_callback", root_text)
        self.assertNotIn("build_domain_agents", root_text)
        self.assertNotIn("ROOT_INSTRUCTION", root_text)
        self.assertNotIn("workflow_state", registry_text)
        self.assertFalse({"load_workflow_state", "update_workflow_state", "answer_pending_question"} & tool_names)

    def test_custom_rpc_success_asks_continue_instead_of_looping_schema(self) -> None:
        from agent.harness.groups import process_turn
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

        with patch("agent.harness.groups.validate_rpc_endpoint") as probe:
            probe.return_value = {"ready": True, "evidence_file": ".agent/evidence/endpoint-probes/demo.json"}
            result = process_turn(state)

        self.assertEqual(result["custom_rpc"]["status"], "method_validated_next")
        self.assertEqual(result["pending_question"]["id"], "custom_rpc_continue")
        self.assertIn("What should happen next", "\n".join(result["visible_response"]))

    def test_new_chain_rest_evidence_does_not_probe_as_jsonrpc_method(self) -> None:
        from agent.harness.groups import process_turn
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
        }
        state["last_user_input"] = "Get Blocks by ID\npath Parameters\nid required\nquery Parameters"

        with (
            patch("agent.harness.groups.extract_rpc_schema_from_evidence") as extract,
            patch("agent.harness.groups.validate_rpc_endpoint") as probe,
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

        from agent.harness.groups import process_turn
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
            patch("agent.harness.groups.extract_rpc_schema_from_evidence") as extract,
            patch("agent.harness.groups.validate_rpc_endpoint") as probe,
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

        from agent.harness.groups import process_turn
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
            patch("agent.harness.groups.extract_rpc_schema_from_evidence") as extract,
            patch("agent.harness.groups.validate_rpc_endpoint") as probe,
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
        from agent.harness.groups import process_turn
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
        state["last_user_input"] = "https://rest-testnet.onflow.org/v1/blocks"

        result = process_turn(state)

        self.assertEqual(result["chain_identity"]["status"], "existing_family_needs_method")
        self.assertNotEqual(result["chain_identity"].get("candidate_method"), "https://rest-testnet.onflow.org/v1/blocks")
        self.assertIn("不像可直接验证的 RPC method 名称", "\n".join(result["visible_response"]))

    def test_new_chain_method_question_accepts_inline_jsonrpc_request(self) -> None:
        from agent.harness.groups import process_turn
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
        }
        state["last_user_input"] = 'method 是 eth_blockNumber，请求是 {"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}'

        ok_probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}
        with patch("agent.harness.groups.validate_rpc_endpoint", return_value=ok_probe):
            result = process_turn(state)

        self.assertEqual(result["chain_identity"]["status"], "existing_family_method_validated_next")
        self.assertEqual(result["chain_identity"]["candidate_method"], "eth_blockNumber")
        self.assertEqual(result["pending_question"]["id"], "new_chain_method_continue")

    def test_docs_excerpt_at_new_chain_method_question_stays_in_current_group(self) -> None:
        from agent.harness.groups import process_turn
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

        with patch("agent.harness.groups.resolve_action_queue") as router:
            result = process_turn(state)

        router.assert_not_called()
        self.assertEqual(result["chain_identity"]["status"], "existing_family_needs_method")
        self.assertNotIn("candidate_method", result["chain_identity"])
        self.assertIn("不像可直接验证的 RPC method 名称", "\n".join(result["visible_response"]))

    def test_partial_chain_change_requires_identity_gate_not_silent_alias(self) -> None:
        from agent.harness.groups import process_turn
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
            patch("agent.harness.groups.resolve_action_queue") as queue,
            patch("agent.harness.groups.resolve_unknown_chain_identity") as identify,
        ):
            queue.return_value = {"actions": [{"type": "change_chain", "chain_text": "Sola", "confidence": "high"}]}
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

    def test_ambiguous_chain_turn_asks_user_to_choose_candidate(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = "我想用 fake-node 测一下，链名可能是 sola 或 solana，QPS 用 quick"

        with patch("agent.harness.groups.resolve_action_queue") as queue:
            queue.return_value = {
                "actions": [
                    {"type": "choose_target_mode", "target_mode": "fake-node", "target_mode_explicit": True, "confidence": "high"},
                    {"type": "choose_chain", "chain_text": "sola", "confidence": "high"},
                    {"type": "set_qps_mode", "qps_mode": "quick", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "chain_ambiguity_confirm")
        rendered = "\n".join(result["visible_response"])
        self.assertIn("solana", rendered)
        self.assertIn("sola", rendered)

        result["last_user_input"] = "1"
        result = process_turn(result)

        self.assertEqual(result["chain_identity"]["canonical"], "solana")
        self.assertEqual(result["qps_profile"]["mode"], "quick")

    def test_chain_ambiguity_confirmation_drops_resolved_chain_actions_but_keeps_followups(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = "solana 还是 bnb 都行，QPS quick"

        with patch("agent.harness.groups.resolve_action_queue") as queue:
            queue.return_value = {
                "actions": [
                    {"type": "choose_chain", "chain_text": "solana", "confidence": "high"},
                    {"type": "choose_chain", "chain_text": "bsc", "confidence": "high"},
                    {"type": "set_qps_mode", "qps_mode": "quick", "confidence": "high"},
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
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = "我想用 fake-node 测一下，链名可能是 sola 或 solana，QPS 用 quick"

        with patch("agent.harness.groups.resolve_action_queue") as queue:
            queue.return_value = {
                "actions": [
                    {"type": "choose_target_mode", "target_mode": "fake-node", "target_mode_explicit": True, "confidence": "high"},
                    {"type": "choose_chain", "chain_text": "solana", "confidence": "high"},
                    {"type": "set_qps_mode", "qps_mode": "quick", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "chain_ambiguity_confirm")
        rendered = "\n".join(result["visible_response"])
        self.assertIn("solana", rendered)
        self.assertIn("sola", rendered)

    def test_pending_question_chain_detour_uses_llm_chain_mention_gate(self) -> None:
        from agent.harness.groups import process_turn
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
            patch("agent.harness.groups.resolve_action_queue", return_value={"actions": [{"type": "unknown", "confidence": "low"}]}),
            patch("agent.harness.groups.resolve_pending_choice", return_value={"matched": False}),
            patch("agent.harness.groups.extract_chain_mention") as mention,
            patch("agent.harness.groups.resolve_unknown_chain_identity") as identify,
        ):
            mention.return_value = {"found": True, "chain_text": "Sola", "confidence": "high", "reason": "user is correcting chain"}
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
        from agent.harness.groups import process_turn
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

        with patch("agent.harness.groups.resolve_action_queue") as queue:
            queue.return_value = {
                "actions": [
                    {
                        "type": "change_chain",
                        "chain_text": "abcd",
                        "adapter_family": "jsonrpc",
                        "chain_exists": True,
                        "confidence": "high",
                    }
                ]
            }
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "chain_change_confirm")
        self.assertIn("jsonrpc", result["pending_question"]["prompt"])
        self.assertIn("endpoint/RPC", result["pending_question"]["prompt"])

    def test_single_custom_rpc_method_accepts_inline_numeric_weight(self) -> None:
        from agent.harness.groups import process_turn
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

        result = process_turn(state)

        self.assertEqual(result["rpc_mode"], "mixed")
        self.assertEqual(result["workload"]["mixed_weights"], {"eth_chainId": 100})
        self.assertTrue(result["workload"]["replace_defaults"])

    def test_compound_custom_rpc_answer_resumes_remaining_action_queue(self) -> None:
        from agent.harness.groups import process_turn
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
            {"type": "set_qps_mode", "qps_mode": "quick", "confidence": "high", "_origin_text": "quick Grafana 不开"},
            {"type": "set_observability", "observability_mode": "disabled", "confidence": "high", "_origin_text": "quick Grafana 不开"},
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

        result = process_turn(state)

        self.assertEqual(result["workload"]["mixed_weights"], {"eth_chainId": 100})
        self.assertEqual(result["qps_profile"]["mode"], "quick")
        self.assertEqual(result["observability"]["mode"], "disabled")
        self.assertEqual(result["pending_question"]["id"], "qps_profile_confirm")

    def test_generated_schema_question_preserves_remaining_action_queue(self) -> None:
        from agent.harness.groups import process_turn
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

    def test_schema_answer_propagates_resume_to_next_custom_rpc_question(self) -> None:
        from agent.harness.groups import process_turn
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
        state["action_queue"] = [{"type": "set_qps_mode", "qps_mode": "quick", "confidence": "high", "_origin_text": "quick"}]
        state["pending_question"] = {
            "id": "custom_rpc_schema_evidence",
            "group": "endpoint_process",
            "kind": "evidence",
            "field": "custom_rpc_schema_evidence",
            "manual_input_allowed": True,
            "resume_action_queue": True,
        }
        state["last_user_input"] = "[]"
        probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}

        with patch("agent.harness.groups.validate_rpc_endpoint", return_value=probe):
            result = process_turn(state)

        self.assertEqual(result["pending_question"]["id"], "custom_rpc_continue")
        self.assertTrue(result["pending_question"].get("resume_action_queue"))

    def test_custom_rpc_schema_accepts_no_parameters_natural_language(self) -> None:
        from agent.harness.groups import process_turn
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

        with patch("agent.harness.groups.validate_rpc_endpoint", return_value=probe):
            result = process_turn(state)

        self.assertEqual(result["custom_rpc"]["schema_draft"]["params"], [])
        self.assertEqual(result["pending_question"]["id"], "custom_rpc_schema_confirm")

    def test_start_custom_rpc_uses_original_turn_as_schema_evidence(self) -> None:
        from agent.harness.groups import _apply_queue_action
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
            "confidence": "high",
        }
        probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}

        with patch("agent.harness.groups.validate_rpc_endpoint", return_value=probe):
            result = _apply_queue_action(state, action, state["last_user_input"])

        self.assertNotIn("LOCAL_RPC_URL", result.get("confirmed_config", {}))
        self.assertEqual(result["custom_rpc"]["params"], [])
        self.assertEqual(result["custom_rpc"]["status"], "validated")
        self.assertEqual(result["rpc_mode"], "mixed")
        self.assertEqual(result["workload"]["mixed_weights"], {"eth_chainId": 100})
        self.assertTrue(result["workload"]["replace_defaults"])
        self.assertIn("自定义 RPC mixed workload 已确认", "\n".join(result["visible_response"]))

    def test_custom_rpc_validation_recovers_inline_weight_hint_after_method_is_known(self) -> None:
        from agent.harness.groups import _validate_rpc_schema
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed", "case": "known"}
        state["custom_rpc"] = {
            "method": "eth_chainId",
            "endpoint": "https://example.invalid/rpc",
            "endpoint_ready": True,
        }
        state["last_user_input"] = "没有参数，mixed 只跑这个 method，权重 100，quick，Grafana 不开"
        probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}

        with patch("agent.harness.groups.validate_rpc_endpoint", return_value=probe):
            result = _validate_rpc_schema(state, [], case="custom_rpc")

        self.assertEqual(result["custom_rpc"]["status"], "validated")
        self.assertEqual(result["workload"]["mixed_weights"], {"eth_chainId": 100})
        self.assertTrue(result["workload"]["replace_defaults"])

    def test_inline_weight_ignores_endpoint_version_numbers(self) -> None:
        from agent.harness.groups import _custom_rpc_inline_workload_hint

        text = "endpoint 是 https://example.invalid/v1/token，没有参数，mixed 只跑这个 method，权重 100"
        hint = _custom_rpc_inline_workload_hint(text, ["eth_chainId"])

        self.assertEqual(hint["weights"], {"eth_chainId": 100})

    def test_custom_rpc_schema_confirmation_uses_original_turn_for_weights(self) -> None:
        from agent.harness.groups import process_turn
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
            "schema_draft": {
                "status": "draft",
                "method": "eth_chainId",
                "params": [],
                "params_json": [],
                "evidence_kind": "jsonrpc_request",
                "transport": "jsonrpc",
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

        with patch("agent.harness.groups.validate_rpc_endpoint", return_value=probe):
            result = process_turn(state)

        self.assertEqual(result["custom_rpc"]["status"], "validated")
        self.assertEqual(result["workload"]["mixed_weights"], {"eth_chainId": 100})
        self.assertIn("自定义 RPC mixed workload 已确认", "\n".join(result["visible_response"]))

    def test_case2_protocol_confirm_consumes_queued_endpoint_and_method(self) -> None:
        from agent.harness.groups import process_turn
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
        }
        state["action_queue"] = [
            {
                "type": "start_custom_rpc",
                "rpc_method": "eth_blockNumber",
                "rpc_endpoint": "https://example.invalid/rpc",
                "_origin_text": source,
                "confidence": "medium",
            }
        ]
        state["last_user_input"] = "1"
        probe = {"ready": True, "status": "ok", "evidence_file": ".agent/evidence/probe.json"}

        with patch("agent.harness.groups.resolve_pending_choice", return_value={"matched": True, "confidence": "high", "selected_value": "confirm_proposed_protocol"}), patch("agent.harness.groups.validate_rpc_endpoint", return_value=probe):
            result = process_turn(state)

        self.assertEqual(result["endpoint_evidence"]["candidate_endpoint"], "https://example.invalid/rpc")
        self.assertIn(result["chain_identity"]["status"], {"existing_family_schema_needs_confirmation", "existing_family_method_validated_next", "existing_family_runtime_choice"})
        self.assertNotEqual(result["pending_question"].get("id"), "new_chain_endpoint")

    def test_case3_handoff_stops_instead_of_falling_back_to_chain_prompt(self) -> None:
        from agent.harness.groups import process_turn
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

        result = process_turn(state)

        self.assertEqual(result["chain_identity"]["status"], "unsupported_family_handoff")
        self.assertEqual(result.get("pending_question"), {})
        self.assertIn("二次开发", "\n".join(result["visible_response"]))
        self.assertNotIn("你想测试哪条链", "\n".join(result["visible_response"]))

    def test_case3_handoff_collects_followup_evidence_without_fallback(self) -> None:
        from agent.harness.groups import process_turn
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
        state["secondary_handoff"] = {"status": "collecting_evidence", "evidence": []}
        state["pending_question"] = {}
        state["last_user_input"] = "protocol: weird-p2p"

        with patch("agent.harness.groups.resolve_action_queue", return_value={"actions": [{"type": "unknown", "confidence": "low"}]}):
            result = process_turn(state)

        self.assertEqual(result["chain_identity"]["status"], "unsupported_family_handoff")
        self.assertEqual(result["secondary_handoff"]["evidence"], ["protocol: weird-p2p"])
        self.assertEqual(result.get("pending_question"), {})
        self.assertIn("二次开发证据", "\n".join(result.get("visible_response") or []))
        self.assertNotIn("你想测试哪条链", "\n".join(result.get("visible_response") or []))

    def test_case3_handoff_evidence_shaped_navigation_routes_through_resolver(self) -> None:
        """Phase 6 item 2: the keyword blocklist `_looks_like_handoff_navigation`

        was removed. A turn that is evidence-shaped (contains an evidence token
        such as `http`) but is actually a navigation/command ("analyze my latest
        report") must be classified by the action-queue resolver and routed
        away, not captured as secondary-development evidence.
        """

        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["latest_job_id"] = "job_demo"
        state["chain_identity"] = {
            "raw": "WeirdP2PChain",
            "canonical": "WeirdP2PChain",
            "adapter_family": "unsupported",
            "status": "unsupported_family_handoff",
            "case": "case3",
        }
        state["secondary_handoff"] = {"status": "collecting_evidence", "evidence": ["protocol: weird-p2p"]}
        state["last_user_input"] = "先别管这些，帮我分析最近一次 job 的 http 报告"

        with patch("agent.harness.groups.resolve_action_queue", return_value={"actions": [{"type": "analyze_report", "confidence": "high"}]}):
            result = process_turn(state)

        # Evidence must NOT grow — the resolver classified this as navigation.
        self.assertEqual(result["secondary_handoff"]["evidence"], ["protocol: weird-p2p"])
        self.assertNotIn("已记录第", "\n".join(result.get("visible_response") or []))

    def test_case3_handoff_generation_request_uses_collected_evidence(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["chain_identity"] = {
            "raw": "WeirdP2PChain",
            "canonical": "WeirdP2PChain",
            "adapter_family": "unsupported",
            "status": "unsupported_family_handoff",
            "case": "case3",
        }
        state["secondary_handoff"] = {"status": "collecting_evidence", "evidence": ["protocol: weird-p2p"]}
        state["last_user_input"] = "我想生成给另一个 AI 的二次开发任务文档"

        # An explicit generation request is a deterministic intent: even if the
        # resolver classifies it as a navigation action (e.g. an extension
        # question), the handoff draft must still be produced, never overridden.
        with patch(
            "agent.harness.groups.resolve_action_queue",
            return_value={"actions": [{"type": "answer_opening_question", "topic": "extension", "confidence": "high"}]},
        ):
            result = process_turn(state)

        self.assertEqual(len(result["secondary_handoff"]["evidence"]), 2)
        self.assertEqual(result.get("pending_question"), {})
        self.assertIn("adapter", "\n".join(result.get("visible_response") or []))
        self.assertIn("二次开发交接草案", "\n".join(result.get("visible_response") or []))
        self.assertIn("protocol: weird-p2p", "\n".join(result.get("visible_response") or []))

    def test_case3_handoff_does_not_capture_report_analysis_request(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["latest_job_id"] = "job_demo"
        state["chain_identity"] = {
            "raw": "WeirdP2PChain",
            "canonical": "WeirdP2PChain",
            "adapter_family": "unsupported",
            "status": "unsupported_family_handoff",
            "case": "case3",
        }
        state["secondary_handoff"] = {"status": "collecting_evidence", "evidence": ["protocol: weird-p2p"]}
        state["last_user_input"] = "先别管之前配置，帮我分析最近一次 job 的报告和日志"

        with patch("agent.harness.groups.resolve_action_queue") as resolver, patch("agent.harness.groups.resume_job") as resume:
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
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["chain_identity"] = {
            "raw": "WeirdP2PChain",
            "canonical": "WeirdP2PChain",
            "adapter_family": "unsupported",
            "status": "unsupported_family_handoff",
            "case": "case3",
        }
        state["secondary_handoff"] = {"status": "collecting_evidence", "evidence": ["protocol: weird-p2p"]}
        state["last_user_input"] = "我需要测试，但是我不知道可以测试什么"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "greeting", "confidence": "medium"}]}
            result = process_turn(state)

        self.assertEqual(result["secondary_handoff"]["evidence"], ["protocol: weird-p2p"])
        self.assertNotIn("二次开发证据", "\n".join(result.get("visible_response") or []))

    def test_case3_handoff_does_not_capture_confusion_question(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["chain_identity"] = {
            "raw": "WeirdP2PChain",
            "canonical": "WeirdP2PChain",
            "adapter_family": "unsupported",
            "status": "unsupported_family_handoff",
            "case": "case3",
        }
        state["secondary_handoff"] = {"status": "collecting_evidence", "evidence": ["protocol: weird-p2p"]}
        state["last_user_input"] = "什么意思？你在讲什么"

        result = process_turn(state)

        self.assertEqual(result["secondary_handoff"]["evidence"], ["protocol: weird-p2p"])
        self.assertNotIn("已记录第", "\n".join(result.get("visible_response") or []))

    def test_pending_config_review_merges_additional_yaml_lines(self) -> None:
        from agent.harness.groups import process_turn
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
        }
        state["last_user_input"] = "CLOUD_ZONE: asia-east1-c"

        result = process_turn(state)

        proposal = result["inferred_config"]["pending_review"]["config_values"]
        self.assertEqual(proposal["CLOUD_REGION"], "asia-east1")
        self.assertEqual(proposal["CLOUD_ZONE"], "asia-east1-c")
        self.assertEqual(result["pending_question"]["id"], "inferred_config_review")
        self.assertIn("CLOUD_ZONE", "\n".join(result["visible_response"]))

        result["last_user_input"] = "MACHINE_TYPE=n2-standard-16"
        merged = process_turn(result)

        proposal = merged["inferred_config"]["pending_review"]["config_values"]
        self.assertEqual(proposal["CLOUD_REGION"], "asia-east1")
        self.assertEqual(proposal["CLOUD_ZONE"], "asia-east1-c")
        self.assertEqual(proposal["MACHINE_TYPE"], "n2-standard-16")
        self.assertEqual(merged["pending_question"]["id"], "inferred_config_review")
        self.assertIn("MACHINE_TYPE", "\n".join(merged["visible_response"]))

    def test_terminal_style_fragmented_config_paste_maps_common_short_fields(self) -> None:
        from agent.harness.groups import process_turn
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
        }
        for line in [
            "zone: asia-east1-c",
            "machine_type: n2-standard-16",
            "ledger_device: vda",
            "data_vol_type: hyperdisk-balanced,",
            "data_vol_size: 926GiB",
            "iops: 20000 IOPS",
            "throughput: 1000 MiB/s",
            "interface: eth0",
            "bandwidth: 100Gbps",
        ]:
            state["last_user_input"] = line
            state = process_turn(state)

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
        from agent.harness.groups import process_turn
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
            {"type": "set_qps_mode", "qps_mode": "quick", "_origin_text": "QPS 用 quick"},
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

    def test_pending_config_review_rerenders_in_current_language(self) -> None:
        from agent.harness.groups import process_turn
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

        result = process_turn(state)
        text = "\n".join(result["visible_response"])

        self.assertIn("我从你粘贴的内容中推断出这些配置候选值", text)
        self.assertNotIn("I inferred these candidate config values", text)

    def test_freeform_traceback_collects_multiple_lines_before_analysis(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = "Traceback (most recent call last):"
        first = process_turn(state)

        self.assertEqual(first["evidence_collection"]["question"]["id"], "freeform_evidence")
        self.assertIn("多行错误/日志证据", "\n".join(first["visible_response"]))

        first["last_user_input"] = '  File "agent/terminal/repl.py", line 123, in run'
        first["language"] = "en"
        second = process_turn(first)

        self.assertEqual(second["evidence_collection"]["question"]["id"], "freeform_evidence")
        self.assertNotIn("AnyChain Benchmark Agent", "\n".join(second["visible_response"]))
        self.assertIn("已记录第 2 行证据", "\n".join(second["visible_response"]))

        second["last_user_input"] = "这个错误是什么意思？"
        third = process_turn(second)

        self.assertEqual(third.get("evidence_collection"), {})
        self.assertTrue(third["evidence_buffer"])
        self.assertIn("Traceback", third["evidence_buffer"][-1]["text"])
        self.assertIn("错误/日志证据", "\n".join(third["visible_response"]))

    def test_log_analysis_request_without_log_asks_for_evidence(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = "这是日志，你可以帮我分析么？"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "analyze_evidence", "confidence": "high"}]}
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertEqual(result.get("evidence_collection"), {})
        self.assertFalse(result.get("evidence_buffer"))
        self.assertIn("请把真实日志", text)
        self.assertNotIn("已开始接收多行错误/日志证据", text)

    def test_saved_log_evidence_followup_analyzes_buffer_not_opening_context(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["evidence_buffer"] = [{
            "text": "Traceback (most recent call last):\n  File \"/workspace/agent/terminal/repl.py\", line 1, in <module>\nRuntimeError: endpoint probe failed"
        }]
        state["last_user_input"] = "这是什么意思？下一步怎么修？"

        result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertEqual(result["active_group"], "error_evidence_analysis")
        self.assertIn("endpoint/probe failed", text)
        self.assertIn("证据预览", text)
        self.assertNotIn("测试前需要准备什么", text)

    def test_multiline_log_paste_is_stored_as_multiple_evidence_lines(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = (
            "Traceback (most recent call last):\n"
            "  File \"agent/terminal/repl.py\", line 123, in run\n"
            "RuntimeError: boom"
        )

        result = process_turn(state)

        lines = result["evidence_collection"]["lines"]
        text = "\n".join(result.get("visible_response") or [])
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[0], "Traceback (most recent call last):")
        self.assertIn("当前收到 3 行", text)

    def test_blank_line_during_log_evidence_collection_does_not_resume_config(self) -> None:
        from agent.harness.groups import process_turn
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

    def test_identity_question_breaks_stale_log_evidence_collection(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["evidence_collection"] = {
            "question": {"id": "freeform_evidence", "kind": "log_evidence"},
            "lines": ["Traceback (most recent call last):", '  File "x.py", line 1'],
            "language": "zh",
        }
        state["last_user_input"] = "你是谁？"

        result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertEqual(result.get("evidence_collection"), {})
        self.assertIn("AnyChain Benchmark Agent", text)
        self.assertNotIn("已记录第", text)

    def test_multiline_rpc_evidence_is_collected_before_protocol_conflict(self) -> None:
        from agent.harness.groups import process_turn
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
        second = process_turn(first)
        self.assertTrue(second["evidence_collection"]["lines"])

        second["last_user_input"] = 'response: {"jsonrpc":"2.0","id":1,"result":"0x10"}'
        with patch("agent.harness.groups.extract_rpc_schema_from_evidence") as extract:
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
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = "如果我需要测试，我都需要做什么，提供什么"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "answer_opening_question", "topic": "requirements", "confidence": "high"}]}
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("测试类型", text)
        self.assertIn("CLOUD_REGION", text)
        self.assertIn("Ledger/data", text)
        self.assertIn("preflight/smoke", text)
        self.assertNotIn("已知链：acala", text)

    def test_opening_correction_question_does_not_repeat_capabilities(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = "你是否理解我的问题？"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "answer_opening_question", "topic": "correction", "confidence": "high"}]}
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("我理解", text)
        self.assertIn("准备什么", text)
        self.assertNotIn("协议族分布", text)

    def test_opening_identity_question_answers_agent_identity_not_chain_list(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = "你是谁"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "answer_opening_question", "topic": "identity", "confidence": "high"}]}
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("AnyChain Benchmark Agent", text)
        self.assertIn("配置、校验、执行和报告分析", text)
        self.assertNotIn("已知链：", text)

    def test_identity_destination_question_uses_llm_identity_topic(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = "你要去哪里？"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "answer_opening_question", "topic": "identity", "confidence": "high"}]}
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("AnyChain Benchmark Agent", text)
        self.assertIn("配置、校验、执行和报告分析", text)
        self.assertNotIn("当前没有待确认问题", text)

    def test_opening_agent_capabilities_are_product_summary_not_raw_chain_list(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = "你能做什么"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
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

        `capabilities` value with no dedicated prompt rule. `groups.py` used
        to normalize it to `supported_chains` (the raw chain list), which
        contradicted the sibling `capability`/`what_can_you_do` normalization
        (agent capabilities). It must map to the agent-capabilities product
        summary, consistent with what the schema text implies.
        """

        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = "capabilities?"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
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

        from agent.harness.groups import process_turn
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

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
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

        from agent.harness.groups import _answer_fits_pending

        question = {
            "id": "CLOUD_ZONE",
            "group": "provider_deployment",
            "kind": "manual_value",
            "field": "CLOUD_ZONE",
            "manual_input_allowed": True,
        }
        colloquial_question = "你有一个环境依赖的检测脚本，这个脚本会帮我推断一些变量，这些变量推断了么"
        self.assertFalse(_answer_fits_pending(colloquial_question, question))

    def test_scalar_value_containing_current_as_substring_is_not_treated_as_a_question(self) -> None:
        """Code review (Phase 1/2 diff) found `_looks_like_user_question`'s

        "current" marker was a plain substring check, newly wired into the
        manual_value/device/url branches. A legitimate single-token answer
        like an API key or device path containing "current" as a substring
        (e.g. "concurrent-tier-01") must not be rejected as a question.
        """

        from agent.harness.groups import _answer_fits_pending

        manual_value_question = {
            "id": "RPC_API_KEY",
            "group": "chain_auxiliary_endpoints",
            "kind": "manual_value",
            "field": "RPC_API_KEY",
            "manual_input_allowed": True,
        }
        self.assertTrue(_answer_fits_pending("concurrent-tier-01", manual_value_question))

        device_question = {
            "id": "LEDGER_DEVICE",
            "group": "ledger_disk",
            "kind": "device",
            "field": "LEDGER_DEVICE",
            "manual_input_allowed": True,
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
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = "fake-node real-node sync-observe 有什么区别"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "answer_opening_question", "topic": "mode_comparison", "confidence": "high"}]}
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("fake-node", text)
        self.assertIn("real-node", text)
        self.assertIn("sync-observe", text)
        self.assertIn("不走 vegeta", text)
        self.assertIn("fake-node 的作用", text)
        self.assertIn("支持多少 QPS", text)
        self.assertIn("real-node benchmark", text)

    def test_opening_performance_goal_guidance_prefers_real_node(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = "我需要观察能支持多少 qps，性能瓶颈在哪里"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "answer_opening_question", "topic": "performance_benchmark_guidance", "confidence": "high"}]}
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("real-node benchmark", text)
        self.assertIn("LOCAL_RPC_URL", text)
        self.assertIn("fake-node 只能验证框架闭环", text)
        self.assertIn("sync-observe", text)

    def test_performance_goal_keeps_guidance_when_llm_extracts_topic_and_chain(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["last_user_input"] = "I want to benchmark BSC max throughput and find bottlenecks, not just test the tool itself."

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "answer_opening_question", "topic": "performance_benchmark_guidance", "confidence": "high"},
                    {"type": "choose_chain", "chain_text": "BSC", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("real-node benchmark", text)
        self.assertIn("fake-node only validates", text)
        self.assertIn("Confirmed chain: `bsc`", text)
        self.assertEqual(result["chain_identity"]["canonical"], "bsc")
        self.assertEqual(result["pending_question"]["id"], "opening_next_action")

    def test_performance_goal_rejects_conflicting_fake_node_target_mode_action(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = "我需要测试 solana 能扛多少 qps，fake-node 可以测这个吗"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "answer_opening_question", "topic": "performance_benchmark_guidance", "confidence": "high"},
                    {"type": "choose_target_mode", "target_mode": "fake-node", "target_mode_explicit": True, "confidence": "high"},
                    {"type": "choose_chain", "chain_text": "solana", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertEqual(result.get("target_mode"), "")
        self.assertEqual(result["chain_identity"]["canonical"], "solana")
        self.assertEqual(result["pending_question"]["id"], "opening_next_action")
        self.assertIn("real-node benchmark", text)

    def test_opening_option_three_enters_sync_observe_without_llm_recommendation(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["active_group"] = "opening"
        state["chain_identity"] = {"raw": "solana", "canonical": "solana", "adapter_family": "jsonrpc", "status": "confirmed"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "solana"}
        state["pending_question"] = {
            "id": "opening_next_action",
            "group": "opening",
            "kind": "numbered_choice",
            "field": "target_mode",
            "manual_input_allowed": False,
            "options": [
                {"label": "启动 fake-node 测试", "value": "fake-node"},
                {"label": "启动 real-node 测试", "value": "real-node"},
                {"label": "观察节点同步", "value": "sync-observe"},
                {"label": "了解支持的链、RPC method 和二次开发方式", "value": "info"},
            ],
        }
        state["last_user_input"] = "3"

        result = process_turn(state)

        self.assertEqual(result["target_mode"], "sync-observe")
        self.assertEqual(result["workflow_mode"], "sync_observe")
        self.assertNotIn("建议先用 fake-node", "\n".join(result.get("visible_response") or []))

    def test_opening_option_answer_discards_stale_consultation_queue(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["active_group"] = "opening"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["pending_question"] = {
            "id": "opening_next_action",
            "group": "opening",
            "kind": "numbered_choice",
            "field": "target_mode",
            "manual_input_allowed": False,
            "resume_action_queue": True,
            "options": [
                {"label": "启动 fake-node 测试", "value": "fake-node"},
                {"label": "启动 real-node 测试", "value": "real-node"},
                {"label": "观察节点同步", "value": "sync-observe"},
                {"label": "了解支持的链、RPC method 和二次开发方式", "value": "info"},
            ],
        }
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
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["chain_identity"] = {"raw": "BSC", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["last_user_input"] = "Why can't fake-node tell me the real bottleneck?"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "choose_target_mode", "target_mode": "fake-node", "target_mode_explicit": True, "confidence": "high"},
                    {"type": "answer_opening_question", "topic": "mode_comparison", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertEqual(result.get("target_mode"), "")
        self.assertIn("fake-node is useful", text)
        self.assertIn("real-node benchmark", text)

    def test_mode_consultation_prunes_conflicting_fake_node_recommendation(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["chain_identity"] = {"raw": "BSC", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["last_user_input"] = "Why can't fake-node tell me the real bottleneck? If I care about block sync speed, which mode should I use?"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
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
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["chain_identity"] = {"raw": "BNB", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["last_user_input"] = "那假节点还有什么意义？它能告诉我真实性能吗？"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "choose_target_mode", "target_mode": "fake-node", "target_mode_explicit": True, "confidence": "high"},
                    {"type": "answer_opening_question", "topic": "mode_comparison", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertEqual(result.get("target_mode"), "")
        self.assertIn("fake-node 的作用", text)
        self.assertIn("不能回答真实节点 QPS", text)

    def test_explicit_sync_observe_selection_survives_consultation_guard(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="en")
        state["chain_identity"] = {"raw": "BSC", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["last_user_input"] = "Then let's observe sync behavior for BSC, but don't run vegeta."

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "choose_target_mode", "target_mode": "sync-observe", "target_mode_explicit": True, "confidence": "high"},
                ]
            }
            result = process_turn(state)

        self.assertEqual(result.get("target_mode"), "sync-observe")
        self.assertEqual(result.get("workflow_mode"), "sync_observe")

    def test_chinese_block_catchup_phrase_selects_sync_observe(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["chain_identity"] = {"raw": "BNB", "canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed"}
        state["confirmed_config"] = {"BLOCKCHAIN_NODE": "bsc"}
        state["last_user_input"] = "那我先观察 bsc 追块，不要跑 vegeta 压测"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "choose_chain", "chain_text": "bsc", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        self.assertEqual(result.get("target_mode"), "sync-observe")
        self.assertEqual(result.get("workflow_mode"), "sync_observe")

    def test_pending_accounts_answer_with_extra_question_applies_answer_then_routes_question(self) -> None:
        from agent.harness.groups import process_turn
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

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "answer_opening_question", "topic": "requirements", "confidence": "high"}]}
            result = process_turn(state)

        self.assertIs(result["confirmed_config"]["has_accounts_device"], False)
        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("LOCAL_RPC_URL", text)
        self.assertIn("endpoint", text)
        self.assertNotIn("这个节点是否有独立的 accounts/state 磁盘", text)

    def test_pending_accounts_plain_natural_answer_only_continues_fallback(self) -> None:
        from agent.harness.groups import process_turn
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

        result = process_turn(state)

        self.assertIs(result["confirmed_config"]["has_accounts_device"], False)
        self.assertNotEqual(result.get("pending_question", {}).get("id"), "has_accounts_device")

    def test_mixed_goal_and_config_turn_preserves_chain_and_target_mode_when_llm_returns_only_config(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = (
            "我要测试 BNB real-node，region asia-east1，zone asia-east1-c，机器 n2-standard-16，"
            "ledger vda，磁盘 hyperdisk-balanced，IOPS 20000，吞吐 1000"
        )

        with (
            patch("agent.harness.groups.resolve_action_queue") as resolver,
            patch("agent.harness.groups.extract_chain_mention") as mention,
        ):
            resolver.return_value = {
                "actions": [
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
            mention.return_value = {"found": True, "chain_text": "BNB", "confidence": "high"}
            result = process_turn(state)

        self.assertEqual(result["target_mode"], "real-node")
        self.assertEqual(result["chain_identity"]["canonical"], "bsc")
        self.assertEqual(result["pending_question"]["id"], "inferred_config_review")
        self.assertIn("CLOUD_REGION", "\n".join(result.get("visible_response") or []))

    def test_opening_current_config_question_reports_state(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "real-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"canonical": "bsc", "adapter_family": "jsonrpc", "status": "confirmed"}
        state["pending_question"] = {"id": "DATA_VOL_SIZE", "group": "ledger_disk", "kind": "confirm_or_value"}
        state["last_user_input"] = "当前链和模式是什么？"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "answer_opening_question", "topic": "current_config", "confidence": "high"}]}
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("chain: `bsc`", text)
        self.assertIn("target_mode: `real-node`", text)
        self.assertIn("DATA_VOL_SIZE", text)
        self.assertNotIn("已知链：", text)

    def test_current_context_question_explains_pending_qps_confirmation(self) -> None:
        from agent.harness.groups import process_turn
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

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "answer_opening_question", "topic": "current_context", "confidence": "high"}]}
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("QPS profile", text)
        self.assertIn("INITIAL_QPS", text)
        self.assertIn("fake-node smoke", text)

    def test_current_context_without_pending_does_not_leak_internal_next_action(self) -> None:
        from agent.harness.groups import process_turn
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

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
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
        from agent.workflows.group_registry import GROUP_ORDER

        self.assertEqual(list(GROUP_ORDER), list(DEFAULT_GROUP_ORDER))
        self.assertEqual(list(ALLOWED_GROUPS), list(DEFAULT_GROUP_ORDER))
        self.assertNotIn("hardware_discovery", DEFAULT_GROUP_ORDER)

    def test_adapter_family_lists_derive_from_single_source(self) -> None:
        """Architecture audit Finding C3: the supported adapter-family set was

        retyped in six places (one with real value drift). They must all now
        derive from `agent.onboarding.families.SUPPORTED_FAMILIES`.
        """

        from agent.harness.groups import SUPPORTED_ADAPTER_FAMILIES, _adapter_family_options
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
        option_values = {opt["value"] for opt in _adapter_family_options("en")}
        self.assertEqual(option_values - {"unsupported"}, canonical)
        # template_drafter partitions the canonical families into two transports.
        self.assertEqual(REST_TRANSPORT_FAMILIES | JSONRPC_TRANSPORT_FAMILIES, canonical)
        self.assertEqual(REST_TRANSPORT_FAMILIES & JSONRPC_TRANSPORT_FAMILIES, set())
        # endpoint_probe's generic-JSON-RPC set must only contain canonical
        # family values (no non-canonical evm/ethereum aliases).
        self.assertTrue(GENERIC_JSONRPC_PROBE_FAMILIES <= canonical)
        self.assertEqual(GENERIC_JSONRPC_PROBE_FAMILIES, {"jsonrpc"})

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

        from agent.harness.groups import _ask_next_blocking_question
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

        from agent.harness.groups import _ask_next_blocking_question
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

        from agent.harness.groups import _build_config_proposal

        proposal = _build_config_proposal({"config_values": {"RPC_API_KEY": "abc123"}})

        self.assertEqual(proposal["config_values"].get("RPC_API_KEY"), "abc123")
        self.assertNotIn("RPC_API_KEY", proposal["unmapped_values"])

    def test_advanced_tuning_is_reachable_and_not_a_dead_end(self) -> None:
        """Architecture audit Finding A: `advanced_tuning` was declared in

        `DEFAULT_GROUP_ORDER` but never implemented in `groups.py`, so a
        user routed there hit a dead end. This is the fixed behavior: the
        group asks a real, documented question grounded in
        `config/user_config.sh` / `config/internal_config.sh` field names.
        """

        from agent.harness.groups import _ask_next_blocking_question

        state = self._fully_configured_state_before_advanced_tuning()
        result = _ask_next_blocking_question(state)

        self.assertEqual(result["pending_question"]["id"], "advanced_tuning_confirm")
        self.assertEqual(result["pending_question"]["group"], "advanced_tuning")
        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("MONITOR_INTERVAL", text)
        self.assertIn("BOTTLENECK_CPU_THRESHOLD", text)

    def test_advanced_tuning_accepting_defaults_advances_to_preflight(self) -> None:
        from agent.harness.groups import process_turn

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
        from agent.harness.groups import process_turn

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

        from agent.harness.groups import process_turn

        state = self._fully_configured_state_before_advanced_tuning()
        # Park on an unrelated, already-satisfied group so activating
        # advanced_tuning is a genuine redirect, not natural progression.
        state["active_group"] = "workload_rpc"
        state["pending_question"] = {}
        state["last_user_input"] = "我想改一下 CPU 瓶颈阈值"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [{"type": "change_group", "group": "advanced_tuning", "confidence": "high"}]
            }
            result = process_turn(state)

        self.assertEqual(result["active_group"], "advanced_tuning")
        self.assertEqual(result["pending_question"]["id"], "advanced_tuning_confirm")

    def test_oracle_and_groups_agree_on_next_group_for_advanced_tuning(self) -> None:
        """Architecture audit Finding B1: `oracle.py` and `groups.py` used to

        reimplement the same next-group precondition chain independently
        and could disagree. Both must now delegate to
        `agent.harness.routing.next_group_and_reason`.
        """

        from agent.harness.groups import _next_group
        from agent.harness.oracle import _next_group_and_reason

        state = self._fully_configured_state_before_advanced_tuning()
        self.assertEqual(_next_group(state), "advanced_tuning")
        self.assertEqual(_next_group_and_reason(state), ("advanced_tuning", "review advanced tuning settings"))

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
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["target_mode"] = "fake-node"
        state["workflow_mode"] = "rpc_benchmark"
        state["chain_identity"] = {"raw": "bsc", "canonical": "bsc", "status": "confirmed"}
        state["observability"] = {"mode": "disabled"}
        state["last_user_input"] = "如何重新开始呢"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "answer_opening_question", "topic": "reset_help", "confidence": "high"}]}
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("完全重新开始", text)
        self.assertIn("清空配置重新开始", text)
        self.assertNotIn("可观测性模式", text)
        self.assertNotIn("preflight_smoke", text)

    def test_reset_session_action_clears_previous_config_but_keeps_startup_facts(self) -> None:
        from agent.harness.groups import process_turn
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

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "reset_session", "confidence": "high"}]}
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("已清空之前的 Agent 配置", text)
        self.assertEqual(result.get("chain_identity"), {})
        self.assertEqual(result.get("confirmed_config"), {})
        self.assertEqual(result.get("target_mode"), "")
        self.assertEqual(result.get("framework_summary", {}).get("chain_count"), 36)
        self.assertEqual((result.get("discovery") or {}).get("cloud", {}).get("provider"), "other")

    def test_change_to_another_chain_without_name_asks_chain_instead_of_preflight(self) -> None:
        from agent.harness.groups import process_turn
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

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "change_group", "group": "chain_identity", "confidence": "high"}]}
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertEqual(result.get("active_group"), "chain_identity")
        self.assertEqual((result.get("pending_question") or {}).get("id"), "chain")
        self.assertIn("你想测试哪条链", text)
        self.assertNotIn("是否运行 preflight", text)

    def test_opening_recommendation_for_unsure_quick_validation(self) -> None:
        from agent.harness.groups import process_turn
        from agent.harness.state import new_state

        state = new_state("unit-thread", language="zh")
        state["last_user_input"] = "我想先随便跑一下，但不知道 fake-node 和 real-node 选哪个"

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "answer_opening_question", "topic": "recommendation", "confidence": "high"}]}
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertIn("建议先用 fake-node smoke", text)
        self.assertIn("solana", text)
        self.assertEqual(result.get("target_mode"), "")

    def test_pending_jump_to_workload_preserves_followup_qps_action(self) -> None:
        from agent.harness.groups import process_turn
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

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "change_group", "group": "workload_rpc", "confidence": "high"},
                    {"type": "set_qps_mode", "qps_mode": "quick", "confidence": "high"},
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
        from agent.harness.groups import process_turn
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

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {"actions": [{"type": "set_qps_mode", "qps_mode": "quick", "confidence": "high"}]}
            first = process_turn(state)

        self.assertEqual(first["pending_question"]["id"], "inferred_config_review")
        self.assertTrue(first["pending_question"].get("resume_action_queue"))
        self.assertEqual(first["action_queue"][0]["type"], "set_qps_mode")

        first["last_user_input"] = "Y"
        second = process_turn(first)

        self.assertEqual(second["confirmed_config"]["CLOUD_REGION"], "us-1")
        self.assertEqual(second["qps_profile"]["mode"], "quick")
        self.assertEqual(second["pending_question"]["id"], "qps_profile_confirm")

    def test_custom_rpc_capability_question_does_not_start_endpoint_workflow(self) -> None:
        from agent.harness.groups import process_turn
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

        with patch("agent.harness.groups.resolve_action_queue") as resolver:
            resolver.return_value = {
                "actions": [
                    {"type": "answer_opening_question", "topic": "config_explanation", "subject": "workload_rpc", "confidence": "high"},
                    {"type": "start_custom_rpc", "confidence": "high"},
                ]
            }
            result = process_turn(state)

        text = "\n".join(result.get("visible_response") or [])
        self.assertNotEqual(result.get("pending_question", {}).get("id"), "custom_rpc_endpoint")
        self.assertNotEqual((result.get("custom_rpc") or {}).get("status"), "needs_endpoint")
        self.assertIn("扩展分三类", text)

    def test_workload_pending_context_uses_user_facing_explanation(self) -> None:
        from agent.harness.groups import process_turn
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

        result = process_turn(state)
        text = "\n".join(result.get("visible_response") or [])

        self.assertIn("默认 RPC workload", text)
        self.assertNotIn("workload_rpc 用于生成", text)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Run product CLI scenarios against the LangGraph Harness.

This runner is intentionally small: it drives the same ``./bin/anychain-agent``
entrypoint a user runs, isolates terminal/checkpoint state per scenario, and
asserts LangGraph state after the conversation. It does not read legacy
``.agent/sessions/*/conversation_state.json``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Scenario:
    name: str
    prompts: list[str]
    assert_state: Callable[[dict[str, Any], str], list[str]]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LangGraph product CLI live matrix.")
    parser.add_argument("--scenario", action="append", help="Run only the named scenario.")
    args = parser.parse_args()

    selected = set(args.scenario or [])
    scenarios = [item for item in _scenarios() if not selected or item.name in selected]
    failures: list[str] = []
    for scenario in scenarios:
        issues = _run_scenario(scenario)
        failures.extend(f"{scenario.name}: {issue}" for issue in issues)

    if failures:
        for issue in failures:
            print(issue)
        return 1
    print(f"langgraph cli matrix ok ({len(scenarios)} scenarios)")
    return 0


def _run_scenario(scenario: Scenario) -> list[str]:
    with tempfile.TemporaryDirectory(prefix=f"anychain-{scenario.name}-") as tmpdir:
        tmp = Path(tmpdir)
        state_file = tmp / "terminal-state.json"
        checkpoint = tmp / "checkpoints.sqlite"
        session_id = f"matrix-{scenario.name}"
        env = os.environ.copy()
        env["ANYCHAIN_AGENT_CHECKPOINT_PATH"] = str(checkpoint)
        command = [
            str(REPO_ROOT / "bin" / "anychain-agent"),
            "--state-file",
            str(state_file),
            "--session-id",
            session_id,
        ]
        for prompt in scenario.prompts:
            command.extend(["--prompt", prompt])
        try:
            proc = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=300,
                check=False,
            )
            transcript = proc.stdout
        except subprocess.TimeoutExpired as exc:
            transcript = exc.stdout or ""
            if isinstance(transcript, bytes):
                transcript = transcript.decode("utf-8", errors="replace")
            log_dir = REPO_ROOT / ".agent" / "live-matrix"
            log_dir.mkdir(parents=True, exist_ok=True)
            transcript_file = log_dir / f"{scenario.name}.timeout.transcript.txt"
            transcript_file.write_text(str(transcript), encoding="utf-8")
            return [f"CLI timed out after 300s; transcript retained for this run: {transcript_file}"]
        if proc.returncode != 0:
            return [f"CLI exited {proc.returncode}\n{transcript}"]
        state = _load_graph_state(session_id, checkpoint)
        issues = scenario.assert_state(state, transcript)
        if issues:
            log_dir = REPO_ROOT / ".agent" / "live-matrix"
            log_dir.mkdir(parents=True, exist_ok=True)
            transcript_file = log_dir / f"{scenario.name}.transcript.txt"
            transcript_file.write_text(transcript, encoding="utf-8")
            issues.append(f"transcript retained for this run: {transcript_file}")
        return issues


def _load_graph_state(session_id: str, checkpoint: Path) -> dict[str, Any]:
    sys.path.insert(0, str(REPO_ROOT / "agent"))
    from harness.graph import AnyChainGraphRuntime

    runtime = AnyChainGraphRuntime(thread_id=session_id, checkpoint_path=checkpoint)
    snapshot = runtime.graph.get_state({"configurable": {"thread_id": session_id}})
    values = getattr(snapshot, "values", None) or {}
    return dict(values)


def _scenarios() -> list[Scenario]:
    return [
        Scenario(
            name="unknown_chain_then_mode_jump",
            prompts=["Hi", "1", "sola", "Y", "change to BNB", "Y", "use real-node instead", "Y", "Y"],
            assert_state=_assert_unknown_chain_then_mode_jump,
        ),
        Scenario(
            name="capability_detour_returns_to_benchmark",
            prompts=["你好", "1", "我需要了解支持的链、RPC 方法和扩展路径", "回到 benchmark 设置", "solana"],
            assert_state=_assert_capability_detour,
        ),
        Scenario(
            name="existing_chain_custom_rpc_replace_defaults",
            prompts=[
                "Hi",
                "1",
                "bsc",
                "us-1",
                "us-1-z",
                "n2",
                "1",
                "hyperdisk-balanced",
                "Y",
                "20000",
                "1000",
                "N",
                "1",
                "100",
                "mixed",
                "2",
                "http://fake-node:19000",
                "eth_blockNumber",
                '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}',
                "Y",
                "2",
                "2",
                "eth_blockNumber=100",
            ],
            assert_state=_assert_existing_chain_custom_rpc_replace_defaults,
        ),
        Scenario(
            name="new_chain_existing_family_custom_rpc_mixed_weights",
            prompts=[
                "你好",
                "1",
                "abcd",
                "1",
                "1",
                "http://fake-node:19000",
                "eth_blockNumber",
                '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}',
                "Y",
                "1",
                "eth_chainId",
                '{"jsonrpc":"2.0","method":"eth_chainId","params":[],"id":1}',
                "Y",
                "2",
                "2",
                "eth_blockNumber=70,eth_chainId=30",
            ],
            assert_state=_assert_new_chain_existing_family_custom_rpc_mixed_weights,
        ),
        Scenario(
            name="group_jump_qps_then_default_resume",
            prompts=[
                "Hi",
                "1",
                "bsc",
                "I want to configure QPS before the environment values",
                "1",
                "Y",
            ],
            assert_state=_assert_group_jump_qps_then_default_resume,
        ),
        Scenario(
            name="interrupted_workload_qps_then_back_to_rpc",
            prompts=[
                "Hi",
                "1",
                "bsc",
                "us-1",
                "us-1-z",
                "n2",
                "1",
                "hyperdisk-balanced",
                "Y",
                "20000",
                "1000",
                "N",
                "1",
                "100",
                "single",
                "I want to adjust QPS before choosing workload defaults",
                "1",
                "Y",
                "go back to RPC config",
            ],
            assert_state=_assert_interrupted_workload_qps_then_back_to_rpc,
        ),
        Scenario(
            name="chain_change_decline_restores_interrupted_group",
            prompts=[
                "Hi",
                "1",
                "bsc",
                "us-1",
                "us-1-z",
                "n2",
                "1",
                "hyperdisk-balanced",
                "Y",
                "20000",
                "1000",
                "N",
                "1",
                "100",
                "single",
                "change to ethereum",
                "N",
            ],
            assert_state=_assert_chain_change_decline_restores_interrupted_group,
        ),
        Scenario(
            name="chain_change_accept_preserves_environment_and_invalidates_workload",
            prompts=[
                "Hi",
                "1",
                "bsc",
                "us-1",
                "us-1-z",
                "n2",
                "1",
                "hyperdisk-balanced",
                "Y",
                "20000",
                "1000",
                "N",
                "1",
                "100",
                "mixed",
                "1",
                "change to ethereum",
                "Y",
            ],
            assert_state=_assert_chain_change_accept_preserves_environment_and_invalidates_workload,
        ),
        Scenario(
            name="rpc_setup_switches_to_sync_observe",
            prompts=[
                "Hi",
                "1",
                "bsc",
                "us-1",
                "us-1-z",
                "n2",
                "1",
                "hyperdisk-balanced",
                "Y",
                "20000",
                "1000",
                "N",
                "1",
                "100",
                "I do not want RPC load now; observe node sync instead",
                "Y",
            ],
            assert_state=_assert_rpc_setup_switches_to_sync_observe,
        ),
        Scenario(
            name="multi_demand_action_queue",
            prompts=[
                "你好",
                "我要用 fake-node 测试 BNB，用 mixed，QPS quick，并开启本地 Grafana",
            ],
            assert_state=_assert_multi_demand_action_queue,
        ),
        Scenario(
            name="multi_demand_without_explicit_target_mode",
            prompts=[
                "你好",
                "我要测试 BNB，用 mixed，QPS quick，并开启本地 Grafana",
            ],
            assert_state=_assert_multi_demand_without_explicit_target_mode,
        ),
        Scenario(
            name="capability_plus_uncertain_benchmark_goal",
            prompts=[
                "我想先随便跑通一下框架，但我不确定应该用 fake-node 还是 real-node；我这边可能测 BNB，也可能只是想看看支持哪些链",
            ],
            assert_state=_assert_capability_plus_uncertain_benchmark_goal,
        ),
        Scenario(
            name="multiline_config_inference_review",
            prompts=[
                "你好",
                "1",
                "solana",
                """这是从另一台机器复制过来的配置，请你先推断，不确定的先列出来让我确认：
cloud:
  region: asia-east1
  zone: asia-east1-c
  machine_type: n2-standard-16
storage:
  ledger_device: vda
  data_vol_type: hyperdisk-balanced
  data_vol_size: 926
  data_vol_max_iops: 20000
  data_vol_max_throughput: 1000
network:
  interface: eth0
  bandwidth_gbps: 100
note: copied from terraform workspace prod-a""",
                "Y",
            ],
            assert_state=_assert_multiline_config_inference_review,
        ),
        Scenario(
            name="partial_pasted_config_resumes_missing_groups",
            prompts=[
                "你好",
                "1",
                "solana",
                """我先贴一部分环境信息，其他不确定的你继续问：
export CLOUD_REGION=asia-east1
export CLOUD_ZONE=asia-east1-c
export MACHINE_TYPE=n2-standard-16
NETWORK_INTERFACE=eth0
NETWORK_MAX_BANDWIDTH_GBPS=100
RPC_MODE=single
unrelated_ticket=INC-12345""",
                "Y",
            ],
            assert_state=_assert_partial_pasted_config_resumes_missing_groups,
        ),
        Scenario(
            name="single_turn_multi_group_with_pasted_config",
            prompts=[
                "你好",
                """我要用 fake-node 测试 BNB，用 mixed，QPS quick，并开启本地 Grafana。下面是从环境里复制的配置：
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
  bandwidth_gbps: 100""",
                "Y",
            ],
            assert_state=_assert_single_turn_multi_group_with_pasted_config,
        ),
        Scenario(
            name="nested_detours_return_to_earliest_incomplete_group",
            prompts=[
                "Hi",
                "1",
                "bsc",
                "us-1",
                "I want to configure QPS first before the rest",
                "1",
                "Y",
                "also turn on local Grafana now",
            ],
            assert_state=_assert_nested_detours_return_to_earliest_incomplete_group,
        ),
        Scenario(
            name="first_turn_direct_goal_with_config_paste",
            prompts=[
                """我要直接测试 BNB，用 fake-node，mixed，quick，并开启本地 Grafana。这是机器信息：
{
  "CLOUD_REGION": "asia-east1",
  "CLOUD_ZONE": "asia-east1-c",
  "MACHINE_TYPE": "n2-standard-16",
  "LEDGER_DEVICE": "vda",
  "DATA_VOL_TYPE": "hyperdisk-balanced",
  "DATA_VOL_SIZE": 926,
  "DATA_VOL_MAX_IOPS": 20000,
  "DATA_VOL_MAX_THROUGHPUT": 1000,
  "NETWORK_INTERFACE": "eth0",
  "NETWORK_MAX_BANDWIDTH_GBPS": 100
}""",
                "Y",
            ],
            assert_state=_assert_first_turn_direct_goal_with_config_paste,
        ),
        Scenario(
            name="language_switch_and_group_jump_preserves_state",
            prompts=[
                "Hi",
                "1",
                "bsc",
                "us-1",
                "请后续用中文回答。I want to configure QPS first and set benchmark mode to quick.",
                "Y",
                "回到 CLOUD_ZONE 配置",
                "asia-east1-c",
            ],
            assert_state=_assert_language_switch_and_group_jump_preserves_state,
        ),
        Scenario(
            name="disk_pending_jump_to_qps_then_resume_disk",
            prompts=[
                "你好",
                "1",
                "bsc",
                """下面是部分环境：
cloud:
  region: asia-east1
  zone: asia-east1-c
  machine_type: n2-standard-16
network:
  interface: eth0
  bandwidth_gbps: 100""",
                "Y",
                "先别问磁盘，我要先把 QPS 改成 quick，然后回到磁盘配置",
                "Y",
            ],
            assert_state=_assert_disk_pending_jump_to_qps_then_resume_disk,
        ),
        Scenario(
            name="triple_detour_returns_to_earliest_incomplete_group",
            prompts=[
                "你好",
                "1",
                "bsc",
                "us-1",
                "切换到 ethereum，然后把 QPS 设成 quick，并开启本地 Grafana",
                "Y",
                "Y",
            ],
            assert_state=_assert_triple_detour_returns_to_earliest_incomplete_group,
        ),
        Scenario(
            name="pending_question_error_paste_goes_to_evidence_analysis",
            prompts=[
                "Hi",
                "1",
                "bsc",
                "Traceback (most recent call last):\n  File \"blockchain_node_benchmark.sh\", line 42\nRuntimeError: endpoint timeout while probing eth_blockNumber",
            ],
            assert_state=_assert_pending_question_error_paste_goes_to_evidence_analysis,
        ),
        Scenario(
            name="english_process_like_user_text_routes_normally",
            prompts=[
                "I want to quickly test BNB with fake-node, but I also need to know which chains and RPC methods are supported first",
            ],
            assert_state=_assert_english_process_like_user_text_routes_normally,
        ),
    ]


def _assert_unknown_chain_then_mode_jump(state: dict[str, Any], transcript: str) -> list[str]:
    issues: list[str] = []
    confirmed = state.get("confirmed_config") or {}
    identity = state.get("chain_identity") or {}
    if identity.get("canonical") != "bsc":
        issues.append(f"expected chain bsc, got {identity}")
    if state.get("target_mode") != "real-node":
        issues.append(f"expected target_mode real-node, got {state.get('target_mode')}")
    if confirmed.get("CLOUD_REGION") in {"Y", "y", "yes"}:
        issues.append(f"CLOUD_REGION was polluted by yes/no answer: {confirmed.get('CLOUD_REGION')!r}")
    if "Confirm switching from `fake-node` to `real-node`" not in transcript:
        issues.append("target mode change was not explicitly confirmed in transcript")
    if "`sola` is not a configured template" not in transcript:
        issues.append("unknown chain identity gate did not appear for partial chain")
    return issues


def _assert_capability_detour(state: dict[str, Any], transcript: str) -> list[str]:
    issues: list[str] = []
    if "36" not in transcript or "RPC" not in transcript:
        issues.append("capability answer did not describe supported chains/RPC methods")
    if "{'chain':" in transcript or '"chain":' in transcript:
        issues.append("capability answer leaked raw chain inventory objects instead of a concise product summary")
    identity = state.get("chain_identity") or {}
    if identity.get("canonical") != "solana":
        issues.append(f"expected benchmark flow to resume and accept solana, got {identity}")
    if state.get("target_mode") != "fake-node":
        issues.append(f"expected fake-node target after returning to benchmark, got {state.get('target_mode')}")
    return issues


def _assert_existing_chain_custom_rpc_replace_defaults(state: dict[str, Any], transcript: str) -> list[str]:
    issues: list[str] = []
    custom = state.get("custom_rpc") or {}
    workload = state.get("workload") or {}
    identity = state.get("chain_identity") or {}
    if identity.get("canonical") != "bsc":
        issues.append(f"expected bsc, got {identity}")
    if custom.get("status") != "validated":
        issues.append(f"expected custom RPC validated, got {custom}")
    if custom.get("scope") != "mixed_replace":
        issues.append(f"expected mixed_replace scope, got {custom.get('scope')!r}")
    if custom.get("weights") != {"eth_blockNumber": 100}:
        issues.append(f"expected custom weights only, got {custom.get('weights')!r}")
    if workload.get("replace_defaults") is not True:
        issues.append(f"expected job-local workload to replace defaults, got {workload}")
    if workload.get("mixed_weights") != {"eth_blockNumber": 100}:
        issues.append(f"expected workload mixed weights to only include custom method, got {workload.get('mixed_weights')!r}")
    if "Endpoint validation passed" not in transcript:
        issues.append("custom endpoint validation did not pass in transcript")
    if "Method/schema validation passed" not in transcript:
        issues.append("custom method schema validation did not pass in transcript")
    return issues


def _assert_new_chain_existing_family_custom_rpc_mixed_weights(state: dict[str, Any], transcript: str) -> list[str]:
    issues: list[str] = []
    identity = state.get("chain_identity") or {}
    workload = state.get("workload") or {}
    pending = state.get("pending_question") or {}
    if identity.get("canonical") != "abcd":
        issues.append(f"expected new chain abcd to remain selected, got {identity}")
    if identity.get("status") != "existing_family_runtime_choice":
        issues.append(f"expected new chain to reach runtime choice after weights, got {identity}")
    if state.get("rpc_mode") != "mixed":
        issues.append(f"expected rpc_mode mixed, got {state.get('rpc_mode')!r}")
    if workload.get("mixed_weights") != {"eth_blockNumber": 70, "eth_chainId": 30}:
        issues.append(f"expected validated mixed weights, got {workload}")
    if workload.get("replace_defaults") is not True or workload.get("job_local_override") is not True:
        issues.append(f"expected job-local replacement workload, got {workload}")
    if pending.get("id") != "new_chain_runtime_choice":
        issues.append(f"expected runtime choice after new-chain workload weights, got {pending}")
    if "新链 RPC method 已验证通过" not in transcript and "New-chain endpoint and method/schema validation passed" not in transcript:
        issues.append("new-chain method validation success was not shown in transcript")
    return issues


def _assert_group_jump_qps_then_default_resume(state: dict[str, Any], transcript: str) -> list[str]:
    issues: list[str] = []
    qps = state.get("qps_profile") or {}
    pending = state.get("pending_question") or {}
    if qps.get("mode") != "quick" or qps.get("confirmed") is not True:
        issues.append(f"expected quick QPS defaults confirmed after group jump, got {qps}")
    if state.get("active_group") != "provider_deployment":
        issues.append(f"expected fallback to provider_deployment after QPS group completion, got {state.get('active_group')}")
    if pending.get("id") != "CLOUD_REGION":
        issues.append(f"expected next blocking question CLOUD_REGION, got {pending}")
    if "Choose benchmark mode" not in transcript:
        issues.append("QPS group was not activated from free-form user jump")
    return issues


def _assert_interrupted_workload_qps_then_back_to_rpc(state: dict[str, Any], transcript: str) -> list[str]:
    issues: list[str] = []
    qps = state.get("qps_profile") or {}
    pending = state.get("pending_question") or {}
    if qps.get("mode") != "quick" or qps.get("confirmed") is not True:
        issues.append(f"expected QPS group to remain confirmed after returning to RPC, got {qps}")
    if state.get("active_group") != "workload_rpc":
        issues.append(f"expected active group workload_rpc, got {state.get('active_group')}")
    if pending.get("id") != "workload_confirm":
        issues.append(f"expected workload_confirm after returning to RPC config, got {pending}")
    if "Use the current chain template default workload for bsc" not in transcript:
        issues.append("RPC workload question was not restored in transcript")
    return issues


def _assert_chain_change_decline_restores_interrupted_group(state: dict[str, Any], transcript: str) -> list[str]:
    issues: list[str] = []
    identity = state.get("chain_identity") or {}
    pending = state.get("pending_question") or {}
    if identity.get("canonical") != "bsc":
        issues.append(f"declined chain change should keep bsc, got {identity}")
    if state.get("active_group") != "workload_rpc":
        issues.append(f"declined chain change should restore workload_rpc, got {state.get('active_group')}")
    if pending.get("id") != "workload_confirm":
        issues.append(f"declined chain change should re-ask workload_confirm, got {pending}")
    if "Confirm switching from `bsc` to `ethereum`" not in transcript:
        issues.append("chain change confirmation did not appear before decline")
    return issues


def _assert_chain_change_accept_preserves_environment_and_invalidates_workload(state: dict[str, Any], transcript: str) -> list[str]:
    issues: list[str] = []
    confirmed = state.get("confirmed_config") or {}
    identity = state.get("chain_identity") or {}
    workload = state.get("workload") or {}
    pending = state.get("pending_question") or {}
    if identity.get("canonical") != "ethereum":
        issues.append(f"accepted chain change should switch to ethereum, got {identity}")
    if confirmed.get("CLOUD_REGION") != "us-1" or confirmed.get("LEDGER_DEVICE") != "vda":
        issues.append(f"chain change should preserve reusable environment values, got {confirmed}")
    if workload.get("confirmed"):
        issues.append(f"chain change should invalidate old workload confirmation, got {workload}")
    if pending.get("group") != "workload_rpc":
        issues.append(f"after chain switch and preserved env, expected workload_rpc question, got {pending}")
    if "Confirm switching from `bsc` to `ethereum`" not in transcript:
        issues.append("chain change confirmation did not appear before accept")
    return issues


def _assert_rpc_setup_switches_to_sync_observe(state: dict[str, Any], transcript: str) -> list[str]:
    issues: list[str] = []
    pending = state.get("pending_question") or {}
    if state.get("target_mode") != "sync-observe":
        issues.append(f"expected target_mode sync-observe, got {state.get('target_mode')}")
    if state.get("workflow_mode") != "sync_observe":
        issues.append(f"expected workflow_mode sync_observe, got {state.get('workflow_mode')}")
    if state.get("rpc_mode"):
        issues.append(f"RPC mode should not remain active after switching to sync-observe: {state.get('rpc_mode')}")
    if state.get("workload", {}).get("confirmed"):
        issues.append(f"RPC workload should not be confirmed after switching to sync-observe: {state.get('workload')}")
    if pending.get("id") != "sync_observe_source":
        issues.append(f"expected sync_observe_source after target-mode switch, got {pending}")
    if "sync-observe" not in transcript:
        issues.append("sync-observe switch did not appear in transcript")
    return issues


def _assert_multi_demand_action_queue(state: dict[str, Any], transcript: str) -> list[str]:
    issues: list[str] = []
    identity = state.get("chain_identity") or {}
    qps = state.get("qps_profile") or {}
    observability = state.get("observability") or {}
    pending = state.get("pending_question") or {}
    if state.get("target_mode") != "fake-node":
        issues.append(f"expected fake-node target mode, got {state.get('target_mode')}")
    if identity.get("canonical") != "bsc":
        issues.append(f"expected bsc from BNB mention, got {identity}")
    if state.get("rpc_mode") != "mixed":
        issues.append(f"expected mixed rpc mode, got {state.get('rpc_mode')}")
    if qps.get("mode") != "quick":
        issues.append(f"expected quick qps mode, got {qps}")
    if observability.get("mode") != "local":
        issues.append(f"expected local observability from Grafana mention, got {observability}")
    if pending.get("id") != "qps_profile_confirm":
        issues.append(f"expected explicit quick QPS to require default-profile confirmation, got {pending}")
    completed = state.get("completed_actions") or []
    if len(completed) < 3:
        issues.append(f"expected multiple completed queued actions, got {completed}")
    if "默认 QPS 配置" not in transcript and "default QPS profile" not in transcript:
        issues.append("transcript did not ask for QPS default-profile confirmation")
    return issues


def _assert_multi_demand_without_explicit_target_mode(state: dict[str, Any], transcript: str) -> list[str]:
    issues: list[str] = []
    identity = state.get("chain_identity") or {}
    pending = state.get("pending_question") or {}
    if state.get("target_mode") == "real-node":
        issues.append("target mode must not default to real-node when the user only says benchmark/test")
    if identity.get("canonical") != "bsc":
        issues.append(f"expected bsc from BNB mention, got {identity}")
    if pending.get("id") != "opening_next_action":
        issues.append(f"expected Harness to ask target mode instead of defaulting, got {pending}")
    if not state.get("action_queue"):
        issues.append("expected remaining queued actions to be preserved until target mode is confirmed")
    return issues


def _assert_capability_plus_uncertain_benchmark_goal(state: dict[str, Any], transcript: str) -> list[str]:
    issues: list[str] = []
    identity = state.get("chain_identity") or {}
    pending = state.get("pending_question") or {}
    if "36" not in transcript or "RPC" not in transcript:
        issues.append("capability summary was not shown for the support/capability part of the request")
    if "{'chain':" in transcript or '"chain":' in transcript:
        issues.append("capability summary leaked raw chain dictionaries")
    if identity.get("canonical") != "bsc":
        issues.append(f"expected BNB mention to be preserved as bsc, got {identity}")
    if state.get("target_mode"):
        issues.append(f"target mode must remain unconfirmed when user says they are unsure, got {state.get('target_mode')}")
    if pending.get("id") != "opening_next_action":
        issues.append(f"expected Harness to continue by asking target mode, got {pending}")
    completed = state.get("completed_actions") or []
    completed_types = [item.get("type") for item in completed]
    if "ask_capabilities" not in completed_types or "choose_chain" not in completed_types:
        issues.append(f"expected capability and chain actions to both complete, got {completed}")
    return issues


def _assert_multiline_config_inference_review(state: dict[str, Any], transcript: str) -> list[str]:
    issues: list[str] = []
    confirmed = state.get("confirmed_config") or {}
    inferred = state.get("inferred_config") or {}
    expected = {
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
    }
    for key, value in expected.items():
        if str(confirmed.get(key) or "") != value:
            issues.append(f"expected {key}={value}, got {confirmed.get(key)!r}")
    if not inferred.get("accepted_reviews"):
        issues.append(f"expected accepted inferred config review, got {inferred}")
    if "CLOUD_REGION" not in transcript or "NETWORK_MAX_BANDWIDTH_GBPS" not in transcript:
        issues.append("inferred config review did not display mapped values")
    pending = state.get("pending_question") or {}
    if pending.get("id") != "has_accounts_device":
        issues.append(f"expected fallback to accounts disk question after inferred config, got {pending}")
    return issues


def _assert_partial_pasted_config_resumes_missing_groups(state: dict[str, Any], transcript: str) -> list[str]:
    issues: list[str] = []
    confirmed = state.get("confirmed_config") or {}
    for key, value in {
        "CLOUD_REGION": "asia-east1",
        "CLOUD_ZONE": "asia-east1-c",
        "MACHINE_TYPE": "n2-standard-16",
        "NETWORK_INTERFACE": "eth0",
        "NETWORK_MAX_BANDWIDTH_GBPS": "100",
    }.items():
        if str(confirmed.get(key) or "") != value:
            issues.append(f"expected {key}={value}, got {confirmed.get(key)!r}")
    if state.get("rpc_mode") != "single":
        issues.append(f"expected RPC_MODE proposal to set rpc_mode=single, got {state.get('rpc_mode')!r}")
    pending = state.get("pending_question") or {}
    if pending.get("id") != "LEDGER_DEVICE":
        issues.append(f"expected fallback to missing ledger disk after partial config, got {pending}")
    if "unrelated_ticket" not in transcript:
        issues.append("unmapped copied value was not shown for user confirmation")
    return issues


def _assert_single_turn_multi_group_with_pasted_config(state: dict[str, Any], transcript: str) -> list[str]:
    issues: list[str] = []
    confirmed = state.get("confirmed_config") or {}
    identity = state.get("chain_identity") or {}
    qps = state.get("qps_profile") or {}
    observability = state.get("observability") or {}
    if state.get("target_mode") != "fake-node":
        issues.append(f"expected fake-node target mode, got {state.get('target_mode')}")
    if identity.get("canonical") != "bsc":
        issues.append(f"expected bsc from BNB mention, got {identity}")
    if state.get("rpc_mode") != "mixed":
        issues.append(f"expected mixed rpc mode, got {state.get('rpc_mode')}")
    if qps.get("mode") != "quick":
        issues.append(f"expected quick qps mode, got {qps}")
    if observability.get("mode") != "local":
        issues.append(f"expected local observability from Grafana mention, got {observability}")
    for key, value in {
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
    }.items():
        if str(confirmed.get(key) or "") != value:
            issues.append(f"expected {key}={value}, got {confirmed.get(key)!r}")
    pending = state.get("pending_question") or {}
    if pending.get("id") != "has_accounts_device":
        issues.append(f"expected accounts disk question after full pasted env config, got {pending}")
    if "CLOUD_REGION" not in transcript or "DATA_VOL_MAX_IOPS" not in transcript:
        issues.append("config proposal review did not show core mapped values")
    return issues


def _assert_nested_detours_return_to_earliest_incomplete_group(state: dict[str, Any], transcript: str) -> list[str]:
    issues: list[str] = []
    confirmed = state.get("confirmed_config") or {}
    qps = state.get("qps_profile") or {}
    observability = state.get("observability") or {}
    pending = state.get("pending_question") or {}
    if confirmed.get("CLOUD_REGION") != "us-1":
        issues.append(f"expected partial provider group to keep CLOUD_REGION=us-1, got {confirmed}")
    if qps.get("mode") != "quick" or qps.get("confirmed") is not True:
        issues.append(f"expected QPS detour to remain confirmed, got {qps}")
    if observability.get("mode") != "local":
        issues.append(f"expected observability detour to remain local, got {observability}")
    if pending.get("id") != "CLOUD_ZONE":
        issues.append(f"expected fallback to earliest incomplete provider field CLOUD_ZONE, got {pending}")
    if "CLOUD_ZONE" not in transcript and "cloud zone" not in transcript.lower():
        issues.append("transcript did not show fallback to cloud zone")
    return issues


def _assert_first_turn_direct_goal_with_config_paste(state: dict[str, Any], transcript: str) -> list[str]:
    issues: list[str] = []
    confirmed = state.get("confirmed_config") or {}
    identity = state.get("chain_identity") or {}
    if state.get("target_mode") != "fake-node":
        issues.append(f"expected fake-node from first turn, got {state.get('target_mode')}")
    if identity.get("canonical") != "bsc":
        issues.append(f"expected bsc from first turn BNB, got {identity}")
    if state.get("rpc_mode") != "mixed":
        issues.append(f"expected mixed from first turn, got {state.get('rpc_mode')}")
    if (state.get("qps_profile") or {}).get("mode") != "quick":
        issues.append(f"expected quick from first turn, got {state.get('qps_profile')}")
    if (state.get("observability") or {}).get("mode") != "local":
        issues.append(f"expected local observability from first turn, got {state.get('observability')}")
    for key in (
        "CLOUD_REGION",
        "CLOUD_ZONE",
        "MACHINE_TYPE",
        "LEDGER_DEVICE",
        "DATA_VOL_TYPE",
        "DATA_VOL_SIZE",
        "DATA_VOL_MAX_IOPS",
        "DATA_VOL_MAX_THROUGHPUT",
        "NETWORK_INTERFACE",
        "NETWORK_MAX_BANDWIDTH_GBPS",
    ):
        if not confirmed.get(key):
            issues.append(f"expected first-turn pasted config to confirm {key}, got {confirmed}")
    pending = state.get("pending_question") or {}
    if pending.get("id") != "has_accounts_device":
        issues.append(f"expected accounts disk question after first-turn config, got {pending}")
    if "CLOUD_REGION" not in transcript or "DATA_VOL_MAX_IOPS" not in transcript:
        issues.append("first-turn config proposal review did not show core mapped values")
    return issues


def _assert_language_switch_and_group_jump_preserves_state(state: dict[str, Any], transcript: str) -> list[str]:
    issues: list[str] = []
    confirmed = state.get("confirmed_config") or {}
    qps = state.get("qps_profile") or {}
    pending = state.get("pending_question") or {}
    if state.get("language") != "zh":
        issues.append(f"expected language to switch to zh after Chinese turn, got {state.get('language')!r}")
    if confirmed.get("CLOUD_REGION") != "us-1":
        issues.append(f"expected CLOUD_REGION to survive QPS detour, got {confirmed}")
    if confirmed.get("CLOUD_ZONE") != "asia-east1-c":
        issues.append(f"expected CLOUD_ZONE after returning to provider group, got {confirmed}")
    if qps.get("mode") != "quick" or qps.get("confirmed") is not True:
        issues.append(f"expected quick QPS to remain confirmed, got {qps}")
    if pending.get("id") != "MACHINE_TYPE":
        issues.append(f"expected fallback to MACHINE_TYPE after cloud zone, got {pending}")
    if "默认 QPS 配置" not in transcript and "default QPS profile" not in transcript:
        issues.append("QPS default confirmation was not visible in transcript")
    return issues


def _assert_disk_pending_jump_to_qps_then_resume_disk(state: dict[str, Any], transcript: str) -> list[str]:
    issues: list[str] = []
    qps = state.get("qps_profile") or {}
    pending = state.get("pending_question") or {}
    confirmed = state.get("confirmed_config") or {}
    if qps.get("mode") != "quick" or qps.get("confirmed") is not True:
        issues.append(f"expected QPS detour to confirm quick, got {qps}")
    if pending.get("id") != "LEDGER_DEVICE":
        issues.append(f"expected fallback to resume ledger disk after QPS detour, got {pending}")
    if confirmed.get("CLOUD_REGION") != "asia-east1" or confirmed.get("NETWORK_INTERFACE") != "eth0":
        issues.append(f"expected pasted provider/network config to survive QPS detour, got {confirmed}")
    if "QPS" not in transcript and "qps" not in transcript:
        issues.append("transcript did not show QPS detour")
    return issues


def _assert_triple_detour_returns_to_earliest_incomplete_group(state: dict[str, Any], transcript: str) -> list[str]:
    issues: list[str] = []
    confirmed = state.get("confirmed_config") or {}
    qps = state.get("qps_profile") or {}
    observability = state.get("observability") or {}
    identity = state.get("chain_identity") or {}
    pending = state.get("pending_question") or {}
    if identity.get("canonical") != "ethereum":
        issues.append(f"expected accepted chain detour to switch to ethereum, got {identity}")
    if confirmed.get("CLOUD_REGION") != "us-1":
        issues.append(f"expected partial provider value to survive triple detour, got {confirmed}")
    if qps.get("mode") != "quick" or qps.get("confirmed") is not True:
        issues.append(f"expected quick QPS to remain confirmed through chain switch, got {qps}")
    if observability.get("mode") != "local":
        issues.append(f"expected local observability to remain confirmed, got {observability}")
    if pending.get("id") != "CLOUD_ZONE":
        issues.append(f"expected fallback to earliest incomplete CLOUD_ZONE, got {pending}")
    if "Confirm switching" not in transcript and "是否确认" not in transcript:
        issues.append("chain switch confirmation did not appear in transcript")
    return issues


def _assert_pending_question_error_paste_goes_to_evidence_analysis(state: dict[str, Any], transcript: str) -> list[str]:
    issues: list[str] = []
    evidence = state.get("evidence_buffer") or []
    pending = state.get("pending_question") or {}
    confirmed = state.get("confirmed_config") or {}
    if not evidence:
        issues.append("expected pasted traceback to be captured as evidence instead of answered as current pending question")
    if confirmed.get("CLOUD_REGION") and "Traceback" in str(confirmed.get("CLOUD_REGION")):
        issues.append(f"traceback polluted CLOUD_REGION: {confirmed.get('CLOUD_REGION')!r}")
    if pending:
        issues.append(f"expected evidence detour to stop the turn without asking a new config question, got {pending}")
    if "evidence" not in transcript.lower() and "证据" not in transcript:
        issues.append("transcript did not acknowledge evidence capture")
    return issues


def _assert_english_process_like_user_text_routes_normally(state: dict[str, Any], transcript: str) -> list[str]:
    issues: list[str] = []
    identity = state.get("chain_identity") or {}
    pending = state.get("pending_question") or {}
    if state.get("target_mode") != "fake-node":
        issues.append(f"expected explicit fake-node target mode, got {state.get('target_mode')}")
    if identity.get("canonical") != "bsc":
        issues.append(f"expected BNB to resolve to bsc, got {identity}")
    if "36" not in transcript or "RPC" not in transcript:
        issues.append("capability summary was not shown for the support request")
    if pending.get("id") != "CLOUD_REGION":
        issues.append(f"expected fallback to provider config after explicit fake-node+BNB, got {pending}")
    return issues


if __name__ == "__main__":
    raise SystemExit(main())

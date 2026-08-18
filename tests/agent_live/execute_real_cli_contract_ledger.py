#!/usr/bin/env python3
"""Execute reviewed non-semantic contract edges through the real Linux CLI."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.agent_live.coverage_evidence import (
    PtyCliTurnRecord,
    TurnObservation,
    build_pty_cli_evidence_artifact,
    content_hash,
    pty_transcript_hash,
    repository_revision,
    validate_runtime_turn_event,
    verify_runtime_postcondition,
    write_evidence_artifact,
)
from tests.agent_live.dynamic_dual_ai_chaos import (
    ChaosRunConfig,
    CompletionExpectation,
    JsonlRuntimeEventStream,
    JsonlTerminalOutcomeStream,
    SubprocessPtyTransport,
    WorkflowCompletion,
    _startup_resume_submission,
    validate_startup_terminal_protocol,
    validate_startup_session_event,
    wait_for_turn_completion,
)
from tests.agent_live.generate_harness_coverage_ledger import (
    build_ledger,
    contract_variant_hash,
)
from tests.agent_live.runtime_checkpoint import seed_runtime_checkpoint
from tests.agent_live.reviewed_execution_cases import reviewed_execution_case


DEFAULT_OUTPUT = Path(".agent/evidence/real-cli-contracts")


def eligible_edges(ledger: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return fixed real-CLI rows backed by one reviewed execution case."""

    return [
        edge
        for edge in ledger.get("edges") or ()
        if bool(((edge.get("evidence") or {}).get("real_cli") or {}).get("required"))
        and not bool(((edge.get("evidence") or {}).get("dynamic_dual_ai") or {}).get("required"))
        and len(edge.get("execution_case_ids") or ()) == 1
    ]


def select_edges(
    ledger: Mapping[str, Any],
    *,
    edge_keys: Iterable[str] = (),
    limit: int = 0,
    shard_index: int = 0,
    shard_count: int = 1,
) -> list[Mapping[str, Any]]:
    """Select a stable, disjoint subset without changing edge semantics."""

    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count")
    selected_keys = {str(item) for item in edge_keys if str(item)}
    edges = [
        edge for edge in eligible_edges(ledger)
        if not selected_keys or str(edge.get("edge_key") or "") in selected_keys
    ]
    edges = [
        edge for index, edge in enumerate(edges)
        if index % shard_count == shard_index
    ]
    if limit > 0:
        edges = edges[:limit]
    return edges


def execute_edge(
    *,
    repo_root: Path,
    edge: Mapping[str, Any],
    revision: Mapping[str, str],
    output_root: Path,
    timeout_seconds: float,
) -> Path:
    resolved = reviewed_execution_case(edge)
    if resolved is None:
        raise RuntimeError(f"real CLI edge has no reviewed execution case: {edge.get('edge_key')}")
    scenario, execution_case = resolved
    if execution_case.case_id not in set(edge.get("execution_case_ids") or ()):
        raise RuntimeError("resolved execution case is not bound to the ledger edge")
    if execution_case.descriptor_hash != str(edge.get("execution_case_hash") or ""):
        raise RuntimeError("resolved execution case hash does not match the ledger edge")
    seed_state = deepcopy(dict(scenario.seed_state or {}))
    user_input = execution_case.resolve_input(os.environ)
    _prepare_execution_case(execution_case.setup_capabilities, seed_state)
    session_id = "real-cli-" + content_hash(edge.get("edge_key"))[:20]
    runtime_root = output_root / "runtime" / session_id
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    runtime_root.mkdir(parents=True)

    config = ChaosRunConfig(
        repo_root=repo_root,
        command=(str(repo_root / "bin" / "anychain-agent"),),
        session_id=session_id,
        session_purpose="real-cli-coverage",
        runtime_root=runtime_root,
        runtime_root_in_process=runtime_root,
        response_timeout_seconds=timeout_seconds,
    )
    seed_receipt = seed_runtime_checkpoint(
        seed_state,
        checkpoint_path=runtime_root / "checkpoints.sqlite",
        session_id=session_id,
        session_purpose=config.session_purpose,
        scenario_id=scenario.scenario_id,
        scenario_state_fingerprint=scenario.state_fingerprint,
    )
    env = _runtime_environment(config, runtime_root)
    transport = SubprocessPtyTransport(config.command, cwd=repo_root)
    event_stream = JsonlRuntimeEventStream(runtime_root / "turn-events.jsonl")
    terminal_stream = JsonlTerminalOutcomeStream(
        runtime_root / "terminal-outcomes.jsonl",
        poll_interval_seconds=config.poll_interval_seconds,
    )
    transcript: list[str] = []
    terminal_stream.mark_process_start()
    event_stream.mark_process_start()
    transport.start(env=env)
    try:
        startup = transport.read_complete_agent_response(timeout_seconds=timeout_seconds)
        transcript.append(startup)
        startup_session = validate_startup_session_event(
            terminal_stream,
            expected_revision=revision,
            expected_provider=config.provider,
            expected_model=config.model,
            expected_session_id=session_id,
            expected_session_purpose=config.session_purpose,
            startup_response=startup,
        )
        provider = startup_session.provider
        model = startup_session.model
        runtime_history, startup_events = event_stream.capture_startup_snapshot()
        combined_events = (*runtime_history, *startup_events)
        if not combined_events:
            raise RuntimeError("real CLI contract requires a baseline runtime event")
        for historical_event in runtime_history:
            validate_runtime_turn_event(historical_event)
        baseline = combined_events[-1]
        startup_outcomes, startup_detours = (
            terminal_stream.capture_startup_snapshot()
        )
        validate_startup_terminal_protocol(
            startup_events,
            startup_outcomes,
            expected_revision=revision,
            session_event=startup_session,
            detours=startup_detours,
            baseline_event=baseline,
        )
        _require_revision(baseline.revision, revision)

        if baseline.pending_question_id != str(edge.get("question_id") or ""):
            resume_submission = _startup_resume_submission(
                baseline.pending_contract,
                desired="continue",
            )
            if not resume_submission:
                raise RuntimeError(
                    "startup did not expose the target or resume contract: "
                    f"{baseline.pending_question_id or '<none>'}"
                )
            transport.submit_bracketed_paste(resume_submission)
            resume_completion = wait_for_turn_completion(
                transport,
                event_stream,
                terminal_stream,
                timeout_seconds=timeout_seconds,
                expectation=CompletionExpectation(
                    submitted_input=resume_submission,
                    session_id=session_id,
                    session_purpose=config.session_purpose,
                    product_authority_id=startup_session.product_authority_id,
                    process_instance_id=startup_session.process_instance_id,
                    product_revision=int(baseline.product_revision),
                    product_checkpoint_thread_id=(
                        baseline.product_checkpoint_thread_id
                    ),
                    product_checkpoint_id=baseline.product_checkpoint_id,
                    product_fingerprint=baseline.after_fingerprint,
                ),
            )
            if not isinstance(resume_completion, WorkflowCompletion):
                raise RuntimeError(
                    "real CLI resume did not produce a workflow completion"
                )
            resumed_response = resume_completion.response
            transcript.extend((f"User> {resume_submission}", resumed_response))
            baseline = resume_completion.event
            _require_revision(baseline.revision, revision)

        _require_target_contract(edge, baseline.pending_contract)
        previous_response = transcript[-1]
        previous_received_ns = time.time_ns()
        submitted_ns = time.time_ns()
        transport.submit_bracketed_paste(user_input)
        completion = wait_for_turn_completion(
            transport,
            event_stream,
            terminal_stream,
            timeout_seconds=timeout_seconds,
            expectation=CompletionExpectation(
                submitted_input=user_input,
                session_id=session_id,
                session_purpose=config.session_purpose,
                product_authority_id=startup_session.product_authority_id,
                process_instance_id=startup_session.process_instance_id,
                product_revision=int(baseline.product_revision),
                product_checkpoint_thread_id=(
                    baseline.product_checkpoint_thread_id
                ),
                product_checkpoint_id=baseline.product_checkpoint_id,
                product_fingerprint=baseline.after_fingerprint,
            ),
        )
        if not isinstance(completion, WorkflowCompletion):
            raise RuntimeError(
                "real CLI edge did not produce a workflow completion"
            )
        response = completion.response
        response_received_ns = time.time_ns()
        committed = completion.event
        _require_revision(committed.revision, revision)
        transcript.extend((f"User> {user_input}", response))

        turn_index = committed.turn_index
        turn = PtyCliTurnRecord(
            session_id=session_id,
            turn_index=turn_index,
            previous_agent_response=previous_response,
            user_message=user_input,
            agent_response=response,
            provider=provider,
            model=model,
            before_fingerprint=committed.before_fingerprint,
            after_fingerprint=committed.after_fingerprint,
            transcript_hash=pty_transcript_hash(
                session_id=session_id,
                turn_index=turn_index,
                previous_agent_response=previous_response,
                user_message=user_input,
                agent_response=response,
            ),
            previous_response_received_at_ns=previous_received_ns,
            user_message_submitted_at_ns=submitted_ns,
            agent_response_received_at_ns=response_received_ns,
        )
        verified = verify_runtime_postcondition(edge, baseline, committed, turn)
        if not verified.passed:
            raise RuntimeError("; ".join(str(item) for item in verified.details.get("errors") or ()))
        observation = TurnObservation(
            seed=0,
            revision=dict(revision),
            target_edge_key=str(edge.get("edge_key") or ""),
            target_contract_hash=str(edge.get("contract_hash") or ""),
            target_variant_hash=str(edge.get("contract_variant_hash") or ""),
            prior_agent_response=previous_response,
            simulator_decision={},
            exact_user_turn=user_input,
            provider=provider,
            model=model,
            before_turn_index=baseline.turn_index,
            after_turn_index=committed.turn_index,
            before_state_fingerprint=committed.before_fingerprint,
            after_state_fingerprint=committed.after_fingerprint,
            pending_contract=dict(baseline.pending_contract),
            runtime_events=(baseline, committed),
            verified_postcondition=verified,
        )
        artifact = build_pty_cli_evidence_artifact(
            edge=edge,
            evidence_class="real_cli",
            revision=revision,
            turn=turn,
            observation=observation,
            execution_case=execution_case.descriptor,
            seed_receipt={
                **seed_receipt.payload,
                "receipt_hash": seed_receipt.receipt_hash,
            },
        )
        path = write_evidence_artifact(artifact, output_root / "evidence")
        (runtime_root / "transcript.txt").write_text(
            "\n".join(transcript).rstrip() + "\n",
            encoding="utf-8",
        )
        return path
    finally:
        transport.close()


def execute_edges(
    *,
    repo_root: Path,
    ledger: Mapping[str, Any],
    output_root: Path,
    edge_keys: Iterable[str] = (),
    limit: int = 0,
    shard_index: int = 0,
    shard_count: int = 1,
    timeout_seconds: float = 180.0,
) -> list[Path]:
    revision = dict(ledger.get("revision") or {})
    edges = select_edges(
        ledger,
        edge_keys=edge_keys,
        limit=limit,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    paths: list[Path] = []
    for index, edge in enumerate(edges, 1):
        print(f"[{index}/{len(edges)}] {edge['edge_key']}", flush=True)
        paths.append(execute_edge(
            repo_root=repo_root,
            edge=edge,
            revision=revision,
            output_root=output_root,
            timeout_seconds=timeout_seconds,
        ))
    return paths


def _runtime_environment(config: ChaosRunConfig, runtime_root: Path) -> dict[str, str]:
    process_root = config.runtime_root_in_process or runtime_root
    env = os.environ.copy()
    env.update({
        "ANYCHAIN_AGENT_CHECKPOINT_PATH": str(process_root / "checkpoints.sqlite"),
        "ANYCHAIN_AGENT_SESSION_ID": config.session_id,
        "ANYCHAIN_AGENT_SESSION_PURPOSE": config.session_purpose,
        "ANYCHAIN_AGENT_JOBS_DIR": str(process_root / "jobs"),
        "ANYCHAIN_AGENT_TURN_EVENT_FILE": str(process_root / "turn-events.jsonl"),
        "ANYCHAIN_AGENT_TERMINAL_OUTCOME_FILE": str(
            process_root / "terminal-outcomes.jsonl"
        ),
    })
    return config.isolated_environment(env)


def _prepare_execution_case(
    capabilities: Iterable[str],
    seed_state: Mapping[str, Any],
) -> None:
    requested = set(capabilities)
    unknown = requested - {"prepared_real_node_plan", "local_jsonrpc_endpoint"}
    if unknown:
        raise RuntimeError(f"unsupported execution-case setup capabilities: {sorted(unknown)}")
    if "prepared_real_node_plan" not in requested:
        return
    plan_file = Path(str(seed_state.get("plan_file") or ""))
    if not plan_file.is_absolute():
        raise RuntimeError("prepared_real_node_plan requires an absolute plan_file")
    from agent.runners.benchmark_pipeline import prepare_benchmark_run

    with tempfile.TemporaryDirectory(prefix="anychain-real-cli-plan-") as output_dir:
        prepared = prepare_benchmark_run(
            source_prompt="controlled real CLI final benchmark coverage",
            chain="ethereum",
            goal="smoke",
            rpc_mode="single",
            use_fake_node=False,
            target_rpc_url="http://geth-dev:8545",
            mainnet_rpc_url_reviewed=True,
            blockchain_process_names=["geth"],
            deployment_type="container",
            cloud_provider="other",
            cloud_region="test-region",
            cloud_zone="test-zone",
            machine_type="test-machine",
            ledger_device="vda",
            data_vol_type="ssd",
            data_vol_size="1",
            data_vol_max_iops="1",
            data_vol_max_throughput="1",
            network_interface="eth0",
            network_max_bandwidth_gbps="1",
            qps_initial=1,
            qps_max=1,
            qps_step=1,
            duration_seconds=3,
            rpc_methods=["eth_blockNumber"],
            observability_enabled=False,
            observability_mode="local",
            observability_auto_stop=True,
            confirmations=[
                "chain_template_reviewed",
                "rpc_param_samples_confirmed",
                "rpc_workload_confirmed",
                "qps_profile_confirmed",
                "observability_choice_confirmed",
                "advanced_config_review",
                "disk_inventory_confirmation",
                "ledger_device_confirmation",
                "benchmark_mode_confirmed",
                "has_accounts_device",
            ],
            output_dir=output_dir,
        )
        data = dict(prepared.get("data") or {})
        generated_path = Path(str(data.get("plan_file") or ""))
        plan = data.get("plan")
        preflight = dict(data.get("preflight") or {})
        if (
            prepared.get("status") != "ok"
            or not preflight.get("passed")
            or not isinstance(plan, Mapping)
            or not str(plan.get("plan_id") or "")
            or not generated_path.is_file()
        ):
            raise RuntimeError("product plan preparation did not produce an approved executable plan")
        serialized = generated_path.read_text(encoding="utf-8")
    plan_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = plan_file.with_suffix(plan_file.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(plan_file)


def _require_revision(observed: Mapping[str, str], expected: Mapping[str, str]) -> None:
    if dict(observed) != dict(expected):
        raise RuntimeError("runtime event repository revision mismatch")


def _require_target_contract(edge: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    if str(contract.get("id") or "") != str(edge.get("question_id") or ""):
        raise RuntimeError("real CLI baseline question does not match the target edge")
    if contract_variant_hash(contract) != str(edge.get("contract_hash") or ""):
        raise RuntimeError("real CLI baseline contract hash does not match the target edge")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--edge-key", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    repo_root = REPO_ROOT
    ledger_path = Path(args.ledger)
    existing = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger = build_ledger(existing=existing, revision=repository_revision(repo_root))
    paths = execute_edges(
        repo_root=repo_root,
        ledger=ledger,
        output_root=Path(args.output),
        edge_keys=args.edge_key,
        limit=args.limit,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        timeout_seconds=args.timeout,
    )
    print(json.dumps({"executed": len(paths), "evidence": [str(path) for path in paths]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

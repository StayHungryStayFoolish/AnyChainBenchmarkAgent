#!/usr/bin/env python3
"""Execute reviewed non-semantic contract edges through the real Linux CLI."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.harness.checkpoints import create_sqlite_checkpointer
from agent.harness.graph import build_graph
from agent.harness.invariants import validate_state
from agent.harness.state import project_checkpoint_state
from tests.agent_live.coverage_evidence import (
    PtyCliTurnRecord,
    TurnObservation,
    build_pty_cli_evidence_artifact,
    content_hash,
    pty_transcript_hash,
    repository_revision,
    verify_runtime_postcondition,
    write_evidence_artifact,
)
from tests.agent_live.dynamic_dual_ai_chaos import (
    ChaosRunConfig,
    JsonlRuntimeEventStream,
    SubprocessPtyTransport,
    _provider_model_from_startup,
)
from tests.agent_live.generate_harness_coverage_ledger import (
    build_ledger,
    contract_variant_hash,
)


DEFAULT_OUTPUT = Path(".agent/evidence/real-cli-contracts")


def eligible_edges(ledger: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return rows whose reviewed deterministic artifact supplies seed and input."""

    return [
        edge
        for edge in ledger.get("edges") or ()
        if bool(((edge.get("evidence") or {}).get("real_cli") or {}).get("required"))
        and not bool(((edge.get("evidence") or {}).get("dynamic_dual_ai") or {}).get("required"))
        and str(edge.get("deterministic_test_id") or "").strip()
    ]


def execute_edge(
    *,
    repo_root: Path,
    edge: Mapping[str, Any],
    revision: Mapping[str, str],
    output_root: Path,
    timeout_seconds: float,
) -> Path:
    deterministic = _load_deterministic_reference(edge)
    turn_evidence = dict(deterministic.get("turn_evidence") or {})
    seed_state = deepcopy(dict(turn_evidence.get("seed") or {}))
    user_input = str(turn_evidence.get("input") if "input" in turn_evidence else "")
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
    _seed_checkpoint(
        seed_state,
        checkpoint_path=runtime_root / "checkpoints.sqlite",
        session_id=session_id,
        session_purpose=config.session_purpose,
    )
    env = _runtime_environment(config, runtime_root)
    transport = SubprocessPtyTransport(config.command, cwd=repo_root)
    event_stream = JsonlRuntimeEventStream(runtime_root / "turn-events.jsonl")
    transcript: list[str] = []
    transport.start(env=env)
    try:
        startup = transport.read_complete_agent_response(timeout_seconds=timeout_seconds)
        transcript.append(startup)
        provider, model = _provider_model_from_startup(startup)
        baseline = event_stream.baseline()
        _require_revision(baseline.revision, revision)

        if baseline.pending_question_id != str(edge.get("question_id") or ""):
            if baseline.pending_question_id != "resume_harness_session":
                raise RuntimeError(
                    "startup did not expose the target or resume contract: "
                    f"{baseline.pending_question_id or '<none>'}"
                )
            transport.submit_bracketed_paste("1")
            resumed_response = transport.read_complete_agent_response(timeout_seconds=timeout_seconds)
            transcript.extend(("User> 1", resumed_response))
            baseline = event_stream.next_event(timeout_seconds=timeout_seconds)
            _require_revision(baseline.revision, revision)

        _require_target_contract(edge, baseline.pending_contract)
        previous_response = transcript[-1]
        previous_received_ns = time.time_ns()
        submitted_ns = time.time_ns()
        transport.submit_bracketed_paste(user_input)
        response = transport.read_complete_agent_response(timeout_seconds=timeout_seconds)
        response_received_ns = time.time_ns()
        committed = event_stream.next_event(timeout_seconds=timeout_seconds)
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
    timeout_seconds: float = 180.0,
) -> list[Path]:
    revision = dict(ledger.get("revision") or {})
    selected_keys = {str(item) for item in edge_keys if str(item)}
    edges = [
        edge for edge in eligible_edges(ledger)
        if not selected_keys or str(edge.get("edge_key") or "") in selected_keys
    ]
    if limit > 0:
        edges = edges[:limit]
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


def _load_deterministic_reference(edge: Mapping[str, Any]) -> Mapping[str, Any]:
    path = Path(str(edge.get("deterministic_test_id") or ""))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("evidence_class") != "deterministic":
        raise ValueError("real CLI seed reference is not deterministic evidence")
    if payload.get("edge_key") != edge.get("edge_key"):
        raise ValueError("real CLI seed reference belongs to another edge")
    return payload


def _seed_checkpoint(
    seed_state: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    session_id: str,
    session_purpose: str,
) -> None:
    state = deepcopy(dict(seed_state))
    state["last_user_input"] = ""
    state["thread_id"] = session_id
    state["session"] = {
        "id": session_id,
        "purpose": session_purpose,
        "created_at": "1970-01-01T00:00:00Z",
        "updated_at": "1970-01-01T00:00:00Z",
    }
    validate_state(state)
    checkpointer = create_sqlite_checkpointer(checkpoint_path)
    try:
        graph = build_graph(checkpointer)
        graph.update_state(
            {"configurable": {"thread_id": session_id}},
            project_checkpoint_state(state),
        )
    finally:
        manager = getattr(checkpointer, "_anychain_context_manager", None)
        if manager is not None:
            manager.__exit__(None, None, None)


def _runtime_environment(config: ChaosRunConfig, runtime_root: Path) -> dict[str, str]:
    process_root = config.runtime_root_in_process or runtime_root
    env = os.environ.copy()
    env.update({
        "ANYCHAIN_AGENT_CHECKPOINT_PATH": str(process_root / "checkpoints.sqlite"),
        "ANYCHAIN_AGENT_SESSION_ID": config.session_id,
        "ANYCHAIN_AGENT_SESSION_PURPOSE": config.session_purpose,
        "ANYCHAIN_AGENT_JOBS_DIR": str(process_root / "jobs"),
        "ANYCHAIN_AGENT_TURN_EVENT_FILE": str(process_root / "turn-events.jsonl"),
    })
    return env


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
        timeout_seconds=args.timeout,
    )
    print(json.dumps({"executed": len(paths), "evidence": [str(path) for path in paths]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

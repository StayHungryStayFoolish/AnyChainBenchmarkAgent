#!/usr/bin/env python3
"""Interactive bridge between the active Codex session and dynamic Chaos.

The bridge does not generate user messages.  It emits a framed simulator
context after the real CLI has produced a complete Agent response, then blocks
until an external Codex actor submits one framed decision for that response.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from tests.agent_live.chaos_scheduler import build_chaos_schedule
from tests.agent_live.coverage_evidence import content_hash, repository_revision
from tests.agent_live.dynamic_dual_ai_chaos import (
    ChaosRunConfig,
    DynamicDualAiChaosRunner,
    SimulatorContext,
    SimulatorDecision,
)
from tests.agent_live.generate_harness_coverage_ledger import build_ledger


CONTEXT_FRAME = "CODEX_SIMULATOR_CONTEXT "
DECISION_FRAME = "CODEX_SIMULATOR_DECISION "
RESULT_FRAME = "CODEX_SIMULATOR_RESULT "


class StdioCodexSimulator:
    """Exchange one revision-bound decision per complete Agent response."""

    def __init__(self, input_stream: Any = sys.stdin, output_stream: Any = sys.stdout) -> None:
        self.input_stream = input_stream
        self.output_stream = output_stream

    def __call__(self, context: SimulatorContext) -> SimulatorDecision:
        response_hash = content_hash(context.previous_agent_response)
        target = context.scheduled_target
        decision_template = {
            "previous_response_hash": response_hash,
            "user_message": "<one dynamically selected user message>",
            "persona": target.persona,
            "goal": target.goal,
            "rationale": "<why this message follows from the complete Agent response>",
            "target_coverage_ids": [target.edge_key],
        }
        payload = {
            "schema_version": 1,
            "session_id": context.session_id,
            "turn_index": context.turn_index,
            "previous_agent_response": context.previous_agent_response,
            "previous_response_hash": response_hash,
            "previous_response_received_at_ns": context.previous_response_received_at_ns,
            "scheduled_target": asdict(context.scheduled_target),
            "transcript": [list(item) for item in context.transcript],
            "decision_contract": {
                "frame_prefix": DECISION_FRAME,
                "schema_version": 1,
                "required_keys": list(decision_template),
                "response_binding_key": "previous_response_hash",
                "response_binding_value": response_hash,
                "immutable_persona": target.persona,
                "immutable_goal": target.goal,
                "coverage_rule": (
                    "Declare only coverage IDs genuinely demanded by user_message; "
                    "the scheduled edge must remain among them."
                ),
                "output_rule": (
                    "Write exactly one line beginning with frame_prefix followed by "
                    "one JSON object; do not write plain user prose or Markdown."
                ),
                "decision_template": decision_template,
            },
        }
        self.output_stream.write(CONTEXT_FRAME + json.dumps(payload, ensure_ascii=False) + "\n")
        self.output_stream.flush()

        line = self.input_stream.readline()
        if not line:
            raise RuntimeError("external Codex simulator closed before providing a decision")
        if not line.startswith(DECISION_FRAME):
            raise RuntimeError("external Codex simulator decision used an invalid frame")
        try:
            decision = json.loads(line[len(DECISION_FRAME):])
        except json.JSONDecodeError as exc:
            raise RuntimeError("external Codex simulator decision is not valid JSON") from exc
        if not isinstance(decision, Mapping):
            raise RuntimeError("external Codex simulator decision must be a JSON object")
        if str(decision.get("previous_response_hash") or "") != response_hash:
            raise RuntimeError("external Codex simulator decision references a stale Agent response")

        return SimulatorDecision(
            user_message=_required_text(decision, "user_message"),
            persona=_required_text(decision, "persona"),
            goal=_required_text(decision, "goal"),
            rationale=_required_text(decision, "rationale"),
            target_coverage_ids=tuple(
                str(item) for item in decision.get("target_coverage_ids") or (target.edge_key,)
            ),
        )


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise RuntimeError(f"external Codex simulator decision requires {key}")
    return value


def _load_targets(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_targets = payload.get("targets") if isinstance(payload, Mapping) else payload
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("target file must contain a non-empty JSON target list")
    if not all(isinstance(item, Mapping) for item in raw_targets):
        raise ValueError("every target must be a JSON object")
    return [dict(item) for item in raw_targets]


def run_bridge(
    *,
    repo_root: Path,
    targets: Sequence[Mapping[str, Any]],
    seed: int,
    session_id: str,
    service: str,
    runtime: str,
    input_stream: Any = sys.stdin,
    output_stream: Any = sys.stdout,
) -> Mapping[str, Any]:
    revision = repository_revision(repo_root)
    ledger = build_ledger(revision=revision)
    schedule = build_chaos_schedule(
        ledger,
        revision=revision,
        seed=seed,
        targets=targets,
    )
    if runtime == "linux":
        config = ChaosRunConfig.linux(
            repo_root,
            session_id=session_id,
            max_turns=len(schedule.targets),
        )
    elif runtime == "docker":
        config = ChaosRunConfig.docker(
            repo_root,
            service=service,
            session_id=session_id,
            max_turns=len(schedule.targets),
        )
    else:
        raise ValueError(f"unsupported bridge runtime: {runtime}")
    runner = DynamicDualAiChaosRunner(
        config,
        StdioCodexSimulator(input_stream, output_stream),
        ledger=ledger,
        schedule=schedule,
        revision=revision,
    )
    result = runner.run()
    payload = {
        "session_id": result.session_id,
        "execution_status": result.execution_status,
        "schedule_path": str(result.schedule_path),
        "schedule_result_path": str(result.schedule_result_path),
        "transcript_path": str(result.transcript_path),
        "evidence_paths": [str(path) for path in result.evidence_paths],
    }
    output_stream.write(RESULT_FRAME + json.dumps(payload, ensure_ascii=False) + "\n")
    output_stream.flush()
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--service", default="bench")
    parser.add_argument("--runtime", choices=("docker", "linux"), default="docker")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run_bridge(
        repo_root=args.repo_root.resolve(),
        targets=_load_targets(args.targets),
        seed=args.seed,
        session_id=args.session_id,
        service=args.service,
        runtime=args.runtime,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

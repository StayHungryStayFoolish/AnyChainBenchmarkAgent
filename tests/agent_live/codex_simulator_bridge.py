#!/usr/bin/env python3
"""Interactive bridge between the active Codex session and dynamic Chaos.

The bridge does not generate user messages.  It emits a framed simulator
context after the real CLI has produced a complete Agent response, then blocks
until an external Codex actor submits one framed decision for that response.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent.utils.redaction import redact
from tests.agent_live.chaos_scheduler import build_chaos_schedule
from tests.agent_live.coverage_evidence import content_hash, repository_revision
from tests.agent_live.dynamic_dual_ai_chaos import (
    ChaosRunConfig,
    DynamicDualAiChaosRunner,
    SimulatorContext,
    SimulatorDecision,
    SimulatorDecisionInvalid,
    SimulatorTerminalClassification,
)
from tests.agent_live.generate_harness_coverage_ledger import build_ledger


CONTEXT_FRAME = "CODEX_SIMULATOR_CONTEXT "
DECISION_FRAME = "CODEX_SIMULATOR_DECISION "
RESULT_FRAME = "CODEX_SIMULATOR_RESULT "
DEFAULT_DECISION_TIMEOUT_SECONDS = 300.0


class CodexSimulatorBridgeError(RuntimeError):
    """Base error raised at the external Codex decision boundary."""


class CodexSimulatorExternalBlockError(CodexSimulatorBridgeError):
    """The external simulator did not deliver a decision."""


class CodexSimulatorDecisionTimeout(CodexSimulatorExternalBlockError, TimeoutError):
    """The external simulator exceeded its decision deadline."""


class CodexSimulatorInputClosed(CodexSimulatorExternalBlockError):
    """The external simulator input closed before a decision arrived."""


class CodexSimulatorProtocolError(CodexSimulatorBridgeError):
    """The external simulator supplied a decision that violates the frame contract."""


class CodexSimulatorInvalidFrame(CodexSimulatorProtocolError):
    """The decision did not use the required frame prefix."""


class CodexSimulatorInvalidJson(CodexSimulatorProtocolError):
    """The framed decision was not a valid JSON object."""


class CodexSimulatorStaleResponse(CodexSimulatorProtocolError):
    """The decision was bound to a different Agent response."""


def _input_generation_rule(contract: Mapping[str, Any]) -> str:
    input_class = str(contract.get("input_class") or "")
    action_type = str(contract.get("action_type") or "")
    postcondition = contract.get("expected_postcondition")
    expected_field = ""
    if isinstance(postcondition, Mapping):
        expected_field = str(postcondition.get("field") or "")
    if input_class == "structured_json_yaml_env_curl" and action_type == "propose_config_values":
        return (
            "The user_message must supply a concrete value for "
            f"{expected_field or 'the expected configuration field'} in a parseable "
            "JSON, YAML, env, shell, curl, or mixed configuration block. It must not ask "
            "the Agent to invent, recommend, or output that user-owned value."
        )
    action_contract = contract.get("simulator_action_contract")
    if isinstance(action_contract, Mapping):
        return (
            "The user_message must directly and consistently request the exact declared action purpose, "
            "supply every user-selected value required by required_arguments, and remain compatible with "
            "the declared effect and constraints. It must not substitute another domain purpose, contradict "
            "the requested state change, or claim the scheduled coverage ID merely because the action name "
            "sounds related. Audit this before submitting the turn. Contract: "
            + json.dumps(dict(action_contract), ensure_ascii=False, sort_keys=True)
        )
    return (
        "The user_message must genuinely exercise coverage_contract.input_class and "
        "supply the evidence required by its action_type and expected_postcondition."
    )


class StdioCodexSimulator:
    """Exchange one revision-bound decision per complete Agent response."""

    def __init__(
        self,
        input_stream: Any = sys.stdin,
        output_stream: Any = sys.stdout,
        *,
        decision_timeout_seconds: float | None = None,
    ) -> None:
        self.input_stream = input_stream
        self.output_stream = output_stream
        self.decision_timeout_seconds = decision_timeout_seconds
        self._input_buffer = bytearray()

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
            "previous_agent_response": str(redact(context.previous_agent_response)),
            "previous_response_hash": response_hash,
            "previous_response_received_at_ns": context.previous_response_received_at_ns,
            "scheduled_target": asdict(context.scheduled_target),
            "coverage_contract": dict(context.coverage_contract),
            "transcript": [
                [str(redact(user)), str(redact(agent))]
                for user, agent in context.transcript
            ],
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
                "input_generation_rule": _input_generation_rule(context.coverage_contract),
                "output_rule": (
                    "Write exactly one line beginning with frame_prefix followed by "
                    "one JSON object; do not write plain user prose or Markdown."
                ),
                "decision_template": decision_template,
            },
        }
        self.output_stream.write(CONTEXT_FRAME + json.dumps(payload, ensure_ascii=False) + "\n")
        self.output_stream.flush()

        line = self._read_decision_line()
        if not line.startswith(DECISION_FRAME):
            raise CodexSimulatorInvalidFrame(
                "external Codex simulator decision used an invalid frame"
            )
        try:
            decision = json.loads(line[len(DECISION_FRAME):])
        except json.JSONDecodeError as exc:
            raise CodexSimulatorInvalidJson(
                "external Codex simulator decision is not valid JSON"
            ) from exc
        if not isinstance(decision, Mapping):
            raise CodexSimulatorInvalidJson(
                "external Codex simulator decision must be a JSON object"
            )
        if str(decision.get("previous_response_hash") or "") != response_hash:
            raise CodexSimulatorStaleResponse(
                "external Codex simulator decision references a stale Agent response"
            )

        return SimulatorDecision(
            user_message=_required_text(decision, "user_message"),
            persona=_required_text(decision, "persona"),
            goal=_required_text(decision, "goal"),
            rationale=_required_text(decision, "rationale"),
            target_coverage_ids=tuple(
                str(item) for item in decision.get("target_coverage_ids") or (target.edge_key,)
            ),
        )

    def _read_decision_line(self) -> str:
        if self.decision_timeout_seconds is None:
            line = self.input_stream.readline()
            if not line:
                raise CodexSimulatorInputClosed(
                    "external Codex simulator closed before providing a decision"
                )
            return str(line)

        timeout = float(self.decision_timeout_seconds)
        if timeout <= 0:
            raise ValueError("decision_timeout_seconds must be greater than zero")
        try:
            fd = int(self.input_stream.fileno())
        except (AttributeError, OSError, TypeError, ValueError):
            # In-memory streams used by callers and tests cannot block on I/O.
            line = self.input_stream.readline()
            if not line:
                raise CodexSimulatorInputClosed(
                    "external Codex simulator closed before providing a decision"
                )
            return str(line)

        deadline = time.monotonic() + timeout
        was_blocking = os.get_blocking(fd)
        os.set_blocking(fd, False)
        try:
            while True:
                buffered = self._pop_buffered_line()
                if buffered is not None:
                    return buffered
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CodexSimulatorDecisionTimeout(
                        f"external Codex simulator decision timed out after {timeout:.1f}s"
                    )
                ready, _, _ = select.select([fd], [], [], remaining)
                if not ready:
                    raise CodexSimulatorDecisionTimeout(
                        f"external Codex simulator decision timed out after {timeout:.1f}s"
                    )
                try:
                    chunk = os.read(fd, 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    raise CodexSimulatorInputClosed(
                        "external Codex simulator closed before providing a decision"
                    )
                self._input_buffer.extend(chunk)
        finally:
            try:
                os.set_blocking(fd, was_blocking)
            except OSError:
                pass

    def _pop_buffered_line(self) -> str | None:
        newline = self._input_buffer.find(b"\n")
        if newline < 0:
            return None
        raw = bytes(self._input_buffer[: newline + 1])
        del self._input_buffer[: newline + 1]
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CodexSimulatorInvalidFrame(
                "external Codex simulator decision is not valid UTF-8"
            ) from exc


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise CodexSimulatorProtocolError(f"external Codex simulator decision requires {key}")
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
    decision_timeout_seconds: float = DEFAULT_DECISION_TIMEOUT_SECONDS,
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
        StdioCodexSimulator(
            input_stream,
            output_stream,
            decision_timeout_seconds=decision_timeout_seconds,
        ),
        ledger=ledger,
        schedule=schedule,
        revision=revision,
    )
    try:
        result = runner.run()
        payload = {
            "session_id": result.session_id,
            "execution_status": result.execution_status,
            "terminal_classification": SimulatorTerminalClassification.PASSED.value,
            "failure_reason": "",
            "schedule_path": str(result.schedule_path),
            "schedule_result_path": str(result.schedule_result_path),
            "transcript_path": str(result.transcript_path),
            "evidence_paths": [str(path) for path in result.evidence_paths],
        }
    except (SimulatorDecisionInvalid, CodexSimulatorProtocolError) as exc:
        runtime_root = Path(config.runtime_root or (
            config.repo_root / ".agent" / "dynamic-chaos" / config.session_id
        )).resolve()
        payload = {
            "session_id": config.session_id,
            "execution_status": "incomplete",
            "terminal_classification": SimulatorTerminalClassification.SIMULATOR_INVALID.value,
            "failure_reason": str(redact(f"{type(exc).__name__}: {exc}")),
            "schedule_path": str(runtime_root / "schedule.json"),
            "schedule_result_path": str(runtime_root / "schedule-result.json"),
            "transcript_path": str(runtime_root / "transcript.txt"),
            "evidence_paths": [],
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
    parser.add_argument(
        "--decision-timeout-seconds",
        type=float,
        default=DEFAULT_DECISION_TIMEOUT_SECONDS,
    )
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
        decision_timeout_seconds=args.decision_timeout_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

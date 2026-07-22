#!/usr/bin/env python3
"""Framed external-Codex entry point for response-driven Journey schedules."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import select
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent.utils.redaction import redact
from tests.agent_live.chaos_scheduler import (
    JourneySchedule,
    build_journey_schedule,
    journey_schedule_payload,
)
from tests.agent_live.coverage_evidence import content_hash, repository_revision
from tests.agent_live.dynamic_dual_ai_chaos import (
    ChaosRunConfig,
    DynamicDualAiJourneyRunner,
    JourneyExternallyBlockedError,
    JourneyOutcomeVerifierRegistry,
    JourneyRunError,
    JourneySimulatorContext,
    JourneySimulatorDecision,
    JourneySimulatorInvalidError,
    JourneyTerminalClassification,
    validate_journey_outcome_verifier_registry,
)
from tests.agent_live.generate_harness_coverage_ledger import build_ledger


CONTEXT_FRAME = "CODEX_JOURNEY_SIMULATOR_CONTEXT "
DECISION_FRAME = "CODEX_JOURNEY_SIMULATOR_DECISION "
RESULT_FRAME = "CODEX_JOURNEY_SIMULATOR_RESULT "
DEFAULT_DECISION_TIMEOUT_SECONDS = 300.0


class JourneySimulatorBridgeError(RuntimeError):
    """Base error at the external Journey simulator boundary."""


class JourneySimulatorDecisionTimeout(
    JourneyExternallyBlockedError,
    JourneySimulatorBridgeError,
    TimeoutError,
):
    """The external Codex actor did not answer before the deadline."""


class JourneySimulatorInputClosed(
    JourneyExternallyBlockedError,
    JourneySimulatorBridgeError,
):
    """The external Codex actor closed its input without a decision."""


class JourneySimulatorProtocolError(
    JourneySimulatorInvalidError,
    JourneySimulatorBridgeError,
):
    """A framed Journey decision violated the external protocol."""


class JourneySimulatorInvalidFrame(JourneySimulatorProtocolError):
    pass


class JourneySimulatorInvalidJson(JourneySimulatorProtocolError):
    pass


class JourneySimulatorStaleResponse(JourneySimulatorProtocolError):
    pass


class StdioCodexJourneySimulator:
    """Exchange one revision-bound decision after each complete Agent response."""

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

    def __call__(self, context: JourneySimulatorContext) -> JourneySimulatorDecision:
        response_hash = content_hash(context.previous_agent_response)
        decision_template = {
            "previous_response_hash": response_hash,
            "user_message": "<one user turn selected from the complete response>",
            "persona": context.schedule.persona,
            "mission": context.schedule.mission,
            "rationale": "<why this turn follows from the observed response>",
            "risk_factor_ids": [],
        }
        payload = {
            "schema_version": 1,
            "session_id": context.session_id,
            "turn_index": context.turn_index,
            "previous_agent_response": str(redact(context.previous_agent_response)),
            "previous_response_hash": response_hash,
            "previous_response_received_at_ns": context.previous_response_received_at_ns,
            "schedule": journey_schedule_payload(context.schedule),
            "transcript": [
                {
                    "user": str(redact(user_message)),
                    "agent": str(redact(agent_response)),
                }
                for user_message, agent_response in context.transcript
            ],
            "observed_edge_keys": list(context.observed_edge_keys),
            "simulator_contract": {
                "frame_prefix": DECISION_FRAME,
                "response_driven_rule": (
                    "Select this turn only from previous_agent_response and prior transcript; "
                    "never predeclare or reuse a future dialogue turn."
                ),
                "immutable_persona": context.schedule.persona,
                "immutable_mission": context.schedule.mission,
                "allowed_risk_factors": list(context.schedule.allowed_risk_factors),
                "output_rule": (
                    "Write exactly one decision frame followed by one JSON object."
                ),
                "decision_template": decision_template,
            },
        }
        self.output_stream.write(CONTEXT_FRAME + json.dumps(payload, ensure_ascii=False) + "\n")
        self.output_stream.flush()

        line = self._read_decision_line()
        if not line.startswith(DECISION_FRAME):
            raise JourneySimulatorInvalidFrame(
                "external Codex Journey decision used an invalid frame"
            )
        try:
            decision = json.loads(line[len(DECISION_FRAME):])
        except json.JSONDecodeError as exc:
            raise JourneySimulatorInvalidJson(
                "external Codex Journey decision is not valid JSON"
            ) from exc
        if not isinstance(decision, Mapping):
            raise JourneySimulatorInvalidJson(
                "external Codex Journey decision must be a JSON object"
            )
        if str(decision.get("previous_response_hash") or "") != response_hash:
            raise JourneySimulatorStaleResponse(
                "external Codex Journey decision references a stale Agent response"
            )
        risk_factor_ids = decision.get("risk_factor_ids") or ()
        if not isinstance(risk_factor_ids, Sequence) or isinstance(
            risk_factor_ids, (str, bytes)
        ):
            raise JourneySimulatorProtocolError("risk_factor_ids must be a JSON array")
        return JourneySimulatorDecision(
            user_message=_required_text(decision, "user_message"),
            persona=_required_text(decision, "persona"),
            mission=_required_text(decision, "mission"),
            rationale=_required_text(decision, "rationale"),
            risk_factor_ids=tuple(str(item) for item in risk_factor_ids),
        )

    def _read_decision_line(self) -> str:
        if self.decision_timeout_seconds is None:
            line = self.input_stream.readline()
            if not line:
                raise JourneySimulatorInputClosed(
                    "external Codex Journey simulator closed before providing a decision"
                )
            return str(line)
        timeout = float(self.decision_timeout_seconds)
        if timeout <= 0:
            raise ValueError("decision_timeout_seconds must be greater than zero")
        try:
            fd = int(self.input_stream.fileno())
        except (AttributeError, OSError, TypeError, ValueError):
            line = self.input_stream.readline()
            if not line:
                raise JourneySimulatorInputClosed(
                    "external Codex Journey simulator closed before providing a decision"
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
                    raise JourneySimulatorDecisionTimeout(
                        f"external Codex Journey decision timed out after {timeout:.1f}s"
                    )
                ready, _, _ = select.select([fd], [], [], remaining)
                if not ready:
                    raise JourneySimulatorDecisionTimeout(
                        f"external Codex Journey decision timed out after {timeout:.1f}s"
                    )
                try:
                    chunk = os.read(fd, 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    raise JourneySimulatorInputClosed(
                        "external Codex Journey simulator closed before providing a decision"
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
            raise JourneySimulatorInvalidFrame(
                "external Codex Journey decision is not valid UTF-8"
            ) from exc


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise JourneySimulatorProtocolError(
            f"external Codex Journey decision requires {key}"
        )
    return value


def load_journey_schedule(path: Path, *, revision: Mapping[str, str]) -> JourneySchedule:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Journey schedule file must contain a JSON object")
    if dict(payload.get("revision") or {}) != dict(revision):
        raise ValueError("Journey schedule file revision does not match the active revision")
    schedule = build_journey_schedule(
        revision=revision,
        seed=int(payload["seed"]),
        journey={
            "journey_id": payload.get("journey_id"),
            "start_scenario": payload.get("start_scenario"),
            "persona": payload.get("persona"),
            "mission": payload.get("mission"),
            "allowed_risk_factors": payload.get("allowed_risk_factors") or (),
            "max_turns": payload.get("max_turns"),
            "terminal_outcome": payload.get("terminal_outcome"),
            "forbidden_outcomes": payload.get("forbidden_outcomes") or (),
        },
    )
    if payload.get("schedule_id") != schedule.schedule_id:
        raise ValueError("Journey schedule file identity is stale or was modified")
    return schedule


def load_journey_definition(
    path: Path,
    *,
    revision: Mapping[str, str],
    seed: int,
    expected_schedule_id: str,
) -> JourneySchedule:
    """Build a Journey from one immutable batch target without future dialogue."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("lane") != "journey":
        raise ValueError("Journey definition must declare lane=journey")
    journey = payload.get("journey")
    if not isinstance(journey, Mapping):
        raise ValueError("Journey definition requires a journey object")
    schedule = build_journey_schedule(
        revision=revision,
        seed=seed,
        journey=journey,
    )
    if schedule.schedule_id != expected_schedule_id:
        raise ValueError("Journey definition does not match the frozen schedule identity")
    return schedule


def load_verifier_registry(import_path: str) -> JourneyOutcomeVerifierRegistry:
    module_name, separator, attribute_name = import_path.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("verifier registry must use module:attribute syntax")
    registry = getattr(importlib.import_module(module_name), attribute_name)
    if not isinstance(registry, JourneyOutcomeVerifierRegistry):
        raise TypeError("imported Journey verifier registry has the wrong type")
    validate_journey_outcome_verifier_registry(registry)
    return registry


def run_bridge(
    *,
    repo_root: Path,
    schedule: JourneySchedule,
    verifier_registry: JourneyOutcomeVerifierRegistry,
    session_id: str,
    service: str,
    runtime: str,
    decision_timeout_seconds: float = DEFAULT_DECISION_TIMEOUT_SECONDS,
    input_stream: Any = sys.stdin,
    output_stream: Any = sys.stdout,
) -> Mapping[str, Any]:
    revision = repository_revision(repo_root)
    if dict(schedule.revision) != revision:
        raise ValueError("Journey schedule revision does not match the repository revision")
    ledger = build_ledger(revision=revision)
    if runtime == "linux":
        config = ChaosRunConfig.linux(
            repo_root, session_id=session_id, max_turns=schedule.max_turns
        )
    elif runtime == "docker":
        config = ChaosRunConfig.docker(
            repo_root, service=service, session_id=session_id, max_turns=schedule.max_turns
        )
    else:
        raise ValueError(f"unsupported Journey bridge runtime: {runtime}")
    runner = DynamicDualAiJourneyRunner(
        config,
        StdioCodexJourneySimulator(
            input_stream,
            output_stream,
            decision_timeout_seconds=decision_timeout_seconds,
        ),
        ledger=ledger,
        schedule=schedule,
        postcondition_verifier_registry=verifier_registry,
        revision=revision,
    )
    try:
        result = runner.run()
        payload = {
            "session_id": result.session_id,
            "terminal_classification": result.terminal_classification.value,
            "schedule_id": schedule.schedule_id,
            "schedule_path": str(result.schedule_path),
            "journey_result_path": str(result.journey_result_path),
            "transcript_path": str(result.transcript_path),
            "evidence_id": result.evidence_id,
            "evidence_path": str(result.evidence_path),
        }
    except JourneyRunError:
        result_path = (config.runtime_root or Path()) / "journey-result.json"
        if not result_path.is_file():
            raise
        result_payload = json.loads(result_path.read_text(encoding="utf-8"))
        payload = {
            "session_id": session_id,
            "terminal_classification": result_payload["terminal_classification"],
            "schedule_id": schedule.schedule_id,
            "schedule_path": str((config.runtime_root or Path()) / "journey-schedule.json"),
            "journey_result_path": str(result_path),
            "transcript_path": str((config.runtime_root or Path()) / "transcript.txt"),
            "evidence_id": result_payload["evidence_id"],
            "evidence_path": result_payload["evidence_path"],
        }
    output_stream.write(RESULT_FRAME + json.dumps(payload, ensure_ascii=False) + "\n")
    output_stream.flush()
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--journey", type=Path)
    source.add_argument("--journey-definition", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--expected-schedule-id")
    parser.add_argument("--verifier-registry", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
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
    repo_root = args.repo_root.resolve()
    revision = repository_revision(repo_root)
    if args.journey_definition is not None:
        if args.seed is None or not args.expected_schedule_id:
            raise ValueError(
                "--journey-definition requires --seed and --expected-schedule-id"
            )
        schedule = load_journey_definition(
            args.journey_definition,
            revision=revision,
            seed=args.seed,
            expected_schedule_id=args.expected_schedule_id,
        )
    else:
        schedule = load_journey_schedule(args.journey, revision=revision)
    payload = run_bridge(
        repo_root=repo_root,
        schedule=schedule,
        verifier_registry=load_verifier_registry(args.verifier_registry),
        session_id=args.session_id,
        service=args.service,
        runtime=args.runtime,
        decision_timeout_seconds=args.decision_timeout_seconds,
    )
    return (
        0
        if payload["terminal_classification"] == JourneyTerminalClassification.PASSED.value
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())

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
SIMULATOR_ATTESTATION_SCHEMA_VERSION = 1


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


def simulator_context_binding(context: Mapping[str, Any]) -> dict[str, Any]:
    """Project the immutable response-bound identity shared by every lane."""

    schedule = context.get("schedule")
    target = context.get("scheduled_target")
    if isinstance(schedule, Mapping):
        control_identity = {
            "lane": "journey",
            "schedule_id": str(schedule.get("schedule_id") or ""),
        }
    elif isinstance(target, Mapping):
        control_identity = {
            "lane": "edge",
            "target_id": str(target.get("target_id") or ""),
            "edge_key": str(target.get("edge_key") or ""),
        }
    else:
        raise ValueError("simulator context has no immutable control identity")
    binding = {
        "session_id": str(context.get("session_id") or ""),
        "turn_index": int(context.get("turn_index") or 0),
        "previous_response_hash": str(
            context.get("previous_response_hash") or ""
        ),
        "previous_response_received_at_ns": int(
            context.get("previous_response_received_at_ns") or 0
        ),
        "control_identity": control_identity,
        "observed_edge_keys": [
            str(item) for item in context.get("observed_edge_keys") or ()
        ],
    }
    if (
        not binding["session_id"]
        or binding["turn_index"] <= 0
        or len(binding["previous_response_hash"]) != 64
        or binding["previous_response_received_at_ns"] <= 0
        or (
            control_identity["lane"] == "journey"
            and not control_identity["schedule_id"]
        )
        or (
            control_identity["lane"] == "edge"
            and not control_identity["edge_key"]
        )
    ):
        raise ValueError("simulator context binding is incomplete")
    return binding


def simulator_context_hash(context: Mapping[str, Any]) -> str:
    return content_hash(simulator_context_binding(context))


def build_simulator_attestation(
    *,
    actor_kind: str,
    task_id: str,
    model: str,
    request_id: str,
    previous_response_hash: str,
    context_hash: str,
    decision_hash: str,
    user_message_hash: str,
    turn_index: int,
    declared_at_ns: int,
) -> dict[str, Any]:
    """Build an auditable declaration, never a cryptographic identity proof."""

    actor = {
        "actor_kind": str(actor_kind).strip(),
        "task_id": str(task_id).strip(),
        "model": str(model).strip(),
    }
    if not all(actor.values()):
        raise ValueError("simulator actor declaration requires kind, task id, and model")
    unsigned = {
        "schema_version": SIMULATOR_ATTESTATION_SCHEMA_VERSION,
        "identity_strength": "auditable_declaration_only",
        "cryptographic_identity_claimed": False,
        "selection_mode": "response_driven",
        "prewritten_future_turns": False,
        "actor": actor,
        "request_id": str(request_id).strip(),
        "previous_response_hash": str(previous_response_hash).strip(),
        "context_hash": str(context_hash).strip(),
        "decision_hash": str(decision_hash).strip(),
        "user_message_hash": str(user_message_hash).strip(),
        "turn_index": int(turn_index),
        "declared_at_ns": int(declared_at_ns),
    }
    for field in (
        "request_id",
        "previous_response_hash",
        "context_hash",
        "decision_hash",
        "user_message_hash",
    ):
        if not str(unsigned[field]):
            raise ValueError(f"simulator attestation requires {field}")
    return {**unsigned, "attestation_id": content_hash(unsigned)}


def validate_simulator_attestation(
    attestation: Mapping[str, Any],
    *,
    previous_response_hash: str,
    decision_hash: str,
    request_id: str = "",
    context_hash: str = "",
    user_message_hash: str = "",
    turn_index: int = 0,
    submitted_at_ns: int = 0,
) -> dict[str, Any]:
    required_fields = {
        "schema_version",
        "identity_strength",
        "cryptographic_identity_claimed",
        "selection_mode",
        "prewritten_future_turns",
        "actor",
        "request_id",
        "previous_response_hash",
        "context_hash",
        "decision_hash",
        "user_message_hash",
        "turn_index",
        "declared_at_ns",
        "attestation_id",
    }
    if set(attestation) != required_fields:
        raise ValueError("simulator attestation schema fields are invalid")
    if attestation.get("schema_version") != SIMULATOR_ATTESTATION_SCHEMA_VERSION:
        raise ValueError("simulator attestation schema is unsupported")
    actor = attestation.get("actor")
    if (
        not isinstance(actor, Mapping)
        or set(actor) != {"actor_kind", "task_id", "model"}
        or not all(
        str(actor.get(field) or "").strip()
        for field in ("actor_kind", "task_id", "model")
        )
    ):
        raise ValueError("simulator attestation actor declaration is incomplete")
    if str(actor.get("actor_kind") or "") != "codex":
        raise ValueError("scripted or undeclared simulator actors do not qualify")
    expected = {
        "identity_strength": "auditable_declaration_only",
        "cryptographic_identity_claimed": False,
        "selection_mode": "response_driven",
        "prewritten_future_turns": False,
    }
    if any(attestation.get(key) != value for key, value in expected.items()):
        raise ValueError("simulator attestation policy is invalid")
    if (
        isinstance(attestation.get("turn_index"), bool)
        or not isinstance(attestation.get("turn_index"), int)
        or int(attestation["turn_index"]) <= 0
    ):
        raise ValueError("simulator attestation turn index is invalid")
    if not str(attestation.get("request_id") or "").strip():
        raise ValueError("simulator attestation request identity is invalid")
    for field in (
        "previous_response_hash",
        "context_hash",
        "decision_hash",
        "user_message_hash",
    ):
        value = str(attestation.get(field) or "")
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"simulator attestation {field} is invalid")
    if (
        isinstance(attestation.get("declared_at_ns"), bool)
        or not isinstance(attestation.get("declared_at_ns"), int)
        or int(attestation["declared_at_ns"]) <= 0
    ):
        raise ValueError("simulator attestation declaration time is invalid")
    if str(attestation.get("previous_response_hash") or "") != str(
        previous_response_hash
    ):
        raise ValueError("simulator attestation is bound to a stale response")
    if str(attestation.get("decision_hash") or "") != str(decision_hash):
        raise ValueError("simulator attestation decision hash is stale")
    expected_optional = {
        "request_id": request_id,
        "context_hash": context_hash,
        "user_message_hash": user_message_hash,
    }
    for field, expected_value in expected_optional.items():
        if expected_value and str(attestation.get(field) or "") != str(
            expected_value
        ):
            raise ValueError(f"simulator attestation {field} is stale")
    if turn_index and attestation.get("turn_index") != turn_index:
        raise ValueError("simulator attestation turn index is stale")
    if (
        submitted_at_ns
        and int(attestation.get("declared_at_ns") or 0) > submitted_at_ns
    ):
        raise ValueError("simulator attestation was declared after submission")
    unsigned = {
        key: value for key, value in attestation.items()
        if key != "attestation_id"
    }
    if str(attestation.get("attestation_id") or "") != content_hash(unsigned):
        raise ValueError("simulator attestation identity is stale")
    return dict(attestation)


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
    if action_type == "change_group":
        target_group = ""
        if isinstance(postcondition, Mapping):
            target_group = str(postcondition.get("target_group") or "")
        return (
            "The user_message must request only generic navigation to the exact target group "
            f"{target_group or '<declared target>'}. It must not request a concrete mutation, "
            "supply a value, or name a registered domain subflow/intake such as custom RPC "
            "catalog setup; those belong to their typed domain actions rather than change_group."
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
    max_turns = sum(
        1 + int(target.continuation_turn_budget)
        for target in schedule.targets
    )
    if runtime == "linux":
        config = ChaosRunConfig.linux(
            repo_root,
            session_id=session_id,
            max_turns=max_turns,
        )
    elif runtime == "docker":
        config = ChaosRunConfig.docker(
            repo_root,
            service=service,
            session_id=session_id,
            max_turns=max_turns,
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

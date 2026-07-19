#!/usr/bin/env python3
"""Response-driven Codex-user / Agent-LLM chaos runner over a real PTY.

This module deliberately has no ``prompts`` or scripted-conversation input.
The simulator is invoked once per turn *after* the complete preceding Agent
response has been observed.  Each decision is then submitted through the same
bracketed-paste terminal path used by an interactive user and recorded with the
existing tamper-evident PTY evidence contract.
"""

from __future__ import annotations

import json
import os
import re
import select
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from tests.agent_live.coverage_evidence import (
    DynamicTurnSelection,
    PtyCliTurnRecord,
    RuntimeTurnEvent,
    TurnObservation,
    build_pty_cli_evidence_artifact,
    pty_transcript_hash,
    repository_revision,
    verify_runtime_postcondition,
    write_evidence_artifact,
)
from tests.agent_live.chaos_scheduler import (
    ChaosSchedule,
    ScheduledCoverageTarget,
    validate_chaos_schedule,
    write_chaos_schedule,
)


_ANSI_CSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
_USER_PROMPT_RE = re.compile(r"(?:^|\n)User>\s*$")
_USER_PROMPT_BOUNDARY_RE = re.compile(r"(?:^|\n)User>[ \t]*(?=\n|$)")
_MODEL_CONFIG_RE = re.compile(
    r"provider\s*=\s*([^,\s]+)\s*,\s*model\s*=\s*([^,\s]+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SimulatorContext:
    """Only information available after the preceding response completed."""

    session_id: str
    turn_index: int
    previous_agent_response: str
    previous_response_received_at_ns: int
    scheduled_target: ScheduledCoverageTarget
    transcript: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class SimulatorDecision:
    """One live user decision and the coverage contract it is targeting."""

    user_message: str
    persona: str
    goal: str
    rationale: str
    target_coverage_ids: tuple[str, ...]


@dataclass(frozen=True)
class ChaosRunConfig:
    repo_root: Path
    command: tuple[str, ...]
    session_id: str = field(default_factory=lambda: f"dynamic-chaos-{uuid.uuid4().hex}")
    session_purpose: str = "dynamic-dual-ai-chaos"
    provider: str = "deepseek"
    model: str = "deepseek-chat"
    max_turns: int = 20
    response_timeout_seconds: float = 180.0
    poll_interval_seconds: float = 0.05
    runtime_root: Path | None = None
    runtime_root_in_process: Path | None = None
    extra_env: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def docker(
        cls,
        repo_root: str | Path,
        *,
        service: str = "bench",
        **changes: Any,
    ) -> "ChaosRunConfig":
        """Run the product CLI in the Linux Docker service through a host PTY."""

        root = Path(repo_root).resolve()
        session_id = str(changes.pop("session_id", f"dynamic-chaos-{uuid.uuid4().hex}"))
        host_runtime = root / ".agent" / "dynamic-chaos" / session_id
        container_runtime = Path("/workspace/.agent/dynamic-chaos") / session_id
        container_env = {
            "ANYCHAIN_AGENT_CHECKPOINT_PATH": str(container_runtime / "checkpoints.sqlite"),
            "ANYCHAIN_AGENT_SESSION_ID": session_id,
            "ANYCHAIN_AGENT_SESSION_PURPOSE": str(
                changes.get("session_purpose", "dynamic-dual-ai-chaos")
            ),
            "ANYCHAIN_AGENT_JOBS_DIR": str(container_runtime / "jobs"),
            "ANYCHAIN_AGENT_TURN_EVENT_FILE": str(container_runtime / "turn-events.jsonl"),
        }
        command: list[str] = ["docker", "compose", "exec"]
        for name, value in container_env.items():
            command.extend(("-e", f"{name}={value}"))
        command.extend((
            service,
            "./bin/anychain-agent",
            "--state-file",
            str(container_runtime / "terminal-session.json"),
            "--language",
            "en",
        ))
        return cls(
            repo_root=root,
            command=tuple(command),
            session_id=session_id,
            runtime_root=host_runtime,
            runtime_root_in_process=container_runtime,
            **changes,
        )

    @classmethod
    def linux(
        cls,
        repo_root: str | Path,
        **changes: Any,
    ) -> "ChaosRunConfig":
        """Run the product CLI directly on a Linux host through a PTY."""

        root = Path(repo_root).resolve()
        session_id = str(changes.pop("session_id", f"dynamic-chaos-{uuid.uuid4().hex}"))
        runtime = root / ".agent" / "dynamic-chaos" / session_id
        return cls(
            repo_root=root,
            command=(
                str(root / "bin" / "anychain-agent"),
                "--state-file",
                str(runtime / "terminal-session.json"),
                "--language",
                "en",
            ),
            session_id=session_id,
            runtime_root=runtime,
            runtime_root_in_process=runtime,
            **changes,
        )


@dataclass(frozen=True)
class ChaosRunResult:
    session_id: str
    transcript_path: Path
    evidence_paths: tuple[Path, ...]
    turns: tuple[PtyCliTurnRecord, ...]
    schedule_path: Path
    schedule_result_path: Path
    execution_status: str


class Simulator(Protocol):
    """External decision boundary; this repository provides no Codex API."""

    def __call__(self, context: SimulatorContext) -> SimulatorDecision | None: ...


class PtyTransport(Protocol):
    def start(self, *, env: Mapping[str, str]) -> None: ...

    def read_complete_agent_response(self, *, timeout_seconds: float) -> str: ...

    def submit_bracketed_paste(self, message: str) -> None: ...

    def close(self) -> None: ...


class RuntimeEventStream(Protocol):
    def baseline(self) -> RuntimeTurnEvent: ...

    def next_event(self, *, timeout_seconds: float) -> RuntimeTurnEvent: ...


class SubprocessPtyTransport:
    """Small stdlib PTY transport suitable for Docker Compose or direct Linux."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path,
        poll_interval_seconds: float = 0.05,
    ) -> None:
        self.command = tuple(command)
        self.cwd = Path(cwd)
        self.poll_interval_seconds = poll_interval_seconds
        self._master_fd: int | None = None
        self._process: subprocess.Popen[bytes] | None = None

    def start(self, *, env: Mapping[str, str]) -> None:
        if self._process is not None:
            raise RuntimeError("PTY transport has already started")
        master_fd, slave_fd = os.openpty()
        try:
            self._process = subprocess.Popen(
                self.command,
                cwd=self.cwd,
                env=dict(env),
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                start_new_session=True,
                close_fds=True,
            )
        finally:
            os.close(slave_fd)
        self._master_fd = master_fd

    def read_complete_agent_response(self, *, timeout_seconds: float) -> str:
        if self._master_fd is None or self._process is None:
            raise RuntimeError("PTY transport is not running")
        deadline = time.monotonic() + timeout_seconds
        captured = bytearray()
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                captured.extend(self._drain())
                raise RuntimeError(
                    f"Agent CLI exited before the next User prompt (exit={self._process.returncode}): "
                    f"{_clean_terminal_text(bytes(captured))[-2000:]}"
                )
            ready, _, _ = select.select(
                [self._master_fd],
                [],
                [],
                min(self.poll_interval_seconds, max(0.0, deadline - time.monotonic())),
            )
            if not ready:
                continue
            try:
                chunk = os.read(self._master_fd, 65536)
            except OSError as exc:
                raise RuntimeError("Agent PTY closed while reading a response") from exc
            if not chunk:
                continue
            captured.extend(chunk)
            cleaned = _clean_terminal_text(bytes(captured))
            if _USER_PROMPT_RE.search(cleaned):
                response = _complete_agent_response(cleaned)
                if response is not None:
                    return response
        raise TimeoutError(
            f"timed out after {timeout_seconds:.1f}s waiting for a complete Agent response; "
            f"partial output: {_clean_terminal_text(bytes(captured))[-2000:]}"
        )

    def submit_bracketed_paste(self, message: str) -> None:
        if self._master_fd is None:
            raise RuntimeError("PTY transport is not running")
        os.write(self._master_fd, encode_bracketed_paste(message))

    def close(self) -> None:
        process = self._process
        master_fd = self._master_fd
        self._process = None
        self._master_fd = None
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=3)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=3)
        if master_fd is not None:
            os.close(master_fd)

    def _drain(self) -> bytes:
        if self._master_fd is None:
            return b""
        result = bytearray()
        while True:
            ready, _, _ = select.select([self._master_fd], [], [], 0)
            if not ready:
                return bytes(result)
            try:
                chunk = os.read(self._master_fd, 65536)
            except OSError:
                return bytes(result)
            if not chunk:
                return bytes(result)
            result.extend(chunk)


class JsonlRuntimeEventStream:
    """Read only newly appended product-process events from the isolated JSONL."""

    def __init__(self, path: str | Path, *, poll_interval_seconds: float = 0.05) -> None:
        self.path = Path(path)
        self.poll_interval_seconds = poll_interval_seconds
        self._events: list[RuntimeTurnEvent] = []
        self._cursor = 0

    def baseline(self) -> RuntimeTurnEvent:
        self._refresh()
        if not self._events:
            raise RuntimeError(
                "dynamic Chaos requires a baseline runtime event; seed the session "
                "through the product runtime before scheduling coverage turns"
            )
        self._cursor = len(self._events)
        return self._events[-1]

    def next_event(self, *, timeout_seconds: float) -> RuntimeTurnEvent:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            self._refresh()
            available = len(self._events) - self._cursor
            if available > 1:
                raise RuntimeError("multiple runtime events advanced for one submitted PTY turn")
            if available == 1:
                event = self._events[self._cursor]
                self._cursor += 1
                return event
            time.sleep(min(self.poll_interval_seconds, max(0.0, deadline - time.monotonic())))
        raise RuntimeError("CLI returned without a new committed runtime turn event")

    def _refresh(self) -> None:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        parsed = [_runtime_event_from_mapping(json.loads(line)) for line in lines if line.strip()]
        if len(parsed) < len(self._events) or parsed[:len(self._events)] != self._events:
            raise RuntimeError("runtime turn event stream was truncated or rewritten")
        self._events = parsed


class DynamicDualAiChaosRunner:
    """Drive a real CLI where every turn is selected from the last response."""

    def __init__(
        self,
        config: ChaosRunConfig,
        simulator: Simulator,
        *,
        ledger: Mapping[str, Any],
        schedule: ChaosSchedule,
        transport: PtyTransport | None = None,
        event_stream: RuntimeEventStream | None = None,
        revision: Mapping[str, str] | None = None,
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.config = config
        self.simulator = simulator
        self.transport = transport or SubprocessPtyTransport(
            config.command,
            cwd=config.repo_root,
            poll_interval_seconds=config.poll_interval_seconds,
        )
        self.ledger = dict(ledger)
        self.schedule = schedule
        self.revision = dict(revision or repository_revision(config.repo_root))
        validate_chaos_schedule(schedule, ledger=self.ledger, revision=self.revision)
        if len(schedule.targets) > config.max_turns:
            raise ValueError("Chaos schedule exceeds max_turns")
        self.edge_index = {
            str(edge.get("edge_key") or ""): edge
            for edge in self.ledger.get("edges") or ()
            if str(edge.get("edge_key") or "")
        }
        default_event_path = (
            (config.runtime_root or config.repo_root / ".agent" / "dynamic-chaos" / config.session_id)
            / "turn-events.jsonl"
        )
        self.event_stream = event_stream or JsonlRuntimeEventStream(
            default_event_path,
            poll_interval_seconds=config.poll_interval_seconds,
        )
        self.clock_ns = clock_ns

    def run(self) -> ChaosRunResult:
        runtime_root = (self.config.runtime_root or (
            self.config.repo_root / ".agent" / "dynamic-chaos" / self.config.session_id
        )).resolve()
        runtime_root.mkdir(parents=True, exist_ok=True)
        evidence_dir = runtime_root / "evidence"
        transcript_path = runtime_root / "transcript.txt"
        schedule_path = write_chaos_schedule(self.schedule, runtime_root / "schedule.json")
        schedule_result_path = runtime_root / "schedule-result.json"
        env = self._isolated_environment(runtime_root)

        seed_scenario_id = str(self.schedule.targets[0].scenario_id or "")
        if seed_scenario_id:
            from tests.agent_live.runtime_checkpoint import (
                reviewed_scenario_state,
                seed_runtime_checkpoint,
            )

            seed_runtime_checkpoint(
                reviewed_scenario_state(seed_scenario_id),
                checkpoint_path=runtime_root / "checkpoints.sqlite",
                session_id=self.config.session_id,
                session_purpose=self.config.session_purpose,
            )

        transcript: list[tuple[str, str]] = []
        transcript_lines: list[str] = []
        evidence_paths: list[Path] = []
        turns: list[PtyCliTurnRecord] = []
        target_results: list[dict[str, Any]] = []
        execution_status = "incomplete"

        self.transport.start(env=env)
        try:
            previous_response = self.transport.read_complete_agent_response(
                timeout_seconds=self.config.response_timeout_seconds
            )
            previous_received_ns = self.clock_ns()
            observed_provider, observed_model = _provider_model_from_startup(previous_response)
            if observed_provider != self.config.provider or observed_model != self.config.model:
                raise RuntimeError(
                    "observed provider/model does not match the configured Chaos boundary: "
                    f"{observed_provider}/{observed_model}"
                )
            baseline_event = self.event_stream.baseline()
            self._validate_event_revision(baseline_event)
            transcript_lines.append(previous_response)

            if seed_scenario_id:
                first_target = self.schedule.targets[0]
                first_edge = self.edge_index[first_target.edge_key]
                expected_question = str(first_edge.get("question_id") or "")
                if baseline_event.pending_question_id == "resume_harness_session":
                    self.transport.submit_bracketed_paste("1")
                    resumed_response = self.transport.read_complete_agent_response(
                        timeout_seconds=self.config.response_timeout_seconds
                    )
                    resumed_event = self.event_stream.next_event(
                        timeout_seconds=self.config.response_timeout_seconds
                    )
                    self._validate_event_revision(resumed_event)
                    transcript.extend((("1", resumed_response),))
                    transcript_lines.extend(("User> 1", resumed_response))
                    previous_response = resumed_response
                    previous_received_ns = self.clock_ns()
                    baseline_event = resumed_event
                if baseline_event.pending_question_id != expected_question:
                    raise RuntimeError(
                        "reviewed checkpoint did not restore the scheduled target contract: "
                        f"expected {expected_question or '<action>'}, got "
                        f"{baseline_event.pending_question_id or '<none>'}"
                    )

            for scheduled_target in self.schedule.targets:
                edge = self.edge_index.get(scheduled_target.edge_key)
                if edge is None:
                    raise RuntimeError(
                        f"scheduled edge disappeared from authoritative ledger: {scheduled_target.edge_key}"
                    )
                context = SimulatorContext(
                    session_id=self.config.session_id,
                    turn_index=baseline_event.turn_index + 1,
                    previous_agent_response=previous_response,
                    previous_response_received_at_ns=previous_received_ns,
                    scheduled_target=scheduled_target,
                    transcript=tuple(transcript),
                )
                decision = self.simulator(context)
                selected_at_ns = self.clock_ns()
                if decision is None:
                    target_results.append({
                        "target_id": scheduled_target.target_id,
                        "edge_key": scheduled_target.edge_key,
                        "status": "externally_blocked",
                        "reason": "external Codex simulator did not provide a decision",
                    })
                    raise RuntimeError("external Codex simulator did not provide a decision")
                _validate_decision(decision, scheduled_target)
                submitted_at_ns = self.clock_ns()
                self.transport.submit_bracketed_paste(decision.user_message)
                response = self.transport.read_complete_agent_response(
                    timeout_seconds=self.config.response_timeout_seconds
                )
                response_received_ns = self.clock_ns()
                committed_event = self.event_stream.next_event(
                    timeout_seconds=self.config.response_timeout_seconds
                )
                self._validate_event_revision(committed_event)

                transcript_data = {
                    "session_id": self.config.session_id,
                    "turn_index": committed_event.turn_index,
                    "previous_agent_response": previous_response,
                    "user_message": decision.user_message,
                    "agent_response": response,
                }
                turn = PtyCliTurnRecord(
                    **transcript_data,
                    provider=observed_provider,
                    model=observed_model,
                    before_fingerprint=committed_event.before_fingerprint,
                    after_fingerprint=committed_event.after_fingerprint,
                    transcript_hash=pty_transcript_hash(**transcript_data),
                    previous_response_received_at_ns=previous_received_ns,
                    user_message_submitted_at_ns=submitted_at_ns,
                    agent_response_received_at_ns=response_received_ns,
                )
                selection = DynamicTurnSelection(
                    persona=decision.persona,
                    goal=decision.goal,
                    selected_message=decision.user_message,
                    rationale=decision.rationale,
                    target_coverage_ids=tuple(decision.target_coverage_ids),
                    selected_at_ns=selected_at_ns,
                )
                verified_postcondition = verify_runtime_postcondition(
                    edge,
                    baseline_event,
                    committed_event,
                    turn,
                )
                if not verified_postcondition.passed:
                    errors = verified_postcondition.details.get("errors") or ()
                    raise ValueError(
                        "turn observation postcondition did not pass: "
                        + "; ".join(str(item) for item in errors)
                    )
                observation = TurnObservation(
                    seed=self.schedule.seed,
                    revision=self.revision,
                    target_edge_key=str(edge.get("edge_key") or ""),
                    target_contract_hash=str(edge.get("contract_hash") or ""),
                    target_variant_hash=str(edge.get("contract_variant_hash") or ""),
                    prior_agent_response=previous_response,
                    simulator_decision={
                        "persona": selection.persona,
                        "goal": selection.goal,
                        "selected_message": selection.selected_message,
                        "rationale": selection.rationale,
                        "target_coverage_ids": list(selection.target_coverage_ids),
                        "selected_at_ns": selection.selected_at_ns,
                        "simulator": selection.simulator,
                        "selection_mode": selection.selection_mode,
                    },
                    exact_user_turn=decision.user_message,
                    provider=observed_provider,
                    model=observed_model,
                    before_turn_index=baseline_event.turn_index,
                    after_turn_index=committed_event.turn_index,
                    before_state_fingerprint=committed_event.before_fingerprint,
                    after_state_fingerprint=committed_event.after_fingerprint,
                    pending_contract=dict(baseline_event.pending_contract),
                    runtime_events=(baseline_event, committed_event),
                    verified_postcondition=verified_postcondition,
                )
                artifact = build_pty_cli_evidence_artifact(
                    edge=edge,
                    evidence_class="dynamic_dual_ai",
                    revision=self.revision,
                    turn=turn,
                    observation=observation,
                    dynamic_selection=selection,
                )
                evidence_path = write_evidence_artifact(artifact, evidence_dir)
                evidence_paths.append(evidence_path)
                lane_evidence = {
                    "dynamic_dual_ai": {
                        "evidence_id": artifact["evidence_id"],
                        "evidence_path": str(evidence_path),
                    }
                }
                real_cli_lane = (edge.get("evidence") or {}).get("real_cli") or {}
                if bool(real_cli_lane.get("required")):
                    real_cli_observation = TurnObservation(
                        seed=observation.seed,
                        revision=observation.revision,
                        target_edge_key=observation.target_edge_key,
                        target_contract_hash=observation.target_contract_hash,
                        target_variant_hash=observation.target_variant_hash,
                        prior_agent_response=observation.prior_agent_response,
                        simulator_decision={},
                        exact_user_turn=observation.exact_user_turn,
                        provider=observation.provider,
                        model=observation.model,
                        before_turn_index=observation.before_turn_index,
                        after_turn_index=observation.after_turn_index,
                        before_state_fingerprint=observation.before_state_fingerprint,
                        after_state_fingerprint=observation.after_state_fingerprint,
                        pending_contract=observation.pending_contract,
                        runtime_events=observation.runtime_events,
                        verified_postcondition=observation.verified_postcondition,
                    )
                    real_cli_artifact = build_pty_cli_evidence_artifact(
                        edge=edge,
                        evidence_class="real_cli",
                        revision=self.revision,
                        turn=turn,
                        observation=real_cli_observation,
                    )
                    real_cli_path = write_evidence_artifact(real_cli_artifact, evidence_dir)
                    evidence_paths.append(real_cli_path)
                    lane_evidence["real_cli"] = {
                        "evidence_id": real_cli_artifact["evidence_id"],
                        "evidence_path": str(real_cli_path),
                    }
                target_results.append({
                    "target_id": scheduled_target.target_id,
                    "edge_key": scheduled_target.edge_key,
                    "status": "passed",
                    "evidence_id": artifact["evidence_id"],
                    "evidence_path": str(evidence_path),
                    "lane_evidence": lane_evidence,
                })
                turns.append(turn)
                transcript.append((decision.user_message, response))
                transcript_lines.extend((f"User> {decision.user_message}", response))
                previous_response = response
                previous_received_ns = response_received_ns
                baseline_event = committed_event
            execution_status = "complete"
        except Exception as exc:
            if not target_results or target_results[-1].get("status") == "passed":
                completed_ids = {item["target_id"] for item in target_results}
                pending = next(
                    (item for item in self.schedule.targets if item.target_id not in completed_ids),
                    None,
                )
                if pending is not None:
                    target_results.append({
                        "target_id": pending.target_id,
                        "edge_key": pending.edge_key,
                        "status": "failed",
                        "reason": f"{type(exc).__name__}: {exc}",
                    })
            raise
        finally:
            self.transport.close()
            transcript_path.write_text(
                "\n".join(transcript_lines).rstrip() + "\n",
                encoding="utf-8",
            )
            schedule_result_path.write_text(
                json.dumps({
                    "schema_version": 1,
                    "schedule_id": self.schedule.schedule_id,
                    "revision": self.revision,
                    "execution_status": execution_status,
                    "required_target_count": len(self.schedule.targets),
                    "passed_target_count": sum(item.get("status") == "passed" for item in target_results),
                    "targets": target_results,
                }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        return ChaosRunResult(
            session_id=self.config.session_id,
            transcript_path=transcript_path,
            evidence_paths=tuple(evidence_paths),
            turns=tuple(turns),
            schedule_path=schedule_path,
            schedule_result_path=schedule_result_path,
            execution_status=execution_status,
        )

    def _validate_event_revision(self, event: RuntimeTurnEvent) -> None:
        if dict(event.revision) != self.revision:
            raise RuntimeError(
                "runtime event revision does not match the scheduled ledger revision"
            )

    def _isolated_environment(self, runtime_root: Path) -> dict[str, str]:
        process_root = self.config.runtime_root_in_process or runtime_root
        env = os.environ.copy()
        env.update({
            "ANYCHAIN_AGENT_CHECKPOINT_PATH": str(process_root / "checkpoints.sqlite"),
            "ANYCHAIN_AGENT_SESSION_ID": self.config.session_id,
            "ANYCHAIN_AGENT_SESSION_PURPOSE": self.config.session_purpose,
            "ANYCHAIN_AGENT_JOBS_DIR": str(process_root / "jobs"),
            "ANYCHAIN_AGENT_TURN_EVENT_FILE": str(process_root / "turn-events.jsonl"),
        })
        env.update({str(key): str(value) for key, value in self.config.extra_env.items()})
        return env



def encode_bracketed_paste(message: str) -> bytes:
    """Encode one user turn without treating embedded newlines as submissions."""

    normalized = str(message).replace("\r\n", "\n").replace("\r", "\n")
    return b"\x1b[200~" + normalized.encode("utf-8") + b"\x1b[201~\r"


def _validate_decision(
    decision: SimulatorDecision,
    target: ScheduledCoverageTarget,
) -> None:
    required = {
        "user_message": decision.user_message,
        "persona": decision.persona,
        "goal": decision.goal,
        "rationale": decision.rationale,
    }
    missing = sorted(name for name, value in required.items() if not str(value or "").strip())
    if missing:
        raise ValueError(f"simulator decision is missing: {', '.join(missing)}")
    if not decision.target_coverage_ids or any(
        not str(item or "").strip() for item in decision.target_coverage_ids
    ):
        raise ValueError("simulator decision requires target coverage ids")
    if target.edge_key not in decision.target_coverage_ids:
        raise ValueError("simulator decision does not target the scheduled ledger edge")
    if decision.persona != target.persona or decision.goal != target.goal:
        raise ValueError("simulator decision changed the scheduled persona or goal")


def _clean_terminal_text(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    return _ANSI_CSI_RE.sub("", text)


def _complete_agent_response(cleaned: str) -> str | None:
    """Return the newest prompt-delimited frame containing Agent output."""

    boundaries = list(_USER_PROMPT_BOUNDARY_RE.finditer(cleaned))
    if not boundaries:
        return None
    current = boundaries[-1]
    frame_start = boundaries[-2].end() if len(boundaries) > 1 else 0
    frame = cleaned[frame_start:current.start()]
    first_agent = frame.find("Agent>")
    if first_agent < 0:
        return None
    response = frame[first_agent:].strip()
    return response or None


def _provider_model_from_startup(response: str) -> tuple[str, str]:
    match = _MODEL_CONFIG_RE.search(response)
    if match is None:
        raise RuntimeError("Agent startup response did not expose provider/model identity")
    return match.group(1), match.group(2)


def _runtime_event_from_mapping(payload: Mapping[str, Any]) -> RuntimeTurnEvent:
    required = {
        "schema_version",
        "event_type",
        "thread_id",
        "session_purpose",
        "before_fingerprint",
        "after_fingerprint",
        "turn_index",
        "active_group",
        "pending_question_id",
        "pending_contract",
        "revision",
        "action_queue_types",
        "admitted_action_types",
        "admitted_action_targets",
        "state_diff_hashes",
        "after_value_hashes",
        "next_result",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise RuntimeError(f"runtime turn event is missing: {', '.join(missing)}")
    return RuntimeTurnEvent(
        schema_version=int(payload["schema_version"]),
        event_type=str(payload["event_type"]),
        thread_id=str(payload["thread_id"]),
        session_purpose=str(payload["session_purpose"]),
        before_fingerprint=str(payload["before_fingerprint"]),
        after_fingerprint=str(payload["after_fingerprint"]),
        turn_index=int(payload["turn_index"]),
        active_group=str(payload["active_group"]),
        pending_question_id=str(payload["pending_question_id"]),
        action_queue_types=tuple(str(item) for item in payload["action_queue_types"]),
        pending_contract=dict(payload["pending_contract"] or {}),
        revision={str(key): str(value) for key, value in dict(payload["revision"] or {}).items()},
        admitted_action_types=tuple(str(item) for item in payload["admitted_action_types"]),
        admitted_action_targets=tuple(
            {
                "type": str(item.get("type") or ""),
                "group": str(item.get("group") or ""),
            }
            for item in payload["admitted_action_targets"]
            if isinstance(item, Mapping)
        ),
        state_diff_hashes={
            str(path): {str(key): str(value) for key, value in dict(hashes).items()}
            for path, hashes in dict(payload["state_diff_hashes"] or {}).items()
        },
        after_value_hashes={
            str(path): str(value)
            for path, value in dict(payload["after_value_hashes"] or {}).items()
        },
        next_result=dict(payload["next_result"] or {}),
    )

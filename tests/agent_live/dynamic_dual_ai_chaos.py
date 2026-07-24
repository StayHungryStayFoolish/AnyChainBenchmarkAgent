#!/usr/bin/env python3
"""Response-driven Codex-user / Agent-LLM chaos runner over a real PTY.

This module deliberately has no ``prompts`` or scripted-conversation input.
The simulator is invoked once per turn *after* the complete preceding Agent
response has been observed.  Each decision is then submitted through the same
bracketed-paste terminal path used by an interactive user and recorded with the
existing tamper-evident PTY evidence contract.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import select
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

from agent.utils.redaction import redact
from tests.agent_live.coverage_evidence import (
    DynamicTurnSelection,
    PtyDiagnosticRecord,
    PtyCliTurnRecord,
    RuntimeTurnEvent,
    TurnObservation,
    VerifiedPostcondition,
    build_pty_cli_evidence_artifact,
    build_pty_diagnostic_artifact,
    content_hash,
    pty_transcript_hash,
    repository_revision,
    verify_runtime_postcondition,
    write_evidence_artifact,
    write_pty_diagnostic_artifact,
)
from tests.agent_live.chaos_scheduler import (
    ChaosSchedule,
    JourneyOutcomeContract,
    JourneySchedule,
    ScheduledCoverageTarget,
    journey_schedule_payload,
    validate_chaos_schedule,
    validate_journey_schedule,
    write_chaos_schedule,
    write_journey_schedule,
)
from agent.harness.plan_coverage import segment_user_turn


_ANSI_CSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
_USER_PROMPT_RE = re.compile(r"(?:^|\n)User>\s*$")
_USER_PROMPT_BOUNDARY_RE = re.compile(r"(?:^|\n)User>[ \t]*(?=\n|$)")
_MODEL_CONFIG_RE = re.compile(
    r"provider\s*=\s*([^,\s]+)\s*,\s*model\s*=\s*([^,\s]+)",
    re.IGNORECASE,
)


def _write_redacted_transcript(path: Path, lines: Sequence[str]) -> None:
    """Persist terminal evidence only after applying the shared secret boundary."""

    original = "\n".join(str(line) for line in lines).rstrip() + "\n"
    path.write_text(str(redact(original)), encoding="utf-8")


@dataclass(frozen=True)
class SimulatorContext:
    """Only information available after the preceding response completed."""

    session_id: str
    turn_index: int
    previous_agent_response: str
    previous_response_received_at_ns: int
    scheduled_target: ScheduledCoverageTarget
    transcript: tuple[tuple[str, str], ...]
    coverage_contract: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulatorDecision:
    """One live user decision and the coverage contract it is targeting."""

    user_message: str
    persona: str
    goal: str
    rationale: str
    target_coverage_ids: tuple[str, ...]


class SimulatorTerminalClassification(str, Enum):
    PASSED = "passed"
    SIMULATOR_INVALID = "simulator_invalid"


class SimulatorDecisionInvalid(ValueError):
    """A simulator turn violated its immutable scheduled contract."""


@dataclass(frozen=True)
class ChaosRunConfig:
    repo_root: Path
    command: tuple[str, ...]
    session_id: str = field(default_factory=lambda: f"dynamic-chaos-{uuid.uuid4().hex}")
    execution_id: str = field(default_factory=lambda: f"chaos-{uuid.uuid4().hex}")
    session_purpose: str = "dynamic-dual-ai-chaos"
    provider: str = "deepseek"
    model: str = "deepseek-chat"
    max_turns: int = 20
    response_timeout_seconds: float = 180.0
    poll_interval_seconds: float = 0.05
    runtime_root: Path | None = None
    runtime_root_in_process: Path | None = None
    extra_env: Mapping[str, str] = field(default_factory=dict)
    transport_kind: str = "direct_pty"

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
        execution_id = str(changes.pop("execution_id", f"chaos-{uuid.uuid4().hex}"))
        changes.setdefault("provider", os.environ.get("LLM_PROVIDER", cls.provider))
        changes.setdefault("model", os.environ.get("LLM_MODEL", cls.model))
        host_runtime = Path(changes.pop(
            "runtime_root",
            root / ".agent" / "dynamic-chaos" / session_id,
        )).resolve()
        container_runtime = Path(changes.pop(
            "runtime_root_in_process",
            Path("/workspace/.agent/dynamic-chaos") / session_id,
        ))
        container_env = {
            "ANYCHAIN_CHAOS_EXECUTION_ID": execution_id,
            "ANYCHAIN_CHAOS_INNER_CLEANUP_RECEIPT_DIR": str(
                container_runtime / "container-cleanup-receipts"
            ),
            "ANYCHAIN_AGENT_CHECKPOINT_PATH": str(container_runtime / "checkpoints.sqlite"),
            "ANYCHAIN_AGENT_SESSION_ID": session_id,
            "ANYCHAIN_AGENT_SESSION_PURPOSE": str(
                changes.get("session_purpose", "dynamic-dual-ai-chaos")
            ),
            "ANYCHAIN_AGENT_JOBS_DIR": str(container_runtime / "jobs"),
            "ANYCHAIN_AGENT_TURN_EVENT_FILE": str(container_runtime / "turn-events.jsonl"),
        }
        command: list[str] = ["docker", "compose", "exec", "-T"]
        for name, value in container_env.items():
            command.extend(("-e", f"{name}={value}"))
        command.extend((
            service,
            "/workspace/.venv-adk/bin/python",
            "-m",
            "tests.agent_live.container_pty_bridge",
            "--cwd",
            "/workspace",
            "--",
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
            execution_id=execution_id,
            runtime_root=host_runtime,
            runtime_root_in_process=container_runtime,
            transport_kind="container_pty_bridge",
            **changes,
        )

    @classmethod
    def linux(
        cls,
        repo_root: str | Path,
        **changes: Any,
    ) -> "ChaosRunConfig":
        """Run the product CLI under the Linux PTY process supervisor."""

        root = Path(repo_root).resolve()
        session_id = str(changes.pop("session_id", f"dynamic-chaos-{uuid.uuid4().hex}"))
        changes.setdefault("provider", os.environ.get("LLM_PROVIDER", cls.provider))
        changes.setdefault("model", os.environ.get("LLM_MODEL", cls.model))
        runtime = root / ".agent" / "dynamic-chaos" / session_id
        return cls(
            repo_root=root,
            command=(
                str(root / ".venv-adk" / "bin" / "python"),
                "-m",
                "tests.agent_live.container_pty_bridge",
                "--cwd",
                str(root),
                "--",
                str(root / "bin" / "anychain-agent"),
                "--state-file",
                str(runtime / "terminal-session.json"),
                "--language",
                "en",
            ),
            session_id=session_id,
            runtime_root=runtime,
            runtime_root_in_process=runtime,
            transport_kind="container_pty_bridge",
            **changes,
        )


@dataclass(frozen=True)
class ChaosRunResult:
    session_id: str
    transcript_path: Path
    evidence_paths: tuple[Path, ...]
    diagnostic_paths: tuple[Path, ...]
    turns: tuple[PtyCliTurnRecord, ...]
    schedule_path: Path
    schedule_result_path: Path
    execution_status: str


@dataclass(frozen=True)
class JourneySimulatorContext:
    """Open-journey context exposed only after one complete Agent response."""

    session_id: str
    turn_index: int
    previous_agent_response: str
    previous_response_received_at_ns: int
    schedule: JourneySchedule
    transcript: tuple[tuple[str, str], ...]
    observed_edge_keys: tuple[str, ...]


@dataclass(frozen=True)
class JourneySimulatorDecision:
    """One response-driven journey turn without a predeclared future edge."""

    user_message: str
    persona: str
    mission: str
    rationale: str
    risk_factor_ids: tuple[str, ...] = ()
    broker_request_id: str = ""
    simulator_attestation: Mapping[str, Any] = field(default_factory=dict)
    variant_binding: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class JourneyObservedEdge:
    """One authoritative edge independently proven after a journey turn."""

    edge_key: str
    verifier_id: str
    observed_coverage_ids: tuple[str, ...]
    details: Mapping[str, Any]


@dataclass(frozen=True)
class JourneyDecisionProvenance:
    """One simulator decision bound to the complete response it followed."""

    turn_index: int
    previous_response_hash: str
    selected_at_ns: int
    submitted_at_ns: int
    user_message_hash: str
    persona: str
    mission: str
    rationale: str
    risk_factor_ids: tuple[str, ...] = ()
    execution_id: str = ""
    obligation_id: str = ""
    broker_request_id: str = ""
    simulator_context_binding: Mapping[str, Any] = field(default_factory=dict)
    simulator_attestation: Mapping[str, Any] = field(default_factory=dict)
    variant_attestation: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JourneyVerifierContext:
    """Immutable facts available to an injected journey postcondition verifier."""

    schedule: JourneySchedule
    initial_event: RuntimeTurnEvent
    current_event: RuntimeTurnEvent
    completed_turns: tuple[PtyCliTurnRecord, ...]
    transcript: tuple[tuple[str, str], ...]
    observed_edge_keys: tuple[str, ...]
    latest_turn: PtyCliTurnRecord | None
    completed_events: tuple[RuntimeTurnEvent, ...] = ()
    completed_decisions: tuple[JourneyDecisionProvenance, ...] = ()
    evaluating_postcondition_id: str = ""
    verifier_input_contract: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JourneyPostconditionResult:
    """Typed result returned by one injected named journey verifier."""

    postcondition_id: str
    satisfied: bool
    details: Mapping[str, Any] = field(default_factory=dict)


JOURNEY_VERIFIER_REGISTRY_SCHEMA_VERSION = 1
JOURNEY_EVIDENCE_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class JourneyPostconditionVerifierDefinition:
    """One named, versioned verifier eligible for formal Journey evidence."""

    postcondition_id: str
    verifier_id: str
    verifier_version: int
    description: str
    verifier: "JourneyPostconditionVerifier"
    implementation_hash: str = ""


@dataclass(frozen=True)
class JourneyOutcomeVerifierRegistry:
    """Immutable authoritative registry bound into every Journey artifact."""

    registry_id: str
    definitions: Mapping[str, JourneyPostconditionVerifierDefinition]
    schema_version: int = JOURNEY_VERIFIER_REGISTRY_SCHEMA_VERSION


@dataclass(frozen=True)
class JourneyVerifierBinding:
    postcondition_id: str
    verifier_id: str
    verifier_version: int
    implementation_hash: str


@dataclass(frozen=True)
class JourneyOutcomeVerification:
    """Evaluation of one terminal or forbidden outcome contract."""

    outcome_id: str
    satisfied: bool
    postconditions: tuple[JourneyPostconditionResult, ...]
    verifier_bindings: tuple[JourneyVerifierBinding, ...]


class JourneyTerminalClassification(str, Enum):
    PASSED = "passed"
    PRODUCT_FAILED = "product_failed"
    SIMULATOR_INVALID = "simulator_invalid"
    INFRASTRUCTURE_INTERRUPTED = "infrastructure_interrupted"
    EXTERNALLY_BLOCKED = "externally_blocked"


class JourneyRunError(RuntimeError):
    classification = JourneyTerminalClassification.INFRASTRUCTURE_INTERRUPTED


class JourneyProductFailure(JourneyRunError):
    classification = JourneyTerminalClassification.PRODUCT_FAILED


class JourneySimulatorInvalidError(JourneyRunError):
    classification = JourneyTerminalClassification.SIMULATOR_INVALID


class JourneyInfrastructureInterruptedError(JourneyRunError):
    classification = JourneyTerminalClassification.INFRASTRUCTURE_INTERRUPTED


class JourneyExternallyBlockedError(JourneyRunError):
    classification = JourneyTerminalClassification.EXTERNALLY_BLOCKED


@dataclass(frozen=True)
class JourneyRunResult:
    session_id: str
    transcript_path: Path
    schedule_path: Path
    journey_result_path: Path
    evidence_path: Path
    evidence_id: str
    turns: tuple[PtyCliTurnRecord, ...]
    observed_edge_keys: tuple[str, ...]
    terminal_classification: JourneyTerminalClassification
    terminal_outcome_id: str

    @property
    def execution_status(self) -> str:
        return self.terminal_classification.value


class Simulator(Protocol):
    """External decision boundary; this repository provides no Codex API."""

    def __call__(self, context: SimulatorContext) -> SimulatorDecision | None: ...


class JourneySimulator(Protocol):
    def __call__(
        self,
        context: JourneySimulatorContext,
    ) -> JourneySimulatorDecision | None: ...


class JourneyPostconditionVerifier(Protocol):
    def __call__(
        self,
        context: JourneyVerifierContext,
    ) -> JourneyPostconditionResult: ...


def build_journey_outcome_verifier_registry(
    definitions: Sequence[JourneyPostconditionVerifierDefinition],
) -> JourneyOutcomeVerifierRegistry:
    """Build an immutable registry whose identity includes verifier code and version."""

    if not definitions:
        raise ValueError("Journey verifier registry requires at least one definition")
    normalized: dict[str, JourneyPostconditionVerifierDefinition] = {}
    for definition in definitions:
        postcondition_id = str(definition.postcondition_id or "").strip()
        verifier_id = str(definition.verifier_id or "").strip()
        description = str(definition.description or "").strip()
        if not postcondition_id or not verifier_id or not description:
            raise ValueError(
                "Journey verifier definitions require postcondition_id, verifier_id, and description"
            )
        if postcondition_id in normalized:
            raise ValueError(f"duplicate Journey postcondition verifier: {postcondition_id}")
        if isinstance(definition.verifier_version, bool) or definition.verifier_version <= 0:
            raise ValueError("Journey verifier_version must be a positive integer")
        verifier = definition.verifier
        if not inspect.isfunction(verifier):
            raise TypeError("formal Journey verifiers must be top-level named functions")
        if verifier.__name__ == "<lambda>" or "<locals>" in verifier.__qualname__:
            raise TypeError("lambda and local-closure Journey verifiers cannot qualify evidence")
        implementation_identity = {
            "module": verifier.__module__,
            "qualname": verifier.__qualname__,
            "source": _journey_verifier_source(verifier),
        }
        implementation_hash = content_hash(implementation_identity)
        if definition.implementation_hash and definition.implementation_hash != implementation_hash:
            raise ValueError(
                f"stale implementation hash for Journey verifier {postcondition_id}"
            )
        normalized[postcondition_id] = JourneyPostconditionVerifierDefinition(
            postcondition_id=postcondition_id,
            verifier_id=verifier_id,
            verifier_version=definition.verifier_version,
            description=description,
            verifier=verifier,
            implementation_hash=implementation_hash,
        )
    registry_id = content_hash({
        "schema_version": JOURNEY_VERIFIER_REGISTRY_SCHEMA_VERSION,
        "definitions": [
            _journey_verifier_definition_payload(normalized[key])
            for key in sorted(normalized)
        ],
    })
    return JourneyOutcomeVerifierRegistry(
        registry_id=registry_id,
        definitions=MappingProxyType(normalized),
    )


def journey_outcome_verifier_registry_payload(
    registry: JourneyOutcomeVerifierRegistry,
) -> dict[str, Any]:
    return {
        "schema_version": registry.schema_version,
        "registry_id": registry.registry_id,
        "definitions": [
            _journey_verifier_definition_payload(registry.definitions[key])
            for key in sorted(registry.definitions)
        ],
    }


def validate_journey_outcome_verifier_registry(
    registry: JourneyOutcomeVerifierRegistry,
) -> None:
    if registry.schema_version != JOURNEY_VERIFIER_REGISTRY_SCHEMA_VERSION:
        raise ValueError("unsupported Journey verifier registry schema")
    rebuilt = build_journey_outcome_verifier_registry(tuple(registry.definitions.values()))
    if rebuilt.registry_id != registry.registry_id:
        raise ValueError("Journey verifier registry identity is stale or was modified")


def _journey_verifier_definition_payload(
    definition: JourneyPostconditionVerifierDefinition,
) -> dict[str, Any]:
    return {
        "postcondition_id": definition.postcondition_id,
        "verifier_id": definition.verifier_id,
        "verifier_version": definition.verifier_version,
        "description": definition.description,
        "implementation_hash": definition.implementation_hash,
    }


def _journey_verifier_source(verifier: Callable[..., Any]) -> str:
    try:
        return inspect.getsource(verifier)
    except (OSError, TypeError):
        code = verifier.__code__
        return repr((code.co_code.hex(), code.co_consts, code.co_names))


class PtyTransport(Protocol):
    def start(self, *, env: Mapping[str, str]) -> None: ...

    def read_complete_agent_response(self, *, timeout_seconds: float) -> str: ...

    def submit_bracketed_paste(self, message: str) -> None: ...

    def send_interrupt(self) -> None: ...

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
        # A real terminal delivers the paste boundary before the user's later
        # Enter key. Give prompt-toolkit one input cycle to leave paste mode;
        # otherwise multiline content can remain in the edit buffer forever.
        time.sleep(self.poll_interval_seconds)
        os.write(self._master_fd, b"\r")

    def send_interrupt(self) -> None:
        if self._master_fd is None:
            raise RuntimeError("PTY transport is not running")
        os.write(self._master_fd, b"\x03")

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


class ContainerPtyBridgeTransport:
    """Control one container-local product PTY over a pipe-framed RPC bridge."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path,
        poll_interval_seconds: float = 0.05,
        execution_id: str | None = None,
        cleanup_receipt_dir: str | Path | None = None,
    ) -> None:
        self.command = tuple(command)
        self.cwd = Path(cwd)
        self.poll_interval_seconds = poll_interval_seconds
        self.execution_id = execution_id or f"chaos-{uuid.uuid4().hex}"
        self.cleanup_receipt_dir = (
            Path(cleanup_receipt_dir) if cleanup_receipt_dir is not None else None
        )
        self.cleanup_receipt: Mapping[str, Any] | None = None
        self._temporary_receipt_dir: tempfile.TemporaryDirectory[str] | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._stdout_buffer = bytearray()
        self._stderr_buffer = bytearray()
        self._stderr_lock = threading.Lock()
        self._stderr_thread: threading.Thread | None = None
        self._stderr_cap_bytes = 64 * 1024

    def start(self, *, env: Mapping[str, str]) -> None:
        if self._process is not None:
            raise RuntimeError("container PTY bridge transport has already started")
        process_env = dict(env)
        process_env.setdefault("ANYCHAIN_CHAOS_EXECUTION_ID", self.execution_id)
        if self.cleanup_receipt_dir is None:
            self._temporary_receipt_dir = tempfile.TemporaryDirectory(
                prefix="anychain-container-cleanup-"
            )
            receipt_dir = Path(self._temporary_receipt_dir.name)
        else:
            receipt_dir = self.cleanup_receipt_dir
        receipt_dir.mkdir(parents=True, exist_ok=True)
        process_env.setdefault(
            "ANYCHAIN_CHAOS_INNER_CLEANUP_RECEIPT_DIR", str(receipt_dir)
        )
        self._process = subprocess.Popen(
            self.command,
            cwd=self.cwd,
            env=process_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            close_fds=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(self._process,),
            name="anychain-container-pty-stderr",
            daemon=True,
        )
        self._stderr_thread.start()

    def read_complete_agent_response(self, *, timeout_seconds: float) -> str:
        result = self._request(
            {"op": "read", "timeout_seconds": timeout_seconds},
            timeout_seconds=timeout_seconds + 5.0,
        )
        response = str(result.get("response") or "")
        if not response:
            raise RuntimeError("container PTY bridge returned an empty Agent response")
        return response

    def submit_bracketed_paste(self, message: str) -> None:
        self._request(
            {"op": "submit", "message": message},
            timeout_seconds=max(5.0, self.poll_interval_seconds * 20),
        )

    def send_interrupt(self) -> None:
        self._request({"op": "interrupt"}, timeout_seconds=5.0)

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None:
            try:
                response = self._request_with_process(
                    process, {"op": "close"}, timeout_seconds=5.0
                )
                self.cleanup_receipt = self._validate_cleanup_receipt(
                    response.get("cleanup_receipt")
                )
                process.wait(timeout=5.0)
            except (OSError, RuntimeError, subprocess.TimeoutExpired):
                self._terminate_process_group(process)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        stderr_thread = self._stderr_thread
        self._stderr_thread = None
        if stderr_thread is not None:
            stderr_thread.join(timeout=1.0)
        temporary = self._temporary_receipt_dir
        self._temporary_receipt_dir = None
        if temporary is not None:
            temporary.cleanup()

    def validated_execution_proof(self) -> Mapping[str, Any]:
        """Return the authoritative container PTY/process cleanup proof."""

        if self.cleanup_receipt_dir is None or self.cleanup_receipt is None:
            raise RuntimeError(
                "container PTY execution has no persistent cleanup receipt"
            )
        from tests.agent_live.container_process_guard import (
            validate_cleanup_receipt_artifact,
        )

        paths = tuple(
            self.cleanup_receipt_dir.glob(
                "container-cleanup-receipt-*.json"
            )
        )
        if len(paths) != 1:
            raise RuntimeError(
                "container PTY execution must produce exactly one cleanup receipt"
            )
        proof = validate_cleanup_receipt_artifact(
            paths[0],
            execution_id=self.execution_id,
            required_roles=(
                "container_bridge",
                "agent_process_group_leader",
            ),
            allowed_roots=(self.cleanup_receipt_dir,),
        )
        comparable = {
            key: proof[key]
            for key in ("receipt_id", "sha256", "cleaned")
        }
        observed = {
            key: self.cleanup_receipt[key]
            for key in ("receipt_id", "sha256", "cleaned")
        }
        if comparable != observed:
            raise RuntimeError(
                "container PTY cleanup summary differs from its receipt"
            )
        return {
            "proof_type": "container_pty_process_guard",
            "transport_kind": "container_pty_bridge",
            **proof,
        }

    def _request(self, payload: Mapping[str, Any], *, timeout_seconds: float) -> dict[str, Any]:
        if self._process is None:
            raise RuntimeError("container PTY bridge transport is not running")
        return self._request_with_process(self._process, payload, timeout_seconds=timeout_seconds)

    def _request_with_process(
        self,
        process: subprocess.Popen[bytes],
        payload: Mapping[str, Any],
        *,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("container PTY bridge pipes are unavailable")
        if process.poll() is not None:
            raise RuntimeError(self._exit_message(process))
        process.stdin.write(json.dumps(dict(payload), ensure_ascii=False).encode("utf-8") + b"\n")
        process.stdin.flush()
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            line = self._pop_stdout_line()
            if line is not None:
                response = json.loads(line.decode("utf-8"))
                if not response.get("ok"):
                    raise RuntimeError(
                        "container PTY bridge operation failed: "
                        f"{response.get('error_type')}: {response.get('error')}"
                    )
                return dict(response)
            if process.poll() is not None:
                raise RuntimeError(self._exit_message(process))
            ready, _, _ = select.select(
                [process.stdout],
                [],
                [],
                min(self.poll_interval_seconds, max(0.0, deadline - time.monotonic())),
            )
            if not ready:
                continue
            chunk = os.read(process.stdout.fileno(), 65536)
            if not chunk:
                continue
            self._stdout_buffer.extend(chunk)
        raise TimeoutError(
            f"timed out after {timeout_seconds:.1f}s waiting for the container PTY bridge"
        )

    @staticmethod
    def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=3.0)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=3.0)

    def _validate_cleanup_receipt(self, value: Any) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise RuntimeError("container PTY bridge returned no cleanup receipt")
        required = {"receipt_id", "path", "sha256", "cleaned"}
        if set(value) != required or not bool(value.get("cleaned")):
            raise RuntimeError("container PTY bridge returned an invalid cleanup receipt")
        if not all(str(value.get(key) or "") for key in ("receipt_id", "path", "sha256")):
            raise RuntimeError("container PTY bridge cleanup receipt is incomplete")
        return dict(value)

    def _pop_stdout_line(self) -> bytes | None:
        newline = self._stdout_buffer.find(b"\n")
        if newline < 0:
            return None
        line = bytes(self._stdout_buffer[:newline])
        del self._stdout_buffer[:newline + 1]
        return line

    def _drain_stderr(self, process: subprocess.Popen[bytes]) -> None:
        if process.stderr is None:
            return
        while True:
            try:
                chunk = process.stderr.read(8192)
            except (OSError, ValueError):
                return
            if not chunk:
                return
            with self._stderr_lock:
                self._stderr_buffer.extend(chunk)
                overflow = len(self._stderr_buffer) - self._stderr_cap_bytes
                if overflow > 0:
                    del self._stderr_buffer[:overflow]

    def _exit_message(self, process: subprocess.Popen[bytes]) -> str:
        with self._stderr_lock:
            stderr = bytes(self._stderr_buffer)
        return (
            f"container PTY bridge exited unexpectedly (exit={process.returncode}): "
            f"{stderr.decode('utf-8', errors='replace')[-2000:]}"
        )


def transport_for_config(config: ChaosRunConfig) -> PtyTransport:
    if config.transport_kind == "container_pty_bridge":
        return ContainerPtyBridgeTransport(
            config.command,
            cwd=config.repo_root,
            poll_interval_seconds=config.poll_interval_seconds,
            execution_id=config.execution_id,
            cleanup_receipt_dir=(
                config.runtime_root / "container-cleanup-receipts"
                if config.runtime_root is not None
                else None
            ),
        )
    if config.transport_kind != "direct_pty":
        raise ValueError(f"unsupported PTY transport kind: {config.transport_kind!r}")
    return SubprocessPtyTransport(
        config.command,
        cwd=config.repo_root,
        poll_interval_seconds=config.poll_interval_seconds,
    )


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
        self.transport = transport or transport_for_config(config)
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
        diagnostic_dir = runtime_root / "diagnostics"
        transcript_path = runtime_root / "transcript.txt"
        schedule_path = write_chaos_schedule(self.schedule, runtime_root / "schedule.json")
        schedule_result_path = runtime_root / "schedule-result.json"
        env = self._isolated_environment(runtime_root)

        seed_scenario_id = str(self.schedule.targets[0].scenario_id or "")
        if seed_scenario_id:
            from tests.agent_live.runtime_checkpoint import (
                reviewed_scenario,
                seed_runtime_checkpoint,
            )

            seed_scenario = reviewed_scenario(seed_scenario_id)
            seed_runtime_checkpoint(
                seed_scenario.seed_state,
                checkpoint_path=runtime_root / "checkpoints.sqlite",
                session_id=self.config.session_id,
                session_purpose=self.config.session_purpose,
                scenario_id=seed_scenario.scenario_id,
                scenario_state_fingerprint=seed_scenario.state_fingerprint,
            )

        transcript: list[tuple[str, str]] = []
        transcript_lines: list[str] = []
        evidence_paths: list[Path] = []
        diagnostic_paths: list[Path] = []
        turns: list[PtyCliTurnRecord] = []
        target_results: list[dict[str, Any]] = []
        execution_status = "incomplete"
        last_complete_response = ""
        last_complete_event: RuntimeTurnEvent | None = None
        last_complete_turn: PtyCliTurnRecord | None = None
        last_complete_selection: DynamicTurnSelection | None = None

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
            last_complete_response = previous_response
            last_complete_event = baseline_event

            if seed_scenario_id:
                first_target = self.schedule.targets[0]
                first_edge = self.edge_index[first_target.edge_key]
                expected_question = str(first_edge.get("question_id") or "")
                if (
                    baseline_event.pending_question_id == "resume_harness_session"
                    and expected_question != "resume_harness_session"
                ):
                    resume_choice = (
                        "2"
                        if str(first_edge.get("edge_type") or "") == "action_transition"
                        else "1"
                    )
                    self.transport.submit_bracketed_paste(resume_choice)
                    resumed_response = self.transport.read_complete_agent_response(
                        timeout_seconds=self.config.response_timeout_seconds
                    )
                    resumed_event = self.event_stream.next_event(
                        timeout_seconds=self.config.response_timeout_seconds
                    )
                    self._validate_event_revision(resumed_event)
                    transcript.extend(((resume_choice, resumed_response),))
                    transcript_lines.extend((f"User> {resume_choice}", resumed_response))
                    previous_response = resumed_response
                    previous_received_ns = self.clock_ns()
                    baseline_event = resumed_event
                    last_complete_response = resumed_response
                    last_complete_event = resumed_event
                    last_complete_turn = None
                    last_complete_selection = None
                if (
                    expected_question
                    and baseline_event.pending_question_id != expected_question
                ):
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
                _require_scheduled_baseline_contract(baseline_event, edge)
                context = SimulatorContext(
                    session_id=self.config.session_id,
                    turn_index=baseline_event.turn_index + 1,
                    previous_agent_response=previous_response,
                    previous_response_received_at_ns=previous_received_ns,
                    scheduled_target=scheduled_target,
                    transcript=tuple(transcript),
                    coverage_contract=dict(edge),
                )
                decision = self.simulator(context)
                selection_observed_at_ns = self.clock_ns()
                if decision is None:
                    target_results.append({
                        "target_id": scheduled_target.target_id,
                        "edge_key": scheduled_target.edge_key,
                        "status": "externally_blocked",
                        "reason": "external Codex simulator did not provide a decision",
                    })
                    raise RuntimeError("external Codex simulator did not provide a decision")
                _validate_decision(decision, scheduled_target, edge)
                submitted_at_ns = self.clock_ns()
                self.transport.submit_bracketed_paste(decision.user_message)
                committed_event = self.event_stream.next_event(
                    timeout_seconds=self.config.response_timeout_seconds
                )
                self._validate_event_revision(committed_event)
                if (
                    committed_event.before_fingerprint
                    != baseline_event.after_fingerprint
                    or committed_event.turn_index
                    != baseline_event.turn_index + 1
                ):
                    raise JourneyInfrastructureInterruptedError(
                        "Journey runtime event lineage skipped or diverged"
                    )
                response = self.transport.read_complete_agent_response(
                    timeout_seconds=self.config.response_timeout_seconds
                )
                response_received_ns = self.clock_ns()

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
                    selected_at_ns=selection_observed_at_ns,
                )
                turns.append(turn)
                transcript.append((decision.user_message, response))
                transcript_lines.extend((f"User> {decision.user_message}", response))
                root_previous_response = previous_response
                root_committed_event = committed_event
                continuation_events: list[RuntimeTurnEvent] = []
                continuation_turns: list[PtyCliTurnRecord] = []
                continuation_selections: list[DynamicTurnSelection] = []
                last_complete_response = response
                last_complete_event = committed_event
                last_complete_turn = turn
                last_complete_selection = selection

                pending_diagnostic = build_pty_diagnostic_artifact(
                    PtyDiagnosticRecord(
                        diagnostic_kind="failed_attempt",
                        verification_status="verification_pending",
                        target_id=scheduled_target.target_id,
                        target_edge_key=scheduled_target.edge_key,
                        revision=self.revision,
                        session_id=self.config.session_id,
                        reason="completed PTY boundary awaiting postcondition verification",
                        last_complete_response=response,
                        last_complete_event=committed_event,
                        completed_turn=turn,
                        dynamic_selection=selection,
                    )
                )
                pending_diagnostic_path = write_pty_diagnostic_artifact(
                    pending_diagnostic,
                    diagnostic_dir,
                )
                try:
                    verified_postcondition = _verify_declared_postconditions(
                        baseline_event,
                        committed_event,
                        turn,
                        edge_index=self.edge_index,
                        target_coverage_ids=selection.target_coverage_ids,
                    )
                except Exception as exc:
                    pending_diagnostic_path.unlink(missing_ok=True)
                    verification_error = build_pty_diagnostic_artifact(
                        PtyDiagnosticRecord(
                            diagnostic_kind="failed_attempt",
                            verification_status="verification_error",
                            target_id=scheduled_target.target_id,
                            target_edge_key=scheduled_target.edge_key,
                            revision=self.revision,
                            session_id=self.config.session_id,
                            reason=f"{type(exc).__name__}: {exc}",
                            last_complete_response=response,
                            last_complete_event=committed_event,
                            completed_turn=turn,
                            dynamic_selection=selection,
                        )
                    )
                    verification_error_path = write_pty_diagnostic_artifact(
                        verification_error,
                        diagnostic_dir,
                    )
                    diagnostic_paths.append(verification_error_path)
                    target_results.append({
                        "target_id": scheduled_target.target_id,
                        "edge_key": scheduled_target.edge_key,
                        "status": "failed",
                        "reason": str(redact(f"{type(exc).__name__}: {exc}")),
                        "diagnostic_path": str(verification_error_path),
                        "diagnostic_id": verification_error["diagnostic_id"],
                    })
                    raise
                while (
                    not verified_postcondition.passed
                    and _verification_requires_continuation(verified_postcondition)
                    and len(continuation_events)
                    < scheduled_target.continuation_turn_budget
                    and len(turns) < self.config.max_turns
                ):
                    continuation_contract = {
                        **dict(edge),
                        "execution_phase": "linked_prerequisite_continuation",
                        "continuation": {
                            "turn_number": len(continuation_events) + 1,
                            "turn_budget": scheduled_target.continuation_turn_budget,
                            "terminal_postcondition": dict(
                                (edge.get("deferred_transition_contract") or {}).get(
                                    "terminal_postcondition"
                                )
                                or edge.get("expected_postcondition")
                                or {}
                            ),
                            "current_pending_question_id": committed_event.pending_question_id,
                            "current_pending_contract": dict(committed_event.pending_contract),
                            "selection_rule": (
                                "Read the complete current Agent response, choose one natural "
                                "next user turn that advances its prerequisite workflow, and do "
                                "not assume or pre-generate later answers."
                            ),
                        },
                    }
                    continuation_context = SimulatorContext(
                        session_id=self.config.session_id,
                        turn_index=committed_event.turn_index + 1,
                        previous_agent_response=response,
                        previous_response_received_at_ns=response_received_ns,
                        scheduled_target=scheduled_target,
                        transcript=tuple(transcript),
                        coverage_contract=continuation_contract,
                    )
                    continuation_decision = self.simulator(continuation_context)
                    continuation_selected_at_ns = self.clock_ns()
                    if continuation_decision is None:
                        raise RuntimeError(
                            "external Codex simulator did not provide a continuation decision"
                        )
                    _validate_continuation_decision(
                        continuation_decision,
                        scheduled_target,
                    )
                    continuation_submitted_at_ns = self.clock_ns()
                    self.transport.submit_bracketed_paste(
                        continuation_decision.user_message
                    )
                    continuation_event = self.event_stream.next_event(
                        timeout_seconds=self.config.response_timeout_seconds
                    )
                    self._validate_event_revision(continuation_event)
                    continuation_response = self.transport.read_complete_agent_response(
                        timeout_seconds=self.config.response_timeout_seconds
                    )
                    continuation_received_ns = self.clock_ns()
                    continuation_transcript_data = {
                        "session_id": self.config.session_id,
                        "turn_index": continuation_event.turn_index,
                        "previous_agent_response": response,
                        "user_message": continuation_decision.user_message,
                        "agent_response": continuation_response,
                    }
                    continuation_turn = PtyCliTurnRecord(
                        **continuation_transcript_data,
                        provider=observed_provider,
                        model=observed_model,
                        before_fingerprint=continuation_event.before_fingerprint,
                        after_fingerprint=continuation_event.after_fingerprint,
                        transcript_hash=pty_transcript_hash(
                            **continuation_transcript_data
                        ),
                        previous_response_received_at_ns=response_received_ns,
                        user_message_submitted_at_ns=continuation_submitted_at_ns,
                        agent_response_received_at_ns=continuation_received_ns,
                    )
                    continuation_selection = DynamicTurnSelection(
                        persona=continuation_decision.persona,
                        goal=continuation_decision.goal,
                        selected_message=continuation_decision.user_message,
                        rationale=continuation_decision.rationale,
                        target_coverage_ids=tuple(
                            continuation_decision.target_coverage_ids
                        ),
                        selected_at_ns=continuation_selected_at_ns,
                    )
                    continuation_events.append(continuation_event)
                    continuation_turns.append(continuation_turn)
                    continuation_selections.append(continuation_selection)
                    turns.append(continuation_turn)
                    transcript.append(
                        (continuation_decision.user_message, continuation_response)
                    )
                    transcript_lines.extend((
                        f"User> {continuation_decision.user_message}",
                        continuation_response,
                    ))
                    response = continuation_response
                    response_received_ns = continuation_received_ns
                    committed_event = continuation_event
                    last_complete_response = response
                    last_complete_event = committed_event
                    last_complete_turn = continuation_turn
                    last_complete_selection = continuation_selection
                    verified_postcondition = _verify_declared_postconditions(
                        baseline_event,
                        root_committed_event,
                        turn,
                        edge_index=self.edge_index,
                        target_coverage_ids=selection.target_coverage_ids,
                        continuation_events=continuation_events,
                    )
                if not verified_postcondition.passed:
                    errors = verified_postcondition.details.get("errors") or ()
                    failure_reason = (
                        "turn observation postcondition did not pass: "
                        + "; ".join(str(item) for item in errors)
                    )
                    pending_diagnostic_path.unlink(missing_ok=True)
                    failed_diagnostic = build_pty_diagnostic_artifact(
                        PtyDiagnosticRecord(
                            diagnostic_kind="failed_attempt",
                            verification_status="postcondition_failed",
                            target_id=scheduled_target.target_id,
                            target_edge_key=scheduled_target.edge_key,
                            revision=self.revision,
                            session_id=self.config.session_id,
                            reason=failure_reason,
                            last_complete_response=response,
                            last_complete_event=committed_event,
                            completed_turn=last_complete_turn,
                            dynamic_selection=last_complete_selection,
                            verified_postcondition=verified_postcondition,
                        )
                    )
                    failed_diagnostic_path = write_pty_diagnostic_artifact(
                        failed_diagnostic,
                        diagnostic_dir,
                    )
                    diagnostic_paths.append(failed_diagnostic_path)
                    target_results.append({
                        "target_id": scheduled_target.target_id,
                        "edge_key": scheduled_target.edge_key,
                        "status": "failed",
                        "reason": str(redact(failure_reason)),
                        "diagnostic_path": str(failed_diagnostic_path),
                        "diagnostic_id": failed_diagnostic["diagnostic_id"],
                    })
                    raise ValueError(
                        failure_reason
                    )
                pending_diagnostic_path.unlink(missing_ok=True)
                observation = TurnObservation(
                    seed=self.schedule.seed,
                    revision=self.revision,
                    target_edge_key=str(edge.get("edge_key") or ""),
                    target_contract_hash=str(edge.get("contract_hash") or ""),
                    target_variant_hash=str(edge.get("contract_variant_hash") or ""),
                    prior_agent_response=root_previous_response,
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
                    before_state_fingerprint=root_committed_event.before_fingerprint,
                    after_state_fingerprint=committed_event.after_fingerprint,
                    pending_contract=dict(baseline_event.pending_contract),
                    runtime_events=(
                        baseline_event,
                        root_committed_event,
                        *continuation_events,
                    ),
                    verified_postcondition=verified_postcondition,
                    continuation_turns=tuple(continuation_turns),
                    continuation_simulator_decisions=tuple({
                        "persona": item.persona,
                        "goal": item.goal,
                        "selected_message": item.selected_message,
                        "rationale": item.rationale,
                        "target_coverage_ids": list(item.target_coverage_ids),
                        "selected_at_ns": item.selected_at_ns,
                        "simulator": item.simulator,
                        "selection_mode": item.selection_mode,
                    } for item in continuation_selections),
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
                        continuation_turns=observation.continuation_turns,
                        continuation_simulator_decisions=(),
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
                    interruption_reason = (
                        f"{type(exc).__name__}: PTY/runtime boundary interrupted before completion"
                    )
                    failed_result = {
                        "target_id": pending.target_id,
                        "edge_key": pending.edge_key,
                        "status": "failed",
                        "reason": interruption_reason,
                    }
                    if last_complete_event is not None and last_complete_response:
                        interruption = build_pty_diagnostic_artifact(
                            PtyDiagnosticRecord(
                                diagnostic_kind="interruption",
                                verification_status="interrupted",
                                target_id=pending.target_id,
                                target_edge_key=pending.edge_key,
                                revision=self.revision,
                                session_id=self.config.session_id,
                                reason=interruption_reason,
                                last_complete_response=last_complete_response,
                                last_complete_event=last_complete_event,
                                completed_turn=last_complete_turn,
                                dynamic_selection=last_complete_selection,
                            )
                        )
                        interruption_path = write_pty_diagnostic_artifact(
                            interruption,
                            diagnostic_dir,
                        )
                        diagnostic_paths.append(interruption_path)
                        failed_result.update({
                            "diagnostic_path": str(interruption_path),
                            "diagnostic_id": interruption["diagnostic_id"],
                        })
                    target_results.append(failed_result)
            raise
        finally:
            self.transport.close()
            _write_redacted_transcript(transcript_path, transcript_lines)
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
            diagnostic_paths=tuple(diagnostic_paths),
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


class DynamicDualAiJourneyRunner:
    """Run an open JourneySchedule without prescribing future user turns."""

    def __init__(
        self,
        config: ChaosRunConfig,
        simulator: JourneySimulator,
        *,
        ledger: Mapping[str, Any],
        schedule: JourneySchedule,
        postcondition_verifier_registry: JourneyOutcomeVerifierRegistry,
        transport: PtyTransport | None = None,
        event_stream: RuntimeEventStream | None = None,
        revision: Mapping[str, str] | None = None,
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.config = config
        self.simulator = simulator
        self.transport = transport or transport_for_config(config)
        self.ledger = dict(ledger)
        self.schedule = schedule
        self.revision = dict(revision or repository_revision(config.repo_root))
        validate_journey_schedule(schedule, revision=self.revision)
        if dict(self.ledger.get("revision") or {}) != self.revision:
            raise ValueError("coverage ledger revision does not match the journey revision")
        self.edge_index = {
            str(edge.get("edge_key") or ""): edge
            for edge in self.ledger.get("edges") or ()
            if str(edge.get("edge_key") or "")
        }
        if not isinstance(postcondition_verifier_registry, JourneyOutcomeVerifierRegistry):
            raise TypeError(
                "Journey runner requires an authoritative JourneyOutcomeVerifierRegistry"
            )
        validate_journey_outcome_verifier_registry(postcondition_verifier_registry)
        self.postcondition_verifier_registry = postcondition_verifier_registry
        self._validate_verifier_contracts()
        default_event_path = (
            (config.runtime_root or config.repo_root / ".agent" / "dynamic-chaos" / config.session_id)
            / "turn-events.jsonl"
        )
        self.event_stream = event_stream or JsonlRuntimeEventStream(
            default_event_path,
            poll_interval_seconds=config.poll_interval_seconds,
        )
        self.clock_ns = clock_ns

    def run(self) -> JourneyRunResult:
        runtime_root = (self.config.runtime_root or (
            self.config.repo_root / ".agent" / "dynamic-chaos" / self.config.session_id
        )).resolve()
        runtime_root.mkdir(parents=True, exist_ok=True)
        transcript_path = runtime_root / "transcript.txt"
        schedule_path = write_journey_schedule(self.schedule, runtime_root / "journey-schedule.json")
        journey_result_path = runtime_root / "journey-result.json"
        evidence_dir = runtime_root / "evidence"
        env = self._isolated_environment(runtime_root)

        from tests.agent_live.runtime_checkpoint import reviewed_scenario, seed_runtime_checkpoint

        start_scenario = reviewed_scenario(self.schedule.start_scenario)
        seed_runtime_checkpoint(
            start_scenario.seed_state,
            checkpoint_path=runtime_root / "checkpoints.sqlite",
            session_id=self.config.session_id,
            session_purpose=self.config.session_purpose,
            scenario_id=start_scenario.scenario_id,
            scenario_state_fingerprint=start_scenario.state_fingerprint,
        )

        transcript: list[tuple[str, str]] = []
        transcript_lines: list[str] = []
        turns: list[PtyCliTurnRecord] = []
        events: list[RuntimeTurnEvent] = []
        decisions: list[JourneyDecisionProvenance] = []
        observed_edge_keys: list[str] = []
        turn_results: list[dict[str, Any]] = []
        terminal_classification = JourneyTerminalClassification.INFRASTRUCTURE_INTERRUPTED
        terminal_outcome_id = ""
        failure_reason = ""
        initial_event: RuntimeTurnEvent | None = None
        initial_verification: dict[str, Any] = {}
        observed_provider = ""
        observed_model = ""
        execution_proof: Mapping[str, Any] = {}
        qualification_reason = "execution_not_completed"

        try:
            self.transport.start(env=env)
            previous_response = self.transport.read_complete_agent_response(
                timeout_seconds=self.config.response_timeout_seconds
            )
            previous_received_ns = self.clock_ns()
            observed_provider, observed_model = _provider_model_from_startup(previous_response)
            if observed_provider != self.config.provider or observed_model != self.config.model:
                raise JourneyInfrastructureInterruptedError(
                    "observed provider/model does not match the configured Chaos boundary: "
                    f"{observed_provider}/{observed_model}"
                )
            baseline_event = self.event_stream.baseline()
            self._validate_event_revision(baseline_event)
            initial_event = baseline_event
            transcript_lines.append(previous_response)

            initial_context = self._verifier_context(
                initial_event=initial_event,
                current_event=baseline_event,
                turns=turns,
                events=events,
                decisions=decisions,
                transcript=transcript,
                observed_edge_keys=observed_edge_keys,
                latest_turn=None,
            )
            initial_forbidden = self._verify_forbidden_outcomes(initial_context)
            initial_terminal = self._verify_outcome(
                self.schedule.terminal_outcome,
                initial_context,
            )
            initial_verification = {
                "terminal_outcome": _journey_outcome_verification_payload(
                    initial_terminal
                ),
                "forbidden_outcomes": [
                    _journey_outcome_verification_payload(item)
                    for item in initial_forbidden
                ],
            }
            initial_violation = _satisfied_journey_outcome(initial_forbidden)
            if initial_violation is not None:
                raise JourneyProductFailure(
                    "journey reached forbidden outcome: " + initial_violation.outcome_id
                )
            if initial_terminal.satisfied:
                terminal_classification = JourneyTerminalClassification.PASSED
                terminal_outcome_id = initial_terminal.outcome_id

            while (
                terminal_classification != JourneyTerminalClassification.PASSED
                and len(turns) < self.schedule.max_turns
            ):
                context = JourneySimulatorContext(
                    session_id=self.config.session_id,
                    turn_index=baseline_event.turn_index + 1,
                    previous_agent_response=previous_response,
                    previous_response_received_at_ns=previous_received_ns,
                    schedule=self.schedule,
                    transcript=tuple(transcript),
                    observed_edge_keys=tuple(observed_edge_keys),
                )
                decision = self.simulator(context)
                selection_observed_at_ns = self.clock_ns()
                if decision is None:
                    raise JourneyExternallyBlockedError(
                        "external Codex simulator did not provide a journey decision"
                    )
                try:
                    _validate_journey_decision(decision, self.schedule)
                except (TypeError, ValueError) as exc:
                    raise JourneySimulatorInvalidError(str(exc)) from exc

                submitted_at_ns = self.clock_ns()
                (
                    simulator_attestation,
                    variant_attestation,
                    selected_at_ns,
                ) = self._validated_decision_attestations(
                    context=context,
                    decision=decision,
                    selection_observed_at_ns=selection_observed_at_ns,
                    submitted_at_ns=submitted_at_ns,
                    prior_decisions=decisions,
                )
                self.transport.submit_bracketed_paste(decision.user_message)
                committed_event = self.event_stream.next_event(
                    timeout_seconds=self.config.response_timeout_seconds
                )
                self._validate_event_revision(committed_event)
                response = self.transport.read_complete_agent_response(
                    timeout_seconds=self.config.response_timeout_seconds
                )
                response_received_ns = self.clock_ns()

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
                turns.append(turn)
                events.append(committed_event)
                from tests.agent_live.codex_simulator_bridge import (
                    simulator_context_binding,
                )

                decisions.append(JourneyDecisionProvenance(
                    turn_index=committed_event.turn_index,
                    previous_response_hash=content_hash(previous_response),
                    selected_at_ns=selected_at_ns,
                    submitted_at_ns=submitted_at_ns,
                    user_message_hash=content_hash(decision.user_message),
                    persona=decision.persona,
                    mission=decision.mission,
                    rationale=decision.rationale,
                    risk_factor_ids=tuple(decision.risk_factor_ids),
                    execution_id=self.config.execution_id,
                    obligation_id=self.schedule.journey_id,
                    broker_request_id=decision.broker_request_id,
                    simulator_context_binding=simulator_context_binding({
                        "session_id": context.session_id,
                        "turn_index": context.turn_index,
                        "previous_response_hash": content_hash(
                            context.previous_agent_response
                        ),
                        "previous_response_received_at_ns": (
                            context.previous_response_received_at_ns
                        ),
                        "schedule": journey_schedule_payload(
                            context.schedule
                        ),
                        "observed_edge_keys": list(
                            context.observed_edge_keys
                        ),
                    }),
                    simulator_attestation=simulator_attestation,
                    variant_attestation=variant_attestation,
                ))
                transcript.append((decision.user_message, response))
                transcript_lines.extend((f"User> {decision.user_message}", response))

                observed = self._observe_edges(baseline_event, committed_event, turn)
                for item in observed:
                    if item.edge_key not in observed_edge_keys:
                        observed_edge_keys.append(item.edge_key)
                verifier_context = self._verifier_context(
                    initial_event=initial_event,
                    current_event=committed_event,
                    turns=turns,
                    events=events,
                    decisions=decisions,
                    transcript=transcript,
                    observed_edge_keys=observed_edge_keys,
                    latest_turn=turn,
                )
                forbidden = self._verify_forbidden_outcomes(verifier_context)
                terminal = self._verify_outcome(
                    self.schedule.terminal_outcome,
                    verifier_context,
                )
                turn_results.append({
                    "turn_index": committed_event.turn_index,
                    "selected_at_ns": selected_at_ns,
                    "turn_identity": {
                        "transcript_hash": turn.transcript_hash,
                        "before_fingerprint": turn.before_fingerprint,
                        "after_fingerprint": turn.after_fingerprint,
                        "previous_response_received_at_ns": (
                            turn.previous_response_received_at_ns
                        ),
                        "user_message_submitted_at_ns": (
                            turn.user_message_submitted_at_ns
                        ),
                        "agent_response_received_at_ns": (
                            turn.agent_response_received_at_ns
                        ),
                    },
                    "decision": redact({
                        "user_message": decision.user_message,
                        "persona": decision.persona,
                        "mission": decision.mission,
                        "rationale": decision.rationale,
                        "risk_factor_ids": list(decision.risk_factor_ids),
                    }),
                    "decision_provenance": {
                        "execution_id": decisions[-1].execution_id,
                        "obligation_id": decisions[-1].obligation_id,
                        "broker_request_id": decisions[-1].broker_request_id,
                        "previous_response_hash": decisions[-1].previous_response_hash,
                        "user_message_hash": decisions[-1].user_message_hash,
                        "selected_at_ns": decisions[-1].selected_at_ns,
                        "submitted_at_ns": decisions[-1].submitted_at_ns,
                        "simulator_context_binding": redact(
                            dict(decisions[-1].simulator_context_binding)
                        ),
                        "simulator_attestation": redact(
                            dict(decisions[-1].simulator_attestation)
                        ),
                        "variant_attestation": redact(
                            dict(decisions[-1].variant_attestation)
                        ),
                    },
                    "observed_edges": [
                        {
                            "edge_key": item.edge_key,
                            "verifier_id": item.verifier_id,
                            "observed_coverage_ids": list(item.observed_coverage_ids),
                            "details": redact(dict(item.details)),
                        }
                        for item in observed
                    ],
                    "terminal_outcome": _journey_outcome_verification_payload(terminal),
                    "forbidden_outcomes": [
                        _journey_outcome_verification_payload(item) for item in forbidden
                    ],
                })
                violated = _satisfied_journey_outcome(forbidden)
                if violated is not None:
                    raise JourneyProductFailure(
                        "journey reached forbidden outcome: " + violated.outcome_id
                    )
                if terminal.satisfied:
                    terminal_classification = JourneyTerminalClassification.PASSED
                    terminal_outcome_id = terminal.outcome_id
                    break

                previous_response = response
                previous_received_ns = response_received_ns
                baseline_event = committed_event

            if terminal_classification != JourneyTerminalClassification.PASSED:
                raise JourneyProductFailure(
                    "journey exhausted max_turns without satisfying terminal outcome: "
                    f"{self.schedule.max_turns}"
                )
        except KeyboardInterrupt as exc:
            terminal_classification = JourneyTerminalClassification.INFRASTRUCTURE_INTERRUPTED
            failure_reason = "KeyboardInterrupt: operator interrupted the Journey"
            raise JourneyInfrastructureInterruptedError(failure_reason) from exc
        except JourneyRunError as exc:
            terminal_classification = exc.classification
            failure_reason = f"{type(exc).__name__}: {exc}"
            raise
        except Exception as exc:
            terminal_classification = JourneyTerminalClassification.INFRASTRUCTURE_INTERRUPTED
            failure_reason = f"{type(exc).__name__}: {exc}"
            raise JourneyInfrastructureInterruptedError(failure_reason) from exc
        finally:
            try:
                self.transport.close()
            except Exception as close_error:
                if not failure_reason:
                    terminal_classification = (
                        JourneyTerminalClassification.INFRASTRUCTURE_INTERRUPTED
                    )
                    failure_reason = f"{type(close_error).__name__}: {close_error}"
            if isinstance(self.transport, ContainerPtyBridgeTransport):
                try:
                    execution_proof = (
                        self.transport.validated_execution_proof()
                    )
                    qualification_reason = "trusted_container_pty_execution"
                except Exception as proof_error:
                    terminal_classification = (
                        JourneyTerminalClassification.INFRASTRUCTURE_INTERRUPTED
                    )
                    terminal_outcome_id = ""
                    failure_reason = (
                        f"{type(proof_error).__name__}: {proof_error}"
                    )
                    qualification_reason = "container_pty_proof_invalid"
            else:
                qualification_reason = "untrusted_test_transport"
            _write_redacted_transcript(transcript_path, transcript_lines)
            safe_transcript_lines = redact(list(transcript_lines))
            qualifying_evidence = bool(
                terminal_classification
                == JourneyTerminalClassification.PASSED
                and execution_proof
            )
            journey_payload = {
                "schema_version": JOURNEY_EVIDENCE_SCHEMA_VERSION,
                "artifact_type": "dynamic_dual_ai_journey_evidence",
                "runner_type": "dynamic_dual_ai_journey",
                "schedule_id": self.schedule.schedule_id,
                "schedule_hash": content_hash(journey_schedule_payload(self.schedule)),
                "journey_id": self.schedule.journey_id,
                "revision": self.revision,
                "verifier_registry": journey_outcome_verifier_registry_payload(
                    self.postcondition_verifier_registry
                ),
                "session_id": self.config.session_id,
                "execution_id": self.config.execution_id,
                "provider": observed_provider or self.config.provider,
                "model": observed_model or self.config.model,
                "terminal_classification": terminal_classification.value,
                "qualifying_evidence": qualifying_evidence,
                "qualification_reason": qualification_reason,
                "execution_proof": dict(execution_proof),
                "terminal_outcome_id": terminal_outcome_id,
                "max_turns": self.schedule.max_turns,
                "completed_turn_count": len(turns),
                "observed_edge_keys": observed_edge_keys,
                "failure_reason": str(redact(failure_reason)),
                "transcript_hash": content_hash(safe_transcript_lines),
                "source_transcript_hash": content_hash(transcript_lines),
                "content_redacted": True,
                "initial_verification": initial_verification,
                "turns": turn_results,
            }
            if (
                self.schedule.verifier_input_contract
                and terminal_classification
                == JourneyTerminalClassification.PASSED
            ):
                journey_payload["retained_artifacts"] = (
                    _write_retained_journey_artifacts(
                        runtime_root=runtime_root,
                        schedule=self.schedule,
                        revision=self.revision,
                        execution_id=self.config.execution_id,
                        initial_event=initial_event,
                        events=events,
                        turns=turns,
                        decisions=decisions,
                    )
                )
            evidence_id = content_hash(journey_payload)
            evidence_artifact = {**journey_payload, "evidence_id": evidence_id}
            evidence_artifact["artifact_hash"] = content_hash(evidence_artifact)
            evidence_path = evidence_dir / f"journey-{evidence_id}.json"
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            evidence_path.write_text(
                json.dumps(
                    evidence_artifact,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ) + "\n",
                encoding="utf-8",
            )
            journey_result_path.write_text(
                json.dumps({
                    "schema_version": 1,
                    "schedule_id": self.schedule.schedule_id,
                    "journey_id": self.schedule.journey_id,
                    "revision": self.revision,
                    "execution_status": terminal_classification.value,
                    "terminal_classification": terminal_classification.value,
                    "terminal_outcome_id": terminal_outcome_id,
                    "max_turns": self.schedule.max_turns,
                    "completed_turn_count": len(turns),
                    "observed_edge_keys": observed_edge_keys,
                    "failure_reason": str(redact(failure_reason)),
                    "evidence_id": evidence_id,
                    "evidence_path": str(evidence_path),
                    "qualifying_evidence": evidence_artifact["qualifying_evidence"],
                    "verifier_registry_id": self.postcondition_verifier_registry.registry_id,
                    "initial_verification": initial_verification,
                    "turns": turn_results,
                }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        return JourneyRunResult(
            session_id=self.config.session_id,
            transcript_path=transcript_path,
            schedule_path=schedule_path,
            journey_result_path=journey_result_path,
            evidence_path=evidence_path,
            evidence_id=evidence_id,
            turns=tuple(turns),
            observed_edge_keys=tuple(observed_edge_keys),
            terminal_classification=terminal_classification,
            terminal_outcome_id=terminal_outcome_id,
        )

    def _validate_verifier_contracts(self) -> None:
        required_ids = {
            *self.schedule.terminal_outcome.required_postcondition_ids,
            *(
                postcondition_id
                for outcome in self.schedule.forbidden_outcomes
                for postcondition_id in outcome.required_postcondition_ids
            ),
        }
        missing = sorted(
            required_ids - set(self.postcondition_verifier_registry.definitions)
        )
        if missing:
            raise ValueError(
                "journey schedule is missing registered postcondition verifiers: "
                + ", ".join(missing)
            )

    def _verify_outcome(
        self,
        outcome: JourneyOutcomeContract,
        context: JourneyVerifierContext,
    ) -> JourneyOutcomeVerification:
        results: list[JourneyPostconditionResult] = []
        bindings: list[JourneyVerifierBinding] = []
        for postcondition_id in outcome.required_postcondition_ids:
            definition = self.postcondition_verifier_registry.definitions[postcondition_id]
            result = definition.verifier(replace(
                context,
                evaluating_postcondition_id=postcondition_id,
            ))
            if not isinstance(result, JourneyPostconditionResult):
                raise TypeError(
                    "journey postcondition verifier must return JourneyPostconditionResult: "
                    + postcondition_id
                )
            if result.postcondition_id != postcondition_id:
                raise ValueError(
                    "journey postcondition verifier returned the wrong id: "
                    f"expected {postcondition_id}, got {result.postcondition_id}"
                )
            results.append(result)
            bindings.append(JourneyVerifierBinding(
                postcondition_id=postcondition_id,
                verifier_id=definition.verifier_id,
                verifier_version=definition.verifier_version,
                implementation_hash=definition.implementation_hash,
            ))
        return JourneyOutcomeVerification(
            outcome_id=outcome.outcome_id,
            satisfied=all(item.satisfied for item in results),
            postconditions=tuple(results),
            verifier_bindings=tuple(bindings),
        )

    def _verify_forbidden_outcomes(
        self,
        context: JourneyVerifierContext,
    ) -> tuple[JourneyOutcomeVerification, ...]:
        return tuple(
            self._verify_outcome(outcome, context)
            for outcome in self.schedule.forbidden_outcomes
        )

    def _observe_edges(
        self,
        baseline_event: RuntimeTurnEvent,
        committed_event: RuntimeTurnEvent,
        turn: PtyCliTurnRecord,
    ) -> tuple[JourneyObservedEdge, ...]:
        observed: list[JourneyObservedEdge] = []
        for edge_key, edge in self.edge_index.items():
            if edge.get("applicable") is False:
                continue
            try:
                _require_scheduled_baseline_contract(baseline_event, edge)
            except RuntimeError:
                continue
            verified = verify_runtime_postcondition(
                edge,
                baseline_event,
                committed_event,
                turn,
            )
            if not verified.passed or edge_key not in verified.observed_coverage_ids:
                continue
            observed.append(JourneyObservedEdge(
                edge_key=edge_key,
                verifier_id=verified.verifier_id,
                observed_coverage_ids=tuple(verified.observed_coverage_ids),
                details=dict(verified.details),
            ))
        return tuple(observed)

    def _verifier_context(
        self,
        *,
        initial_event: RuntimeTurnEvent,
        current_event: RuntimeTurnEvent,
        turns: Sequence[PtyCliTurnRecord],
        events: Sequence[RuntimeTurnEvent],
        decisions: Sequence[JourneyDecisionProvenance],
        transcript: Sequence[tuple[str, str]],
        observed_edge_keys: Sequence[str],
        latest_turn: PtyCliTurnRecord | None,
    ) -> JourneyVerifierContext:
        return JourneyVerifierContext(
            schedule=self.schedule,
            initial_event=initial_event,
            current_event=current_event,
            completed_turns=tuple(turns),
            transcript=tuple(transcript),
            observed_edge_keys=tuple(observed_edge_keys),
            latest_turn=latest_turn,
            completed_events=tuple(events),
            completed_decisions=tuple(decisions),
            verifier_input_contract=dict(self.schedule.verifier_input_contract),
        )

    def _validated_decision_attestations(
        self,
        *,
        context: JourneySimulatorContext,
        decision: JourneySimulatorDecision,
        selection_observed_at_ns: int,
        submitted_at_ns: int,
        prior_decisions: Sequence[JourneyDecisionProvenance],
    ) -> tuple[Mapping[str, Any], Mapping[str, Any], int]:
        verifier_input = dict(self.schedule.verifier_input_contract)
        from tests.agent_live.codex_simulator_bridge import (
            simulator_context_hash,
            validate_simulator_attestation,
        )
        if not decision.simulator_attestation:
            if verifier_input:
                raise JourneySimulatorInvalidError(
                    "retained regression decision has no simulator attestation"
                )
            return {}, {}, selection_observed_at_ns
        unsigned_decision = _journey_unsigned_decision(
            context=context,
            decision=decision,
        )
        request_id = str(decision.broker_request_id or "")
        if not request_id:
            raise JourneySimulatorInvalidError(
                "attested Journey decision has no broker request identity"
            )
        if any(
            request_id
            == str(
                item.simulator_attestation.get("request_id") or ""
            )
            for item in prior_decisions
        ):
            raise JourneySimulatorInvalidError(
                "Journey simulator broker request was replayed"
            )
        context_payload = {
            "session_id": context.session_id,
            "turn_index": context.turn_index,
            "previous_response_hash": content_hash(
                context.previous_agent_response
            ),
            "previous_response_received_at_ns": (
                context.previous_response_received_at_ns
            ),
            "schedule": journey_schedule_payload(context.schedule),
            "observed_edge_keys": list(context.observed_edge_keys),
        }
        try:
            simulator_attestation = validate_simulator_attestation(
                decision.simulator_attestation,
                previous_response_hash=content_hash(
                    context.previous_agent_response
                ),
                decision_hash=content_hash(unsigned_decision),
                request_id=request_id,
                context_hash=simulator_context_hash(context_payload),
                user_message_hash=content_hash(decision.user_message),
                turn_index=context.turn_index,
                submitted_at_ns=submitted_at_ns,
            )
        except ValueError as exc:
            raise JourneySimulatorInvalidError(str(exc)) from exc
        declared_at_ns = int(simulator_attestation["declared_at_ns"])
        if (
            declared_at_ns < context.previous_response_received_at_ns
            or declared_at_ns > selection_observed_at_ns
        ):
            raise JourneySimulatorInvalidError(
                "simulator declaration time is outside the response-selection interval"
            )
        if not verifier_input:
            return simulator_attestation, {}, declared_at_ns
        from tests.agent_live.retained_regression_attestations import (
            build_variant_attestation,
            validate_verifier_input_contract,
        )

        contract = validate_verifier_input_contract(verifier_input)
        binding = dict(decision.variant_binding)
        if set(binding) != {"source_step_id", "semantic_role"}:
            raise JourneySimulatorInvalidError(
                "retained regression decision requires an exact variant binding"
            )
        source_step = next(
            (
                dict(step)
                for step in contract["source_contract"]["source_steps"]
                if step.get("step_id") == binding["source_step_id"]
            ),
            None,
        )
        if (
            source_step is None
            or binding["source_step_id"]
            not in contract["variant_contract"]["allowed_source_step_ids"]
            or binding["semantic_role"] != source_step["semantic_role"]
        ):
            raise JourneySimulatorInvalidError(
                "retained regression variant binding is outside the frozen contract"
            )
        variant_attestation = build_variant_attestation(
            actor=simulator_attestation["actor"],
            verifier_input_contract=contract,
            execution_binding={
                "execution_id": self.config.execution_id,
                "session_id": context.session_id,
                "schedule_id": context.schedule.schedule_id,
                "obligation_id": context.schedule.journey_id,
                "broker_request_id": request_id,
                "simulator_attestation_id": simulator_attestation[
                    "attestation_id"
                ],
            },
            source_step_id=binding["source_step_id"],
            semantic_role=binding["semantic_role"],
            turn_index=context.turn_index,
            previous_response_hash=content_hash(
                context.previous_agent_response
            ),
            user_message_hash=content_hash(decision.user_message),
            selected_at_ns=declared_at_ns,
            declared_at_ns=declared_at_ns,
        )
        return simulator_attestation, variant_attestation, declared_at_ns

    def _validate_event_revision(self, event: RuntimeTurnEvent) -> None:
        if dict(event.revision) != self.revision:
            raise RuntimeError(
                "runtime event revision does not match the scheduled journey revision"
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


def _validate_journey_decision(
    decision: JourneySimulatorDecision,
    schedule: JourneySchedule,
) -> None:
    required = {
        "user_message": decision.user_message,
        "persona": decision.persona,
        "mission": decision.mission,
        "rationale": decision.rationale,
    }
    missing = sorted(name for name, value in required.items() if not str(value or "").strip())
    if missing:
        raise ValueError(f"journey simulator decision is missing: {', '.join(missing)}")
    if decision.persona != schedule.persona or decision.mission != schedule.mission:
        raise ValueError("journey simulator decision changed the scheduled persona or mission")
    risk_factor_ids = tuple(str(item).strip() for item in decision.risk_factor_ids)
    if any(not item for item in risk_factor_ids) or len(risk_factor_ids) != len(set(risk_factor_ids)):
        raise ValueError("journey simulator decision has invalid risk factor ids")
    unexpected = sorted(set(risk_factor_ids) - set(schedule.allowed_risk_factors))
    if unexpected:
        raise ValueError(
            "journey simulator selected undeclared risk factors: " + ", ".join(unexpected)
        )
    if schedule.verifier_input_contract:
        if not decision.simulator_attestation:
            raise ValueError(
                "retained regression Journey decision lacks simulator attestation"
            )
        if not decision.variant_binding:
            raise ValueError(
                "retained regression Journey decision lacks variant binding"
            )
    elif decision.variant_binding:
        raise ValueError(
            "generic Journey decision cannot declare a retained variant binding"
        )


def _journey_unsigned_decision(
    *,
    context: JourneySimulatorContext,
    decision: JourneySimulatorDecision,
) -> dict[str, Any]:
    unsigned = {
        "previous_response_hash": content_hash(
            context.previous_agent_response
        ),
        "user_message": decision.user_message,
        "persona": decision.persona,
        "mission": decision.mission,
        "rationale": decision.rationale,
        "risk_factor_ids": list(decision.risk_factor_ids),
    }
    if decision.variant_binding:
        unsigned["variant_binding"] = dict(decision.variant_binding)
    if decision.broker_request_id:
        unsigned["broker_request_id"] = decision.broker_request_id
    return unsigned


def _write_retained_journey_artifacts(
    *,
    runtime_root: Path,
    schedule: JourneySchedule,
    revision: Mapping[str, str],
    execution_id: str,
    initial_event: RuntimeTurnEvent | None,
    events: Sequence[RuntimeTurnEvent],
    turns: Sequence[PtyCliTurnRecord],
    decisions: Sequence[JourneyDecisionProvenance],
) -> dict[str, Any]:
    if initial_event is None or len(events) != len(turns) or len(turns) != len(
        decisions
    ):
        raise JourneyInfrastructureInterruptedError(
            "retained Journey artifacts have incomplete runtime lineage"
        )
    identity = {
        "obligation_id": schedule.journey_id,
        "execution_id": str(execution_id),
        "revision": dict(revision),
        "schedule_id": schedule.schedule_id,
        "verifier_input_contract_hash": content_hash(
            dict(schedule.verifier_input_contract)
        ),
    }
    payloads = {
        "transcript": {
            "turns": [
                {
                    "turn": redact(asdict(turn)),
                    "decision": redact(asdict(decision)),
                }
                for turn, decision in zip(turns, decisions)
            ],
        },
        "runtime_events": {
            "initial_event": asdict(initial_event),
            "events": [asdict(event) for event in events],
        },
        "checkpoint_diff": {
            "before_fingerprint": initial_event.after_fingerprint,
            "after_fingerprint": (
                events[-1].after_fingerprint
                if events
                else initial_event.after_fingerprint
            ),
            "material_state_diff_hashes": [
                dict(event.material_state_diff_hashes)
                for event in events
            ],
        },
    }
    references: dict[str, Any] = {}
    artifact_root = runtime_root / "retained-artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    for role, role_payload in payloads.items():
        unsigned = {
            "schema_version": 1,
            "artifact_type": f"retained_regression_{role}",
            "identity": identity,
            **role_payload,
        }
        artifact = {**unsigned, "artifact_hash": content_hash(unsigned)}
        path = artifact_root / f"{role}.json"
        path.write_text(
            json.dumps(
                artifact,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        references[role] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "artifact_hash": artifact["artifact_hash"],
        }
    return references


def _journey_outcome_verification_payload(
    verification: JourneyOutcomeVerification,
) -> dict[str, Any]:
    return {
        "outcome_id": verification.outcome_id,
        "satisfied": verification.satisfied,
        "postconditions": [
            {
                "postcondition_id": item.postcondition_id,
                "satisfied": item.satisfied,
                "details": redact(dict(item.details)),
                "verifier": {
                    "verifier_id": binding.verifier_id,
                    "verifier_version": binding.verifier_version,
                    "implementation_hash": binding.implementation_hash,
                },
            }
            for item, binding in zip(
                verification.postconditions,
                verification.verifier_bindings,
                strict=True,
            )
        ],
    }


def _satisfied_journey_outcome(
    outcomes: Sequence[JourneyOutcomeVerification],
) -> JourneyOutcomeVerification | None:
    return next((item for item in outcomes if item.satisfied), None)


def validate_journey_evidence_artifact(
    artifact: Mapping[str, Any],
    *,
    schedule: JourneySchedule,
    verifier_registry: JourneyOutcomeVerifierRegistry,
    revision: Mapping[str, str],
) -> None:
    """Fail closed unless an artifact is authentic and qualifies this Journey."""

    validate_journey_schedule(schedule, revision=revision)
    validate_journey_outcome_verifier_registry(verifier_registry)
    payload = dict(artifact)
    artifact_hash = str(payload.pop("artifact_hash", ""))
    if not artifact_hash or content_hash(payload) != artifact_hash:
        raise ValueError("Journey evidence artifact hash is invalid")
    evidence_id = str(payload.pop("evidence_id", ""))
    if not evidence_id or content_hash(payload) != evidence_id:
        raise ValueError("Journey evidence content address is invalid")
    if int(payload.get("schema_version", 0)) != JOURNEY_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("unsupported Journey evidence schema")
    if payload.get("artifact_type") != "dynamic_dual_ai_journey_evidence":
        raise ValueError("invalid Journey evidence artifact type")
    if payload.get("content_redacted") is not True:
        raise ValueError("Journey evidence does not declare redaction")
    if re.fullmatch(r"[0-9a-f]{64}", str(payload.get("source_transcript_hash") or "")) is None:
        raise ValueError("Journey source transcript hash is invalid")
    if dict(payload.get("revision") or {}) != dict(revision):
        raise ValueError("Journey evidence revision does not match the active revision")
    if payload.get("schedule_id") != schedule.schedule_id:
        raise ValueError("Journey evidence schedule id does not match")
    if payload.get("schedule_hash") != content_hash(journey_schedule_payload(schedule)):
        raise ValueError("Journey evidence schedule hash does not match")
    expected_registry = journey_outcome_verifier_registry_payload(verifier_registry)
    if payload.get("verifier_registry") != expected_registry:
        raise ValueError("Journey evidence verifier registry does not match")
    try:
        classification = JourneyTerminalClassification(
            str(payload.get("terminal_classification") or "")
        )
    except ValueError as exc:
        raise ValueError("Journey evidence terminal classification is invalid") from exc
    if classification != JourneyTerminalClassification.PASSED:
        raise ValueError("non-passing Journey artifact cannot qualify as evidence")
    if payload.get("qualifying_evidence") is not True:
        raise ValueError("passing Journey artifact is not marked as qualifying evidence")
    if payload.get("qualification_reason") != "trusted_container_pty_execution":
        raise ValueError("Journey evidence has no trusted PTY qualification")
    execution_id = str(payload.get("execution_id") or "")
    execution_proof = payload.get("execution_proof")
    if not execution_id or not isinstance(execution_proof, Mapping):
        raise ValueError("Journey evidence lacks an execution-bound PTY proof")
    if (
        execution_proof.get("proof_type") != "container_pty_process_guard"
        or execution_proof.get("transport_kind") != "container_pty_bridge"
        or execution_proof.get("execution_id") != execution_id
    ):
        raise ValueError("Journey PTY proof identity is invalid")
    from tests.agent_live.container_process_guard import (
        validate_cleanup_receipt_artifact,
    )

    proof_path = Path(str(execution_proof.get("path") or ""))
    validated_proof = validate_cleanup_receipt_artifact(
        proof_path,
        execution_id=execution_id,
        required_roles=(
            "container_bridge",
            "agent_process_group_leader",
        ),
        allowed_roots=(proof_path.parent,),
    )
    expected_proof = {
        "proof_type": "container_pty_process_guard",
        "transport_kind": "container_pty_bridge",
        **validated_proof,
    }
    if dict(execution_proof) != expected_proof:
        raise ValueError("Journey PTY proof differs from its receipt")
    if payload.get("terminal_outcome_id") != schedule.terminal_outcome.outcome_id:
        raise ValueError("Journey evidence terminal outcome does not match")
    verifications = [payload.get("initial_verification") or {}]
    verifications.extend(payload.get("turns") or ())
    terminal = next(
        (
            item.get("terminal_outcome")
            for item in reversed(verifications)
            if isinstance(item, Mapping)
            and isinstance(item.get("terminal_outcome"), Mapping)
            and item["terminal_outcome"].get("satisfied") is True
        ),
        None,
    )
    if not isinstance(terminal, Mapping):
        raise ValueError("Journey evidence lacks a satisfied terminal verification")
    if terminal.get("outcome_id") != schedule.terminal_outcome.outcome_id:
        raise ValueError("Journey terminal verification identifies the wrong outcome")
    expected_postconditions = set(schedule.terminal_outcome.required_postcondition_ids)
    observed_postconditions = {
        str(item.get("postcondition_id") or "")
        for item in terminal.get("postconditions") or ()
        if isinstance(item, Mapping) and item.get("satisfied") is True
    }
    if observed_postconditions != expected_postconditions:
        raise ValueError("Journey terminal postcondition evidence is incomplete")
    for turn in payload.get("turns") or ():
        if not isinstance(turn, Mapping):
            raise ValueError("Journey turn evidence is invalid")
        provenance = turn.get("decision_provenance")
        identity = turn.get("turn_identity")
        if not isinstance(provenance, Mapping) or not isinstance(identity, Mapping):
            raise ValueError("Journey turn lacks response-bound decision provenance")
        if (
            provenance.get("execution_id") != execution_id
            or provenance.get("obligation_id") != schedule.journey_id
            or not str(provenance.get("broker_request_id") or "")
            or not isinstance(
                provenance.get("simulator_context_binding"), Mapping
            )
        ):
            raise ValueError(
                "Journey decision provenance execution binding is invalid"
            )
        for field in ("previous_response_hash", "user_message_hash"):
            if re.fullmatch(r"[0-9a-f]{64}", str(provenance.get(field) or "")) is None:
                raise ValueError("Journey decision provenance hash is invalid")
        selected_at_ns = provenance.get("selected_at_ns")
        submitted_at_ns = provenance.get("submitted_at_ns")
        if (
            isinstance(selected_at_ns, bool)
            or not isinstance(selected_at_ns, int)
            or isinstance(submitted_at_ns, bool)
            or not isinstance(submitted_at_ns, int)
            or selected_at_ns <= 0
            or submitted_at_ns < selected_at_ns
            or submitted_at_ns != identity.get("user_message_submitted_at_ns")
            or selected_at_ns != turn.get("selected_at_ns")
        ):
            raise ValueError("Journey decision provenance timing is invalid")



def _require_scheduled_baseline_contract(
    event: RuntimeTurnEvent,
    edge: Mapping[str, Any],
) -> None:
    """Bind every scheduled edge to its exact runtime baseline contract."""

    edge_type = str(edge.get("edge_type") or "")
    if edge_type == "action_transition":
        # Action transitions are semantic interruptions. A real product CLI
        # may overlay a startup/resume or fallback question on the reviewed
        # checkpoint before Codex submits that action. Product admission and
        # the transition postcondition, not the old question's answer list,
        # decide whether the interruption is valid.
        return
    expected_question = str(edge.get("question_id") or "")
    if event.pending_question_id != expected_question:
        raise RuntimeError(
            "scheduled target question does not match the runtime baseline: "
            f"expected {expected_question or '<none>'}, got "
            f"{event.pending_question_id or '<none>'}"
        )
    from tests.agent_live.generate_harness_coverage_ledger import contract_variant_hash

    if contract_variant_hash(event.pending_contract) != str(edge.get("contract_hash") or ""):
        raise RuntimeError("scheduled target contract hash does not match the runtime baseline")


def encode_bracketed_paste(message: str) -> bytes:
    """Encode one paste event without treating embedded newlines as submissions."""

    normalized = str(message).replace("\r\n", "\n").replace("\r", "\n")
    return b"\x1b[200~" + normalized.encode("utf-8") + b"\x1b[201~"


def _validate_decision(
    decision: SimulatorDecision,
    target: ScheduledCoverageTarget,
    coverage_contract: Mapping[str, Any],
) -> None:
    required = {
        "user_message": decision.user_message,
        "persona": decision.persona,
        "goal": decision.goal,
        "rationale": decision.rationale,
    }
    missing = sorted(name for name, value in required.items() if not str(value or "").strip())
    if missing:
        raise SimulatorDecisionInvalid(f"simulator decision is missing: {', '.join(missing)}")
    if not decision.target_coverage_ids or any(
        not str(item or "").strip() for item in decision.target_coverage_ids
    ):
        raise SimulatorDecisionInvalid("simulator decision requires target coverage ids")
    if target.edge_key not in decision.target_coverage_ids:
        raise SimulatorDecisionInvalid("simulator decision does not target the scheduled ledger edge")
    if decision.persona != target.persona or decision.goal != target.goal:
        raise SimulatorDecisionInvalid("simulator decision changed the scheduled persona or goal")
    _validate_declared_input_class(decision.user_message, coverage_contract)


def _validate_declared_input_class(
    user_message: str,
    coverage_contract: Mapping[str, Any],
) -> None:
    """Reject simulator evidence whose transport shape contradicts its lane."""

    input_class = str(coverage_contract.get("input_class") or "")
    clauses = segment_user_turn(user_message)
    shapes = {clause.input_shape for clause in clauses}
    if input_class == "multiline_prose":
        if "\n" not in user_message or not clauses or shapes != {"prose"}:
            raise SimulatorDecisionInvalid(
                "simulator decision does not exercise multiline_prose: "
                "expected multiple prose lines without a structured data block"
            )
    elif input_class == "structured_json_yaml_env_curl":
        if "structured" not in shapes:
            raise SimulatorDecisionInvalid(
                "simulator decision does not exercise structured_json_yaml_env_curl: "
                "expected a parseable structured input region"
            )


def _verify_declared_postconditions(
    baseline_event: RuntimeTurnEvent,
    committed_event: RuntimeTurnEvent,
    turn: PtyCliTurnRecord,
    *,
    edge_index: Mapping[str, Mapping[str, Any]],
    target_coverage_ids: Sequence[str],
    continuation_events: Sequence[RuntimeTurnEvent] = (),
) -> VerifiedPostcondition:
    """Verify every coverage claim attached to one response-driven user turn."""

    edge_keys = tuple(dict.fromkeys(str(item) for item in target_coverage_ids))
    missing = tuple(edge_key for edge_key in edge_keys if edge_key not in edge_index)
    if missing:
        raise ValueError(f"simulator declared unknown coverage ids: {', '.join(missing)}")
    results = []
    for edge_key in edge_keys:
        arguments = (
            edge_index[edge_key],
            baseline_event,
            committed_event,
            turn,
        )
        if continuation_events:
            result = verify_runtime_postcondition(
                *arguments,
                continuation_events=continuation_events,
            )
        else:
            result = verify_runtime_postcondition(*arguments)
        results.append(result)
    primary = results[0]
    errors = [
        f"{edge_key}: {error}"
        for edge_key, result in zip(edge_keys, results)
        for error in result.details.get("errors") or ()
    ]
    details = {
        "declared_target_results": {
            edge_key: dict(result.details)
            for edge_key, result in zip(edge_keys, results)
        },
        "errors": errors,
    }
    return VerifiedPostcondition(
        verifier_id="dynamic-declared-target-set-v1",
        passed=all(result.passed for result in results),
        observed_coverage_ids=tuple(dict.fromkeys(
            coverage_id
            for result in results
            for coverage_id in result.observed_coverage_ids
        )),
        admitted_typed_actions=tuple(dict.fromkeys(
            action_type
            for result in results
            for action_type in result.admitted_typed_actions
        )),
        state_diff=dict(primary.state_diff),
        next_question_or_result=dict(primary.next_question_or_result),
        details=details,
        job_artifacts=tuple(
            artifact
            for result in results
            for artifact in result.job_artifacts
        ),
    )


def _verification_requires_continuation(
    verification: VerifiedPostcondition,
) -> bool:
    """Return whether every failing target is safely awaiting its linked terminal."""

    target_results = tuple(
        dict(item)
        for item in (
            verification.details.get("declared_target_results") or {}
        ).values()
    )
    if not target_results:
        return False
    awaiting = False
    for result in target_results:
        errors = tuple(result.get("errors") or ())
        if not errors:
            continue
        if result.get("continuation_eligible") is not True:
            return False
        awaiting = True
    return awaiting


def _validate_continuation_decision(
    decision: SimulatorDecision,
    target: ScheduledCoverageTarget,
) -> None:
    """Validate a response-driven continuation without reusing the entry input lane."""

    required = {
        "user_message": decision.user_message,
        "persona": decision.persona,
        "goal": decision.goal,
        "rationale": decision.rationale,
    }
    missing = sorted(name for name, value in required.items() if not str(value or "").strip())
    if missing:
        raise SimulatorDecisionInvalid(
            f"continuation simulator decision is missing: {', '.join(missing)}"
        )
    if target.edge_key not in decision.target_coverage_ids:
        raise SimulatorDecisionInvalid(
            "continuation simulator decision lost the scheduled ledger edge"
        )
    if decision.persona != target.persona or decision.goal != target.goal:
        raise SimulatorDecisionInvalid(
            "continuation simulator decision changed the scheduled persona or goal"
        )


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
    if int(payload["schema_version"]) == 3:
        required_v3 = {
            "admitted_action_provenance",
            "turn_receipt_summary",
            "pending_transition",
            "render_manifest",
            "execution_receipt_summary",
            "control_receipts",
            "material_state_diff_hashes",
        }
        missing_v3 = sorted(required_v3 - set(payload))
        if missing_v3:
            raise RuntimeError(
                "runtime turn event v3 is missing: " + ", ".join(missing_v3)
            )
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
        admitted_action_provenance=tuple(
            dict(item)
            for item in payload.get("admitted_action_provenance") or ()
            if isinstance(item, Mapping)
        ),
        turn_receipt_summary=dict(payload.get("turn_receipt_summary") or {}),
        pending_transition=dict(payload.get("pending_transition") or {}),
        render_manifest=dict(payload.get("render_manifest") or {}),
        execution_receipt_summary=dict(
            payload.get("execution_receipt_summary") or {}
        ),
        control_receipts=tuple(
            dict(item)
            for item in payload.get("control_receipts") or ()
            if isinstance(item, Mapping)
        ),
        state_diff_hashes={
            str(path): {str(key): str(value) for key, value in dict(hashes).items()}
            for path, hashes in dict(payload["state_diff_hashes"] or {}).items()
        },
        material_state_diff_hashes={
            str(path): {
                str(key): str(value)
                for key, value in dict(hashes).items()
            }
            for path, hashes in dict(
                payload.get("material_state_diff_hashes") or {}
            ).items()
        },
        after_value_hashes={
            str(path): str(value)
            for path, value in dict(payload["after_value_hashes"] or {}).items()
        },
        next_result=dict(payload["next_result"] or {}),
    )

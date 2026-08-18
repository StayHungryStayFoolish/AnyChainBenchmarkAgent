"""Filesystem rendezvous for a real external Codex Chaos user.

The broker publishes only after a complete Agent response is available.  A
Codex operator reads that response-bound request and submits exactly one next
turn; no future dialogue is stored in the batch target or schedule.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import select
import signal
import stat
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent.utils.redaction import redact
from tests.agent_live.batch_orchestrator import (
    ExternalDecisionBlocked,
    load_frozen_manifest,
    run_batch,
)
from tests.agent_live.coverage_evidence import (
    PtyAuthoritySigner,
    import_pty_authority_private_key,
)
from tests.agent_live.codex_simulator_bridge import (
    build_simulator_attestation,
    simulator_context_hash,
)


BROKER_SCHEMA_VERSION = 2


class FilesystemDecisionBroker:
    """Publish immutable response contexts and consume bound decisions."""

    def __init__(
        self,
        root: str | Path,
        *,
        batch_id: str,
        timeout_seconds: float = 230.0,
        poll_seconds: float = 0.05,
    ) -> None:
        if timeout_seconds <= 0 or poll_seconds <= 0:
            raise ValueError("broker timeouts must be positive")
        self.root = Path(root).resolve()
        self.batch_id = str(batch_id).strip()
        if not self.batch_id:
            raise ValueError("broker requires a batch id")
        self.timeout_seconds = float(timeout_seconds)
        self.poll_seconds = float(poll_seconds)
        self._lock = threading.Lock()
        self._sequence_by_shard: dict[str, int] = {}
        self._closed = threading.Event()

    def close(self) -> None:
        """Release every pending decision wait during controller shutdown."""

        self._closed.set()

    async def __call__(self, shard_id: str, context: Mapping[str, Any]) -> Mapping[str, Any]:
        response_hash = str(context.get("previous_response_hash") or "").strip()
        if len(response_hash) != 64:
            raise ExternalDecisionBlocked("broker context has no valid response hash")
        with self._lock:
            sequence = self._sequence_by_shard.get(shard_id, 0) + 1
            self._sequence_by_shard[shard_id] = sequence
        channel_id = _content_hash({
            "batch_id": self.batch_id,
            "shard_id": str(shard_id),
            "sequence": sequence,
            "previous_response_hash": response_hash,
        })
        channel_path = self.root / "channels" / f"{channel_id}.fifo"
        channel_path.parent.mkdir(parents=True, exist_ok=True)
        os.mkfifo(channel_path, 0o600)
        channel_fd = os.open(
            channel_path,
            os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0),
        )
        if not stat.S_ISFIFO(os.fstat(channel_fd).st_mode):
            os.close(channel_fd)
            channel_path.unlink(missing_ok=True)
            raise ExternalDecisionBlocked("broker decision channel is not a FIFO")
        unsigned = {
            "schema_version": BROKER_SCHEMA_VERSION,
            "batch_id": self.batch_id,
            "shard_id": str(shard_id),
            "sequence": sequence,
            "previous_response_hash": response_hash,
            "context_hash": _content_hash(context),
            "context": redact(dict(context)),
            "control_identity": _control_identity(context),
            "decision_channel": str(channel_path),
            "created_at_ns": time.time_ns(),
        }
        request_id = _content_hash(unsigned)
        request = {"request_id": request_id, **unsigned}
        _write_immutable_json(self.root / "requests" / f"{request_id}.json", request)

        committed_path = self.root / "committed" / f"{request_id}.json"
        deadline = time.monotonic() + self.timeout_seconds
        wire_buffer = bytearray()
        try:
            while time.monotonic() < deadline:
                if self._closed.is_set():
                    raise ExternalDecisionBlocked(
                        "external decision broker is shutting down"
                    )
                try:
                    chunk = os.read(channel_fd, 65536)
                except BlockingIOError:
                    chunk = b""
                if chunk:
                    wire_buffer.extend(chunk)
                while b"\n" in wire_buffer:
                    raw, _, remainder = wire_buffer.partition(b"\n")
                    wire_buffer = bytearray(remainder)
                    if not raw.strip():
                        continue
                    try:
                        execution_payload = json.loads(raw.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        # A failed writer may leave an incomplete frame. The
                        # next leading delimiter makes that fragment safely
                        # discardable before a complete bound frame arrives.
                        continue
                    decision_record = execution_payload.get("record")
                    if not isinstance(decision_record, Mapping):
                        raise ExternalDecisionBlocked(
                            "external execution payload has no decision record"
                        )
                    decision = _validate_decision_record(
                        request, decision_record, execution_payload
                    )
                    receipt_unsigned = {
                        "schema_version": BROKER_SCHEMA_VERSION,
                        "request_id": request_id,
                        "decision_hash": _content_hash(decision),
                        "consumed_at_ns": time.time_ns(),
                    }
                    receipt = {
                        "receipt_id": _content_hash(receipt_unsigned),
                        **receipt_unsigned,
                    }
                    bundle_unsigned = {
                        "schema_version": BROKER_SCHEMA_VERSION,
                        "request_id": request_id,
                        "decision_record": decision_record,
                        "simulator_attestation": decision.get(
                            "simulator_attestation"
                        ),
                        "consumed_receipt": receipt,
                    }
                    _write_immutable_json(
                        committed_path,
                        {
                            "bundle_id": _content_hash(bundle_unsigned),
                            **bundle_unsigned,
                        },
                    )
                    return decision
                await asyncio.sleep(self.poll_seconds)
        finally:
            os.close(channel_fd)
            channel_path.unlink(missing_ok=True)
        raise ExternalDecisionBlocked(
            f"external Codex decision timed out for request {request_id}"
        )


def pending_requests(root: str | Path) -> tuple[Mapping[str, Any], ...]:
    broker_root = Path(root).resolve()
    committed = broker_root / "committed"
    channels = broker_root / "channels"
    rows = []
    for path in sorted((broker_root / "requests").glob("*.json")):
        if (committed / path.name).exists():
            continue
        request = _load_json(path)
        channel_path = Path(str(request.get("decision_channel") or ""))
        try:
            channel_path.resolve().relative_to(channels.resolve())
            channel_is_live = _channel_has_reader(channel_path)
        except (ValueError, OSError):
            channel_is_live = False
        if channel_is_live:
            rows.append(request)
    return tuple(rows)


def submit_decision(
    root: str | Path,
    *,
    request_id: str,
    user_message: str,
    rationale: str,
    risk_factor_ids: Sequence[str] = (),
    actor_kind: str = "",
    actor_task_id: str = "",
    actor_model: str = "",
    source_step_id: str = "",
    semantic_role: str = "",
) -> Path:
    broker_root = Path(root).resolve()
    request = _load_json(broker_root / "requests" / f"{request_id}.json")
    _validate_request_record(request, request_id=request_id)
    control_identity = request.get("control_identity")
    if not isinstance(control_identity, Mapping):
        raise ValueError("broker request has no immutable control identity")
    response_hash = str(request.get("previous_response_hash") or "")
    message = str(user_message).strip()
    reason = str(rationale).strip()
    if not message or not reason:
        raise ValueError("decision requires a user message and rationale")
    if control_identity.get("lane") == "journey":
        schedule = control_identity.get("schedule")
        if not isinstance(schedule, Mapping):
            raise ValueError("Journey broker request has no immutable schedule identity")
        decision = {
            "previous_response_hash": response_hash,
            "broker_request_id": request_id,
            "user_message": message,
            "persona": str(schedule.get("persona") or ""),
            "mission": str(schedule.get("mission") or ""),
            "rationale": reason,
            "risk_factor_ids": [str(item) for item in risk_factor_ids],
        }
        verifier_input = schedule.get("verifier_input_contract") or {}
        binding = {
            "source_step_id": str(source_step_id).strip(),
            "semantic_role": str(semantic_role).strip(),
        }
        if verifier_input:
            if not all(binding.values()):
                raise ValueError(
                    "retained regression decision requires source step and semantic role"
                )
            decision["variant_binding"] = binding
        elif any(binding.values()):
            raise ValueError(
                "generic Journey decision cannot declare a retained variant binding"
            )
    else:
        target = control_identity.get("scheduled_target")
        if not isinstance(target, Mapping):
            raise ValueError("edge broker request has no scheduled target")
        decision = {
            "previous_response_hash": response_hash,
            "broker_request_id": request_id,
            "user_message": message,
            "persona": str(target.get("persona") or ""),
            "goal": str(target.get("goal") or ""),
            "rationale": reason,
            "target_coverage_ids": [str(target.get("edge_key") or "")],
        }
    actor_declaration = {
        "actor_kind": str(actor_kind).strip(),
        "task_id": str(actor_task_id).strip(),
        "model": str(actor_model).strip(),
    }
    if any(actor_declaration.values()) and not all(actor_declaration.values()):
        raise ValueError("simulator actor declaration requires kind, task id, and model")
    if actor_declaration["actor_kind"]:
        attestation = build_simulator_attestation(
            actor_kind=actor_declaration["actor_kind"],
            task_id=actor_declaration["task_id"],
            model=actor_declaration["model"],
            request_id=request_id,
            previous_response_hash=response_hash,
            context_hash=simulator_context_hash(
                request.get("context") or {}
            ),
            decision_hash=_content_hash(decision),
            user_message_hash=_content_hash(message),
            turn_index=int((request.get("context") or {}).get("turn_index") or 0),
            declared_at_ns=time.time_ns(),
        )
        decision["simulator_attestation"] = attestation
    execution_payload_hash = _content_hash(decision)
    audit_decision = {
        **decision,
        "user_message": str(redact(message)),
        "rationale": str(redact(reason)),
    }
    record_unsigned = {
        "schema_version": BROKER_SCHEMA_VERSION,
        "request_id": request_id,
        "previous_response_hash": response_hash,
        "decision": audit_decision,
        "execution_payload_hash": execution_payload_hash,
        "submitted_at_ns": time.time_ns(),
    }
    record = {"record_id": _content_hash(record_unsigned), **record_unsigned}
    path = broker_root / "committed" / f"{request_id}.json"
    channel_path = Path(str(request.get("decision_channel") or ""))
    expected_channel = (broker_root / "channels").resolve()
    try:
        channel_path.resolve().relative_to(expected_channel)
    except (ValueError, OSError) as exc:
        raise ValueError("broker decision channel escaped its root") from exc
    wire = {
        "request_id": request_id,
        "previous_response_hash": response_hash,
        "execution_payload_hash": execution_payload_hash,
        "decision": decision,
        "record": record,
    }
    claim_descriptor = _acquire_request_claim(
        broker_root / "claims" / f"{request_id}.lock"
    )
    descriptor: int | None = None
    try:
        descriptor = _open_channel_writer(channel_path)
        if path.exists():
            raise FileExistsError(f"decision already submitted for request {request_id}")
        payload = (_canonical_json(wire) + "\n").encode("utf-8")
        _write_channel_payload_descriptor(
            descriptor, b"\n" + payload
        )
        deadline = time.monotonic() + 5.0
        while not path.is_file() and time.monotonic() < deadline:
            if not channel_path.exists():
                raise ExternalDecisionBlocked(
                    "broker worker rejected the external decision"
                )
            time.sleep(0.01)
        if not path.is_file():
            raise TimeoutError("broker worker did not acknowledge the external decision")
        bundle = _load_committed_bundle(path, request_id=request_id)
        committed_record = bundle.get("decision_record")
        if not isinstance(committed_record, Mapping):
            raise ExternalDecisionBlocked("committed broker bundle has no decision record")
        if (
            str(committed_record.get("execution_payload_hash") or "")
            != execution_payload_hash
        ):
            raise ExternalDecisionBlocked(
                "broker acknowledgement belongs to another external decision"
            )
    finally:
        try:
            if descriptor is not None:
                os.close(descriptor)
        finally:
            os.close(claim_descriptor)
    return path


def _validate_decision_record(
    request: Mapping[str, Any],
    record: Mapping[str, Any],
    execution_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    unsigned = {key: value for key, value in record.items() if key != "record_id"}
    if str(record.get("record_id") or "") != _content_hash(unsigned):
        raise ExternalDecisionBlocked("external decision record identity is stale")
    if str(record.get("request_id") or "") != str(request.get("request_id") or ""):
        raise ExternalDecisionBlocked("external decision belongs to another request")
    expected = str(request.get("previous_response_hash") or "")
    if str(record.get("previous_response_hash") or "") != expected:
        raise ExternalDecisionBlocked("external decision record is bound to a stale response")
    if str(execution_payload.get("request_id") or "") != str(request.get("request_id") or ""):
        raise ExternalDecisionBlocked("external execution payload belongs to another request")
    if str(execution_payload.get("previous_response_hash") or "") != expected:
        raise ExternalDecisionBlocked("external execution payload is bound to a stale response")
    decision = execution_payload.get("decision")
    if not isinstance(decision, Mapping):
        raise ExternalDecisionBlocked("external decision record has no decision object")
    if str(decision.get("previous_response_hash") or "") != expected:
        raise ExternalDecisionBlocked("external decision is bound to a stale response")
    expected_payload_hash = str(record.get("execution_payload_hash") or "")
    if str(execution_payload.get("execution_payload_hash") or "") != expected_payload_hash:
        raise ExternalDecisionBlocked("external execution payload identity is stale")
    if _content_hash(decision) != expected_payload_hash:
        raise ExternalDecisionBlocked("external execution decision was modified")
    expected_audit = {
        **decision,
        "user_message": str(redact(str(decision.get("user_message") or ""))),
        "rationale": str(redact(str(decision.get("rationale") or ""))),
    }
    if record.get("decision") != expected_audit:
        raise ExternalDecisionBlocked("external decision audit record does not match execution")
    return dict(decision)


def _validate_request_record(request: Mapping[str, Any], *, request_id: str) -> None:
    if request.get("schema_version") != BROKER_SCHEMA_VERSION:
        raise ValueError("unsupported broker request schema")
    unsigned = {key: value for key, value in request.items() if key != "request_id"}
    if str(request.get("request_id") or "") != _content_hash(unsigned):
        raise ValueError("broker request identity is stale")
    if str(request.get("request_id") or "") != str(request_id):
        raise ValueError("broker request id does not match its path")


def _load_committed_bundle(path: Path, *, request_id: str) -> Mapping[str, Any]:
    bundle = _load_json(path)
    if bundle.get("schema_version") != BROKER_SCHEMA_VERSION:
        raise ExternalDecisionBlocked("unsupported committed broker bundle schema")
    unsigned = {key: value for key, value in bundle.items() if key != "bundle_id"}
    if str(bundle.get("bundle_id") or "") != _content_hash(unsigned):
        raise ExternalDecisionBlocked("committed broker bundle identity is stale")
    if str(bundle.get("request_id") or "") != request_id:
        raise ExternalDecisionBlocked("committed broker bundle belongs to another request")
    receipt = bundle.get("consumed_receipt")
    if not isinstance(receipt, Mapping):
        raise ExternalDecisionBlocked("committed broker bundle has no consumed receipt")
    receipt_unsigned = {key: value for key, value in receipt.items() if key != "receipt_id"}
    if str(receipt.get("receipt_id") or "") != _content_hash(receipt_unsigned):
        raise ExternalDecisionBlocked("committed broker receipt identity is stale")
    record = bundle.get("decision_record")
    if not isinstance(record, Mapping):
        raise ExternalDecisionBlocked("committed broker bundle has no decision record")
    if str(record.get("request_id") or "") != request_id:
        raise ExternalDecisionBlocked("committed decision belongs to another request")
    if str(receipt.get("request_id") or "") != request_id:
        raise ExternalDecisionBlocked("committed receipt belongs to another request")
    if (
        str(receipt.get("decision_hash") or "")
        != str(record.get("execution_payload_hash") or "")
    ):
        raise ExternalDecisionBlocked("committed receipt does not match its decision")
    decision = record.get("decision")
    if not isinstance(decision, Mapping):
        raise ExternalDecisionBlocked("committed decision record has no audit decision")
    if bundle.get("simulator_attestation") != decision.get("simulator_attestation"):
        raise ExternalDecisionBlocked("committed attestation does not match its decision")
    return bundle


def _open_channel_writer(path: Path) -> int:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0),
    )
    if not stat.S_ISFIFO(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError("external decision channel is not a FIFO")
    return descriptor


def _acquire_request_claim(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _channel_has_reader(path: Path) -> bool:
    descriptor = _open_channel_writer(path)
    os.close(descriptor)
    return True


def _write_channel_payload_descriptor(
    descriptor: int,
    payload: bytes,
    *,
    timeout_seconds: float = 5.0,
) -> None:
    view = memoryview(payload)
    deadline = time.monotonic() + timeout_seconds
    while view:
        try:
            written = os.write(descriptor, view)
        except BlockingIOError:
            written = 0
        if written:
            view = view[written:]
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("external decision channel write timed out")
        select.select((), (descriptor,), (), min(remaining, 0.05))


def _control_identity(context: Mapping[str, Any]) -> dict[str, Any]:
    """Project immutable scheduler identity outside the redaction data plane."""

    schedule = context.get("schedule")
    if isinstance(schedule, Mapping):
        return {"lane": "journey", "schedule": dict(schedule)}
    target = context.get("scheduled_target")
    if not isinstance(target, Mapping):
        raise ExternalDecisionBlocked("broker context has no scheduled control identity")
    return {"lane": "edge", "scheduled_target": dict(target)}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _content_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode()
    temporary = path.parent / (
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp"
    )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    linked = False
    temporary_removed = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        linked = True
        temporary.unlink()
        temporary_removed = True
        _fsync_directory(path.parent)
    except BaseException:
        if linked:
            path.unlink(missing_ok=True)
        if not temporary_removed:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        if linked or temporary_removed:
            _fsync_directory(path.parent)
        raise


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--manifest", required=True)
    run.add_argument("--result-index", required=True)
    run.add_argument("--broker-root", required=True)
    run.add_argument("--decision-timeout", type=float, default=230.0)
    run.add_argument("--authority-key-fd", type=int, required=True)
    pending = subparsers.add_parser("pending")
    pending.add_argument("--broker-root", required=True)
    submit = subparsers.add_parser("submit")
    submit.add_argument("--broker-root", required=True)
    submit.add_argument("--request-id", required=True)
    submit.add_argument("--message", required=True)
    submit.add_argument("--rationale", required=True)
    submit.add_argument("--risk-factor", action="append", default=[])
    submit.add_argument("--actor-kind", choices=("codex", "script"), default="")
    submit.add_argument("--actor-task-id", default="")
    submit.add_argument("--actor-model", default="")
    submit.add_argument("--source-step-id", default="")
    submit.add_argument("--semantic-role", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "pending":
        print(json.dumps(pending_requests(args.broker_root), ensure_ascii=False, indent=2))
        return 0
    if args.command == "submit":
        print(submit_decision(
            args.broker_root,
            request_id=args.request_id,
            user_message=args.message,
            rationale=args.rationale,
            risk_factor_ids=args.risk_factor,
            actor_kind=args.actor_kind,
            actor_task_id=args.actor_task_id,
            actor_model=args.actor_model,
            source_step_id=args.source_step_id,
            semantic_role=args.semantic_role,
        ))
        return 0
    manifest = load_frozen_manifest(args.manifest)
    try:
        private_key = os.read(args.authority_key_fd, 33)
    finally:
        os.close(args.authority_key_fd)
    if len(private_key) != 32:
        raise ValueError("controller authority pipe did not contain one Ed25519 key")
    authority_signer = import_pty_authority_private_key(
        private_key,
        expected_public_key_b64=manifest.pty_authority_public_key_b64,
    )
    broker = FilesystemDecisionBroker(
        args.broker_root,
        batch_id=manifest.batch_id,
        timeout_seconds=args.decision_timeout,
    )
    interrupted_by = asyncio.run(_run_with_signal_cleanup(
        manifest,
        broker=broker,
        result_index_path=args.result_index,
        authority_signer=authority_signer,
    ))
    return 128 + interrupted_by if interrupted_by else 0


async def _run_with_signal_cleanup(
    manifest: Any,
    *,
    broker: FilesystemDecisionBroker,
    result_index_path: str | Path,
    authority_signer: PtyAuthoritySigner,
) -> int:
    """Translate process signals into one awaited batch cancellation path."""

    loop = asyncio.get_running_loop()
    interrupted_by = 0
    interruption_event = asyncio.Event()

    def interrupt(signum: int) -> None:
        nonlocal interrupted_by
        if interrupted_by:
            return
        interrupted_by = signum
        close_broker = getattr(broker, "close", None)
        if callable(close_broker):
            close_broker()
        interruption_event.set()

    installed: list[int] = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, interrupt, signum)
            installed.append(signum)
        except (NotImplementedError, RuntimeError):
            continue
    try:
        await run_batch(
            manifest,
            broker=broker,
            result_index_path=result_index_path,
            interruption_event=interruption_event,
            authority_signer=authority_signer,
        )
    finally:
        for signum in installed:
            loop.remove_signal_handler(signum)
    return interrupted_by


if __name__ == "__main__":
    raise SystemExit(main())

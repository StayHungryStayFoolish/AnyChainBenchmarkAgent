"""Filesystem rendezvous for a real external Codex Chaos user.

The broker publishes only after a complete Agent response is available.  A
Codex operator reads that response-bound request and submits exactly one next
turn; no future dialogue is stored in the batch target or schedule.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import signal
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


BROKER_SCHEMA_VERSION = 1


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

    async def __call__(self, shard_id: str, context: Mapping[str, Any]) -> Mapping[str, Any]:
        response_hash = str(context.get("previous_response_hash") or "").strip()
        if len(response_hash) != 64:
            raise ExternalDecisionBlocked("broker context has no valid response hash")
        with self._lock:
            sequence = self._sequence_by_shard.get(shard_id, 0) + 1
            self._sequence_by_shard[shard_id] = sequence
        unsigned = {
            "schema_version": BROKER_SCHEMA_VERSION,
            "batch_id": self.batch_id,
            "shard_id": str(shard_id),
            "sequence": sequence,
            "previous_response_hash": response_hash,
            "context_hash": _content_hash(context),
            "context": redact(dict(context)),
            "created_at_ns": time.time_ns(),
        }
        request_id = _content_hash(unsigned)
        request = {"request_id": request_id, **unsigned}
        _write_immutable_json(self.root / "requests" / f"{request_id}.json", request)

        decision_path = self.root / "decisions" / f"{request_id}.json"
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            if decision_path.is_file():
                decision_record = _load_json(decision_path)
                decision = _validate_decision_record(request, decision_record)
                receipt_unsigned = {
                    "schema_version": BROKER_SCHEMA_VERSION,
                    "request_id": request_id,
                    "decision_hash": _content_hash(decision),
                    "consumed_at_ns": time.time_ns(),
                }
                _write_immutable_json(
                    self.root / "consumed" / f"{request_id}.json",
                    {"receipt_id": _content_hash(receipt_unsigned), **receipt_unsigned},
                )
                return decision
            await asyncio.sleep(self.poll_seconds)
        raise ExternalDecisionBlocked(
            f"external Codex decision timed out for request {request_id}"
        )


def pending_requests(root: str | Path) -> tuple[Mapping[str, Any], ...]:
    broker_root = Path(root).resolve()
    consumed = broker_root / "consumed"
    rows = []
    for path in sorted((broker_root / "requests").glob("*.json")):
        if not (consumed / path.name).exists():
            rows.append(_load_json(path))
    return tuple(rows)


def submit_decision(
    root: str | Path,
    *,
    request_id: str,
    user_message: str,
    rationale: str,
    risk_factor_ids: Sequence[str] = (),
) -> Path:
    broker_root = Path(root).resolve()
    request = _load_json(broker_root / "requests" / f"{request_id}.json")
    context = request.get("context")
    if not isinstance(context, Mapping):
        raise ValueError("broker request has no context")
    response_hash = str(request.get("previous_response_hash") or "")
    message = str(redact(str(user_message))).strip()
    reason = str(redact(str(rationale))).strip()
    if not message or not reason:
        raise ValueError("decision requires a user message and rationale")
    if isinstance(context.get("schedule"), Mapping):
        schedule = context["schedule"]
        decision = {
            "previous_response_hash": response_hash,
            "user_message": message,
            "persona": str(schedule.get("persona") or ""),
            "mission": str(schedule.get("mission") or ""),
            "rationale": reason,
            "risk_factor_ids": [str(item) for item in risk_factor_ids],
        }
    else:
        target = context.get("scheduled_target")
        if not isinstance(target, Mapping):
            raise ValueError("edge broker request has no scheduled target")
        decision = {
            "previous_response_hash": response_hash,
            "user_message": message,
            "persona": str(target.get("persona") or ""),
            "goal": str(target.get("goal") or ""),
            "rationale": reason,
            "target_coverage_ids": [str(target.get("edge_key") or "")],
        }
    record_unsigned = {
        "schema_version": BROKER_SCHEMA_VERSION,
        "request_id": request_id,
        "previous_response_hash": response_hash,
        "decision": decision,
        "submitted_at_ns": time.time_ns(),
    }
    record = {"record_id": _content_hash(record_unsigned), **record_unsigned}
    path = broker_root / "decisions" / f"{request_id}.json"
    _write_immutable_json(path, record)
    return path


def _validate_decision_record(
    request: Mapping[str, Any], record: Mapping[str, Any]
) -> Mapping[str, Any]:
    unsigned = {key: value for key, value in record.items() if key != "record_id"}
    if str(record.get("record_id") or "") != _content_hash(unsigned):
        raise ExternalDecisionBlocked("external decision record identity is stale")
    if str(record.get("request_id") or "") != str(request.get("request_id") or ""):
        raise ExternalDecisionBlocked("external decision belongs to another request")
    expected = str(request.get("previous_response_hash") or "")
    if str(record.get("previous_response_hash") or "") != expected:
        raise ExternalDecisionBlocked("external decision record is bound to a stale response")
    decision = record.get("decision")
    if not isinstance(decision, Mapping):
        raise ExternalDecisionBlocked("external decision record has no decision object")
    if str(decision.get("previous_response_hash") or "") != expected:
        raise ExternalDecisionBlocked("external decision is bound to a stale response")
    return dict(decision)


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
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--manifest", required=True)
    run.add_argument("--result-index", required=True)
    run.add_argument("--broker-root", required=True)
    run.add_argument("--decision-timeout", type=float, default=230.0)
    pending = subparsers.add_parser("pending")
    pending.add_argument("--broker-root", required=True)
    submit = subparsers.add_parser("submit")
    submit.add_argument("--broker-root", required=True)
    submit.add_argument("--request-id", required=True)
    submit.add_argument("--message", required=True)
    submit.add_argument("--rationale", required=True)
    submit.add_argument("--risk-factor", action="append", default=[])
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
        ))
        return 0
    manifest = load_frozen_manifest(args.manifest)
    broker = FilesystemDecisionBroker(
        args.broker_root,
        batch_id=manifest.batch_id,
        timeout_seconds=args.decision_timeout,
    )
    interrupted_by = asyncio.run(_run_with_signal_cleanup(
        manifest,
        broker=broker,
        result_index_path=args.result_index,
    ))
    return 128 + interrupted_by if interrupted_by else 0


async def _run_with_signal_cleanup(
    manifest: Any,
    *,
    broker: FilesystemDecisionBroker,
    result_index_path: str | Path,
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
        )
    finally:
        for signum in installed:
            loop.remove_signal_handler(signum)
    return interrupted_by


if __name__ == "__main__":
    raise SystemExit(main())

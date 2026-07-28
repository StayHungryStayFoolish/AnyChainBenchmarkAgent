#!/usr/bin/env python3
"""Linux container process cleanup bound to one exact Chaos execution id."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


EXECUTION_ID_ENV = "ANYCHAIN_CHAOS_EXECUTION_ID"
RECEIPT_DIR_ENV = "ANYCHAIN_CHAOS_INNER_CLEANUP_RECEIPT_DIR"
RECEIPT_SCHEMA_VERSION = 1


class ProcessIdentityError(RuntimeError):
    """Raised when a process identity cannot be proved from procfs."""


class CleanupReceiptError(RuntimeError):
    """Raised when an immutable cleanup receipt cannot be persisted."""


class _ProcessGone(ProcessLookupError):
    pass


def validate_cleanup_receipt_artifact(
    path: str | Path,
    *,
    execution_id: str,
    required_roles: tuple[str, ...],
    allowed_roots: tuple[str | Path, ...],
) -> dict[str, Any]:
    """Validate one content-addressed process-guard receipt from disk."""

    receipt_path = Path(path).resolve()
    roots = tuple(Path(root).resolve() for root in allowed_roots)
    if not roots or not any(
        receipt_path == root or root in receipt_path.parents for root in roots
    ):
        raise ValueError("process guard receipt is outside its allowed roots")
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load process guard receipt: {exc}") from exc
    required = {
        "receipt_id", "schema_version", "execution_id", "container_id",
        "pid_namespace_inode", "bridge_identity", "registered_processes",
        "term_decisions", "kill_decisions", "reap_results", "survivor_scans",
        "zero_survivor_scans", "errors", "cleaned", "started_at_ns",
        "finished_at_ns",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError(
            "process guard receipt fields do not match the authoritative schema"
        )
    if payload["schema_version"] != RECEIPT_SCHEMA_VERSION:
        raise ValueError("unsupported process guard receipt schema")
    if payload["execution_id"] != execution_id:
        raise ValueError("process guard receipt execution id mismatch")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_id"}
    receipt_id = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    if payload["receipt_id"] != receipt_id:
        raise ValueError("process guard receipt content hash is stale")
    if receipt_path.name != f"container-cleanup-receipt-{receipt_id}.json":
        raise ValueError("process guard receipt filename is not content-addressed")
    if not str(payload["container_id"]):
        raise ValueError("process guard receipt has no container/host identity")
    if int(payload["pid_namespace_inode"]) <= 0:
        raise ValueError("process guard receipt has no PID namespace identity")

    registered_rows = payload["registered_processes"]
    if not isinstance(registered_rows, list) or not registered_rows:
        raise ValueError("process guard receipt has no registered processes")
    registered: set[tuple[int, int, int, int]] = set()
    roles: set[str] = set()
    for row in registered_rows:
        identity = _receipt_identity(row)
        if identity in registered:
            raise ValueError("process guard receipt repeats a registered identity")
        registered.add(identity)
        row_roles = row.get("roles") if isinstance(row, Mapping) else None
        if not isinstance(row_roles, list) or not all(
            isinstance(role, str) and role for role in row_roles
        ):
            raise ValueError("registered process roles are invalid")
        roles.update(row_roles)
    if not set(required_roles).issubset(roles):
        raise ValueError("process guard receipt is missing required registered roles")
    if _receipt_identity(payload["bridge_identity"]) not in registered:
        raise ValueError("process guard bridge/observer identity was not registered")

    term_identities = _validate_receipt_signal_decisions(
        payload["term_decisions"],
        registered,
        expected_signal="SIGTERM",
    )
    kill_identities = _validate_receipt_signal_decisions(
        payload["kill_decisions"],
        registered,
        expected_signal="SIGKILL",
    )
    if not kill_identities.issubset(term_identities):
        raise ValueError("process guard issued KILL without a TERM decision")

    zero_scans = payload["zero_survivor_scans"]
    if not isinstance(zero_scans, list) or len(zero_scans) != 2:
        raise ValueError("process guard receipt lacks two stable zero-survivor scans")
    for scan in zero_scans:
        if not isinstance(scan, Mapping) or set(scan) != {
            "scan_index", "purpose", "scanned_at_ns", "survivors", "scan_errors"
        }:
            raise ValueError("process guard zero-survivor scan is malformed")
        if (
            scan["purpose"] != "zero_survivor_verification"
            or scan["survivors"]
            or scan["scan_errors"]
        ):
            raise ValueError("process guard zero-survivor scans are not clean")
    if (
        int(zero_scans[1]["scan_index"]) <= int(zero_scans[0]["scan_index"])
        or int(zero_scans[1]["scanned_at_ns"])
        <= int(zero_scans[0]["scanned_at_ns"])
    ):
        raise ValueError(
            "process guard zero-survivor scans are not stable and ordered"
        )
    survivor_scans = payload["survivor_scans"]
    if not isinstance(survivor_scans, list) or survivor_scans[-2:] != zero_scans:
        raise ValueError(
            "process guard final scans differ from its zero-survivor proof"
        )

    reap_results = payload["reap_results"]
    if not isinstance(reap_results, list) or not all(
        isinstance(row, Mapping)
        and set(row) == {"pid", "reaped", "return_code", "error"}
        and int(row.get("pid") or 0) > 0
        and isinstance(row.get("reaped"), bool)
        for row in reap_results
    ):
        raise ValueError("process guard reap results are malformed")
    if payload["cleaned"]:
        if payload["errors"]:
            raise ValueError("clean process guard receipt contains errors")
        if any(not bool(row.get("reaped")) for row in reap_results):
            raise ValueError(
                "clean process guard receipt contains an unreaped process"
            )
    if payload["cleaned"] is not True:
        raise ValueError("process guard receipt does not prove cleanup")

    return {
        "receipt_id": receipt_id,
        "path": str(receipt_path),
        "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "execution_id": execution_id,
        "container_id": str(payload["container_id"]),
        "pid_namespace_inode": int(payload["pid_namespace_inode"]),
        "registered_processes": registered_rows,
        "term_decisions": payload["term_decisions"],
        "kill_decisions": payload["kill_decisions"],
        "reap_results": payload["reap_results"],
        "zero_survivor_scans": zero_scans,
        "errors": payload["errors"],
        "cleaned": True,
    }


def _validate_receipt_signal_decisions(
    rows: Any,
    registered: set[tuple[int, int, int, int]],
    *,
    expected_signal: str,
) -> set[tuple[int, int, int, int]]:
    if not isinstance(rows, list):
        raise ValueError("process guard signal decisions are not a list")
    identities: set[tuple[int, int, int, int]] = set()
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {"identity", "signal", "sent", "outcome", "decided_at_ns"}
            or row.get("signal") != expected_signal
        ):
            raise ValueError("process guard signal decision is malformed")
        if (
            not isinstance(row.get("sent"), bool)
            or int(row.get("decided_at_ns") or 0) <= 0
        ):
            raise ValueError("process guard signal decision metadata is malformed")
        if row["sent"] != (row.get("outcome") == "sent"):
            raise ValueError(
                "process guard signal outcome contradicts its sent flag"
            )
        identity = _receipt_identity(row.get("identity"))
        if identity not in registered:
            raise ValueError("process guard signaled an unregistered identity")
        identities.add(identity)
    return identities


def _receipt_identity(value: Any) -> tuple[int, int, int, int]:
    if not isinstance(value, Mapping):
        raise ValueError("process identity is not an object")
    try:
        identity = (
            int(value["pid"]),
            int(value["pgid"]),
            int(value["start_ticks"]),
            int(value["pid_namespace_inode"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("process identity fields are invalid") from exc
    if any(item <= 0 for item in identity):
        raise ValueError("process identity fields must be positive")
    return identity


@dataclass(frozen=True, order=True)
class ProcessIdentity:
    pid: int
    pgid: int
    start_ticks: int
    pid_namespace_inode: int

    def payload(self) -> dict[str, int]:
        return {
            "pid": self.pid,
            "pgid": self.pgid,
            "start_ticks": self.start_ticks,
            "pid_namespace_inode": self.pid_namespace_inode,
        }


@dataclass(frozen=True)
class CleanupReceiptArtifact:
    path: Path
    payload: Mapping[str, Any]

    @property
    def receipt_id(self) -> str:
        return str(self.payload["receipt_id"])

    @property
    def cleaned(self) -> bool:
        return bool(self.payload["cleaned"])

    @property
    def file_sha256(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()


class ContainerProcessGuard:
    """Register, terminate, and prove absence of one container execution tree."""

    def __init__(
        self,
        execution_id: str,
        *,
        receipt_dir: str | Path,
        proc_root: str | Path = "/proc",
        term_grace_seconds: float = 1.0,
        kill_grace_seconds: float = 1.0,
        scan_interval_seconds: float = 0.05,
        cleanup_rounds: int = 2,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        time_ns: Callable[[], int] = time.time_ns,
        send_signal: Callable[[int, int], None] = os.kill,
        container_id: str | None = None,
    ) -> None:
        if not execution_id or "\0" in execution_id:
            raise ValueError(f"{EXECUTION_ID_ENV} must be a non-empty environment value")
        if term_grace_seconds < 0 or kill_grace_seconds < 0:
            raise ValueError("cleanup grace periods cannot be negative")
        if scan_interval_seconds <= 0:
            raise ValueError("cleanup scan interval must be positive")
        if cleanup_rounds < 1:
            raise ValueError("cleanup rounds must be positive")

        self.execution_id = execution_id
        self.receipt_dir = Path(receipt_dir)
        self.proc_root = Path(proc_root)
        self.term_grace_seconds = term_grace_seconds
        self.kill_grace_seconds = kill_grace_seconds
        self.scan_interval_seconds = scan_interval_seconds
        self.cleanup_rounds = cleanup_rounds
        self._sleep = sleep
        self._monotonic = monotonic
        self._time_ns = time_ns
        self._send_signal = send_signal
        self.container_id = container_id or self._read_container_id()
        self.bridge_identity = self._read_identity(os.getpid())
        self.pid_namespace_inode = self.bridge_identity.pid_namespace_inode
        self._ancestor_pids = self._read_ancestor_pids(self.bridge_identity.pid)
        self._registered: dict[ProcessIdentity, set[str]] = {
            self.bridge_identity: {"container_bridge"}
        }
        self._term_attempted: set[ProcessIdentity] = set()
        self._term_decisions: list[dict[str, Any]] = []
        self._kill_decisions: list[dict[str, Any]] = []
        self._scans: list[dict[str, Any]] = []
        self._errors: list[str] = []

    @classmethod
    def from_environment(
        cls,
        env: Mapping[str, str] | None = None,
        **changes: Any,
    ) -> "ContainerProcessGuard":
        values = os.environ if env is None else env
        execution_id = str(values.get(EXECUTION_ID_ENV) or "")
        receipt_dir = str(values.get(RECEIPT_DIR_ENV) or "")
        if not receipt_dir:
            checkpoint = str(values.get("ANYCHAIN_AGENT_CHECKPOINT_PATH") or "")
            if checkpoint:
                receipt_dir = str(Path(checkpoint).parent / "container-cleanup-receipts")
        if not receipt_dir:
            raise ValueError(
                f"{RECEIPT_DIR_ENV} or ANYCHAIN_AGENT_CHECKPOINT_PATH is required"
            )
        return cls(execution_id, receipt_dir=receipt_dir, **changes)

    def register_pid(self, pid: int, *, role: str) -> ProcessIdentity:
        if pid <= 0 or not role:
            raise ValueError("registered processes require a positive pid and role")
        identity = self._read_identity(pid)
        self._registered.setdefault(identity, set()).add(role)
        return identity

    def cleanup(
        self,
        *,
        reapers: Mapping[int, Callable[[], int | None]] | None = None,
    ) -> CleanupReceiptArtifact:
        started_at_ns = self._time_ns()
        reaper_callbacks = dict(reapers or {})
        zero_scans: tuple[dict[str, Any], dict[str, Any]] | tuple[()] = ()

        for _round in range(self.cleanup_rounds):
            survivors = self._scan_survivors(reaper_callbacks)
            self._send_term_to_new(survivors)
            survivors = self._wait_for_survivors(
                self.term_grace_seconds,
                reaper_callbacks,
                send_term_to_new=True,
            )
            self._send_kill(survivors)
            self._wait_for_survivors(
                self.kill_grace_seconds,
                reaper_callbacks,
                send_term_to_new=False,
            )

            first_survivors, first = self._record_scan(
                reaper_callbacks, purpose="zero_survivor_verification"
            )
            if first_survivors:
                continue
            self._sleep(self.scan_interval_seconds)
            second_survivors, second = self._record_scan(
                reaper_callbacks, purpose="zero_survivor_verification"
            )
            if not second_survivors:
                zero_scans = (first, second)
                break

        reap_results = self._wait_for_reapers(
            reaper_callbacks,
            timeout_seconds=self.kill_grace_seconds,
        )
        cleaned = bool(zero_scans) and not self._errors and all(
            result["reaped"] for result in reap_results
        )
        finished_at_ns = self._time_ns()
        unsigned = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "execution_id": self.execution_id,
            "container_id": self.container_id,
            "pid_namespace_inode": self.pid_namespace_inode,
            "bridge_identity": self.bridge_identity.payload(),
            "registered_processes": self._registered_payload(),
            "term_decisions": self._term_decisions,
            "kill_decisions": self._kill_decisions,
            "reap_results": reap_results,
            "survivor_scans": self._scans,
            "zero_survivor_scans": list(zero_scans),
            "errors": self._errors,
            "cleaned": cleaned,
            "started_at_ns": started_at_ns,
            "finished_at_ns": finished_at_ns,
        }
        return self._write_receipt(unsigned)

    def _wait_for_reapers(
        self,
        reapers: Mapping[int, Callable[[], int | None]],
        *,
        timeout_seconds: float,
    ) -> list[dict[str, Any]]:
        """Allow parent runtimes to publish child exit status after disappearance."""

        deadline = self._monotonic() + timeout_seconds
        results = self._reap_results(reapers)
        while (
            any(not result["reaped"] for result in results)
            and self._monotonic() < deadline
        ):
            self._sleep(
                min(
                    self.scan_interval_seconds,
                    max(0.0, deadline - self._monotonic()),
                )
            )
            results = self._reap_results(reapers)
        return results

    def _wait_for_survivors(
        self,
        timeout_seconds: float,
        reapers: Mapping[int, Callable[[], int | None]],
        *,
        send_term_to_new: bool,
    ) -> tuple[ProcessIdentity, ...]:
        deadline = self._monotonic() + timeout_seconds
        survivors = self._scan_survivors(reapers)
        while survivors and self._monotonic() < deadline:
            if send_term_to_new:
                self._send_term_to_new(survivors)
            self._sleep(min(self.scan_interval_seconds, max(0.0, deadline - self._monotonic())))
            survivors = self._scan_survivors(reapers)
        return survivors

    def _send_term_to_new(self, identities: tuple[ProcessIdentity, ...]) -> None:
        for identity in identities:
            if identity in self._term_attempted:
                continue
            self._term_attempted.add(identity)
            self._term_decisions.append(self._signal_decision(identity, signal.SIGTERM))

    def _send_kill(self, identities: tuple[ProcessIdentity, ...]) -> None:
        for identity in identities:
            if identity not in self._term_attempted:
                self._errors.append(
                    f"refused SIGKILL for pid {identity.pid} before a registered SIGTERM decision"
                )
                continue
            self._kill_decisions.append(self._signal_decision(identity, signal.SIGKILL))

    def _signal_decision(self, identity: ProcessIdentity, signum: int) -> dict[str, Any]:
        decision: dict[str, Any] = {
            "identity": identity.payload(),
            "signal": signal.Signals(signum).name,
            "sent": False,
            "outcome": "",
            "decided_at_ns": self._time_ns(),
        }
        try:
            current = self._read_identity(identity.pid)
        except _ProcessGone:
            decision["outcome"] = "already_gone"
            return decision
        except ProcessIdentityError as exc:
            decision["outcome"] = "identity_unreadable"
            self._errors.append(str(exc))
            return decision
        if current != identity:
            decision["outcome"] = "identity_changed"
            return decision
        try:
            self._send_signal(identity.pid, signum)
        except ProcessLookupError:
            decision["outcome"] = "already_gone"
        except OSError as exc:
            decision["outcome"] = "signal_failed"
            self._errors.append(
                f"{decision['signal']} failed for registered pid {identity.pid}: {exc}"
            )
        else:
            decision["sent"] = True
            decision["outcome"] = "sent"
        return decision

    def _scan_survivors(
        self,
        reapers: Mapping[int, Callable[[], int | None]],
    ) -> tuple[ProcessIdentity, ...]:
        identities, _record = self._record_scan(reapers, purpose="cleanup")
        return identities

    def _record_scan(
        self,
        reapers: Mapping[int, Callable[[], int | None]],
        *,
        purpose: str,
    ) -> tuple[tuple[ProcessIdentity, ...], dict[str, Any]]:
        self._poll_reapers(reapers)
        matches, scan_errors = self._matching_execution_identities()
        for error in scan_errors:
            if error not in self._errors:
                self._errors.append(error)
        for identity in matches:
            self._registered.setdefault(identity, set()).add("execution_id_match")

        survivors = set(matches)
        survivors.discard(self.bridge_identity)
        for identity in self._registered:
            if identity == self.bridge_identity:
                continue
            try:
                if self._read_identity(identity.pid) == identity:
                    survivors.add(identity)
            except _ProcessGone:
                continue
            except ProcessIdentityError as exc:
                error = str(exc)
                if error not in self._errors:
                    self._errors.append(error)
                survivors.add(identity)

        ordered = tuple(sorted(survivors))
        recorded = {
            "scan_index": len(self._scans) + 1,
            "purpose": purpose,
            "scanned_at_ns": self._time_ns(),
            "survivors": [identity.payload() for identity in ordered],
            "scan_errors": list(scan_errors),
        }
        self._scans.append(recorded)
        return ordered, recorded

    def _matching_execution_identities(
        self,
    ) -> tuple[tuple[ProcessIdentity, ...], tuple[str, ...]]:
        token = f"{EXECUTION_ID_ENV}={self.execution_id}".encode("utf-8")
        matches: list[ProcessIdentity] = []
        errors: list[str] = []
        try:
            entries = tuple(self.proc_root.iterdir())
        except OSError as exc:
            return (), (f"cannot enumerate {self.proc_root}: {exc}",)
        for entry in entries:
            if not entry.name.isdecimal():
                continue
            pid = int(entry.name)
            if pid in self._ancestor_pids:
                continue
            try:
                before = self._read_identity(pid)
                environ = (entry / "environ").read_bytes()
                after = self._read_identity(pid)
            except (_ProcessGone, FileNotFoundError, ProcessLookupError):
                continue
            except (OSError, ProcessIdentityError) as exc:
                errors.append(f"cannot inspect proc identity/environment for pid {pid}: {exc}")
                continue
            if before != after:
                errors.append(f"pid {pid} changed identity during execution-id scan")
                continue
            if token in environ.split(b"\0"):
                matches.append(after)
        return tuple(sorted(set(matches))), tuple(errors)

    def _read_ancestor_pids(self, pid: int) -> frozenset[int]:
        """Return supervisors that must never be claimed by this child guard."""

        ancestors: set[int] = set()
        current = pid
        while current > 1:
            status = self.proc_root / str(current) / "status"
            try:
                lines = status.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                break
            parent_line = next(
                (line for line in lines if line.startswith("PPid:")), ""
            )
            try:
                parent = int(parent_line.split(":", 1)[1].strip())
            except (IndexError, ValueError):
                break
            if parent <= 0 or parent == current:
                break
            ancestors.add(parent)
            current = parent
        return frozenset(ancestors)

    def _read_identity(self, pid: int) -> ProcessIdentity:
        process_root = self.proc_root / str(pid)
        try:
            stat_text = (process_root / "stat").read_text(encoding="utf-8")
            namespace_inode = (process_root / "ns" / "pid").stat().st_ino
        except (FileNotFoundError, ProcessLookupError) as exc:
            raise _ProcessGone(pid) from exc
        except OSError as exc:
            raise ProcessIdentityError(f"cannot read registered identity for pid {pid}: {exc}") from exc
        closing_paren = stat_text.rfind(")")
        fields = stat_text[closing_paren + 2:].split() if closing_paren >= 0 else []
        if len(fields) <= 19:
            raise ProcessIdentityError(f"malformed proc stat for pid {pid}")
        try:
            pgid = int(fields[2])
            start_ticks = int(fields[19])
        except ValueError as exc:
            raise ProcessIdentityError(f"malformed proc identity fields for pid {pid}") from exc
        return ProcessIdentity(
            pid=pid,
            pgid=pgid,
            start_ticks=start_ticks,
            pid_namespace_inode=namespace_inode,
        )

    def _poll_reapers(self, reapers: Mapping[int, Callable[[], int | None]]) -> None:
        for pid, reap in reapers.items():
            try:
                reap()
            except Exception as exc:
                error = f"reaper failed for pid {pid}: {type(exc).__name__}: {exc}"
                if error not in self._errors:
                    self._errors.append(error)

    def _reap_results(
        self,
        reapers: Mapping[int, Callable[[], int | None]],
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for pid, reap in sorted(reapers.items()):
            try:
                return_code = reap()
            except Exception as exc:
                error = f"final reaper failed for pid {pid}: {type(exc).__name__}: {exc}"
                self._errors.append(error)
                results.append({"pid": pid, "reaped": False, "return_code": None, "error": error})
            else:
                results.append({
                    "pid": pid,
                    "reaped": return_code is not None,
                    "return_code": return_code,
                    "error": "",
                })
        return results

    def _registered_payload(self) -> list[dict[str, Any]]:
        return [
            {**identity.payload(), "roles": sorted(roles)}
            for identity, roles in sorted(self._registered.items())
        ]

    def _write_receipt(self, unsigned: Mapping[str, Any]) -> CleanupReceiptArtifact:
        receipt_id = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
        payload = {"receipt_id": receipt_id, **dict(unsigned)}
        encoded = _canonical_json(payload) + b"\n"
        try:
            self.receipt_dir.mkdir(parents=True, exist_ok=True)
            path = self.receipt_dir / f"container-cleanup-receipt-{receipt_id}.json"
            try:
                with path.open("xb") as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
            except FileExistsError:
                if path.read_bytes() != encoded:
                    raise CleanupReceiptError(f"existing cleanup receipt differs: {path}")
        except (OSError, CleanupReceiptError) as exc:
            raise CleanupReceiptError(f"cannot persist container cleanup receipt: {exc}") from exc
        return CleanupReceiptArtifact(path=path, payload=payload)

    @staticmethod
    def _read_container_id() -> str:
        try:
            value = Path("/etc/hostname").read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ProcessIdentityError(f"cannot read container id from /etc/hostname: {exc}") from exc
        if not value:
            raise ProcessIdentityError("container id from /etc/hostname is empty")
        return value


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")

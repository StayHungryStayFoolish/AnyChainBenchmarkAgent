#!/usr/bin/env python3
"""Container-local PTY supervisor for formal Agent CLI evidence.

The host talks to this process over newline-delimited JSON on ordinary pipes.
This process alone owns the PTY used by the real product CLI, avoiding the
unsupported host-PTY-around-Docker-TTY topology that loses prompt-toolkit
frames after multiline redraws.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

from tests.agent_live.container_process_guard import ContainerProcessGuard
from tests.agent_live.dynamic_dual_ai_chaos import SubprocessPtyTransport


def _write(payload: Mapping[str, Any]) -> None:
    sys.stdout.write(json.dumps(dict(payload), ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def _command(values: Sequence[str]) -> tuple[str, ...]:
    command = tuple(values[1:] if values and values[0] == "--" else values)
    if not command:
        raise ValueError("container PTY bridge requires a product CLI command")
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    guard = ContainerProcessGuard.from_environment()
    transport = SubprocessPtyTransport(_command(args.command), cwd=Path(args.cwd))
    agent_process = None
    close_requested = False
    terminal_error: dict[str, Any] | None = None
    exit_code = 0

    def stop(_signum: int, _frame: object) -> None:
        raise SystemExit(128 + _signum)

    try:
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGHUP, stop)
        transport.start(env=os.environ)
        agent_process = transport._process
        if agent_process is None:
            raise RuntimeError("container PTY transport did not expose its Agent process")
        guard.register_pid(agent_process.pid, role="agent_process_group_leader")
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                request = json.loads(line)
                operation = str(request.get("op") or "")
                if operation == "read":
                    response = transport.read_complete_agent_response(
                        timeout_seconds=float(request.get("timeout_seconds") or 0),
                    )
                    _write({"ok": True, "response": response})
                elif operation == "submit":
                    transport.submit_bracketed_paste(str(request.get("message") or ""))
                    _write({"ok": True})
                elif operation == "interrupt":
                    transport.send_interrupt()
                    _write({"ok": True})
                elif operation == "close":
                    close_requested = True
                    break
                else:
                    raise ValueError(f"unsupported container PTY bridge operation: {operation!r}")
            except Exception as exc:
                terminal_error = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                exit_code = 1
                break
    finally:
        try:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
        except ValueError:
            pass
        cleanup_artifact = None
        cleanup_error = ""
        cleanup_process = agent_process or transport._process
        if agent_process is None and cleanup_process is not None:
            try:
                guard.register_pid(
                    cleanup_process.pid,
                    role="agent_process_group_leader",
                )
            except Exception as exc:
                cleanup_error = f"late Agent registration failed: {type(exc).__name__}: {exc}"
        try:
            reapers = (
                {cleanup_process.pid: cleanup_process.poll}
                if cleanup_process is not None
                else {}
            )
            cleanup_artifact = guard.cleanup(reapers=reapers)
            if not cleanup_artifact.cleaned:
                cleanup_error = "container process cleanup proof is incomplete"
        except Exception as exc:
            cleanup_error = f"{type(exc).__name__}: {exc}"

        try:
            if cleanup_process is None or cleanup_process.poll() is not None:
                transport.close()
            else:
                cleanup_error = cleanup_error or "registered Agent process remains alive"
        except Exception as exc:
            cleanup_error = cleanup_error or f"transport close failed: {type(exc).__name__}: {exc}"

        if cleanup_error:
            exit_code = 1
        if close_requested or terminal_error is not None:
            response = terminal_error or {
                "ok": True,
                "error_type": "",
                "error": "",
            }
            if cleanup_error:
                response = {
                    "ok": False,
                    "error_type": "ContainerCleanupError",
                    "error": cleanup_error,
                    "operation_error": terminal_error,
                }
            if cleanup_artifact is not None:
                response["cleanup_receipt"] = {
                    "receipt_id": cleanup_artifact.receipt_id,
                    "path": str(cleanup_artifact.path),
                    "sha256": cleanup_artifact.file_sha256,
                    "cleaned": cleanup_artifact.cleaned,
                }
            _write(response)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

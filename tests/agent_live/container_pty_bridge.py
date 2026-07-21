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
    transport = SubprocessPtyTransport(_command(args.command), cwd=Path(args.cwd))

    def stop(_signum: int, _frame: object) -> None:
        raise SystemExit(128 + _signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGHUP, stop)
    transport.start(env=os.environ)
    try:
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
                    _write({"ok": True})
                    return 0
                else:
                    raise ValueError(f"unsupported container PTY bridge operation: {operation!r}")
            except Exception as exc:
                _write({
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })
    finally:
        transport.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

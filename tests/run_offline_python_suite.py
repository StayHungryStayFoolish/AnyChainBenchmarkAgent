#!/usr/bin/env python3
"""Run the ordinary Python regression suite without external network access."""

from __future__ import annotations

import ipaddress
import socket
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _is_loopback_host(host: object) -> bool:
    value = str(host or "").strip().strip("[]").lower()
    if value in {"localhost", "localhost.localdomain"}:
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def _guarded_connect(original):
    def connect(sock, address):
        host = address[0] if isinstance(address, tuple) and address else address
        if not _is_loopback_host(host):
            raise AssertionError(
                f"ordinary regression attempted external network access: {host}"
            )
        return original(sock, address)

    return connect


def main() -> int:
    suite = unittest.defaultTestLoader.discover(str(TESTS), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=1)
    with (
        patch.object(
            socket.socket,
            "connect",
            _guarded_connect(socket.socket.connect),
        ),
        patch.object(
            socket.socket,
            "connect_ex",
            _guarded_connect(socket.socket.connect_ex),
        ),
    ):
        result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())

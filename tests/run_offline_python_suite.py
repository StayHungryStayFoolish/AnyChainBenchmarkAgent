#!/usr/bin/env python3
"""Run the ordinary Python regression suite without external network access."""

from __future__ import annotations

import ipaddress
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
PROCESS_TREE_GUARD_SOURCE = (
    TESTS / "agent_live" / "offline_network_guard.c"
)
PROVIDER_CREDENTIAL_NAMES = frozenset({
    "ANTHROPIC_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
})
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


def _remove_provider_credentials(environment: dict[str, str]) -> tuple[str, ...]:
    removed = []
    for name in tuple(environment):
        upper = name.upper()
        if (
            name in PROVIDER_CREDENTIAL_NAMES
            or upper.endswith("_API_KEY")
            or upper.endswith("_ACCESS_TOKEN")
            or upper.endswith("_AUTH_TOKEN")
        ):
            environment.pop(name, None)
            removed.append(name)
    return tuple(sorted(removed))


def _compile_process_tree_guard(output_dir: Path) -> Path:
    if platform.system() != "Linux":
        raise RuntimeError("offline process-tree guard requires Linux")
    library = output_dir / "libanychain_offline_network_guard.so"
    completed = subprocess.run(
        (
            "cc",
            "-shared",
            "-fPIC",
            "-O2",
            "-Wall",
            "-Wextra",
            "-o",
            str(library),
            str(PROCESS_TREE_GUARD_SOURCE),
            "-ldl",
        ),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not library.is_file():
        raise RuntimeError(
            "cannot compile offline process-tree network guard: "
            + completed.stderr.strip()
        )
    return library


def _install_process_tree_guard(library: Path) -> tuple[str, ...]:
    removed = _remove_provider_credentials(os.environ)
    existing = str(os.environ.get("LD_PRELOAD") or "").strip()
    os.environ["LD_PRELOAD"] = (
        f"{library}:{existing}" if existing else str(library)
    )
    os.environ["ANYCHAIN_OFFLINE_NETWORK_GUARD"] = "ld_preload_v1"
    return removed


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="anychain-offline-") as tmpdir:
        library = _compile_process_tree_guard(Path(tmpdir))
        removed = _install_process_tree_guard(library)
        print(json.dumps({
            "offline_isolation": {
                "schema_version": 1,
                "platform": "linux",
                "parent_python_socket_guard": True,
                "process_tree_guard": "ld_preload_connect_sendto_sendmsg",
                "loopback_allowed": True,
                "provider_credentials_removed": list(removed),
            }
        }, sort_keys=True))
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

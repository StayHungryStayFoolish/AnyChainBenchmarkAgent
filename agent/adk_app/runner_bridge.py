"""ADK Runner availability diagnostics.

The product terminal workflow is owned by ``agent.harness``. This module is
kept only for startup/status compatibility with the ADK app package; it must
not run ADK turns, mutate benchmark workflow state, or sanitize model output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RunnerBridgeStatus:
    available: bool
    reason: str
    runner_import: str = "google.adk.runners.Runner"

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "runner_import": self.runner_import,
        }


def runner_bridge_status() -> RunnerBridgeStatus:
    try:
        from google.adk.runners import Runner  # type: ignore  # noqa: F401
    except Exception as exc:
        return RunnerBridgeStatus(False, f"ADK Runner is unavailable: {type(exc).__name__}: {exc}")
    return RunnerBridgeStatus(True, "ADK Runner is importable")


def require_runner_available() -> None:
    status = runner_bridge_status()
    if not status.available:
        raise RuntimeError(status.reason)


def run_text_once(*_args: Any, **_kwargs: Any) -> str:
    raise RuntimeError("ADK text execution was retired; use agent.harness.AnyChainGraphRuntime.")


class ADKRunnerBridge:
    """Retired compatibility shell.

    Keeping the name prevents stale import sites from failing at import time,
    while instantiation fails loudly so old runtime control cannot come back.
    """

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("ADKRunnerBridge was retired; use agent.harness.AnyChainGraphRuntime.")


def sanitize_adk_text(text: str) -> str:
    """Legacy no-op compatibility helper for old tests and tooling imports."""
    return str(text).strip()

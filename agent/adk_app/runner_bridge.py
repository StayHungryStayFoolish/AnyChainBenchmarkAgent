"""ADK Runner boundary for the product terminal.

The product terminal must not shell out to ``adk run``. This bridge owns the
optional in-process ADK Runner integration. When ADK is not installed, it
reports an unavailable status instead of falling back to the old custom Agent
brain.
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

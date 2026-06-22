"""ADK availability and model integration helpers."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class ADKStatus:
    available: bool
    import_name: str = "google.adk"
    reason: str = ""
    python_version: str = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    python_supported: bool = sys.version_info >= (3, 10)

    def as_dict(self) -> dict[str, str | bool]:
        return {
            "available": self.available,
            "import_name": self.import_name,
            "reason": self.reason,
            "python_version": self.python_version,
            "python_supported": self.python_supported,
        }


def is_adk_available() -> bool:
    """Return whether the optional Google ADK package is importable."""
    try:
        return importlib.util.find_spec("google.adk") is not None
    except ModuleNotFoundError:
        return False


def adk_status() -> ADKStatus:
    """Return a safe ADK availability payload for diagnostics."""
    if is_adk_available():
        reason = "google.adk is importable"
        if sys.version_info < (3, 10):
            reason = "google.adk is importable, but Python 3.10+ is recommended for ADK runtime features"
        return ADKStatus(available=True, reason=reason)
    return ADKStatus(
        available=False,
        reason="google-adk is not installed; install it in an isolated Python 3.10+ environment",
    )

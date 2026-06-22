"""ADK orchestration layer for AnyChain Benchmark Agent.

This package is intentionally thin. It wraps the existing deterministic
benchmark engine as ADK tools instead of reimplementing benchmark behavior.
"""

from .models import adk_status, is_adk_available

__all__ = ["adk_status", "is_adk_available"]


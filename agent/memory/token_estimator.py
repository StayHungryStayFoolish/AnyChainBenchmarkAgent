"""Small local token estimator used for chat compaction thresholds."""

from __future__ import annotations


DEFAULT_CONTEXT_WINDOW_TOKENS = 1_000_000
DEFAULT_COMPACT_TRIGGER_RATIO = 0.7
DEFAULT_COMPACT_TURN_THRESHOLD = 40
DEFAULT_COMPACT_KEEP_RECENT_TURNS = 8


def estimate_tokens(text: str) -> int:
    """Return a rough token estimate without requiring tokenizer dependencies."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def compact_token_threshold(context_window_tokens: int, trigger_ratio: float) -> int:
    """Compute the token threshold that triggers compaction."""
    return max(1, int(context_window_tokens * trigger_ratio))

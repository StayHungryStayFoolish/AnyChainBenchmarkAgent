"""Persistent checkpoint helpers for the LangGraph Harness."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any


DEFAULT_CHECKPOINT_PATH = Path(".agent/langgraph/checkpoints.sqlite")


def default_checkpoint_path() -> Path:
    return Path(os.environ.get("ANYCHAIN_AGENT_CHECKPOINT_PATH") or DEFAULT_CHECKPOINT_PATH)


def ensure_langgraph_environment() -> None:
    os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
    warnings.filterwarnings(
        "ignore",
        message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.*",
        category=UserWarning,
    )


def create_sqlite_checkpointer(path: str | Path = DEFAULT_CHECKPOINT_PATH) -> Any:
    ensure_langgraph_environment()
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    from langgraph.checkpoint.sqlite import SqliteSaver

    manager = SqliteSaver.from_conn_string(str(checkpoint_path))
    saver = manager.__enter__()
    setattr(saver, "_anychain_context_manager", manager)
    return saver

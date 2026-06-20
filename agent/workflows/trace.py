"""File-backed workflow trace for Agent decisions."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class WorkflowTrace:
    def __init__(self, output_dir: str | Path):
        self.path = Path(output_dir) / "workflow_trace.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        user_input: str,
        intent: dict[str, Any] | None,
        workflow: str,
        tools: list[str] | None = None,
        prompt_bundle: str = "",
        artifacts: dict[str, str] | None = None,
        fallback: str = "",
        next_actions: list[str] | None = None,
    ) -> None:
        event = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "user_input": user_input,
            "intent": intent or {},
            "workflow": workflow,
            "tools": tools or [],
            "prompt_bundle": prompt_bundle,
            "artifacts": artifacts or {},
            "fallback": fallback,
            "next_actions": next_actions or [],
        }
        with self.path.open("a", encoding="utf-8") as handle:
            json.dump(event, handle, sort_keys=True)
            handle.write("\n")


def read_trace(path: str | Path, limit: int = 20) -> list[dict[str, Any]]:
    trace_file = Path(path)
    if not trace_file.is_file():
        return []
    rows = []
    for line in trace_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows[-limit:]

"""Persist compacted chat memory snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_memory(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_memory(path: str | Path) -> dict[str, Any] | None:
    target = Path(path)
    if not target.is_file():
        return None
    return json.loads(target.read_text(encoding="utf-8"))

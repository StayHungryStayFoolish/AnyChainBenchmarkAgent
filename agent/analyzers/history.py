"""Experiment history reader for archived benchmark runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def list_history(limit: int = 10, archives_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(archives_dir) if archives_dir else REPO_ROOT / "archives"
    summaries = []
    if root.is_dir():
        for path in sorted(root.glob("*/test_summary.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            summaries.append({
                "run_id": payload.get("run_id", path.parent.name),
                "benchmark_mode": payload.get("benchmark_mode"),
                "max_successful_qps": payload.get("max_successful_qps"),
                "bottleneck_detected": payload.get("bottleneck_detected"),
                "bottleneck_types": payload.get("bottleneck_types", []),
                "duration_minutes": payload.get("duration_minutes"),
                "archived_at": payload.get("archived_at"),
                "summary_file": str(path),
            })
            if len(summaries) >= limit:
                break
    return {"archives_dir": str(root), "runs": summaries}


def compare_latest(archives_dir: str | Path | None = None) -> dict[str, Any]:
    history = list_history(limit=2, archives_dir=archives_dir)
    runs = history["runs"]
    if len(runs) < 2:
        return {"status": "insufficient_history", "runs": runs}
    current, previous = runs[0], runs[1]
    return {
        "status": "compared",
        "current": current,
        "previous": previous,
        "delta": {
            "max_successful_qps": _delta(current.get("max_successful_qps"), previous.get("max_successful_qps")),
            "bottleneck_changed": current.get("bottleneck_types") != previous.get("bottleneck_types"),
        },
    }


def _delta(current: Any, previous: Any) -> Any:
    try:
        return float(current) - float(previous)
    except (TypeError, ValueError):
        return None

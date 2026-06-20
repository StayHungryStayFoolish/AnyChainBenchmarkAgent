#!/usr/bin/env python3
"""Detached benchmark worker used by Agent jobs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from runners.artifacts import write_artifact_index  # noqa: E402
from runners.materialize import load_runtime_env_file  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    try:
        os.setsid()
    except OSError:
        pass
    parser = argparse.ArgumentParser(description="Run a detached AnyChain benchmark job")
    parser.add_argument("--job-file", required=True)
    args = parser.parse_args(argv)

    job_file = Path(args.job_file)
    job = _read_json(job_file)
    plan_file = Path(job["plan_file"])
    plan = _read_json(plan_file)
    run_dir = Path(job["run_dir"])
    runtime_env_file = job.get("runtime_env_file", "")
    log_file = run_dir / "benchmark.log"

    try:
        env = load_runtime_env_file(runtime_env_file)
        command = plan["execution"]["command"]
        with log_file.open("w", encoding="utf-8") as handle:
            handle.write(f"[anychain-agent] detached worker pid={os.getpid()} started at {_now()}\n")
            handle.flush()
            completed = subprocess.run(
                command,
                cwd=plan["execution"].get("working_dir", str(REPO_ROOT)),
                env={**os.environ, **env},
                text=True,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        job["status"] = "completed" if completed.returncode == 0 else "failed"
        job["exit_code"] = completed.returncode
        if completed.returncode != 0:
            job["error"] = f"benchmark command exited with {completed.returncode}"
    except Exception as exc:  # pragma: no cover - defensive detached worker guard
        job["status"] = "failed"
        job["error"] = str(exc)

    job["updated_at"] = _now()
    job["artifact_index"] = write_artifact_index(run_dir, job, plan)
    _write_json(job_file, job)
    _notify_status(job)
    return 0 if job["status"] == "completed" else 1


def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _notify_status(job: dict) -> None:
    webhook = os.environ.get("AGENT_NOTIFY_WEBHOOK_URL", "").strip()
    if not webhook:
        return
    events = {
        item.strip()
        for item in os.environ.get("AGENT_NOTIFY_ON", "completed,failed").split(",")
        if item.strip()
    }
    if job.get("status") not in events:
        return
    try:
        from urllib import request as urlrequest

        payload = json.dumps({
            "job_id": job.get("job_id"),
            "status": job.get("status"),
            "updated_at": job.get("updated_at"),
            "run_dir": job.get("run_dir"),
            "artifact_index": job.get("artifact_index", ""),
            "error": job.get("error", ""),
        }).encode("utf-8")
        req = urlrequest.Request(webhook, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urlrequest.urlopen(req, timeout=5):  # nosec B310 - user-configured notification endpoint
            pass
    except Exception:
        return


if __name__ == "__main__":
    raise SystemExit(main())

"""Exact terminal job commands.

Natural-language job questions must stay outside this module and flow through
the LangGraph Harness. This module only handles stable shell-like terminal
commands.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

try:
    from ..runners.job_manager import get_job, list_jobs, tail_job_log
    from .language import t
except ImportError:  # product script adds agent/ to sys.path
    from runners.job_manager import get_job, list_jobs, tail_job_log
    from terminal.language import t


class JobCommandState(Protocol):
    language: str
    latest_job_id: str


class JobCommandIO(Protocol):
    def agent(self, language: str, message: str) -> None:
        ...


class JobCommandHandler:
    def __init__(self, state: JobCommandState, io: JobCommandIO) -> None:
        self.state = state
        self.io = io

    def handle_jobs_command(self, stripped: str, lowered: str) -> bool:
        if not _is_exact_command(stripped, lowered, {"jobs", "任务"}):
            return False
        self._jobs()
        return True

    def handle_status_command(self, stripped: str, lowered: str) -> bool:
        parts = stripped.split()
        if not parts:
            return False
        if parts[0].lower() not in {"status"} and parts[0] not in {"状态"}:
            return False
        if len(parts) <= 1:
            self._status()
            return True
        job_id = parts[1]
        try:
            job = get_job(job_id)
        except Exception:
            self.io.agent(self.state.language, t(self.state.language, "job_not_found", job_id=job_id))
            return True
        self._emit_job_status(job)
        return True

    def handle_log_command(self, stripped: str, lowered: str) -> bool:
        parts = stripped.split()
        if not parts:
            return False
        command = parts[0].lower()
        if command not in {"logs", "log", "follow"}:
            return False
        job_id = parts[1] if len(parts) > 1 else self.state.latest_job_id
        if not job_id:
            jobs = list_jobs(limit=1)
            job_id = jobs[0].get("job_id", "") if jobs else ""
        if not job_id:
            self.io.agent(self.state.language, t(self.state.language, "jobs_empty"))
            return True
        if command == "follow":
            self._follow_logs(job_id)
        else:
            self._logs(job_id)
        return True

    def _jobs(self) -> None:
        jobs = list_jobs(limit=5)
        if not jobs:
            self.io.agent(self.state.language, t(self.state.language, "jobs_empty"))
            return
        lines = [t(self.state.language, "jobs_header")]
        for job in jobs:
            lines.append(f"{job.get('job_id')}  {job.get('status')}  {job.get('updated_at')}")
        self.io.agent(self.state.language, "\n".join(lines))

    def _status(self) -> None:
        jobs = list_jobs(limit=1)
        if not jobs:
            self.io.agent(self.state.language, t(self.state.language, "jobs_empty"))
            return
        self._emit_job_status(jobs[0])

    def _emit_job_status(self, job: dict) -> None:
        self.io.agent(
            self.state.language,
            t(self.state.language, "job_found", job_id=job.get("job_id", ""), status=job.get("status", "unknown")),
        )

    def _logs(self, job_id: str) -> None:
        payload = tail_job_log(job_id, lines=80)
        self.io.agent(
            self.state.language,
            t(self.state.language, "log_path", job_id=job_id, path=payload.get("log_file", "")),
        )
        if not payload.get("exists"):
            self.io.agent(self.state.language, t(self.state.language, "log_missing"))
            return
        lines = payload.get("lines", [])
        if lines:
            self.io.agent(self.state.language, "\n".join(lines))
        else:
            self.io.agent(self.state.language, t(self.state.language, "log_empty"))

    def _follow_logs(self, job_id: str) -> None:
        try:
            job = get_job(job_id)
        except Exception:
            self.io.agent(self.state.language, t(self.state.language, "job_not_found", job_id=job_id))
            return
        log_file = Path(job["run_dir"]) / "benchmark.log"
        self.io.agent(
            self.state.language,
            t(self.state.language, "follow_start", job_id=job_id, path=str(log_file)),
        )
        position = 0
        try:
            while True:
                if log_file.is_file():
                    with log_file.open("r", encoding="utf-8", errors="replace") as handle:
                        handle.seek(position)
                        chunk = handle.read()
                        position = handle.tell()
                    if chunk:
                        self.io.agent(self.state.language, chunk.rstrip())
                try:
                    status = get_job(job_id).get("status", "unknown")
                except Exception:
                    status = "unknown"
                if status in {"completed", "failed"}:
                    self.io.agent(self.state.language, t(self.state.language, "follow_done", status=status))
                    return
                time.sleep(2)
        except KeyboardInterrupt:
            self.io.agent(self.state.language, t(self.state.language, "follow_stopped", path=str(log_file)))


def _is_exact_command(stripped: str, lowered: str, aliases: set[str]) -> bool:
    normalized_aliases = {alias.lower() for alias in aliases}
    return lowered in normalized_aliases or stripped in aliases

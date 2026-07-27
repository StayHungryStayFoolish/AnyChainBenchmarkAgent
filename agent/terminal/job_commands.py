"""Exact terminal job commands.

Natural-language job questions must stay outside this module and flow through
the LangGraph Harness. This module only handles stable shell-like terminal
commands.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Protocol

from agent.harness.failures import failure_record_from_job, render_failure_summary
from agent.runners.job_manager import get_job, list_jobs, tail_job_log
from agent.terminal.language import t


class JobCommandState(Protocol):
    language: str
    latest_job_id: str


@dataclass(frozen=True)
class JobCommandPlan:
    command_name: str
    effect_class: str
    job_id: str = ""


FollowStopReason = Literal[
    "completed",
    "cancelled",
    "deadline",
    "limit_reached",
    "not_found",
    "interrupted",
    "error",
]


@dataclass(frozen=True)
class FollowEvent:
    message: str
    terminal: bool = False
    stop_reason: FollowStopReason | None = None
    failed: bool = False


class JobCommandHandler:
    def __init__(self, state: JobCommandState) -> None:
        self.state = state

    def plan_command(
        self,
        stripped: str,
        lowered: str,
    ) -> JobCommandPlan | None:
        if _is_exact_command(stripped, lowered, {"jobs", "任务"}):
            return JobCommandPlan("jobs", "read_only")
        parts = stripped.split()
        if not parts:
            return None
        command = parts[0].lower()
        if command in {"status"} or parts[0] in {"状态"}:
            return JobCommandPlan(
                "status",
                "read_only",
                parts[1] if len(parts) > 1 else "",
            )
        if command not in {"logs", "log", "follow"}:
            return None
        job_id = parts[1] if len(parts) > 1 else self.state.latest_job_id
        if not job_id:
            jobs = list_jobs(limit=1)
            job_id = str(jobs[0].get("job_id") or "") if jobs else ""
        return JobCommandPlan(
            "follow" if command == "follow" else "logs",
            "streaming_observation" if command == "follow" else "read_only",
            job_id,
        )

    def execute(self, plan: JobCommandPlan) -> tuple[str, ...]:
        if plan.command_name == "jobs":
            return self._jobs()
        if plan.command_name == "status":
            return self._status(plan.job_id)
        if plan.command_name == "logs":
            return self._logs(plan.job_id)
        if plan.command_name == "follow":
            raise ValueError("follow requires the streaming terminal executor")
        raise ValueError(f"unsupported job command plan: {plan.command_name}")

    def _jobs(self) -> tuple[str, ...]:
        jobs = list_jobs(limit=5)
        if not jobs:
            return (t(self.state.language, "jobs_empty"),)
        lines = [t(self.state.language, "jobs_header")]
        for job in jobs:
            lines.append(f"{job.get('job_id')}  {job.get('status')}  {job.get('updated_at')}")
        return ("\n".join(lines),)

    def _status(self, job_id: str) -> tuple[str, ...]:
        if job_id:
            try:
                return (self._job_status_message(get_job(job_id)),)
            except Exception:
                return (
                    t(
                        self.state.language,
                        "job_not_found",
                        job_id=job_id,
                    ),
                )
        jobs = list_jobs(limit=1)
        if not jobs:
            return (t(self.state.language, "jobs_empty"),)
        return (self._job_status_message(jobs[0]),)

    def _job_status_message(self, job: dict) -> str:
        message = t(
            self.state.language,
            "job_found",
            job_id=job.get("job_id", ""),
            status=job.get("status", "unknown"),
        )
        if str(job.get("status") or "") in {"failed", "partial"}:
            message += "\n" + render_failure_summary(failure_record_from_job(job), self.state.language)
        return message

    def _logs(self, job_id: str) -> tuple[str, ...]:
        if not job_id:
            return (t(self.state.language, "jobs_empty"),)
        try:
            payload = tail_job_log(job_id, lines=80)
        except Exception:
            return (
                t(
                    self.state.language,
                    "job_not_found",
                    job_id=job_id,
                ),
            )
        messages = [
            t(
                self.state.language,
                "log_path",
                job_id=job_id,
                path=payload.get("log_file", ""),
            )
        ]
        if not payload.get("exists"):
            messages.append(t(self.state.language, "log_missing"))
            return tuple(messages)
        lines = payload.get("lines", [])
        if lines:
            messages.append("\n".join(lines))
        else:
            messages.append(t(self.state.language, "log_empty"))
        return tuple(messages)

    def iter_follow_events(
        self,
        job_id: str,
        *,
        poll_interval_seconds: float = 2.0,
        max_duration_seconds: float = 3600.0,
        max_log_bytes: int = 8 * 1024 * 1024,
        max_log_chunks: int = 4096,
    ) -> Iterator[FollowEvent]:
        """Yield data chunks followed by exactly one typed terminal event."""

        if not job_id:
            yield FollowEvent(
                t(self.state.language, "jobs_empty"),
                terminal=True,
                stop_reason="not_found",
                failed=True,
            )
            return
        try:
            job = get_job(job_id)
        except Exception:
            yield FollowEvent(
                t(
                    self.state.language,
                    "job_not_found",
                    job_id=job_id,
                ),
                terminal=True,
                stop_reason="not_found",
                failed=True,
            )
            return
        log_file = Path(job["run_dir"]) / "benchmark.log"
        yield FollowEvent(
            t(
                self.state.language,
                "follow_start",
                job_id=job_id,
                path=str(log_file),
            )
        )
        position = 0
        started_at = time.monotonic()
        observed_bytes = 0
        observed_chunks = 0
        while True:
            deadline_reached = (
                time.monotonic() - started_at >= max_duration_seconds
            )
            capacity_reached = (
                observed_bytes >= max_log_bytes
                or observed_chunks >= max_log_chunks
            )
            if deadline_reached or capacity_reached:
                yield FollowEvent(
                    t(
                        self.state.language,
                        "follow_limit_reached",
                        job_id=job_id,
                    ),
                    terminal=True,
                    stop_reason=(
                        "deadline" if deadline_reached else "limit_reached"
                    ),
                )
                return
            if log_file.is_file():
                with log_file.open("r", encoding="utf-8", errors="replace") as handle:
                    handle.seek(position)
                    remaining = max(0, max_log_bytes - observed_bytes)
                    chunk = handle.read(min(65536, remaining))
                    position = handle.tell()
                if chunk:
                    observed_bytes += len(chunk.encode("utf-8"))
                    observed_chunks += 1
                    yield FollowEvent(chunk.rstrip())
            try:
                status = get_job(job_id).get("status", "unknown")
            except Exception:
                status = "unknown"
            if status in {"completed", "failed", "partial"}:
                yield FollowEvent(
                    t(
                        self.state.language,
                        "follow_done",
                        status=status,
                    ),
                    terminal=True,
                    stop_reason="completed",
                )
                return
            time.sleep(max(0.01, float(poll_interval_seconds)))


def _is_exact_command(stripped: str, lowered: str, aliases: set[str]) -> bool:
    normalized_aliases = {alias.lower() for alias in aliases}
    return lowered in normalized_aliases or stripped in aliases

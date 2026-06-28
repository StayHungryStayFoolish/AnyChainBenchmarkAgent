#!/usr/bin/env python3
"""Product terminal for AnyChain Benchmark Agent.

The terminal owns stable input/output, startup diagnostics, dependency
installation consent, and job recovery notices. Conversation planning and
workflow orchestration are delegated to the in-process Google ADK Runner.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "agent"
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from adk_app.models import adk_status  # noqa: E402
from adk_app.runner_bridge import ADKRunnerBridge, runner_bridge_status  # noqa: E402
from adk_app.state import load_startup_state, preserved_state_for_adk  # noqa: E402
from adk_app.tools.web_research import web_research_status  # noqa: E402
from diagnostics.doctor import run_doctor  # noqa: E402
from knowledge.framework_capabilities import load_framework_capabilities  # noqa: E402
from knowledge.framework_context import load_framework_context  # noqa: E402
from llm.config import load_llm_config  # noqa: E402
from runners.job_manager import get_job, list_jobs, tail_job_log  # noqa: E402
from terminal.io import OutputOnlyIO, TerminalIO  # noqa: E402
from terminal.language import detect_language, t  # noqa: E402
from workflows.conversation_state import DEFAULT_SESSION_ID, load_workflow_state  # noqa: E402


@dataclass
class TerminalSession:
    """Small terminal shell state, not a benchmark workflow state machine."""

    language: str = "en"
    current_question_id: str = ""
    latest_job_id: str = ""
    discovery: dict[str, Any] = field(default_factory=dict)
    framework_summary: dict[str, Any] = field(default_factory=dict)
    pending_missing_dependencies: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TerminalSession":
        allowed = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in payload.items() if key in allowed})


class TerminalSessionStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or ".agent/terminal/session.json")

    def load(self) -> TerminalSession:
        try:
            return TerminalSession.from_dict(json.loads(self.path.read_text(encoding="utf-8")))
        except Exception:
            return TerminalSession()

    def save(self, state: TerminalSession) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(asdict(state), indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    store = TerminalSessionStore(args.state_file)
    state = store.load()
    if args.language:
        state.language = args.language
    elif args.prompt:
        state.language = detect_language(args.prompt[0], state.language)
    io = OutputOnlyIO() if args.prompt else TerminalIO()
    app = AnyChainTerminal(state=state, store=store, io=io, session_id=args.session_id)
    if args.prompt:
        app.startup()
        for prompt in args.prompt:
            app.handle_user_text(prompt)
            store.save(state)
        return 0
    return app.run()


class AnyChainTerminal:
    def __init__(
        self,
        state: TerminalSession | None = None,
        store: TerminalSessionStore | None = None,
        io: TerminalIO | OutputOnlyIO | None = None,
        bridge_factory: Callable[[], ADKRunnerBridge] | None = None,
        session_id: str | None = None,
    ) -> None:
        self.state = state or TerminalSession()
        self.store = store or TerminalSessionStore()
        self.io = io or TerminalIO()
        self.session_id = session_id or DEFAULT_SESSION_ID
        self._bridge_factory = bridge_factory or (lambda: ADKRunnerBridge(session_id=self.session_id))
        self._bridge: ADKRunnerBridge | None = None
        self._startup_state: dict[str, Any] = {}
        self._llm_config = load_llm_config()
        self._web_research_status: dict[str, Any] = {}
        self._adk_available = False

    def run(self) -> int:
        self.startup()
        while True:
            try:
                text = self.io.input(self.state.language).strip()
            except KeyboardInterrupt:
                self.io.agent(self.state.language, t(self.state.language, "ctrl_c_exit"))
                self.store.save(self.state)
                return 130
            except EOFError:
                self.io.agent(self.state.language, t(self.state.language, "bye"))
                self.store.save(self.state)
                return 0

            if not text:
                continue
            if text.lower() in {"exit", "quit", "q"}:
                self.io.agent(self.state.language, t(self.state.language, "bye"))
                self.store.save(self.state)
                return 0
            self.handle_user_text(text)
            self.store.save(self.state)

    def startup(self) -> None:
        self._llm_config = load_llm_config()
        self._startup_state = load_startup_state()
        latest_job = self._startup_state.get("latest_job") or {}
        if latest_job:
            self.state.latest_job_id = latest_job.get("job_id", "")

        self.io.agent(self.state.language, t(self.state.language, "welcome"))
        self.io.agent(
            self.state.language,
            t(
                self.state.language,
                "mode",
                provider=self._llm_config.provider,
                model=self._llm_config.model,
                auth_mode=self._llm_config.auth_mode,
            ),
        )
        self._web_research_status = web_research_status(self._llm_config).as_dict()
        self.io.agent(
            self.state.language,
            t(self.state.language, "web_research", status=self._web_research_status.get("reason", "unknown")),
        )
        errors = self._llm_config.validate()
        if errors:
            self.io.agent(self.state.language, t(self.state.language, "llm_config_warning", errors="; ".join(errors)))

        status = adk_status().as_dict()
        bridge_status = runner_bridge_status().as_dict()
        self._adk_available = bool(status.get("available") and bridge_status.get("available"))
        self.io.agent(self.state.language, t(self.state.language, "adk", status=bridge_status.get("reason", status)))
        self._load_framework_context()
        self._startup_doctor()

        if latest_job:
            self.io.agent(
                self.state.language,
                t(
                    self.state.language,
                    "job_found",
                    job_id=latest_job.get("job_id", ""),
                    status=latest_job.get("status", "unknown"),
                ),
            )
            next_actions = self._startup_state.get("next_actions") or []
            if next_actions:
                self.io.agent(self.state.language, t(self.state.language, "job_next_actions", actions=", ".join(next_actions)))
        else:
            self.io.agent(self.state.language, t(self.state.language, "job_none"))

        if not self._adk_available:
            self.state.current_question_id = "install_agent_runtime"
            self.state.pending_missing_dependencies = ["google-adk"]
            self.io.agent(self.state.language, t(self.state.language, "adk_missing_hint"))
            self.io.agent(self.state.language, t(self.state.language, "agent_runtime_offer"))
        else:
            self._ensure_bridge()
        self.io.agent(self.state.language, t(self.state.language, "help"))

    def handle_user_text(self, text: str) -> None:
        self.state.language = detect_language(text, self.state.language)
        stripped = text.strip()
        lowered = stripped.lower()
        if _is_shell_command(stripped, lowered, {"exit", "quit", "q"}):
            self.io.agent(self.state.language, t(self.state.language, "bye"))
            return
        if _is_shell_command(stripped, lowered, {"help", "?", "帮助", "？"}):
            self.io.agent(self.state.language, t(self.state.language, "help"))
            return
        if _is_shell_command(stripped, lowered, {"doctor", "环境检查"}):
            self._doctor()
            return
        if _is_shell_command(stripped, lowered, {"jobs", "任务"}):
            self._jobs()
            return
        if _is_shell_command(stripped, lowered, {"status", "状态"}):
            self._status()
            return
        if self._handle_log_command(stripped, lowered):
            return
        if self._handle_pending_confirmation(lowered):
            return
        if not self._adk_available:
            self.io.agent(self.state.language, t(self.state.language, "adk_missing_hint"))
            self.state.current_question_id = "install_agent_runtime"
            self.io.agent(self.state.language, t(self.state.language, "agent_runtime_offer"))
            return
        try:
            response = self._ensure_bridge().run_text(text, state_delta=self._state_delta())
        except Exception:
            self.io.agent(self.state.language, t(self.state.language, "adk_runtime_error"))
            return
        if response:
            self.io.agent(self.state.language, response)
        else:
            self.io.agent(self.state.language, t(self.state.language, "unknown"))

    def _handle_pending_confirmation(self, lowered: str) -> bool:
        if self.state.current_question_id == "install_agent_runtime":
            if lowered in {"y", "yes", "确认", "是", ""}:
                self._install_agent_runtime()
                return True
            if lowered in {"n", "no", "否"}:
                self.state.current_question_id = ""
                self.io.agent(self.state.language, t(self.state.language, "agent_runtime_declined"))
                return True
        if self.state.current_question_id == "install_dependencies":
            if lowered in {"y", "yes", "确认", "是", ""}:
                self._install_dependencies()
                return True
            if lowered in {"n", "no", "否"}:
                self.state.current_question_id = ""
                self.io.agent(self.state.language, t(self.state.language, "dependency_declined"))
                return True
        return False

    def _ensure_bridge(self) -> ADKRunnerBridge:
        if self._bridge is None:
            self._bridge = self._bridge_factory()
        return self._bridge

    def _state_delta(self) -> dict[str, Any]:
        return {
            "terminal_language": self.state.language,
            "startup": self._startup_state,
            "job_state": preserved_state_for_adk(self._startup_state),
            "workflow_state": load_workflow_state(session_id=self.session_id),
            "llm_config": self._llm_config.safe_dict(),
            "web_research": self._web_research_status,
            "discovery": self.state.discovery,
            "framework_summary": self.state.framework_summary,
        }

    def _startup_doctor(self) -> None:
        self.io.agent(self.state.language, t(self.state.language, "startup_doctor_start"))
        self._emit_doctor_summary(run_doctor(), startup=True)

    def _doctor(self) -> None:
        self.io.agent(self.state.language, t(self.state.language, "doctor_start"))
        self._emit_doctor_summary(run_doctor(), startup=False)

    def _emit_doctor_summary(self, report: dict[str, Any], startup: bool) -> None:
        caps = report.get("capabilities", {})
        env = report.get("environment", {})
        self.state.discovery = env
        cloud = env.get("cloud", {})
        deployment = env.get("deployment", {})
        missing = env.get("dependencies", {}).get("missing_required", [])
        if not missing and self.state.current_question_id == "install_dependencies":
            self.state.current_question_id = ""
        self.state.pending_missing_dependencies = list(missing)
        if startup:
            self.io.agent(
                self.state.language,
                t(
                    self.state.language,
                    "startup_doctor_summary",
                    status=report.get("status", "unknown"),
                    cloud=cloud.get("provider", "unknown"),
                    deployment=deployment.get("type", "unknown"),
                    missing=", ".join(missing) if missing else "<none>",
                    chains=caps.get("chain_count", "?"),
                    methods=caps.get("unique_rpc_method_count", "?"),
                ),
            )
            summary = _format_environment_inference(env)
            if summary:
                self.io.agent(self.state.language, t(self.state.language, "environment_inference_summary", summary=summary))
        else:
            self.io.agent(
                self.state.language,
                t(
                    self.state.language,
                    "doctor_summary",
                    status=report.get("status", "unknown"),
                    missing=", ".join(missing) if missing else "<none>",
                    chains=caps.get("chain_count", "?"),
                    methods=caps.get("unique_rpc_method_count", "?"),
                ),
            )
        if missing and self.state.current_question_id != "install_agent_runtime":
            self.state.current_question_id = "install_dependencies"
            self.io.agent(self.state.language, t(self.state.language, "dependency_offer", missing=", ".join(missing)))

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
        latest = jobs[0]
        self.io.agent(
            self.state.language,
            t(self.state.language, "job_found", job_id=latest.get("job_id", ""), status=latest.get("status", "unknown")),
        )

    def _handle_log_command(self, stripped: str, lowered: str) -> bool:
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

    def _install_agent_runtime(self) -> None:
        self.io.agent(self.state.language, t(self.state.language, "agent_runtime_install_start"))
        config = load_llm_config()
        command = ["bash", "scripts/install_agent_deps.sh", "--yes"]
        if (
            config.provider in {"gemini", "claude"}
            and config.auth_mode in {"google_adc", "service_account_impersonation"}
            and not shutil.which("gcloud")
        ):
            command.append("--with-gcloud")
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.state.current_question_id = ""
        self.io.agent(self.state.language, t(self.state.language, "agent_runtime_install_done", exit_code=completed.returncode))
        self._adk_available = bool(adk_status().available and runner_bridge_status().available)
        if self._adk_available:
            self._bridge = None
            self._ensure_bridge()
        self._startup_doctor()

    def _install_dependencies(self) -> None:
        self.io.agent(self.state.language, t(self.state.language, "dependency_install_start"))
        completed = subprocess.run(
            ["bash", "scripts/install_deps.sh", "--yes"],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.state.current_question_id = ""
        self.io.agent(self.state.language, t(self.state.language, "dependency_install_done", exit_code=completed.returncode))
        self._startup_doctor()

    def _load_framework_context(self) -> None:
        context = load_framework_context(language=self.state.language)
        summary = context.get("capability_summary", {})
        capabilities = load_framework_capabilities()
        self.state.framework_summary = {
            **summary,
            "chain_count": capabilities.get("chain_count", summary.get("chain_count")),
            "family_count": capabilities.get("family_count", summary.get("family_count")),
            "unique_rpc_method_count": capabilities.get("unique_rpc_method_count", summary.get("unique_rpc_method_count")),
        }
        self.io.agent(
            self.state.language,
            t(
                self.state.language,
                "framework_context_loaded",
                chains=self.state.framework_summary.get("chain_count", "?"),
                families=self.state.framework_summary.get("family_count", "?"),
                methods=self.state.framework_summary.get("unique_rpc_method_count", "?"),
                fixtures=summary.get("fake_node_fixture_file_count", "?"),
            ),
        )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AnyChain Benchmark Agent product terminal.")
    parser.add_argument("--prompt", action="append", help="Run one or more scripted user turns, then exit.")
    parser.add_argument("--language", choices=["zh", "en"], default=None, help="Initial terminal language.")
    parser.add_argument("--state-file", help="Use an alternate terminal state file for tests or isolated sessions.")
    parser.add_argument(
        "--session-id",
        default=os.environ.get("ANYCHAIN_AGENT_SESSION_ID", DEFAULT_SESSION_ID),
        help="ADK/workflow session id. Tests and live matrices should isolate it per scenario.",
    )
    return parser.parse_args(argv)


def _format_environment_inference(env: dict[str, Any]) -> str:
    cloud = env.get("cloud", {}) or {}
    deployment = env.get("deployment", {}) or {}
    host = env.get("host", {}) or {}
    network = env.get("network", {}) or {}
    disks = env.get("disks", {}) or {}
    lines = [
        f"- CLOUD_PROVIDER: {cloud.get('provider') or '<needs confirmation>'}",
        f"- deployment: {cloud.get('platform') or deployment.get('type') or '<needs confirmation>'}",
        f"- CLOUD_REGION: {cloud.get('region') or '<needs confirmation>'}",
        f"- CLOUD_ZONE: {cloud.get('zone') or '<needs confirmation>'}",
        f"- MACHINE_TYPE: {cloud.get('machine_type') or '<needs confirmation>'}",
        f"- CPU: {host.get('cpu_count') or '<unknown>'}",
        f"- Memory: {_format_memory(host.get('memory_gib'))}",
        f"- NETWORK_INTERFACE: {network.get('default_interface') or '<needs confirmation>'}",
        f"- LEDGER_DEVICE candidate: {disks.get('proposed_ledger_device') or '<needs confirmation>'}",
        f"- ACCOUNTS_DEVICE candidate: {disks.get('proposed_accounts_device') or '<none detected>'}",
    ]
    candidates = disks.get("candidates") or []
    if candidates:
        lines.append("- Disk candidates:")
        for index, item in enumerate(candidates[:8], start=1):
            lines.append(
                f"  [{index}] {item.get('name') or '<unknown>'} "
                f"type={item.get('type') or '<unknown>'} "
                f"size={item.get('size') or '<unknown>'} "
                f"mount={item.get('mountpoint') or '<none>'} "
                f"label={item.get('label') or '<none>'}"
            )
        if len(candidates) > 8:
            lines.append(f"  ... {len(candidates) - 8} more")
    else:
        lines.append("- Disk candidates: <none detected; ADK should ask manually if needed>")
    return "\n".join(lines)


def _is_shell_command(stripped: str, lowered: str, aliases: set[str]) -> bool:
    """Return true only for exact stable terminal commands.

    Natural-language requests that merely contain words such as "status" or
    "环境检查" must go to ADK for intent handling.
    """
    normalized_aliases = {alias.lower() for alias in aliases}
    return lowered in normalized_aliases or stripped in aliases


def _format_memory(value: Any) -> str:
    if value in {None, ""}:
        return "<unknown>"
    return f"{value} GiB"


if __name__ == "__main__":
    raise SystemExit(main())

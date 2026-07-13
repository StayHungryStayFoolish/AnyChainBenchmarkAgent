#!/usr/bin/env python3
"""Product terminal for AnyChain Benchmark Agent.

The terminal owns stable input/output, startup diagnostics, dependency
installation consent, exact shell commands, and LangGraph event rendering.
Benchmark workflow orchestration belongs to the LangGraph Harness.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "agent"
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from diagnostics.adk_status import adk_status  # noqa: E402
from diagnostics.doctor import run_doctor  # noqa: E402
from harness.graph import AnyChainGraphRuntime  # noqa: E402
from knowledge.framework_capabilities import load_framework_capabilities  # noqa: E402
from knowledge.framework_context import load_framework_context  # noqa: E402
from llm.config import load_llm_config  # noqa: E402
from llm.search_grounding import web_research_status  # noqa: E402
from runners.job_manager import list_jobs  # noqa: E402
from terminal.io import OutputOnlyIO, TerminalIO  # noqa: E402
from terminal.job_commands import JobCommandHandler  # noqa: E402
from terminal.language import detect_language, t  # noqa: E402
from terminal.startup_state import load_startup_state  # noqa: E402
from utils.redaction import redact  # noqa: E402


DEFAULT_AGENT_SESSION_ID = "default"


@dataclass
class TerminalSession:
    """Terminal shell state, not benchmark workflow state."""

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
    app = AnyChainTerminal(
        state=state,
        store=store,
        io=io,
        session_id=args.session_id,
        checkpoint_path=args.checkpoint_path,
        fresh_session=args.fresh_session,
        session_purpose=args.session_purpose,
    )
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
        session_id: str | None = None,
        checkpoint_path: str | Path | None = None,
        fresh_session: bool = False,
        session_purpose: str = "user",
    ) -> None:
        self.state = state or TerminalSession()
        self.store = store or TerminalSessionStore()
        self.io = io or TerminalIO()
        self.session_id = session_id or DEFAULT_AGENT_SESSION_ID
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.fresh_session = bool(fresh_session)
        self.session_purpose = session_purpose or "user"
        self._harness: AnyChainGraphRuntime | None = None
        self._startup_state: dict[str, Any] = {}
        self._llm_config = load_llm_config()
        self._web_research_status: dict[str, Any] = {}
        self._adk_available = False
        self._job_commands = JobCommandHandler(self.state, self.io)

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
        self.state.latest_job_id = latest_job.get("job_id", "") if latest_job else ""

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
        bridge_status = _adk_runner_status()
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
            # `_startup_doctor` may already have set an "install_dependencies"
            # offer (missing vegeta etc.). Do not clobber it here, or the "[Y/n]"
            # prompt becomes dead — the user's "y" would never reach
            # `_install_dependencies` and would fall through to the harness.
            deps_offer_pending = self.state.current_question_id == "install_dependencies"
            if not deps_offer_pending:
                self.state.current_question_id = ""
                self.state.pending_missing_dependencies = []
            self._ensure_harness()
            if self.fresh_session:
                self._ensure_harness().reset(language=self.state.language)
            elif not deps_offer_pending:
                self._offer_harness_resume_if_needed()
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
        if self._job_commands.handle_jobs_command(stripped, lowered):
            return
        if self._job_commands.handle_status_command(stripped, lowered):
            return
        if self._job_commands.handle_log_command(stripped, lowered):
            return
        if self._handle_pending_confirmation(lowered):
            return
        if not self._adk_available:
            self.io.agent(self.state.language, t(self.state.language, "adk_missing_hint"))
            self.state.current_question_id = "install_agent_runtime"
            self.io.agent(self.state.language, t(self.state.language, "agent_runtime_offer"))
            return

        try:
            self.io.agent(self.state.language, t(self.state.language, "thinking"))
            graph_state = self._ensure_harness().invoke(
                text,
                language=self.state.language,
                context={
                    "discovery": self.state.discovery,
                    "framework_summary": self.state.framework_summary,
                    "latest_job_id": self.state.latest_job_id,
                    "web_research": self._web_research_status,
                },
            )
        except (KeyboardInterrupt, asyncio.CancelledError):
            self.io.agent(self.state.language, t(self.state.language, "turn_cancelled"))
            return
        except Exception as exc:
            _debug_exception("harness_turn", exc)
            self.io.agent(self.state.language, _adk_error_message(self.state.language, exc))
            return

        messages = graph_state.get("visible_response") or []
        if not messages:
            self.io.agent(self.state.language, t(self.state.language, "unknown"))
            return
        for message in messages:
            self.io.agent(self.state.language, str(message))

        # A turn may have submitted a new job (fake-node smoke, benchmark). Refresh
        # the latest-job hint from disk so "analyze the latest job" and the injected
        # context point at the just-submitted job, not the startup-detected one.
        try:
            recent = list_jobs(limit=1)
            if recent:
                self.state.latest_job_id = str(recent[0].get("job_id") or "") or self.state.latest_job_id
        except Exception:
            pass

    def _handle_pending_confirmation(self, lowered: str) -> bool:
        if self.state.current_question_id == "resume_harness_session":
            if lowered in {"1", "continue"}:
                self.state.current_question_id = ""
                snapshot = self._ensure_harness().snapshot()
                pending = snapshot.get("pending_question") or {}
                prompt = str(pending.get("prompt") or "").strip()
                next_step = (
                    t(self.state.language, "resume_pending_next", prompt=prompt)
                    if prompt
                    else t(self.state.language, "resume_no_pending_next")
                )
                self.io.agent(self.state.language, t(self.state.language, "resume_session_continue", next_step=next_step))
                return True
            if lowered in {"2", "modify", "change"}:
                self.state.current_question_id = ""
                self._ensure_harness().update({"pending_question": {}, "evidence_collection": {}, "active_group": "", "visible_response": []})
                self.io.agent(self.state.language, t(self.state.language, "resume_session_modify"))
                return True
            if lowered in {"3", "clear", "reset", "fresh"}:
                self.state.current_question_id = ""
                self._ensure_harness().reset(language=self.state.language)
                self.io.agent(self.state.language, t(self.state.language, "resume_session_clear"))
                return True
            if not lowered.isdigit():
                if _is_resume_session_explanation_request(lowered):
                    self.io.agent(self.state.language, t(self.state.language, "resume_session_explain"))
                    return True
                self.state.current_question_id = ""
                self._ensure_harness().update({"pending_question": {}, "evidence_collection": {}, "active_group": "", "visible_response": []})
                return False
            self.io.agent(self.state.language, t(self.state.language, "resume_session_invalid"))
            return True
        if self.state.current_question_id == "install_agent_runtime":
            if lowered in {"y", "yes", ""}:
                self._install_agent_runtime()
                return True
            if lowered in {"n", "no"}:
                self.state.current_question_id = ""
                self.io.agent(self.state.language, t(self.state.language, "agent_runtime_declined"))
                return True
        if self.state.current_question_id == "install_dependencies":
            if lowered in {"y", "yes", ""}:
                self._install_dependencies()
                return True
            if lowered in {"n", "no"}:
                self.state.current_question_id = ""
                self.io.agent(self.state.language, t(self.state.language, "dependency_declined"))
                return True
        return False

    def _ensure_harness(self) -> AnyChainGraphRuntime:
        if self._harness is None:
            self._harness = AnyChainGraphRuntime(
                thread_id=self.session_id,
                checkpoint_path=self.checkpoint_path,
                session_purpose=self.session_purpose,
            )
        return self._harness

    def _offer_harness_resume_if_needed(self) -> None:
        if isinstance(self.io, OutputOnlyIO):
            return
        snapshot = self._ensure_harness().snapshot()
        session = snapshot.get("session") or {}
        snapshot_purpose = str(session.get("purpose") or "user")
        if self.session_purpose == "user" and snapshot_purpose != "user":
            return
        if snapshot.get("evidence_collection"):
            snapshot = self._ensure_harness().update({"evidence_collection": {}})
        if not _has_resumable_harness_state(snapshot):
            return
        self.state.current_question_id = "resume_harness_session"
        self.io.agent(self.state.language, t(self.state.language, "resume_session_offer", summary=_format_harness_resume_summary(snapshot)))

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
        self._adk_available = bool(adk_status().available and _adk_runner_status().get("available"))
        if self._adk_available:
            self._harness = None
            self._ensure_harness()
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
            "chains": capabilities.get("chains", []),
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
        default=os.environ.get("ANYCHAIN_AGENT_SESSION_ID", DEFAULT_AGENT_SESSION_ID),
        help="Agent session id. Tests and live matrices should isolate it per scenario.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=os.environ.get("ANYCHAIN_AGENT_CHECKPOINT_PATH"),
        help="Use an explicit LangGraph checkpoint database. Tests must isolate this per scenario.",
    )
    parser.add_argument(
        "--fresh-session",
        action="store_true",
        help="Reset the selected Harness thread before handling user turns.",
    )
    parser.add_argument(
        "--session-purpose",
        choices=["user", "chaos", "live-matrix", "dev"],
        default=os.environ.get("ANYCHAIN_AGENT_SESSION_PURPOSE", "user"),
        help="Label the runtime session so test/dev checkpoints cannot be resumed as user state.",
    )
    return parser.parse_args(argv)


def _format_environment_inference(env: dict[str, Any]) -> str:
    cloud = env.get("cloud", {}) or {}
    deployment = env.get("deployment", {}) or {}
    host = env.get("host", {}) or {}
    network = env.get("network", {}) or {}
    disks = env.get("disks", {}) or {}
    deployment_value = _known_value(cloud.get("platform")) or _known_value(deployment.get("type")) or "<needs confirmation>"
    lines = [
        f"- CLOUD_PROVIDER: {cloud.get('provider') or '<needs confirmation>'}",
        f"- deployment: {deployment_value}",
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
        lines.append("- Disk candidates: <none detected; Agent should ask manually if needed>")
    interfaces = _usable_interfaces(list(network.get("interfaces") or []), str(network.get("default_interface") or ""))
    if interfaces:
        lines.append("- Network interface candidates:")
        for index, name in enumerate(interfaces[:8], start=1):
            suffix = " (default)" if name == network.get("default_interface") else ""
            lines.append(f"  [{index}] {name}{suffix}")
        if len(interfaces) > 8:
            lines.append(f"  ... {len(interfaces) - 8} more")
    return "\n".join(lines)


def _has_resumable_harness_state(snapshot: dict[str, Any]) -> bool:
    if not snapshot:
        return False
    identity = snapshot.get("chain_identity") or {}
    confirmed = snapshot.get("confirmed_config") or {}
    pending = snapshot.get("pending_question") or {}
    pending_id = str(pending.get("id") or "").strip()
    meaningful_pending = bool(pending) and pending_id not in {"opening_next_action"}
    return any(
        [
            bool(snapshot.get("target_mode")),
            bool(identity.get("canonical")),
            bool(confirmed),
            bool(snapshot.get("rpc_mode")),
            bool(snapshot.get("qps_profile")),
            bool(snapshot.get("observability")),
            bool(snapshot.get("sync_observe")),
            meaningful_pending,
        ]
    )


def _format_harness_resume_summary(snapshot: dict[str, Any]) -> str:
    identity = snapshot.get("chain_identity") or {}
    confirmed = snapshot.get("confirmed_config") or {}
    pending = snapshot.get("pending_question") or {}
    qps = snapshot.get("qps_profile") or {}
    observability = snapshot.get("observability") or {}
    session = snapshot.get("session") or {}
    lines = [
        f"- session purpose: {session.get('purpose') or '<unknown>'}",
        f"- target_mode: {snapshot.get('target_mode') or '<not selected>'}",
        f"- workflow_mode: {snapshot.get('workflow_mode') or '<not selected>'}",
        f"- chain: {identity.get('canonical') or '<not selected>'}",
        f"- rpc_mode: {snapshot.get('rpc_mode') or '<not selected>'}",
        f"- qps: {qps.get('mode') or '<not selected>'}",
        f"- observability: {observability.get('mode') or '<not selected>'}",
        f"- confirmed fields: {', '.join(sorted(confirmed.keys())) if confirmed else '<none>'}",
        f"- pending question: {pending.get('id') or '<none>'}",
    ]
    return "\n".join(lines)


def _is_resume_session_explanation_request(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return False
    return any(
        marker in lowered
        for marker in (
            "这个",
            "这是什么",
            "什么意思",
            "做什么",
            "what is this",
            "what does this mean",
            "what is it",
            "explain",
        )
    )


def _known_value(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"unknown", "none", "null"}:
        return ""
    return text


def _is_shell_command(stripped: str, lowered: str, aliases: set[str]) -> bool:
    """Return true only for exact stable terminal commands."""

    normalized_aliases = {alias.lower() for alias in aliases}
    return lowered in normalized_aliases or stripped in aliases


def _adk_runner_status() -> dict[str, Any]:
    try:
        if importlib.util.find_spec("google.adk.runners") is None:
            return {"available": False, "reason": "ADK Runner is unavailable: google.adk.runners is not importable"}
        from google.adk.runners import Runner  # type: ignore  # noqa: F401
    except Exception as exc:
        return {"available": False, "reason": f"ADK Runner is unavailable: {type(exc).__name__}: {exc}"}
    return {"available": True, "reason": "ADK Runner is importable"}


def _format_memory(value: Any) -> str:
    if value in {None, ""}:
        return "<unknown>"
    return f"{value} GiB"


def _usable_interfaces(interfaces: list[str], default: str = "") -> list[str]:
    blocked_exact = {"lo", "bonding_masters", "sit0", "tunl0", "gre0", "gretap0", "erspan0", "ip_vti0", "ip6_vti0", "ip6gre0", "ip6tnl0"}
    output = []
    for raw in interfaces:
        name = str(raw or "").strip()
        if not name:
            continue
        if name == default:
            output.append(name)
            continue
        if name in blocked_exact:
            continue
        output.append(name)
    if default and default not in output:
        output.insert(0, default)
    return output or interfaces


def _debug_exception(scope: str, exc: Exception) -> None:
    if os.environ.get("ANYCHAIN_DEBUG") not in {"1", "true", "TRUE", "yes", "YES"}:
        return
    message = redact(f"{type(exc).__name__}: {exc}")
    print(f"[anychain-debug] {scope}: {message}", file=sys.stderr)


def _adk_error_message(language: str, exc: Exception) -> str:
    message = str(exc).lower()
    if "insufficient balance" in message or "insufficient quota" in message:
        return t(language, "llm_billing_error")
    return t(language, "adk_runtime_error")


if __name__ == "__main__":
    raise SystemExit(main())

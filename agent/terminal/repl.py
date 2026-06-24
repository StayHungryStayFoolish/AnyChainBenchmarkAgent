#!/usr/bin/env python3
"""Product CLI for AnyChain Benchmark Agent.

This is intentionally not a wrapper around ``adk run``. ADK remains the Agent
runtime, but the user-facing terminal experience is owned here so benchmark
workflows can provide stable prompts, confirmations, progress, and recovery.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "agent"
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from adk_app.models import adk_status  # noqa: E402
from adk_app.state import load_startup_state  # noqa: E402
from diagnostics.doctor import run_doctor  # noqa: E402
from knowledge.framework_capabilities import load_framework_capabilities  # noqa: E402
from knowledge.framework_context import load_framework_context  # noqa: E402
from llm.config import load_llm_config  # noqa: E402
from runners.job_manager import list_jobs  # noqa: E402
from terminal.io import OutputOnlyIO, TerminalIO  # noqa: E402
from terminal.language import detect_language, t  # noqa: E402
from terminal.responder import answer_conversation, should_answer_as_conversation  # noqa: E402
from workflows.benchmark_wizard import BenchmarkWizard  # noqa: E402
from workflows.planning_bridge import prepare_plan_from_state, submit_mock_smoke_from_plan  # noqa: E402
from workflows.state import WorkflowState, WorkflowStateStore  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    store = WorkflowStateStore(args.state_file) if args.state_file else WorkflowStateStore()
    state = store.load()
    state.language = args.language or state.language
    if args.prompt:
        app = AnyChainTerminal(state=state, store=store, io=OutputOnlyIO())
        for prompt in args.prompt:
            app.handle_user_text(prompt)
            store.save(state)
        return 0
    app = AnyChainTerminal(state=state, store=store)
    return app.run()


class AnyChainTerminal:
    def __init__(self, state: WorkflowState | None = None, store: WorkflowStateStore | None = None, io: TerminalIO | None = None) -> None:
        self.state = state or WorkflowState()
        self.store = store or WorkflowStateStore()
        self.io = io or TerminalIO()
        self.capabilities: dict[str, Any] | None = None
        self.discovery: dict[str, Any] | None = None
        self.wizard = BenchmarkWizard(self.state, discovery=self.discovery)

    def run(self) -> int:
        self._startup()
        while True:
            try:
                text = self.io.input(self.state.language).strip()
            except KeyboardInterrupt:
                self.io.agent(self.state.language, t(self.state.language, "ctrl_c_exit"))
                self.store.save(self.state)
                return 130
            except EOFError:
                self.io.agent(self.state.language, t(self.state.language, "bye"))
                return 0

            if not text:
                continue
            if text.lower() in {"exit", "quit", "q"}:
                self.io.agent(self.state.language, t(self.state.language, "bye"))
                self.store.save(self.state)
                return 0
            self.handle_user_text(text)
            self.store.save(self.state)

    def handle_user_text(self, text: str) -> None:
        self.state.language = detect_language(text, self.state.language)
        lowered = text.lower()
        if lowered in {"reset", "restart", "new", "重新开始", "重置"}:
            self._reset_workflow()
            return
        if lowered in {"help", "?"} or text in {"帮助", "？"}:
            self.io.agent(self.state.language, t(self.state.language, "help"))
            return
        if lowered == "doctor" or "环境检查" in text:
            self._doctor()
            return
        if lowered == "jobs" or "任务" in text:
            self._jobs()
            return
        if lowered == "status" or "状态" in text:
            self._status()
            return
        if self.state.current_question_id == "install_agent_runtime":
            if lowered in {"y", "yes", "确认", "是"}:
                self._install_agent_runtime()
                return
            if lowered in {"n", "no", "否"}:
                self.state.stage = "agent_runtime_install_declined"
                self.state.current_question_id = ""
                self.io.agent(self.state.language, t(self.state.language, "agent_runtime_declined"))
                return
        if self.state.current_question_id == "install_dependencies":
            if lowered in {"y", "yes", "确认", "是"}:
                self._install_dependencies()
                return
            if lowered in {"n", "no", "否"}:
                self.state.stage = "dependency_install_declined"
                self.state.current_question_id = ""
                self.io.agent(self.state.language, t(self.state.language, "dependency_declined"))
                return
        if self.state.current_question_id == "smoke_confirmation":
            if lowered in {"y", "yes", "确认", "是"}:
                self._prepare_smoke()
                return
            if lowered in {"n", "no", "否"}:
                self.state.stage = "smoke_declined"
                self.state.current_question_id = ""
                self.io.agent(self.state.language, t(self.state.language, "smoke_declined"))
                return
        if self.state.current_question_id == "run_mock_smoke":
            if lowered in {"y", "yes", "确认", "是"}:
                self._run_mock_smoke()
                return
            if lowered in {"n", "no", "否"}:
                self.state.stage = "mock_smoke_declined"
                self.state.current_question_id = ""
                self.io.agent(self.state.language, t(self.state.language, "smoke_declined"))
                return
        if should_answer_as_conversation(text):
            self.io.agent(self.state.language, answer_conversation(text, self.state, load_llm_config()))
            return
        response = self.wizard.handle(text)
        if response.handled:
            for message in response.messages:
                self.io.agent(self.state.language, message)
            return
        self.io.agent(self.state.language, t(self.state.language, "unknown"))

    def _reset_workflow(self) -> None:
        language = self.state.language
        job_id = self.state.job_id
        self.state.intent = ""
        self.state.stage = "start"
        self.state.current_question_id = ""
        self.state.confirmed_values.clear()
        self.state.defaulted_values.clear()
        self.state.skipped_values.clear()
        self.state.missing_blockers.clear()
        self.state.pending_confirmations.clear()
        self.state.plan_file = ""
        self.state.runtime_env_file = ""
        self.state.job_id = job_id
        self.state.language = language
        self.io.agent(self.state.language, t(self.state.language, "state_reset"))

    def _startup(self) -> None:
        config = load_llm_config()
        status = adk_status().as_dict()
        startup = load_startup_state()
        latest_job = startup.get("latest_job", {})
        next_actions = startup.get("next_actions", [])
        if latest_job:
            self.state.job_id = latest_job.get("job_id", "")
        self.io.agent(self.state.language, t(self.state.language, "welcome"))
        self.io.agent(
            self.state.language,
            t(self.state.language, "mode", provider=config.provider, model=config.model, auth_mode=config.auth_mode),
        )
        config_errors = config.validate()
        if config_errors:
            self.io.agent(self.state.language, t(self.state.language, "llm_config_warning", errors="; ".join(config_errors)))
        self.io.agent(self.state.language, t(self.state.language, "adk", status=status.get("reason", status.get("available"))))
        if not status.get("available"):
            self.io.agent(self.state.language, t(self.state.language, "adk_missing_hint"))
            self.state.stage = "agent_runtime_install_confirmation"
            self.state.current_question_id = "install_agent_runtime"
            self.state.missing_blockers = ["google-adk"]
            self.io.agent(self.state.language, t(self.state.language, "agent_runtime_offer"))
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
            if next_actions:
                self.io.agent(self.state.language, t(self.state.language, "job_next_actions", actions=", ".join(next_actions)))
        else:
            self.io.agent(self.state.language, t(self.state.language, "job_none"))
        self.io.agent(self.state.language, t(self.state.language, "help"))

    def _startup_doctor(self) -> None:
        self.io.agent(self.state.language, t(self.state.language, "startup_doctor_start"))
        report = run_doctor()
        self._emit_doctor_summary(report, startup=True)

    def _doctor(self) -> None:
        self.io.agent(self.state.language, t(self.state.language, "doctor_start"))
        report = run_doctor()
        self._emit_doctor_summary(report, startup=False)

    def _emit_doctor_summary(self, report: dict[str, Any], startup: bool) -> None:
        caps = report.get("capabilities", {})
        env = report.get("environment", {})
        self.discovery = env
        self.wizard = BenchmarkWizard(self.state, discovery=self.discovery)
        cloud = env.get("cloud", {})
        deployment = env.get("deployment", {})
        missing = report.get("environment", {}).get("dependencies", {}).get("missing_required", [])
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
            inference_summary = _format_environment_inference(env)
            if inference_summary:
                self.io.agent(
                    self.state.language,
                    t(self.state.language, "environment_inference_summary", summary=inference_summary),
                )
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
            self.state.stage = "dependency_install_confirmation"
            self.state.current_question_id = "install_dependencies"
            self.state.missing_blockers = list(missing)
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
        self.state.stage = "agent_runtime_install_completed" if completed.returncode == 0 else "agent_runtime_install_failed"
        self.state.current_question_id = ""
        self.state.confirmed_values["agent_runtime_install_exit_code"] = completed.returncode
        self.io.agent(self.state.language, t(self.state.language, "agent_runtime_install_done", exit_code=completed.returncode))
        self._startup_doctor()

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
        latest = jobs[0] if jobs else {}
        if not latest:
            self.io.agent(self.state.language, t(self.state.language, "jobs_empty"))
            return
        self.io.agent(
            self.state.language,
            t(self.state.language, "job_found", job_id=latest.get("job_id", ""), status=latest.get("status", "unknown")),
        )

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
        self.state.stage = "dependency_install_completed" if completed.returncode == 0 else "dependency_install_failed"
        self.state.current_question_id = ""
        self.state.confirmed_values["dependency_install_exit_code"] = completed.returncode
        self.io.agent(self.state.language, t(self.state.language, "dependency_install_done", exit_code=completed.returncode))
        self._startup_doctor()

    def _prepare_smoke(self) -> None:
        self.io.agent(self.state.language, t(self.state.language, "prepare_start"))
        prepared = prepare_plan_from_state(self.state)
        self.state.plan_file = prepared["plan_file"]
        preflight = prepared["preflight"]
        if preflight.get("warnings"):
            self.io.agent(
                self.state.language,
                t(self.state.language, "prepare_warnings", warnings="; ".join(preflight.get("warnings", []))),
            )
        if not preflight.get("passed"):
            blockers = [
                f"{check.get('name')}: {check.get('detail')}"
                for check in preflight.get("checks", [])
                if not check.get("passed")
            ]
            self.state.stage = "preflight_blocked"
            self.state.current_question_id = ""
            self.state.missing_blockers = blockers
            self.io.agent(
                self.state.language,
                t(
                    self.state.language,
                    "prepare_blocked",
                    blockers="; ".join(blockers) if blockers else "<unknown>",
                    plan_file=prepared["plan_file"],
                ),
            )
            return
        self.state.stage = "mock_smoke_confirmation"
        self.state.current_question_id = "run_mock_smoke"
        self.io.agent(
            self.state.language,
            t(
                self.state.language,
                "prepare_ok",
                plan_file=prepared["plan_file"],
                runbook_file=prepared["runbook_file"],
            ),
        )

    def _run_mock_smoke(self) -> None:
        if not self.state.plan_file:
            self.state.stage = "ready_for_smoke"
            self.state.current_question_id = "smoke_confirmation"
            self.io.agent(self.state.language, t(self.state.language, "prepare_smoke_offer"))
            return
        self.io.agent(self.state.language, t(self.state.language, "mock_smoke_start"))
        job = submit_mock_smoke_from_plan(self.state.plan_file)
        self.state.stage = "mock_smoke_completed" if job.get("status") == "completed" else "mock_smoke_failed"
        self.state.current_question_id = ""
        self.state.job_id = job.get("job_id", "")
        self.state.runtime_env_file = job.get("runtime_env_file", "")
        self.io.agent(
            self.state.language,
            t(
                self.state.language,
                "mock_smoke_done",
                job_id=job.get("job_id", ""),
                status=job.get("status", "unknown"),
                runtime_env_file=job.get("runtime_env_file", ""),
                artifact_index=job.get("artifact_index", ""),
            ),
        )

    def _load_capabilities(self) -> dict[str, Any]:
        if self.capabilities is None:
            self.capabilities = load_framework_capabilities()
        return self.capabilities

    def _load_framework_context(self) -> None:
        context = load_framework_context(language=self.state.language)
        summary = context.get("capability_summary", {})
        self.capabilities = load_framework_capabilities()
        self.state.defaulted_values["framework_context_loaded"] = True
        self.state.defaulted_values["framework_context_summary"] = summary
        self.io.agent(
            self.state.language,
            t(
                self.state.language,
                "framework_context_loaded",
                chains=summary.get("chain_count", "?"),
                families=summary.get("family_count", "?"),
                methods=summary.get("unique_rpc_method_count", "?"),
                fixtures=summary.get("fake_node_fixture_file_count", "?"),
            ),
        )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AnyChain Benchmark Agent product CLI.")
    parser.add_argument("--prompt", action="append", help="Run one or more scripted user turns, then exit.")
    parser.add_argument("--language", choices=["zh", "en"], default="en", help="Initial terminal language.")
    parser.add_argument("--state-file", help="Use an alternate workflow state file for tests or isolated sessions.")
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
    ]

    proposed_ledger = disks.get("proposed_ledger_device") or "<needs confirmation>"
    proposed_accounts = disks.get("proposed_accounts_device") or "<none detected>"
    lines.append(f"- LEDGER_DEVICE candidate: {proposed_ledger}")
    lines.append(f"- ACCOUNTS_DEVICE candidate: {proposed_accounts}")

    candidates = disks.get("candidates") or []
    if candidates:
        lines.append("- Disk candidates:")
        for index, item in enumerate(candidates[:8], start=1):
            lines.append(
                "  "
                f"[{index}] {item.get('name') or '<unknown>'} "
                f"type={item.get('type') or '<unknown>'} "
                f"size={item.get('size') or '<unknown>'} "
                f"mount={item.get('mountpoint') or '<none>'} "
                f"label={item.get('label') or '<none>'}"
            )
        if len(candidates) > 8:
            lines.append(f"  ... {len(candidates) - 8} more")
    else:
        lines.append("- Disk candidates: <none detected; will ask manually>")

    return "\n".join(lines)


def _format_memory(value: Any) -> str:
    if value in {None, ""}:
        return "<unknown>"
    return f"{value} GiB"


if __name__ == "__main__":
    raise SystemExit(main())

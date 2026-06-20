"""Terminal chat loop for AnyChain Benchmark Agent."""

from __future__ import annotations

import json
import os
import sys
from io import UnsupportedOperation
from pathlib import Path
from typing import Any, TextIO

from analyzers.artifact_qa import answer_artifact_question
from analyzers.result_analyzer import analyze_job
from diagnostics.doctor import format_doctor_report, run_doctor
from discovery.environment import discover_environment
from knowledge.framework_capabilities import answer_capability_question
from knowledge.gap_analyzer import answer_gap_question
from knowledge.loader import load_knowledge_provider, provider_status
from llm.orchestrator import synthesize_with_fallback
from llm.runtime import LLMRuntime, detect_llm_runtime
from memory.compactor import compact_session_state, should_auto_compact
from memory.session_store import read_memory, write_memory
from memory.token_estimator import (
    DEFAULT_COMPACT_KEEP_RECENT_TURNS,
    DEFAULT_COMPACT_TRIGGER_RATIO,
    DEFAULT_COMPACT_TURN_THRESHOLD,
    DEFAULT_CONTEXT_WINDOW_TOKENS,
)
from planners.preflight import run_preflight
from planners.diff import diff_plans
from planners.request_modifier import apply_request_modification, looks_like_plan_modification
from planners.risk import score_plan_risk
from planners.strategy_planner import generate_plan, write_json
from qa.framework_answers import answer_framework_question, out_of_scope_response
from qa.intent_router import route_intent
from qa.llm_drafter import draft_request_with_llm
from qa.request_drafter import draft_request
from runners.job_manager import get_job, list_jobs, submit_job, tail_job_log
from runners.runbook import render_runbook
from utils.redaction import redact
from workflows.checklist import apply_checklist_answer, format_checklist, next_blocker
from workflows.onboarding_request import answer_onboarding_request
from workflows.router import prompt_bundle_for_workflow, select_workflow
from workflows.trace import WorkflowTrace, read_trace


class ChatSession:
    """Stateful terminal session built on the deterministic Agent primitives."""

    def __init__(
        self,
        output_dir: str | Path = ".agent/chat",
        llm_provider: Any | None = None,
        llm_runtime: LLMRuntime | None = None,
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir = self.output_dir / "jobs"
        self.llm_runtime = llm_runtime or detect_llm_runtime(mock_provider=llm_provider)
        self.knowledge_status = provider_status()
        self.use_llm = self.llm_runtime.enabled
        self.llm_provider = self.llm_runtime.provider
        self.context_window_tokens = _env_int("AGENT_CONTEXT_WINDOW_TOKENS", DEFAULT_CONTEXT_WINDOW_TOKENS)
        self.compact_trigger_ratio = _env_float("AGENT_COMPACT_TRIGGER_RATIO", DEFAULT_COMPACT_TRIGGER_RATIO)
        self.compact_turn_threshold = _env_int("AGENT_COMPACT_TURN_THRESHOLD", DEFAULT_COMPACT_TURN_THRESHOLD)
        self.compact_keep_recent_turns = _env_int("AGENT_COMPACT_KEEP_RECENT_TURNS", DEFAULT_COMPACT_KEEP_RECENT_TURNS)
        self.discovery: dict[str, Any] | None = None
        self.request: dict[str, Any] | None = None
        self.plan: dict[str, Any] | None = None
        self.preflight: dict[str, Any] | None = None
        self.turns: list[dict[str, str]] = []
        self.memory_file = self.output_dir / "memory.json"
        self.memory: dict[str, Any] | None = read_memory(self.memory_file)
        self.request_file = self.output_dir / "request.json"
        self.plan_file = self.output_dir / "plan.json"
        self.runbook_file = self.output_dir / "runbook.md"
        self.job: dict[str, Any] | None = self._restore_latest_job()
        self.trace = WorkflowTrace(self.output_dir)

    def greeting(self) -> str:
        return (
            "AnyChain Benchmark Agent\n"
            f"{self.llm_runtime.banner()}\n"
            f"Knowledge base: {self.knowledge_status['provider']} "
            f"({'enabled' if self.knowledge_status['enabled'] else 'local repo only'})\n"
            f"{self._latest_job_banner()}"
            "Type a benchmark goal, ask a framework question, or use: "
            "help, doctor, plan, preflight, run mock, run, status, analyze, compact, memory, exit."
        )

    def handle(self, text: str) -> str:
        text = text.strip()
        if not text:
            return "Tell me what you want to test, or type `help`."
        self._record_turn("user", text)
        command = text.lower()
        if command in {"help", "?"}:
            return self._respond(self._help())
        if command in {"exit", "quit", "q"}:
            return "__EXIT__"
        if command in {"compact", "summarize context", "compress context"}:
            return self._respond(self._compact(reason="manual"))
        if command in {"memory", "context", "show memory"}:
            return self._respond(self._show_memory())
        route = route_intent(text, provider=self.llm_provider, use_llm=self.use_llm)
        workflow = select_workflow(
            text,
            command=command,
            route=route,
            has_plan=bool(self.plan),
            has_job=bool(self.job),
            plan=self.plan,
            is_plan_edit=bool(self.plan and self.request and looks_like_plan_modification(text)),
        )
        response = self._run_workflow(workflow, text, route)
        self._trace(text, route, workflow, response)
        return self._respond(response)

    def _run_workflow(self, workflow: str, text: str, route: dict[str, Any]) -> str:
        command = text.lower()
        if workflow == "doctor":
            return self._doctor()
        if workflow == "discover":
            self.discovery = discover_environment()
            return self._format_discovery()
        if workflow == "show_plan":
            return self._show_plan()
        if workflow == "preflight":
            return self._run_preflight()
        if workflow == "checklist":
            return self._show_checklist()
        if workflow == "risk":
            return self._risk_score()
        if workflow == "runbook":
            return self._show_runbook()
        if workflow == "submit_mock":
            return self._submit(mock=True, approved=True)
        if workflow == "submit_real":
            return self._submit(mock=False, approved=False)
        if workflow == "submit_real_confirmed":
            return self._submit(mock=False, approved=True)
        if workflow == "status":
            return self._status()
        if workflow == "logs":
            return self._logs()
        if workflow == "analyze":
            return self._analyze()
        if workflow == "trace":
            return self._show_trace()
        if workflow == "artifact_analysis":
            question = text[3:].strip() if command.startswith("qa ") else text
            return self._artifact_qa(question)
        if workflow == "onboarding_request":
            return self._onboarding_request(text)
        if workflow == "plan_edit":
            return self._modify_plan(text)
        if workflow == "checklist_answer":
            return self._apply_checklist_answer(text)
        if workflow == "out_of_scope":
            return out_of_scope_response(text)["answer"]
        if workflow == "benchmark_request":
            return self._create_plan(text)
        if workflow == "framework_question":
            question = text[4:].strip() if command.startswith("ask ") else text
            return self._answer_question(question, route=route)
        return "I can help with benchmark planning, framework capabilities, job status, and result analysis."

    def _onboarding_request(self, text: str) -> str:
        return answer_onboarding_request(
            text,
            llm_provider=self.llm_provider if self.use_llm else None,
        )

    def _help(self) -> str:
        return (
            "Examples:\n"
            "- What chains and RPC methods do you support?\n"
            "- Create a Solana fake-node smoke benchmark at 1 QPS\n"
            "- Create an Ethereum weighted mixed workload plan\n"
            "- Set max qps to 5000\n"
            "- Change mixed weights to getSlot 70%, getBlockHeight 30%\n"
            "- plan\n"
            "- preflight\n"
            "- doctor\n"
            "- run mock\n"
            "- status\n"
            "- analyze\n"
            "- compact\n"
            "- memory\n"
            "- qa What evidence was generated?\n\n"
            "Real benchmark execution requires `yes run` or `confirm run` after reviewing the plan.\n"
            "Default real execution is detached/background so long benchmarks continue if the Agent terminal disconnects.\n"
            "Say `run in foreground` before `yes run` if you want the benchmark tied to this terminal."
        )

    def _doctor(self) -> str:
        self.discovery = discover_environment()
        return format_doctor_report(run_doctor(self.discovery))

    def _answer_question(self, question: str, route: dict[str, Any] | None = None) -> str:
        route = route or route_intent(question, provider=self.llm_provider, use_llm=self.use_llm)
        if route["intent"] == "out_of_scope":
            return out_of_scope_response(question)["answer"]
        capability_answer = answer_capability_question(question)
        if capability_answer:
            return capability_answer["answer"]
        gap_answer = answer_gap_question(question)
        if gap_answer:
            return gap_answer["answer"]
        framework_answer = answer_framework_question(question)
        if framework_answer.get("answer"):
            return framework_answer["answer"]
        kb_answer = self._answer_from_knowledge_base(question)
        if kb_answer:
            return kb_answer
        if route["intent"] == "benchmark_request":
            return self._create_plan(question)
        return "I can help with benchmark planning, framework capabilities, job status, and result analysis."

    def _answer_from_knowledge_base(self, question: str) -> str:
        if not self.knowledge_status.get("enabled") or self.knowledge_status.get("error"):
            return ""
        try:
            provider = load_knowledge_provider()
            results = provider.search(question)
        except Exception:
            return ""
        if not results:
            return ""
        lines = ["Knowledge Base matches:"]
        for item in results[:5]:
            title = item.get("title") or item.get("source") or "result"
            text = item.get("text") or item.get("summary") or item.get("content") or ""
            lines.append(f"- {title}: {text}")
        fallback = "\n".join(lines)
        return synthesize_with_fallback(
            self.llm_provider if self.use_llm else None,
            "kb_answer",
            json.dumps({"question": question, "results": results[:5]}, indent=2, sort_keys=True),
            fallback,
            max_tokens=1000,
        )

    def _create_plan(self, prompt: str) -> str:
        request = (
            draft_request_with_llm(prompt, provider=self.llm_provider)
            if self.use_llm or self.llm_provider
            else draft_request(prompt)
        )
        self.discovery = discover_environment()
        request["discovery"] = self.discovery
        self.request = request
        self.plan = generate_plan(request, discovery=self.discovery)
        self.preflight = run_preflight(self.plan)
        write_json(self.request_file, self.request)
        write_json(self.plan_file, self.plan)
        self.runbook_file.write_text(render_runbook(self.plan, self.preflight), encoding="utf-8")
        return self._plan_summary(prefix="Created a benchmark plan.")

    def _modify_plan(self, text: str) -> str:
        if not self.request or not self.plan:
            return "No active plan yet. Describe the benchmark you want to run."
        updated_request, changes = apply_request_modification(self.request, text)
        if not changes:
            return "I could not identify a supported plan edit. Try `set max qps to 5000` or `getSlot 70%, getBlockHeight 30%`."
        old_plan = self.plan
        self.request = updated_request
        self.plan = generate_plan(self.request, discovery=self.discovery)
        self.preflight = run_preflight(self.plan)
        plan_diff = diff_plans(old_plan, self.plan)
        self.plan["plan_diff"] = plan_diff
        write_json(self.request_file, self.request)
        write_json(self.plan_file, self.plan)
        self.runbook_file.write_text(render_runbook(self.plan, self.preflight), encoding="utf-8")
        return "\n".join([
            "Updated the current benchmark plan.",
            "Changes:",
            *[f"- {change}" for change in changes],
            self._plan_summary(prefix="Updated plan summary."),
        ])

    def _apply_checklist_answer(self, text: str) -> str:
        if not self.request or not self.plan:
            return "No active checklist yet. Describe the benchmark you want to run."
        blocker = next_blocker(self.plan)
        if not blocker:
            return "No blocking checklist item is waiting for an answer."
        answer = text.strip()
        if answer.lower().startswith("answer "):
            answer = answer[7:].strip()
        updated_request, changes = apply_checklist_answer(self.request, blocker, answer)
        if not changes:
            return f"I could not apply that answer to `{blocker['id']}`. {blocker.get('prompt', '')}"
        old_plan = self.plan
        self.request = updated_request
        self.plan = generate_plan(self.request, discovery=self.discovery)
        self.preflight = run_preflight(self.plan)
        self.plan["plan_diff"] = diff_plans(old_plan, self.plan)
        write_json(self.request_file, self.request)
        write_json(self.plan_file, self.plan)
        self.runbook_file.write_text(render_runbook(self.plan, self.preflight), encoding="utf-8")
        return "\n".join([
            f"Applied answer for `{blocker['id']}`.",
            *[f"- {change}" for change in changes],
            self._plan_summary(prefix="Updated plan summary."),
        ])

    def _show_plan(self) -> str:
        if not self.plan:
            return "No active plan yet. Describe the benchmark you want to run."
        return self._plan_summary(prefix="Current benchmark plan.")

    def _show_checklist(self) -> str:
        return format_checklist(self.plan)

    def _plan_summary(self, prefix: str) -> str:
        assert self.plan is not None
        plan = self.plan
        required_questions = plan.get("required_questions", [])
        blockers = [q for q in required_questions if q.get("severity") == "blocker"]
        lines = [
            prefix,
            f"- plan_id: {plan.get('plan_id')}",
            f"- chain: {plan.get('chain') or '<missing>'}",
            f"- strategy: {plan.get('strategy')}",
            f"- rpc_mode: {plan.get('rpc_mode')}",
            f"- fake-node: {plan.get('use_fake_node')}",
            f"- execution_mode: {plan.get('execution', {}).get('runner_mode', 'foreground')}",
            f"- command: {' '.join(plan.get('execution', {}).get('command', []))}",
            f"- request: {self.request_file}",
            f"- plan: {self.plan_file}",
            f"- runbook: {self.runbook_file}",
        ]
        if blockers:
            lines.append("- blockers: " + ", ".join(q["id"] for q in blockers))
        lines.append("Next: type `preflight`, `run mock`, or `yes run` after reviewing the runbook.")
        return "\n".join(lines)

    def _run_preflight(self) -> str:
        if not self.plan:
            return "No active plan yet."
        self.preflight = run_preflight(self.plan)
        status = "passed" if self.preflight.get("passed") else "failed"
        lines = [f"Preflight {status}.", json.dumps(self.preflight, indent=2, sort_keys=True)]
        if not self.preflight.get("passed"):
            lines.append(self._show_checklist())
        return "\n".join(lines)

    def _risk_score(self) -> str:
        if not self.plan:
            return "No active plan yet."
        risk = score_plan_risk(self.plan)
        return json.dumps(risk, indent=2, sort_keys=True)

    def _show_runbook(self) -> str:
        if not self.plan:
            return "No active plan yet."
        self.runbook_file.write_text(render_runbook(self.plan, self.preflight), encoding="utf-8")
        return f"Runbook written to {self.runbook_file}"

    def _submit(self, mock: bool, approved: bool) -> str:
        if not self.plan:
            return "No active plan yet."
        if not mock and not approved:
            return (
                "Real benchmark execution requires explicit confirmation. Type `yes run` after reviewing the plan.\n"
                "Default execution mode is detached/background. The benchmark can continue if the Agent terminal disconnects. "
                "Type `run in foreground` before confirmation if you want terminal-bound execution."
            )
        self.job = submit_job(self.plan_file, jobs_dir=self.jobs_dir, mock=mock, approved=approved)
        mode = "mock" if mock else "real"
        return (
            f"Submitted {mode} job {self.job['job_id']} with status {self.job['status']}.\n"
            f"- job_dir: {self.job['run_dir']}\n"
            f"- artifact_index: {self.job['artifact_index']}\n"
            "Next: type `status`, `analyze`, or `qa <question>`."
        )

    def _status(self) -> str:
        if not self.job:
            return "No job has been submitted in this session."
        self.job = get_job(self.job["job_id"], jobs_dir=self.jobs_dir)
        return json.dumps(self.job, indent=2, sort_keys=True)

    def _logs(self) -> str:
        if not self.job:
            return "No job has been submitted in this session."
        payload = tail_job_log(self.job["job_id"], jobs_dir=self.jobs_dir)
        if not payload["exists"]:
            return f"No benchmark.log exists yet for {self.job['job_id']} at {payload['log_file']}."
        return "\n".join(payload["lines"][-80:])

    def _analyze(self) -> str:
        if not self.job:
            return "No job has been submitted in this session."
        self.job = get_job(self.job["job_id"], jobs_dir=self.jobs_dir)
        analysis = analyze_job(self.job)
        return json.dumps(analysis, indent=2, sort_keys=True)

    def _artifact_qa(self, question: str) -> str:
        if not self.job:
            return "No job has been submitted in this session."
        answer = answer_artifact_question(
            question,
            job=self.job,
            llm_provider=self.llm_provider if self.use_llm else None,
        )
        return answer["answer"]

    def _compact(self, reason: str = "manual") -> str:
        self.memory = compact_session_state(
            turns=self.turns,
            request=self.request,
            plan=self.plan,
            job=self.job,
            discovery=self.discovery,
            previous_summary=self.memory,
            keep_recent=self.compact_keep_recent_turns,
            context_window_tokens=self.context_window_tokens,
            trigger_ratio=self.compact_trigger_ratio,
            turn_threshold=self.compact_turn_threshold,
            reason=reason,
        )
        write_memory(self.memory_file, self.memory)
        recent = self.memory.get("recent_turns", [])
        self.turns = list(recent)
        return (
            f"Context compacted to {self.memory_file}.\n"
            f"- compacted_turn_count: {self.memory.get('compacted_turn_count')}\n"
            f"- recent_turns_kept: {len(recent)}\n"
            f"- summary: {self.memory.get('summary')}"
        )

    def _show_memory(self) -> str:
        if not self.memory:
            return "No compacted memory yet. Type `compact` to summarize the current session."
        return json.dumps(self.memory, indent=2, sort_keys=True)

    def _show_trace(self) -> str:
        rows = read_trace(self.trace.path, limit=10)
        if not rows:
            return f"No workflow trace entries yet. Trace file: {self.trace.path}"
        return json.dumps({"trace_file": str(self.trace.path), "events": rows}, indent=2, sort_keys=True)

    def _format_discovery(self) -> str:
        assert self.discovery is not None
        cloud = self.discovery.get("cloud", {})
        deployment = self.discovery.get("deployment", {})
        dependencies = self.discovery.get("dependencies", {})
        return (
            "Discovery complete.\n"
            f"- cloud: {cloud.get('provider')} / {cloud.get('platform')}\n"
            f"- deployment: {deployment.get('type')}\n"
            f"- dependency_mode: {dependencies.get('mode')}\n"
            f"- missing_required: {', '.join(dependencies.get('missing_required', [])) or '<none>'}"
        )

    @staticmethod
    def _looks_like_question(text: str) -> bool:
        lowered = text.lower().strip()
        return lowered.endswith("?") or lowered.startswith(("what ", "how ", "does ", "do ", "can ", "which ", "why "))

    def _record_turn(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})

    def _restore_latest_job(self) -> dict[str, Any] | None:
        latest = list_jobs(self.jobs_dir, limit=1)
        if latest:
            try:
                return get_job(latest[0]["job_id"], jobs_dir=self.jobs_dir)
            except Exception:
                return latest[0]
        preserved = (self.memory or {}).get("preserved_state", {})
        job_id = preserved.get("job_id")
        if job_id:
            try:
                return get_job(job_id, jobs_dir=self.jobs_dir)
            except Exception:
                return None
        return None

    def _latest_job_banner(self) -> str:
        if not self.job:
            return ""
        status = self.job.get("status", "unknown")
        job_id = self.job.get("job_id", "<unknown>")
        run_dir = self.job.get("run_dir", "")
        if status == "running":
            next_step = "Next: type `status` or `logs`; if the terminal disconnected, verify whether the runner is still alive."
        elif status == "completed":
            next_step = "Next: type `analyze` or `qa <question>`."
        elif status == "failed":
            next_step = "Next: type `logs` or `qa <question>`."
        else:
            next_step = "Next: type `status`."
        return f"Latest job: {job_id} ({status}). {next_step}\nJob dir: {run_dir}\n"

    def _respond(self, response: str) -> str:
        self._record_turn("assistant", response)
        if should_auto_compact(
            turns=self.turns,
            context_window_tokens=self.context_window_tokens,
            trigger_ratio=self.compact_trigger_ratio,
            turn_threshold=self.compact_turn_threshold,
        ):
            compact_note = self._compact(reason="auto")
            response = f"{response}\n\n{compact_note}"
        return response

    def _trace(self, text: str, route: dict[str, Any], workflow: str, response: str) -> None:
        tools = _tools_for_workflow(workflow)
        artifacts = {
            "request_file": str(self.request_file) if self.request_file.exists() else "",
            "plan_file": str(self.plan_file) if self.plan_file.exists() else "",
            "runbook_file": str(self.runbook_file) if self.runbook_file.exists() else "",
            "job_id": self.job.get("job_id", "") if self.job else "",
            "artifact_index": self.job.get("artifact_index", "") if self.job else "",
        }
        self.trace.record(
            user_input=text,
            intent=route,
            workflow=workflow,
            tools=tools,
            prompt_bundle=prompt_bundle_for_workflow(workflow),
            artifacts={key: value for key, value in artifacts.items() if value},
            fallback="" if self.use_llm else "deterministic/offline",
            next_actions=_next_actions_for_workflow(workflow, response),
        )


def _tools_for_workflow(workflow: str) -> list[str]:
    return {
        "doctor": ["discover_environment", "run_doctor"],
        "discover": ["discover_environment"],
        "benchmark_request": ["draft_request", "discover_environment", "generate_plan", "run_preflight"],
        "framework_question": ["route_intent", "capabilities_or_docs_or_kb"],
        "artifact_analysis": ["answer_artifact_question", "diagnose_artifacts"],
        "plan_edit": ["apply_request_modification", "generate_plan", "run_preflight"],
        "checklist_answer": ["apply_checklist_answer", "generate_plan", "run_preflight"],
        "checklist": ["format_checklist"],
        "preflight": ["run_preflight"],
        "submit_mock": ["submit_job"],
        "submit_real": ["submit_job"],
        "submit_real_confirmed": ["submit_job"],
        "status": ["get_job"],
        "logs": ["tail_job_log"],
        "analyze": ["analyze_job"],
    }.get(workflow, [])


def _next_actions_for_workflow(workflow: str, response: str) -> list[str]:
    if workflow in {"benchmark_request", "plan_edit", "checklist_answer"}:
        return ["preflight", "run mock", "yes run"]
    if workflow == "checklist":
        return ["answer <value>", "preflight"]
    if workflow == "submit_mock":
        return ["status", "analyze", "qa <question>"]
    if workflow == "artifact_analysis":
        return ["analyze", "logs"]
    if workflow == "preflight" and "failed" in response.lower():
        return ["answer checklist blockers", "plan"]
    return []


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "")
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name, "")
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def run_chat(
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
    prompt: str | None = None,
    output_dir: str | Path = ".agent/chat",
    llm_provider: Any | None = None,
) -> int:
    session = ChatSession(output_dir=output_dir, llm_provider=llm_provider)
    print(session.greeting(), file=output_stream)
    if prompt:
        print(f"> {prompt}", file=output_stream)
        response = session.handle(prompt)
        if response != "__EXIT__":
            _write_redacted_response(output_stream, response)
        return 0
    while True:
        print("> ", end="", file=output_stream, flush=True)
        line = input_stream.readline()
        if not line:
            print("", file=output_stream)
            return 0
        response = session.handle(line)
        if response == "__EXIT__":
            print("Goodbye.", file=output_stream)
            return 0
        _write_redacted_response(output_stream, response)


def _write_redacted_response(output_stream: TextIO, response: str) -> None:
    redacted = f"{redact(response)}\n".encode("utf-8", errors="replace")
    try:
        os.write(output_stream.fileno(), redacted)
    except (AttributeError, OSError, UnsupportedOperation):
        output_stream.write("[response omitted: output stream has no file descriptor]\n")

"""Terminal chat loop for AnyChain Benchmark Agent."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, TextIO

from analyzers.artifact_qa import answer_artifact_question
from analyzers.result_analyzer import analyze_job
from diagnostics.doctor import format_doctor_report, run_doctor
from discovery.environment import discover_environment
from knowledge.framework_capabilities import answer_capability_question
from knowledge.gap_analyzer import answer_gap_question
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
from runners.job_manager import get_job, submit_job
from runners.runbook import render_runbook


class ChatSession:
    """Stateful terminal session built on the deterministic Agent primitives."""

    def __init__(
        self,
        output_dir: str | Path = ".agent/chat",
        use_llm: bool = False,
        llm_provider: Any | None = None,
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir = self.output_dir / "jobs"
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        self.context_window_tokens = _env_int("AGENT_CONTEXT_WINDOW_TOKENS", DEFAULT_CONTEXT_WINDOW_TOKENS)
        self.compact_trigger_ratio = _env_float("AGENT_COMPACT_TRIGGER_RATIO", DEFAULT_COMPACT_TRIGGER_RATIO)
        self.compact_turn_threshold = _env_int("AGENT_COMPACT_TURN_THRESHOLD", DEFAULT_COMPACT_TURN_THRESHOLD)
        self.compact_keep_recent_turns = _env_int("AGENT_COMPACT_KEEP_RECENT_TURNS", DEFAULT_COMPACT_KEEP_RECENT_TURNS)
        self.discovery: dict[str, Any] | None = None
        self.request: dict[str, Any] | None = None
        self.plan: dict[str, Any] | None = None
        self.preflight: dict[str, Any] | None = None
        self.job: dict[str, Any] | None = None
        self.turns: list[dict[str, str]] = []
        self.memory_file = self.output_dir / "memory.json"
        self.memory: dict[str, Any] | None = read_memory(self.memory_file)
        self.request_file = self.output_dir / "request.json"
        self.plan_file = self.output_dir / "plan.json"
        self.runbook_file = self.output_dir / "runbook.md"

    def greeting(self) -> str:
        return (
            "AnyChain Benchmark Agent\n"
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
        if command in {"doctor", "diagnose", "readiness", "check readiness"}:
            return self._respond(self._doctor())
        if command in {"discover", "detect environment"}:
            self.discovery = discover_environment()
            return self._respond(self._format_discovery())
        if command in {"plan", "show plan", "current plan"}:
            return self._respond(self._show_plan())
        if command in {"preflight", "check plan"}:
            return self._respond(self._run_preflight())
        if command in {"risk", "risk score", "score risk"}:
            return self._respond(self._risk_score())
        if command in {"runbook", "show runbook"}:
            return self._respond(self._show_runbook())
        if command in {"run mock", "mock", "submit mock", "execute mock"}:
            return self._respond(self._submit(mock=True, approved=True))
        if command in {"run", "execute", "submit", "run real", "execute real"}:
            return self._respond(self._submit(mock=False, approved=False))
        if command.startswith("yes run") or command.startswith("confirm run"):
            return self._respond(self._submit(mock=False, approved=True))
        if command in {"status", "job status"}:
            return self._respond(self._status())
        if command in {"analyze", "analyse", "analyze result", "analysis"}:
            return self._respond(self._analyze())
        if command.startswith("ask "):
            return self._respond(self._answer_question(text[4:].strip()))
        if command.startswith("qa "):
            return self._respond(self._artifact_qa(text[3:].strip()))
        if self.plan and self.request and looks_like_plan_modification(text):
            return self._respond(self._modify_plan(text))
        if self._looks_like_question(text):
            return self._respond(self._answer_question(text))
        return self._respond(self._create_plan(text))

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
            "Real benchmark execution requires `yes run` or `confirm run` after reviewing the plan."
        )

    def _doctor(self) -> str:
        self.discovery = discover_environment()
        return format_doctor_report(run_doctor(self.discovery))

    def _answer_question(self, question: str) -> str:
        route = route_intent(question, provider=self.llm_provider, use_llm=self.use_llm)
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
        if route["intent"] == "benchmark_request":
            return self._create_plan(question)
        return "I can help with benchmark planning, framework capabilities, job status, and result analysis."

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

    def _show_plan(self) -> str:
        if not self.plan:
            return "No active plan yet. Describe the benchmark you want to run."
        return self._plan_summary(prefix="Current benchmark plan.")

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
        return f"Preflight {status}.\n{json.dumps(self.preflight, indent=2, sort_keys=True)}"

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
            return "Real benchmark execution requires explicit confirmation. Type `yes run` after reviewing the plan."
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

    def _analyze(self) -> str:
        if not self.job:
            return "No job has been submitted in this session."
        self.job = get_job(self.job["job_id"], jobs_dir=self.jobs_dir)
        analysis = analyze_job(self.job)
        return json.dumps(analysis, indent=2, sort_keys=True)

    def _artifact_qa(self, question: str) -> str:
        if not self.job:
            return "No job has been submitted in this session."
        answer = answer_artifact_question(question, job=self.job)
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
    use_llm: bool = False,
    llm_provider: Any | None = None,
) -> int:
    session = ChatSession(output_dir=output_dir, use_llm=use_llm, llm_provider=llm_provider)
    print(session.greeting(), file=output_stream)
    if prompt:
        print(f"> {prompt}", file=output_stream)
        response = session.handle(prompt)
        if response != "__EXIT__":
            print(response, file=output_stream)
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
        print(response, file=output_stream)

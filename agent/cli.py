#!/usr/bin/env python3
"""AnyChain Benchmark Agent command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from analyzers.result_analyzer import analyze_job
from analyzers.history import compare_latest, list_history
from analyzers.artifact_qa import answer_artifact_question
from diagnostics.doctor import run_doctor
from discovery.environment import discover_environment
from knowledge.gap_analyzer import analyze_capability_gap, answer_gap_question
from knowledge.framework_capabilities import answer_capability_question, load_framework_capabilities
from llm.config import load_llm_config
from llm.google_auth import credential_plan
from llm.providers import provider_from_config
from llm.types import LLMMessage, LLMRequest
from onboarding.chain_onboarding import generate_onboarding_package
from planners.preflight import run_preflight
from planners.diff import diff_plans
from planners.risk import score_plan_risk
from planners.strategy_planner import generate_plan, load_json, validate_plan_shape, write_json
from qa.framework_answers import answer_framework_question, out_of_scope_response
from qa.intent_router import route_intent
from qa.llm_drafter import draft_request_with_llm
from qa.request_drafter import draft_request
from runners.job_manager import get_job, submit_job
from runners.runbook import render_runbook
from wizard import run_wizard
from chat import run_chat


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AnyChain Benchmark Agent control-plane CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    draft = sub.add_parser("draft-request", help="Draft a request JSON from a natural-language prompt")
    draft.add_argument("--prompt", required=True)
    draft.add_argument("--output")
    draft.add_argument("--use-llm", action="store_true", help="Use configured LLM provider with deterministic fallback")
    draft.add_argument("--mock-llm", action="store_true", help="Use the offline fake LLM provider")

    route = sub.add_parser("route-intent", help="Classify a prompt before planning")
    route.add_argument("--prompt", required=True)
    route.add_argument("--use-llm", action="store_true")
    route.add_argument("--mock-llm", action="store_true")

    ask = sub.add_parser("ask", help="Answer framework questions from local docs or report out-of-scope")
    ask.add_argument("--prompt", required=True)
    ask.add_argument("--use-llm", action="store_true")
    ask.add_argument("--mock-llm", action="store_true")

    capabilities = sub.add_parser("capabilities", help="Show dynamic framework capability inventory")
    capabilities.add_argument("--output")

    gap = sub.add_parser("gap-analysis", help="Analyze chain/RPC method support gaps")
    gap.add_argument("--chain", required=True)
    gap.add_argument("--method", action="append", default=[])

    onboarding = sub.add_parser("onboarding-plan", help="Generate plugin-style chain/RPC workload onboarding plan")
    onboarding.add_argument("--chain", required=True)
    onboarding.add_argument("--method", action="append", default=[])
    onboarding.add_argument("--adapter-family")
    onboarding.add_argument("--rpc-mode", default="mixed", choices=["single", "mixed"])

    risk = sub.add_parser("risk-score", help="Score risk for a benchmark plan")
    risk.add_argument("--plan", required=True)

    artifact_qa = sub.add_parser("artifact-qa", help="Answer questions using job artifacts")
    artifact_qa.add_argument("--question", required=True)
    artifact_qa.add_argument("--job-id")
    artifact_qa.add_argument("--jobs-dir")
    artifact_qa.add_argument("--artifact-index")

    plan_cmd = sub.add_parser("plan", help="Generate a benchmark plan from a request JSON")
    plan_cmd.add_argument("--request", required=True)
    plan_cmd.add_argument("--output")
    plan_cmd.add_argument("--dry-run", action="store_true")
    plan_cmd.add_argument("--discover", action="store_true", help="Run read-only discovery before generating the plan")

    preflight = sub.add_parser("preflight", help="Run preflight checks for a plan")
    preflight.add_argument("--plan", required=True)

    discover = sub.add_parser("discover", help="Run read-only environment discovery")
    discover.add_argument("--output")

    doctor = sub.add_parser("doctor", help="Run read-only Agent readiness diagnostics")
    doctor.add_argument("--output")

    llm_config = sub.add_parser("llm-config", help="Validate LLM provider and auth configuration")
    llm_config.add_argument("--output")

    llm_smoke = sub.add_parser("llm-smoke", help="Run a model provider smoke test")
    llm_smoke.add_argument("--prompt", default='Return JSON only: {"ok": true}')
    llm_smoke.add_argument("--mock", action="store_true", help="Use the offline fake provider")

    validate = sub.add_parser("validate-plan", help="Validate plan shape")
    validate.add_argument("plan")

    diff = sub.add_parser("diff-plan", help="Compare two benchmark plans")
    diff.add_argument("--old", required=True)
    diff.add_argument("--new", required=True)

    submit = sub.add_parser("submit", help="Submit a benchmark job")
    submit.add_argument("--plan", required=True)
    submit.add_argument("--jobs-dir")
    submit.add_argument("--mock", action="store_true", help="Complete a lifecycle-only mock job")
    submit.add_argument("--approved", action="store_true", help="Confirm approval checkpoints for real execution")

    status = sub.add_parser("status", help="Show job status")
    status.add_argument("--job-id", required=True)
    status.add_argument("--jobs-dir")

    analyze = sub.add_parser("analyze", help="Analyze a job")
    analyze.add_argument("--job-id", required=True)
    analyze.add_argument("--jobs-dir")

    history = sub.add_parser("history", help="List or compare archived benchmark runs")
    history.add_argument("--archives-dir")
    history.add_argument("--limit", type=int, default=10)
    history.add_argument("--compare-latest", action="store_true")

    runbook = sub.add_parser("runbook", help="Render a runbook for a plan")
    runbook.add_argument("--plan", required=True)
    runbook.add_argument("--output")

    wizard = sub.add_parser("wizard", help="Prompt-first interactive benchmark wizard")
    wizard.add_argument("--prompt")
    wizard.add_argument("--output-dir")
    wizard.add_argument("--yes", action="store_true", help="Auto-approve for non-interactive tests")
    wizard.add_argument("--mock", action="store_true", help="Submit a lifecycle-only mock job")
    wizard.add_argument("--quiet", action="store_true", help="Suppress human-readable wizard output on stderr")
    wizard.add_argument("--answers-file", help="JSON object with answers keyed by required question id")
    wizard.add_argument("--use-llm", action="store_true", help="Use configured LLM provider to draft the request")
    wizard.add_argument("--mock-llm", action="store_true", help="Use the offline fake LLM provider")

    chat = sub.add_parser("chat", help="Start the terminal Agent chat session")
    chat.add_argument("--prompt", help="Run one prompt and exit")
    chat.add_argument("--output-dir", default=".agent/chat")
    chat.add_argument("--use-llm", action="store_true", help="Use configured LLM provider for request drafting/routing")
    chat.add_argument("--mock-llm", action="store_true", help="Use the offline fake LLM provider")

    args = parser.parse_args(argv)

    if args.command == "draft-request":
        provider = _fake_provider() if args.mock_llm else None
        payload = draft_request_with_llm(args.prompt, provider=provider) if (args.use_llm or args.mock_llm) else draft_request(args.prompt)
        return _emit(payload, args.output)

    if args.command == "route-intent":
        provider = _fake_provider() if args.mock_llm else None
        return _emit(route_intent(args.prompt, provider=provider, use_llm=args.use_llm or args.mock_llm), None)

    if args.command == "ask":
        provider = _fake_provider() if args.mock_llm else None
        route_payload = route_intent(args.prompt, provider=provider, use_llm=args.use_llm or args.mock_llm)
        if route_payload["intent"] == "out_of_scope":
            return _emit({**route_payload, **out_of_scope_response(args.prompt)}, None)
        if route_payload["intent"] == "benchmark_request":
            request = draft_request_with_llm(args.prompt, provider=provider) if (args.use_llm or args.mock_llm) else draft_request(args.prompt)
            return _emit({"intent": "benchmark_request", "request": request, "route": route_payload}, None)
        capability_answer = answer_capability_question(args.prompt)
        if capability_answer:
            return _emit({**capability_answer, "route": route_payload}, None)
        gap_answer = answer_gap_question(args.prompt)
        if gap_answer:
            return _emit({**gap_answer, "route": route_payload}, None)
        answer = answer_framework_question(args.prompt)
        return _emit({**answer, "route": route_payload}, None)

    if args.command == "capabilities":
        return _emit(load_framework_capabilities(), args.output)

    if args.command == "gap-analysis":
        return _emit(analyze_capability_gap(args.chain, args.method), None)

    if args.command == "onboarding-plan":
        return _emit(
            generate_onboarding_package(
                args.chain,
                methods=args.method,
                adapter_family=args.adapter_family,
                rpc_mode=args.rpc_mode,
            ),
            None,
        )

    if args.command == "risk-score":
        return _emit(score_plan_risk(load_json(args.plan)), None)

    if args.command == "artifact-qa":
        job = None
        if args.job_id:
            job = get_job(args.job_id, jobs_dir=args.jobs_dir) if args.jobs_dir else get_job(args.job_id)
        return _emit(answer_artifact_question(args.question, job=job, artifact_index=args.artifact_index), None)

    if args.command == "plan":
        request = load_json(args.request)
        discovery = discover_environment() if args.discover else None
        if discovery:
            request["discovery"] = discovery
        payload = generate_plan(request, discovery=discovery)
        if args.dry_run:
            _print_dry_run(payload)
        return _emit(payload, args.output)

    if args.command == "preflight":
        return _emit(run_preflight(load_json(args.plan)), None)

    if args.command == "discover":
        payload = discover_environment()
        return _emit(payload, args.output)

    if args.command == "doctor":
        return _emit(run_doctor(), args.output)

    if args.command == "llm-config":
        config = load_llm_config()
        payload = {
            "llm": config.safe_dict(),
            "google_credential_plan": credential_plan(config).safe_dict() if config.provider.startswith("vertex_") else {},
        }
        return _emit(payload, args.output)

    if args.command == "llm-smoke":
        provider = _fake_provider() if args.mock else provider_from_config()
        response = provider.complete(
            LLMRequest(
                messages=[
                    LLMMessage(role="system", content="Return a compact JSON object for a provider smoke test."),
                    LLMMessage(role="user", content=args.prompt),
                ],
                temperature=0,
                max_tokens=512,
            )
        )
        return _emit({"provider": response.provider, "model": response.model, "text": response.text}, None)

    if args.command == "validate-plan":
        errors = validate_plan_shape(load_json(args.plan))
        payload = {"valid": not errors, "errors": errors}
        _emit(payload, None)
        return 0 if not errors else 1

    if args.command == "diff-plan":
        return _emit(diff_plans(load_json(args.old), load_json(args.new)), None)

    if args.command == "submit":
        if args.jobs_dir:
            payload = submit_job(args.plan, jobs_dir=args.jobs_dir, mock=args.mock, approved=args.approved)
        else:
            payload = submit_job(args.plan, mock=args.mock, approved=args.approved)
        return _emit(payload, None)

    if args.command == "status":
        job = get_job(args.job_id, jobs_dir=args.jobs_dir) if args.jobs_dir else get_job(args.job_id)
        return _emit(job, None)

    if args.command == "analyze":
        job = get_job(args.job_id, jobs_dir=args.jobs_dir) if args.jobs_dir else get_job(args.job_id)
        return _emit(analyze_job(job), None)

    if args.command == "history":
        payload = compare_latest(args.archives_dir) if args.compare_latest else list_history(args.limit, args.archives_dir)
        return _emit(payload, None)

    if args.command == "runbook":
        plan = load_json(args.plan)
        text = render_runbook(plan)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(text, encoding="utf-8")
        print(text, end="")
        return 0

    if args.command == "wizard":
        payload = run_wizard(
            prompt=args.prompt,
            output_dir=args.output_dir,
            yes=args.yes,
            mock=args.mock,
            answers=load_json(args.answers_file) if args.answers_file else None,
            quiet=args.quiet,
            use_llm=args.use_llm,
            llm_provider=_fake_provider() if args.mock_llm else None,
        )
        return _emit(payload, None)

    if args.command == "chat":
        return run_chat(
            prompt=args.prompt,
            output_dir=args.output_dir,
            use_llm=args.use_llm,
            llm_provider=_fake_provider() if args.mock_llm else None,
        )

    parser.error(f"unsupported command: {args.command}")
    return 2


def _emit(payload: dict, output: str | None) -> int:
    if output:
        write_json(output, payload)
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def _print_dry_run(plan: dict) -> None:
    print("Dry run", file=sys.stderr)
    print(f"  Plan ID: {plan['plan_id']}", file=sys.stderr)
    print(f"  Chain: {plan.get('chain') or '<missing>'}", file=sys.stderr)
    print(f"  Strategy: {plan['strategy']}", file=sys.stderr)
    print(f"  RPC mode: {plan['rpc_mode']}", file=sys.stderr)
    print(f"  fake-node: {plan['use_fake_node']}", file=sys.stderr)
    print(f"  Command: {' '.join(plan['execution']['command'])}", file=sys.stderr)
    if plan["required_inputs"]:
        print(f"  Missing required inputs: {', '.join(plan['required_inputs'])}", file=sys.stderr)


def _fake_provider():
    from llm.config import LLMConfig
    from llm.providers import FakeProvider

    return FakeProvider(LLMConfig(provider="fake", model="fake"))


if __name__ == "__main__":
    raise SystemExit(main())

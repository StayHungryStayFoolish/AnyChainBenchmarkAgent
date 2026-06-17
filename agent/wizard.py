"""Interactive Agent wizard helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from discovery.environment import discover_environment
from planners.preflight import run_preflight
from planners.strategy_planner import generate_plan, write_json
from qa.llm_drafter import draft_request_with_llm
from qa.request_drafter import draft_request
from runners.job_manager import submit_job
from runners.runbook import render_runbook
from utils.redaction import redact


def run_wizard(
    prompt: str | None = None,
    output_dir: str | Path | None = None,
    yes: bool = False,
    mock: bool = False,
    answers: dict[str, str] | None = None,
    discovery_override: dict[str, Any] | None = None,
    quiet: bool = False,
    use_llm: bool = False,
    llm_provider: Any | None = None,
) -> dict[str, Any]:
    prompt = prompt or _ask("Describe the benchmark goal")
    request = draft_request_with_llm(prompt, provider=llm_provider) if use_llm or llm_provider else draft_request(prompt)
    discovery = discovery_override or discover_environment()
    request["discovery"] = discovery

    initial_plan = generate_plan(request, discovery=discovery)
    request = _apply_required_answers(request, initial_plan, answers=answers or {}, yes=yes)
    plan = generate_plan(request, discovery=discovery)
    preflight = run_preflight(plan)
    runbook = render_runbook(plan, preflight)

    output_path = Path(output_dir or ".agent/wizard").resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    request_file = output_path / "request.json"
    plan_file = output_path / "plan.json"
    runbook_file = output_path / "runbook.md"
    write_json(request_file, request)
    write_json(plan_file, plan)
    runbook_file.write_text(runbook, encoding="utf-8")

    if not quiet:
        print(runbook, file=sys.stderr)
    blockers = [q for q in plan.get("required_questions", []) if q.get("severity") == "blocker"]
    if blockers and not yes:
        return {
            "status": "needs_input",
            "request_file": str(request_file),
            "plan_file": str(plan_file),
            "runbook_file": str(runbook_file),
            "missing": plan["required_inputs"],
            "required_questions": plan.get("required_questions", []),
        }

    approved = yes or (False if quiet else _confirm("Submit this benchmark plan?"))
    if not approved:
        return {
            "status": "planned",
            "request_file": str(request_file),
            "plan_file": str(plan_file),
            "runbook_file": str(runbook_file),
            "preflight": preflight,
        }

    job = submit_job(plan_file, jobs_dir=output_path / "jobs", mock=mock, approved=approved)
    return {
        "status": "submitted",
        "request_file": str(request_file),
        "plan_file": str(plan_file),
        "runbook_file": str(runbook_file),
        "job": redact(job),
    }


def _apply_required_answers(
    request: dict[str, Any],
    plan: dict[str, Any],
    answers: dict[str, str],
    yes: bool = False,
) -> dict[str, Any]:
    updated = dict(request)
    confirmations = set(updated.get("confirmations", []))
    for question in plan.get("required_questions", []):
        qid = question["id"]
        if yes:
            _apply_yes_default(updated, confirmations, question)
            continue
        answer = answers.get(qid)
        if answer is None:
            if qid in {"mixed_weights_confirmation", "stress_execution_confirmation"}:
                if _confirm(question["prompt"]):
                    answer = "yes"
            else:
                answer = _ask(_question_label(question))
        _apply_answer(updated, confirmations, qid, answer or "")
    if confirmations:
        updated["confirmations"] = sorted(confirmations)
    return updated


def _apply_yes_default(updated: dict[str, Any], confirmations: set[str], question: dict[str, Any]) -> None:
    qid = question["id"]
    if qid == "ledger_device_confirmation":
        disks = updated.get("discovery", {}).get("disks", {})
        proposed = disks.get("proposed_ledger_device")
        if proposed:
            updated["ledger_device"] = proposed
            confirmations.add(qid)
    elif qid.endswith("_confirmation"):
        confirmations.add(qid)


def _apply_answer(updated: dict[str, Any], confirmations: set[str], qid: str, answer: str) -> None:
    answer = answer.strip()
    if not answer:
        return
    if qid == "chain":
        updated["chain"] = answer.lower()
    elif qid == "local_rpc_url":
        if answer.lower() in {"fake-node", "fake_node", "fake node"}:
            updated["use_fake_node"] = True
        else:
            updated["local_rpc_url"] = answer
    elif qid == "ledger_device_confirmation":
        updated["ledger_device"] = answer
        confirmations.add(qid)
    elif qid == "ledger_device":
        updated["ledger_device"] = answer
    elif qid == "blockchain_process_names":
        updated["blockchain_process_names"] = [item.strip() for item in answer.replace(",", " ").split() if item.strip()]
    elif qid == "data_vol_max_iops":
        updated["data_vol_max_iops"] = answer
    elif qid == "data_vol_max_throughput":
        updated["data_vol_max_throughput"] = answer
    elif qid == "network_max_bandwidth_gbps":
        updated["network_max_bandwidth_gbps"] = answer
    elif qid == "rpc_mode":
        if answer.lower() in {"single", "mixed"}:
            updated["rpc_mode"] = answer.lower()
    elif qid == "dependency_mode_confirmation":
        if answer in {"audit", "isolated", "managed"}:
            updated["dependency_mode"] = answer
            confirmations.add(qid)
    elif qid in {"mixed_weights_confirmation", "stress_execution_confirmation"}:
        if answer.lower() in {"y", "yes", "confirm", "confirmed", "true"}:
            confirmations.add(qid)


def _question_label(question: dict[str, Any]) -> str:
    label = question.get("prompt", question["id"])
    if question.get("candidates"):
        label += f" Candidates: {', '.join(question['candidates'])}"
    if question.get("missing"):
        label += f" Missing: {', '.join(question['missing'])}"
    return label


def _ask(label: str) -> str:
    print(f"{label}: ", end="", file=sys.stderr, flush=True)
    return sys.stdin.readline().strip()


def _confirm(label: str) -> bool:
    print(f"{label} [y/N]: ", end="", file=sys.stderr, flush=True)
    answer = sys.stdin.readline().strip().lower()
    return answer in {"y", "yes"}

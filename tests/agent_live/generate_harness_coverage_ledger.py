#!/usr/bin/env python3
"""Generate the Harness edge-coverage ledger from live runtime contracts.

Catalog discovery proves only that an edge exists.  Execution evidence is
tracked independently and is preserved only while the edge's stable contract
identity remains unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from tests.agent_live.coverage_evidence import (
    content_hash,
    load_valid_evidence_reference,
    repository_revision,
)
from tests.agent_live.harness_contract_scenarios import (
    action_transition_scenarios,
    manual_input_case,
    question_scenarios,
)
from agent.harness.domains.environment import CONFIRMABLE_CONFIG_FIELDS


EVIDENCE_CLASSES = (
    "catalog",
    "deterministic",
    "real_cli",
    "dynamic_dual_ai",
    "real_execution",
)
LEDGER_SCHEMA_VERSION = 5
EVIDENCE_STATUSES = (
    "not_run",
    "passed",
    "failed",
    "not_applicable",
    "externally_blocked",
)
LEGACY_EVIDENCE_FIELDS = {
    "deterministic": "deterministic_test_id",
    "real_cli": "fixed_cli_scenario_id",
    "dynamic_dual_ai": "dynamic_chaos_round_id",
    "real_execution": "real_execution_evidence",
}
MANUAL_INPUT_CLASSES = (
    "valid_literal",
    "invalid_literal",
    "empty_whitespace",
    "trimmed_whitespace_punctuation",
    "natural_language_answer",
    "multiline_prose",
    "structured_json_yaml_env_curl",
    "partial_request_response_evidence",
    "out_of_range_numeric",
    "reachable_url",
    "unreachable_or_mismatched_url",
)
REAL_EXECUTION_ACTIONS = frozenset({"approve_preflight_smoke", "approve_final_benchmark"})
SEMANTIC_COORDINATOR_ACTIONS = frozenset({
    "change_group",
    "go_back",
    "queue_workflow_goal",
    "activate_next_workflow_goal",
    "discard_next_workflow_goal",
})
RUNNER_CONTRACTS = {
    "deterministic": {
        "status": "implemented",
        "producer": "tests/agent_live/execute_harness_contract_ledger.py",
        "artifact_schema": "compiled_transition_evidence.v3",
        "gap": "",
    },
    "real_cli": {
        "status": "implemented",
        "producer": "tests/agent_live/execute_real_cli_contract_ledger.py",
        "artifact_schema": "real_cli_evidence.v4",
        "gap": "",
    },
    "dynamic_dual_ai": {
        "status": "implemented",
        "producer": "tests/agent_live/dynamic_dual_ai_chaos.py",
        "artifact_schema": "dynamic_chaos_transcript.v1",
        "gap": "",
    },
    "real_execution": {
        "status": "implemented",
        "producer": "tests/agent_live/execute_real_execution_ledger.py",
        "artifact_schema": "real_execution_evidence.v1",
        "gap": "",
    },
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def contract_variant_payload(question: Mapping[str, Any]) -> dict[str, Any]:
    """Return the language- and run-independent part of a question contract."""

    options = []
    for option in question.get("options") or []:
        action = option.get("action") if isinstance(option.get("action"), Mapping) else {}
        options.append({
            "id": str(option.get("id") or ""),
            "value": option.get("value"),
            "action": dict(action),
            "expected_patch": dict(option.get("expected_patch") or {}),
            "return_policy": str(option.get("return_policy") or "fallback"),
            "semantic_action": str(option.get("semantic_action") or ""),
        })
    return {
        "contract_version": question.get("contract_version"),
        "group": str(question.get("group") or ""),
        "question_id": str(question.get("id") or ""),
        "kind": str(question.get("kind") or ""),
        "field": str(question.get("field") or ""),
        "manual_input_allowed": bool(question.get("manual_input_allowed")),
        "accepted_action_types": sorted(str(item) for item in question.get("accepted_action_types") or []),
        "queue_barrier": bool(question.get("queue_barrier")),
        "validation": dict(question.get("validation") or {}),
        "requires_capabilities": sorted(str(item) for item in question.get("requires_capabilities") or []),
        "options": options,
    }


def contract_variant_hash(question: Mapping[str, Any]) -> str:
    payload = _canonical_json(contract_variant_payload(question)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def _edge_key(
    group: str,
    question_id: str,
    variant_hash: str,
    input_class: str,
    option_or_action: str,
) -> str:
    parts = (group, question_id, variant_hash, input_class, option_or_action)
    return "::".join(str(part).replace("::", "%3A%3A") for part in parts)


def evidence_applicability(
    edge: Mapping[str, Any],
    evidence_class: str,
) -> tuple[bool, str]:
    """Return the proof obligation for one evidence lane.

    Global edge applicability describes the product contract.  It is not a
    substitute for lane applicability: exact parsing belongs in deterministic
    and PTY lanes, semantic interpretation belongs in PTY/dynamic lanes, and a
    side effect is proved once by its action-level real-execution obligation.
    """

    if evidence_class not in EVIDENCE_CLASSES:
        raise ValueError(f"unknown evidence class: {evidence_class}")
    if evidence_class == "catalog":
        return True, "inventory row discovered from an authoritative runtime contract"
    if not edge.get("applicable", True):
        return False, f"product contract not applicable: {edge.get('applicability_reason') or 'unspecified'}"

    edge_type = str(edge.get("edge_type") or "")
    input_class = str(edge.get("input_class") or "")
    has_executable_scenario = bool(edge.get("executable_scenario_ids"))

    if evidence_class == "deterministic":
        if edge.get("real_execution_required") or edge.get("action_type") in REAL_EXECUTION_ACTIONS:
            return False, "side-effecting action is proved by the real_execution obligation"
        if edge_type == "action_transition":
            return False, "standalone registry transition has no reviewed state preconditions"
        if not has_executable_scenario:
            return False, "no reviewed executable seed exists for this contract variant"
        if input_class == "exact_option":
            return True, "exact typed option must execute through the compiled graph"
        if edge_type == "manual_input" and edge.get("deterministic_case_available"):
            behavior = "admission" if edge.get("expected_admitted") else "rejection"
            return True, f"reviewed local input case must prove deterministic {behavior}"
        return False, "semantic or integration input is not a deterministic single-turn proof"

    if evidence_class == "real_cli":
        if edge_type == "action_transition":
            return False, "action registry rows are not direct terminal inputs"
        if input_class == "empty_whitespace":
            return False, "prompt-toolkit ignores whitespace without submitting a workflow turn"
        return True, "real PTY must prove terminal transport and committed workflow behavior"

    if evidence_class == "dynamic_dual_ai":
        if (
            edge_type == "action_transition"
            and str(edge.get("action_type") or "") in SEMANTIC_COORDINATOR_ACTIONS
        ):
            return True, "response-driven simulator must prove semantic workflow control selection"
        semantic_classes = {
            "natural_language_option",
            "natural_language_answer",
            "multiline_prose",
            "structured_json_yaml_env_curl",
            "partial_request_response_evidence",
        }
        if input_class in semantic_classes:
            return True, "response-driven simulator must prove semantic interpretation in context"
        return False, "deterministic syntax or non-semantic registry behavior does not require an LLM simulator"

    if evidence_class == "real_execution":
        if edge_type == "action_transition" and edge.get("real_execution_required"):
            return True, "one action-level side-effect obligation avoids duplicating equivalent user expressions"
        if edge.get("real_execution_required"):
            return False, "linked action-level real-execution obligation is authoritative"
        return False, "edge has no external execution side effect"
    raise AssertionError("unreachable evidence class")


def _evidence(edge: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for evidence_class in EVIDENCE_CLASSES:
        required, reason = evidence_applicability(edge, evidence_class)
        result[evidence_class] = {
            "required": required,
            "applicability_reason": reason,
            "status": "passed" if evidence_class == "catalog" else (
                "not_run" if required else "not_applicable"
            ),
            "evidence_ids": [],
        }
    return result


def derive_overall_status(edge: Mapping[str, Any]) -> str:
    """Derive status without allowing catalog discovery to imply execution."""

    if not edge.get("applicable", True):
        return "not_applicable"
    statuses = [
        str((edge.get("evidence") or {}).get(name, {}).get("status") or "not_run")
        for name in EVIDENCE_CLASSES
        if name != "catalog"
        if bool((edge.get("evidence") or {}).get(name, {}).get("required"))
    ]
    if any(status == "failed" for status in statuses):
        return "failed"
    if any(status == "not_run" for status in statuses):
        return "not_run"
    if any(status == "externally_blocked" for status in statuses):
        return "externally_blocked"
    return "passed" if statuses and all(status == "passed" for status in statuses) else "not_run"


def _manual_class_applicability(question: Mapping[str, Any], input_class: str) -> tuple[bool, str]:
    validation = question.get("validation") if isinstance(question.get("validation"), Mapping) else {}
    value_type = str(validation.get("value_type") or "")
    kind = str(question.get("kind") or "")
    input_mode = str(validation.get("input_mode") or "")
    if input_class == "empty_whitespace":
        return False, "prompt-toolkit ignores whitespace without submitting a workflow turn"
    if input_class == "out_of_range_numeric":
        applies = value_type in {"positive_number", "positive_integer"}
        return applies, "numeric validation contract" if applies else "question is not numeric"
    if input_class in {"reachable_url", "unreachable_or_mismatched_url"}:
        applies = kind == "url"
        return applies, "URL input contract" if applies else "question is not a URL contract"
    if input_class == "partial_request_response_evidence":
        applies = kind == "evidence" or input_mode == "rpc_method_or_schema_evidence"
        return applies, "RPC/evidence contract" if applies else "question does not collect request/response evidence"
    return True, "required manual-input equivalence class"


def _question_contracts() -> list[dict[str, Any]]:
    variants: dict[tuple[str, str, str], dict[str, Any]] = {}
    for scenario in question_scenarios("en"):
        question = scenario.question
        scenario_id = scenario.scenario_id
        group = str(question.get("group") or "")
        question_id = str(question.get("id") or "")
        contract_hash = contract_variant_hash(question)
        variant_hash = content_hash({
            "contract_hash": contract_hash,
            "state_fingerprint": scenario.state_fingerprint,
            "scenario_id": scenario_id,
        })[:20]
        key = (group, question_id, variant_hash)
        accepted_action_types = {
            str(item).strip()
            for item in question.get("accepted_action_types") or ()
            if str(item).strip()
        }
        undeclared_manual_actions = {
            str(action_type).strip()
            for action_type in (scenario.manual_action_overrides or {}).values()
            if str(action_type).strip() not in accepted_action_types
        }
        if undeclared_manual_actions:
            raise ValueError(
                f"scenario {scenario_id} declares manual actions outside the "
                f"question contract: {sorted(undeclared_manual_actions)}"
            )
        if key not in variants:
            variants[key] = {
                "group": group,
                "question_id": question_id,
                "contract_variant_hash": variant_hash,
                "contract_hash": contract_hash,
                "state_fingerprint": scenario.state_fingerprint,
                "scenario_ids": [],
                "executable_scenario_ids": [],
                "manual_postcondition_path": "",
                "manual_next_question_ids": [],
                "option_postcondition_overrides": {},
                "option_relation_overrides": {},
                "manual_input_overrides": {},
                "manual_action_overrides": {},
                "contract": contract_variant_payload(question),
            }
        variants[key]["scenario_ids"].append(scenario_id)
        if scenario.executable:
            variants[key]["executable_scenario_ids"].append(scenario_id)
            variants[key]["manual_postcondition_path"] = scenario.manual_postcondition_path
            variants[key]["manual_next_question_ids"] = list(
                scenario.manual_next_question_ids
            )
            variants[key]["option_postcondition_overrides"] = deepcopy(
                dict(scenario.option_postcondition_overrides or {})
            )
            variants[key]["option_relation_overrides"] = deepcopy(
                dict(scenario.option_relation_overrides or {})
            )
            variants[key]["manual_input_overrides"] = deepcopy(
                dict(scenario.manual_input_overrides or {})
            )
            variants[key]["manual_action_overrides"] = deepcopy(
                dict(scenario.manual_action_overrides or {})
            )
    for variant in variants.values():
        variant["scenario_ids"] = sorted(set(variant["scenario_ids"]))
        variant["executable_scenario_ids"] = sorted(set(variant["executable_scenario_ids"]))
    return sorted(
        variants.values(),
        key=lambda item: (item["group"], item["question_id"], item["contract_variant_hash"]),
    )


def _new_edge(
    *,
    group: str,
    question_id: str,
    variant_hash: str,
    contract_hash: str = "",
    state_fingerprint: str = "",
    input_class: str,
    option_or_action: str,
    edge_type: str,
    owner: str,
    scenario_ids: Iterable[str] = (),
    option: Mapping[str, Any] | None = None,
    action_type: str = "",
    applicable: bool = True,
    applicability_reason: str = "",
    preconditions: Mapping[str, Any] | None = None,
    expected_postcondition: Mapping[str, Any] | None = None,
    expected_state_relations: Iterable[Mapping[str, Any]] = (),
    return_policy: str = "fallback",
    executable_scenario_ids: Iterable[str] = (),
    deterministic_case_available: bool = False,
    expected_admitted: bool | None = None,
    interrupts_pending_contract: bool = False,
) -> dict[str, Any]:
    execution_required = (
        edge_type == "action_transition" and action_type in REAL_EXECUTION_ACTIONS
    )
    edge = {
        "edge_key": _edge_key(group, question_id, variant_hash, input_class, option_or_action),
        "edge_type": edge_type,
        "group": group,
        "question_id": question_id,
        "contract_variant_hash": variant_hash,
        "contract_hash": contract_hash,
        "state_fingerprint": state_fingerprint,
        "input_class": input_class,
        "option_or_action": option_or_action,
        "option_id": str((option or {}).get("id") or ""),
        "option_value": (option or {}).get("value"),
        "action_type": action_type,
        "owner": owner,
        "preconditions": dict(preconditions or {}),
        "expected_postcondition": dict(expected_postcondition or {}),
        "expected_state_relations": [dict(item) for item in expected_state_relations],
        "return_policy": return_policy,
        "applicable": applicable,
        "applicability_reason": applicability_reason,
        "catalog_scenario_ids": sorted(set(scenario_ids)),
        "executable_scenario_ids": sorted(set(executable_scenario_ids)),
        "deterministic_case_available": bool(deterministic_case_available),
        "expected_admitted": expected_admitted,
        "interrupts_pending_contract": bool(interrupts_pending_contract),
        "execution_case_ids": [],
        "execution_case_hash": "",
        "deterministic_test_id": "",
        "fixed_cli_scenario_id": "",
        "dynamic_chaos_round_id": "",
        "real_execution_evidence": "",
        "real_execution_required": execution_required,
        "overall_status": "not_run",
    }
    edge["evidence"] = _evidence(edge)
    edge["evidence"]["catalog"]["evidence_ids"] = sorted(set(scenario_ids))
    edge["overall_status"] = derive_overall_status(edge)
    return edge


def _merge_existing_evidence(
    edges: list[dict[str, Any]],
    existing: Mapping[str, Any] | None,
    *,
    revision: Mapping[str, str],
) -> None:
    if existing and int(existing.get("schema_version") or 0) != LEDGER_SCHEMA_VERSION:
        return
    existing_by_key = {
        str(edge.get("edge_key") or ""): edge
        for edge in (existing or {}).get("edges") or []
        if edge.get("edge_key")
    }
    for edge in edges:
        previous = existing_by_key.get(edge["edge_key"])
        if not previous:
            continue
        previous_evidence = previous.get("evidence") or {}
        for evidence_class in EVIDENCE_CLASSES:
            if evidence_class == "catalog":
                continue
            old = previous_evidence.get(evidence_class) or {}
            status = str(old.get("status") or "")
            if status not in EVIDENCE_STATUSES:
                continue
            if edge["evidence"][evidence_class]["status"] == "not_applicable":
                continue
            if status == "not_applicable":
                continue
            references = [str(item) for item in old.get("evidence_ids") or []]
            valid_references: list[str] = []
            for reference in references:
                artifact, _reason = load_valid_evidence_reference(
                    reference,
                    edge=edge,
                    revision=revision,
                )
                artifact_outcome = str((artifact or {}).get("outcome") or "")
                if (
                    artifact
                    and artifact.get("evidence_class") == evidence_class
                    and artifact_outcome == status
                ):
                    valid_references.append(reference)
            if status in {"passed", "failed", "externally_blocked"} and not valid_references:
                continue
            edge["evidence"][evidence_class] = {
                "required": True,
                "applicability_reason": edge["evidence"][evidence_class]["applicability_reason"],
                "status": status,
                "evidence_ids": sorted(set(valid_references)),
            }
            legacy_field = LEGACY_EVIDENCE_FIELDS[evidence_class]
            edge[legacy_field] = valid_references[-1] if valid_references else ""
        edge["overall_status"] = derive_overall_status(edge)


def _aggregate_status(edges: Iterable[Mapping[str, Any]]) -> str:
    statuses = [str(edge.get("overall_status") or "not_run") for edge in edges]
    applicable = [status for status in statuses if status != "not_applicable"]
    if not applicable:
        return "not_applicable"
    if any(status == "failed" for status in applicable):
        return "failed"
    if any(status == "not_run" for status in applicable):
        return "not_run"
    if any(status == "externally_blocked" for status in applicable):
        return "externally_blocked"
    return "passed" if all(status == "passed" for status in applicable) else "not_run"


def refresh_ledger_status(ledger: dict[str, Any]) -> dict[str, Any]:
    """Recompute derived views after an evidence executor updates edge rows."""

    edges = list(ledger.get("edges") or [])
    edge_by_key = {str(edge.get("edge_key") or ""): edge for edge in edges}
    for edge in edges:
        edge["overall_status"] = derive_overall_status(edge)
    for group in ledger.get("groups") or []:
        group_edges = [edge for edge in edges if edge.get("group") == group.get("group")]
        group["status"] = _aggregate_status(group_edges)
        for question in group.get("questions") or []:
            question_edges = [edge_by_key[key] for key in question.get("edge_keys") or [] if key in edge_by_key]
            question["status"] = _aggregate_status(question_edges)
            for option in question.get("options") or []:
                option_edges = [edge_by_key[key] for key in option.get("edge_keys") or [] if key in edge_by_key]
                option["status"] = _aggregate_status(option_edges)
    for action in ledger.get("actions") or []:
        edge = edge_by_key.get(str(action.get("edge_key") or ""))
        if edge:
            action["status"] = edge["overall_status"]
            action["deterministic_test_id"] = edge.get("deterministic_test_id") or ""
            action["fixed_cli_scenario_id"] = edge.get("fixed_cli_scenario_id") or ""
            action["dynamic_chaos_round_id"] = edge.get("dynamic_chaos_round_id") or ""
            action["real_execution_evidence"] = edge.get("real_execution_evidence") or ""
    summary = ledger.setdefault("summary", {})
    summary["overall_status"] = _aggregate_status(edges)
    summary["evidence_status_counts"] = {
        evidence_class: {
            status: sum(
                1 for edge in edges
                if edge["evidence"][evidence_class]["status"] == status
            )
            for status in EVIDENCE_STATUSES
        }
        for evidence_class in EVIDENCE_CLASSES
    }
    summary["execution_closure"] = {
        evidence_class: _execution_closure(edges, evidence_class)
        for evidence_class in EVIDENCE_CLASSES
        if evidence_class != "catalog"
    }
    return ledger


def ingest_evidence_artifacts(
    ledger: Mapping[str, Any],
    references: Iterable[str | Path],
) -> dict[str, Any]:
    """Atomically attach validated execution artifacts to their ledger lanes.

    Evidence producers do not own the coverage ledger.  This boundary resolves
    each artifact against the authoritative edge inventory and validates the
    complete artifact before any derived ledger state is changed.
    """

    candidate = deepcopy(dict(ledger))
    revision = dict(candidate.get("revision") or {})
    if not str(revision.get("commit") or "").strip() or not str(
        revision.get("worktree_hash") or ""
    ).strip():
        raise ValueError("coverage ledger has no repository revision identity")
    edge_by_key = {
        str(edge.get("edge_key") or ""): edge
        for edge in candidate.get("edges") or ()
        if str(edge.get("edge_key") or "")
    }
    normalized = [str(Path(reference).resolve()) for reference in references]
    if not normalized:
        raise ValueError("no evidence artifacts were provided")

    validated: list[tuple[dict[str, Any], str, str, str]] = []
    batch_outcomes: dict[tuple[str, str], str] = {}
    for reference in normalized:
        try:
            raw = json.loads(Path(reference).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load evidence artifact {reference}: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise ValueError(f"evidence artifact is not an object: {reference}")
        edge_key = str(raw.get("edge_key") or "")
        edge = edge_by_key.get(edge_key)
        if edge is None:
            raise ValueError(f"evidence artifact references an unknown edge: {edge_key or '<empty>'}")
        evidence_class = str(raw.get("evidence_class") or "")
        if evidence_class == "catalog" or evidence_class not in EVIDENCE_CLASSES:
            raise ValueError(f"evidence artifact has an invalid execution lane: {evidence_class or '<empty>'}")
        lane = (edge.get("evidence") or {}).get(evidence_class) or {}
        if not bool(lane.get("required")):
            raise ValueError(
                f"edge is not required for {evidence_class}: "
                f"{lane.get('applicability_reason') or 'unspecified'}"
            )
        artifact, reason = load_valid_evidence_reference(
            reference,
            edge=edge,
            revision=revision,
        )
        if artifact is None:
            raise ValueError(f"invalid evidence artifact {reference}: {reason}")
        outcome = str(artifact.get("outcome") or "")
        if outcome not in {"passed", "failed", "externally_blocked"}:
            raise ValueError(f"unsupported evidence outcome: {outcome or '<empty>'}")
        key = (edge_key, evidence_class)
        previous_batch_outcome = batch_outcomes.setdefault(key, outcome)
        if previous_batch_outcome != outcome:
            raise ValueError(
                f"conflicting evidence outcomes for {edge_key} [{evidence_class}]: "
                f"{previous_batch_outcome} vs {outcome}"
            )
        current_status = str(lane.get("status") or "not_run")
        if current_status not in {"not_run", outcome}:
            raise ValueError(
                f"evidence outcome conflicts with the ledger for {edge_key} "
                f"[{evidence_class}]: {current_status} vs {outcome}"
            )
        validated.append((edge, evidence_class, outcome, reference))

    for edge, evidence_class, outcome, reference in validated:
        lane = edge["evidence"][evidence_class]
        lane["status"] = outcome
        lane["evidence_ids"] = sorted(set([
            *(str(item) for item in lane.get("evidence_ids") or ()),
            reference,
        ]))
        edge[LEGACY_EVIDENCE_FIELDS[evidence_class]] = lane["evidence_ids"][-1]
    return refresh_ledger_status(candidate)


def _execution_closure(
    edges: Iterable[Mapping[str, Any]],
    evidence_class: str,
) -> dict[str, Any]:
    required = [
        edge for edge in edges
        if bool((edge.get("evidence") or {}).get(evidence_class, {}).get("required"))
    ]
    counts = {
        status: sum(
            str((edge.get("evidence") or {}).get(evidence_class, {}).get("status") or "not_run")
            == status
            for edge in required
        )
        for status in ("passed", "failed", "not_run", "externally_blocked")
    }
    if counts["failed"]:
        status = "failed"
    elif counts["not_run"] or counts["externally_blocked"]:
        status = "incomplete"
    else:
        status = "complete"
    return {
        "required_denominator": len(required),
        "observed_pass": counts["passed"],
        "observed_fail": counts["failed"],
        "not_run": counts["not_run"],
        "externally_blocked": counts["externally_blocked"],
        "open_required": counts["failed"] + counts["not_run"] + counts["externally_blocked"],
        "status": status,
    }


def execution_exit_code(ledger: Mapping[str, Any], evidence_class: str) -> int:
    """Return nonzero until every required row for the runner is observed-pass."""

    closure = _execution_closure(ledger.get("edges") or (), evidence_class)
    if closure["status"] == "failed":
        return 1
    if closure["status"] == "incomplete":
        return 2
    return 0


def build_ledger(
    existing: Mapping[str, Any] | None = None,
    *,
    revision: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    from agent.harness.action_registry import ACTION_SPECS
    from agent.workflows.group_registry import GROUPS

    group_by_name = {group.name: group for group in GROUPS}
    active_revision = dict(revision or repository_revision(REPO_ROOT))
    registered_questions = {
        (group.name, question_id)
        for group in GROUPS
        for question_id in group.questions
    }
    variants = _question_contracts()
    discovered_questions = {(item["group"], item["question_id"]) for item in variants}
    runtime_only_questions = sorted(discovered_questions - registered_questions)
    uncataloged_questions = sorted(registered_questions - discovered_questions)

    edges: list[dict[str, Any]] = []
    question_views: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for variant in variants:
        group = variant["group"]
        owner = str(getattr(group_by_name.get(group), "owner", "runtime"))
        contract = variant["contract"]
        variant_edges: list[dict[str, Any]] = []
        for option in contract.get("options") or []:
            action_type = str((option.get("action") or {}).get("type") or "answer_pending")
            postcondition_overrides = variant.get("option_postcondition_overrides") or {}
            option_id = str(option.get("id") or "")
            for input_class in ("exact_option", "natural_language_option"):
                variant_edges.append(_new_edge(
                    group=group,
                    question_id=variant["question_id"],
                    variant_hash=variant["contract_variant_hash"],
                    contract_hash=variant["contract_hash"],
                    state_fingerprint=variant["state_fingerprint"],
                    input_class=input_class,
                    option_or_action=f"option:{option.get('id')}",
                    edge_type="question_option",
                    owner=owner,
                    scenario_ids=variant["scenario_ids"],
                    executable_scenario_ids=variant["executable_scenario_ids"],
                    option=option,
                    action_type=action_type,
                    expected_postcondition=(
                        postcondition_overrides[option_id]
                        if option_id in postcondition_overrides
                        else option.get("expected_patch") or {}
                    ),
                    expected_state_relations=(
                        (variant.get("option_relation_overrides") or {}).get(
                            str(option.get("id") or "")
                        )
                        or ()
                    ),
                    return_policy=str(option.get("return_policy") or "fallback"),
                    applicability_reason="visible option contract",
                ))
        if contract.get("manual_input_allowed"):
            for input_class in MANUAL_INPUT_CLASSES:
                applicable, reason = _manual_class_applicability(contract, input_class)
                concrete_case = (
                    (variant.get("manual_input_overrides") or {}).get(input_class)
                    or manual_input_case(contract, input_class)
                )
                field = str(contract.get("field") or "")
                structured_config_interrupt = (
                    input_class == "structured_json_yaml_env_curl"
                    and field.upper() in CONFIRMABLE_CONFIG_FIELDS
                )
                action_type = (
                    "propose_config_values"
                    if structured_config_interrupt
                    else str(
                        (variant.get("manual_action_overrides") or {}).get(input_class)
                        or "answer_pending"
                    )
                )
                postcondition_path = (
                    f"inferred_config.pending_review.config_values.{field.upper()}"
                    if structured_config_interrupt
                    else variant.get("manual_postcondition_path")
                    or f"confirmed_config.{field}"
                )
                next_question_ids = (
                    ["inferred_config_review"]
                    if structured_config_interrupt
                    else list(variant.get("manual_next_question_ids") or [])
                )
                variant_edges.append(_new_edge(
                    group=group,
                    question_id=variant["question_id"],
                    variant_hash=variant["contract_variant_hash"],
                    contract_hash=variant["contract_hash"],
                    state_fingerprint=variant["state_fingerprint"],
                    input_class=input_class,
                    option_or_action=f"action:{action_type}",
                    edge_type="manual_input",
                    owner=owner,
                    scenario_ids=variant["scenario_ids"],
                    executable_scenario_ids=variant["executable_scenario_ids"],
                    action_type=action_type,
                    applicable=applicable,
                    applicability_reason=reason,
                    preconditions={"validation": contract.get("validation") or {}},
                    expected_postcondition={
                        "field": field,
                        "path": postcondition_path,
                        "next_question_ids": next_question_ids,
                    },
                    deterministic_case_available=concrete_case is not None,
                    expected_admitted=(
                        concrete_case.expected_admitted if concrete_case is not None else None
                    ),
                    interrupts_pending_contract=structured_config_interrupt,
                ))
        edges.extend(variant_edges)
        option_views = []
        for option in contract.get("options") or []:
            option_edge_keys = [
                edge["edge_key"]
                for edge in variant_edges
                if edge["option_id"] == str(option.get("id") or "")
            ]
            option_views.append({
                **deepcopy(option),
                "edge_keys": option_edge_keys,
                "deterministic_test_id": "",
                "fixed_cli_scenario_id": "",
                "dynamic_chaos_round_id": "",
                "real_execution_evidence": "",
                "status": _aggregate_status(
                    edge for edge in variant_edges if edge["edge_key"] in option_edge_keys
                ),
            })
        question_views[group].append({
            "question_id": variant["question_id"],
            "owner": owner,
            "registered": (group, variant["question_id"]) in registered_questions,
            "runtime_only": (group, variant["question_id"]) in runtime_only_questions,
            "contract_variant_hash": variant["contract_variant_hash"],
            "scenario_ids": variant["scenario_ids"],
            "executable_scenario_ids": variant["executable_scenario_ids"],
            "scenario_id": variant["scenario_ids"][0] if variant["scenario_ids"] else "",
            "cataloged_runtime_scenario": True,
            "kind": contract.get("kind"),
            "manual_input_allowed": bool(contract.get("manual_input_allowed")),
            "options": option_views,
            "edge_keys": [edge["edge_key"] for edge in variant_edges],
            "status": _aggregate_status(variant_edges),
        })

    action_scenario_ids: dict[str, list[str]] = defaultdict(list)
    for scenario in action_transition_scenarios("en"):
        action_scenario_ids[scenario.action_type].append(scenario.scenario_id)

    action_rows: list[dict[str, Any]] = []
    for spec in ACTION_SPECS:
        action_group = spec.target_group or f"@action_only/{spec.owner}"
        action_hash = hashlib.sha256(_canonical_json({
            "action_type": spec.action_type,
            "owner": spec.owner,
            "arguments": list(spec.arguments),
            "execution_phase": spec.execution_phase,
            "target_group": spec.target_group,
            "merge_identity": list(spec.merge_identity),
            "preserve_pending": spec.preserve_pending,
            "mutation_dimension": spec.mutation_dimension,
            "allows_followup_actions": spec.allows_followup_actions,
            "requires_capabilities": list(spec.requires_capabilities),
            "provides_capabilities": list(spec.provides_capabilities),
            "suppressed_by": list(spec.suppressed_by),
        }).encode("utf-8")).hexdigest()
        action_contract_hash = action_hash
        action_variant_hash = content_hash({
            "contract_hash": action_contract_hash,
            "runtime_variant": "action-registry",
        })[:20]
        edge = _new_edge(
            group=action_group,
            question_id="",
            variant_hash=action_variant_hash,
            contract_hash=action_contract_hash,
            state_fingerprint="action-registry",
            input_class="action_only_transition",
            option_or_action=f"action:{spec.action_type}",
            edge_type="action_transition",
            owner=spec.owner,
            action_type=spec.action_type,
            applicability_reason="accepted action registry transition",
            preconditions={"requires_capabilities": list(spec.requires_capabilities)},
            expected_postcondition={
                "target_group": spec.target_group,
                "provides_capabilities": list(spec.provides_capabilities),
            },
            scenario_ids=action_scenario_ids.get(spec.action_type, ()),
            executable_scenario_ids=action_scenario_ids.get(spec.action_type, ()),
        )
        edges.append(edge)
        action_rows.append({
            "action_type": spec.action_type,
            "owner": spec.owner,
            "purpose": spec.purpose,
            "arguments": list(spec.arguments),
            "target_group": spec.target_group,
            "precondition_evidence": "",
            "postcondition_evidence": "",
            "return_policy_evidence": "",
            "deterministic_test_id": "",
            "fixed_cli_scenario_id": "",
            "dynamic_chaos_round_id": "",
            "real_execution_evidence": "",
            "edge_key": edge["edge_key"],
            "status": edge["overall_status"],
        })

    from tests.agent_live.reviewed_execution_cases import bind_reviewed_execution_case

    for edge in edges:
        bind_reviewed_execution_case(edge)
        edge["evidence"] = _evidence(edge)
        edge["evidence"]["catalog"]["evidence_ids"] = list(edge["catalog_scenario_ids"])
        edge["overall_status"] = derive_overall_status(edge)
        deterministic_required = bool(edge["evidence"]["deterministic"]["required"])
        fixed_real_cli_required = (
            bool(edge["evidence"]["real_cli"]["required"])
            and not bool(edge["evidence"]["dynamic_dual_ai"]["required"])
        )
        if (deterministic_required or fixed_real_cli_required) and len(edge["execution_case_ids"]) != 1:
            raise RuntimeError(
                "required execution edge does not have exactly one reviewed case: "
                f"{edge['edge_key']}"
            )

    edges.sort(key=lambda edge: edge["edge_key"])
    _merge_existing_evidence(edges, existing, revision=active_revision)
    edge_by_key = {edge["edge_key"]: edge for edge in edges}
    for variants_for_group in question_views.values():
        for question in variants_for_group:
            question["status"] = _aggregate_status(edge_by_key[key] for key in question["edge_keys"])
            for option in question["options"]:
                option["status"] = _aggregate_status(
                    edge_by_key[key] for key in option["edge_keys"]
                )
    for action in action_rows:
        edge = edge_by_key[action["edge_key"]]
        action["status"] = edge["overall_status"]
        for legacy_field in (
            "deterministic_test_id",
            "fixed_cli_scenario_id",
            "dynamic_chaos_round_id",
            "real_execution_evidence",
        ):
            action[legacy_field] = edge[legacy_field]

    groups = []
    for group in GROUPS:
        group_edges = [edge for edge in edges if edge["group"] == group.name]
        groups.append({
            "group": group.name,
            "owner": group.owner,
            "fields": list(group.fields),
            "registered_question_ids": list(group.questions),
            "questions": sorted(
                question_views.get(group.name, []),
                key=lambda question: (question["question_id"], question["contract_variant_hash"]),
            ),
            "status": _aggregate_status(group_edges),
        })

    evidence_status_counts = {
        evidence_class: {
            status: sum(
                1 for edge in edges
                if edge["evidence"][evidence_class]["status"] == status
            )
            for status in EVIDENCE_STATUSES
        }
        for evidence_class in EVIDENCE_CLASSES
    }
    distinct_options = {
        (variant["group"], variant["question_id"], variant["contract_variant_hash"], str(option.get("id") or ""))
        for variant in variants
        for option in variant["contract"].get("options") or []
    }
    summary = {
        "groups": len(GROUPS),
        "registered_questions": len(registered_questions),
        "cataloged_questions": len(discovered_questions),
        "question_contract_variants": len(variants),
        "runtime_only_questions": len(runtime_only_questions),
        "rendered_options": len(distinct_options),
        "manual_input_contract_variants": sum(
            1 for variant in variants if variant["contract"].get("manual_input_allowed")
        ),
        "manual_input_equivalence_rows": sum(edge["edge_type"] == "manual_input" for edge in edges),
        "actions": len(ACTION_SPECS),
        "action_only_transitions": sum(edge["edge_type"] == "action_transition" for edge in edges),
        "edges": len(edges),
        "inventory_denominator": len(edges),
        "behavior_applicable": sum(bool(edge.get("applicable", True)) for edge in edges),
        "uncataloged_questions": len(uncataloged_questions),
        "overall_status": _aggregate_status(edges),
        "evidence_status_counts": evidence_status_counts,
        "execution_closure": {
            evidence_class: _execution_closure(edges, evidence_class)
            for evidence_class in EVIDENCE_CLASSES
            if evidence_class != "catalog"
        },
    }
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "source": "live question factories + runtime-only factories + group/action registries",
        "revision": active_revision,
        "status_values": list(EVIDENCE_STATUSES),
        "evidence_classes": list(EVIDENCE_CLASSES),
        "manual_input_classes": list(MANUAL_INPUT_CLASSES),
        "runner_contracts": deepcopy(RUNNER_CONTRACTS),
        "stable_edge_identity": "group + question_id + contract_variant_hash + input_class + option/action",
        "summary": summary,
        "uncataloged_questions": [f"{group}:{question}" for group, question in uncataloged_questions],
        "runtime_only_questions": [f"{group}:{question}" for group, question in runtime_only_questions],
        "groups": groups,
        "actions": action_rows,
        "edges": edges,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / ".agent" / "evidence" / "harness-coverage-ledger.json",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        nargs="+",
        default=(),
        help="validated execution artifacts to ingest atomically after catalog refresh",
    )
    args = parser.parse_args()
    existing: Mapping[str, Any] | None = None
    if args.output.exists():
        try:
            loaded = json.loads(args.output.read_text(encoding="utf-8"))
            existing = loaded if isinstance(loaded, Mapping) else None
        except (OSError, json.JSONDecodeError):
            existing = None
    ledger = build_ledger(existing)
    if args.evidence:
        ledger = ingest_evidence_artifacts(ledger, args.evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(ledger["summary"], sort_keys=True))
    if ledger["uncataloged_questions"]:
        print("uncataloged runtime scenarios:")
        for item in ledger["uncataloged_questions"]:
            print(f"- {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

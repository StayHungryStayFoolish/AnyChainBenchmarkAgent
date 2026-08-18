#!/usr/bin/env python3
"""Execute reviewed contract scenarios through the compiled product graph.

Every row declared required for the deterministic lane must produce a
revision-bound artifact. A missing seed/input/verifier is a ledger
inconsistency, never a silent ``not_run``.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from tests.agent_live.coverage_events import capture_coverage_events, observe_compiled_graph_turn
from agent.harness.invariants import validate_state
from agent.harness.questions import normalize_scalar
from agent.harness.secret_refs import materialize_state_secret_references
from tests.agent_live.coverage_evidence import (
    COMPILED_GRAPH_RUNNER,
    build_evidence_artifact,
    content_hash,
    repository_revision,
    write_evidence_artifact,
)
from tests.agent_live.generate_harness_coverage_ledger import (
    build_ledger,
    derive_overall_status,
    execution_exit_code,
    refresh_ledger_status,
)
from tests.agent_live.graph_turn import (
    invoke_product_graph_turn,
    reviewed_execution_planner,
)
from tests.agent_live.harness_contract_scenarios import QuestionScenario
from tests.agent_live.reviewed_execution_cases import reviewed_execution_case


def execute_ledger(
    existing: Mapping[str, Any] | None = None,
    *,
    artifact_dir: str | Path,
    revision: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    active_revision = dict(revision or repository_revision(REPO_ROOT))
    ledger = build_ledger(existing, revision=active_revision)
    required_edges = [
        edge for edge in ledger["edges"]
        if bool((edge.get("evidence") or {}).get("deterministic", {}).get("required"))
    ]
    for edge in required_edges:
        execution = reviewed_execution_case(edge)
        if execution is None:
            raise RuntimeError(
                "deterministic ledger inconsistency: required edge has no executable case: "
                f"{edge.get('edge_key')}"
            )
        scenario, execution_case = execution
        input_value = execution_case.resolve_input()
        expected_admitted = execution_case.expected_admitted
        if edge.get("real_execution_required"):
            raise RuntimeError(
                f"deterministic ledger incorrectly requires a side effect: {edge.get('edge_key')}"
            )
        seed_state = deepcopy(dict(scenario.seed_state or {}))
        before_state = _prepare_question_turn(scenario, input_value)
        after_state: Mapping[str, Any] = before_state
        outcome = "passed"
        exit_status = 0
        error = ""
        with capture_coverage_events() as events:
            try:
                with reviewed_execution_planner(
                    expected_input=input_value,
                    expected_admitted=expected_admitted,
                    reviewed_value=execution_case.expected_value,
                ):
                    observation = observe_compiled_graph_turn(
                        invoke_product_graph_turn,
                        before_state,
                        edge_key=str(edge["edge_key"]),
                        input_value=input_value,
                        component=COMPILED_GRAPH_RUNNER,
                    )
                after_state = observation.after
                validate_state(dict(after_state))
                _verify_postcondition(
                    edge,
                    input_value,
                    before_state,
                    after_state,
                    expected_admitted=expected_admitted,
                )
            except Exception as exc:
                outcome = "failed"
                exit_status = 1
                error = f"{type(exc).__name__}: {exc}"
        artifact = build_evidence_artifact(
            edge=edge,
            evidence_class="deterministic",
            scenario_id=scenario.scenario_id,
            runner_type=COMPILED_GRAPH_RUNNER,
            revision=active_revision,
            input_value=input_value,
            seed_state=seed_state,
            before_state=before_state,
            after_state=after_state,
            events=[event.as_dict() for event in events],
            exit_status=exit_status,
            outcome=outcome,
            error=error,
            admitted=expected_admitted,
        )
        artifact_path = write_evidence_artifact(artifact, artifact_dir).resolve()
        edge["evidence"]["deterministic"] = {
            "required": True,
            "applicability_reason": edge["evidence"]["deterministic"]["applicability_reason"],
            "status": outcome,
            "evidence_ids": [str(artifact_path)],
        }
        edge["deterministic_test_id"] = str(artifact_path)
        edge["overall_status"] = derive_overall_status(edge)
    return refresh_ledger_status(ledger)


def _prepare_question_turn(scenario: QuestionScenario, input_value: str) -> Mapping[str, Any]:
    state = deepcopy(dict(scenario.seed_state or {}))
    state["pending_question"] = deepcopy(dict(scenario.question))
    state["active_group"] = str(scenario.question.get("group") or state.get("active_group") or "opening")
    state["last_user_input"] = input_value
    return state


def _verify_postcondition(
    edge: Mapping[str, Any],
    input_value: Any,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    expected_admitted: bool,
) -> None:
    if not expected_admitted:
        _verify_rejection(edge, before, after)
        return
    if edge.get("edge_type") == "manual_input":
        expected = dict(edge.get("expected_postcondition") or {})
        field = str(expected.get("field") or "")
        if not field:
            raise AssertionError("manual edge has no expected field")
        path = str(expected.get("path") or f"confirmed_config.{field}")
        actual = _read_path(after, path)
        actual = materialize_state_secret_references(actual, after)
        if "value" in expected:
            if actual != expected["value"]:
                raise AssertionError(f"manual postcondition {path} mismatch: {actual!r}")
            return
        if normalize_scalar(str(actual)) != normalize_scalar(str(input_value)):
            raise AssertionError(f"manual postcondition {path} was not applied: {actual!r}")
        return
    expected = dict(edge.get("expected_postcondition") or {})
    relations = tuple(edge.get("expected_state_relations") or ())
    if edge.get("edge_type") != "question_option":
        raise AssertionError("unsupported deterministic edge type")
    _verify_state_relations(relations, before, after)
    if expected:
        mismatches = {
            path: {"expected": value, "actual": _read_path(after, path)}
            for path, value in expected.items()
            if _read_path(after, path) != value
        }
        if mismatches:
            raise AssertionError(f"option postcondition mismatch: {mismatches}")
    elif edge.get("return_policy") == "stop_after_response":
        if not after.get("visible_response"):
            raise AssertionError("response option produced no visible response")
        if after.get("pending_question"):
            raise AssertionError("response option did not consume its pending question")
    elif relations:
        pass
    else:
        raise AssertionError("option edge has neither a state nor response postcondition")
    if dict(before) == dict(after):
        raise AssertionError("compiled graph turn did not change state")


def _verify_state_relations(
    relations: tuple[Mapping[str, Any], ...],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> None:
    for relation in relations:
        kind = str(relation.get("kind") or "")
        if kind == "path_absent_after":
            path = str(relation.get("path") or "")
            if not path or _path_exists(after, path):
                raise AssertionError(f"state relation path still exists: {relation}")
            continue
        if kind == "path_equals_before_path":
            after_path = str(relation.get("after_path") or "")
            before_path = str(relation.get("before_path") or "")
            if not _path_exists(after, after_path) or not _path_exists(before, before_path):
                raise AssertionError(f"state relation path is absent: {relation}")
            actual = _read_path(after, after_path)
            expected = _read_path(before, before_path)
            if actual != expected:
                raise AssertionError(f"state relation mismatch: {relation}")
            continue
        if kind == "prefix_equals_before_prefix":
            after_prefix = str(relation.get("after_prefix") or "")
            before_prefix = str(relation.get("before_prefix") or "")
            if not _path_exists(after, after_prefix) or not _path_exists(before, before_prefix):
                raise AssertionError(f"state relation prefix is absent: {relation}")
            actual = _read_path(after, after_prefix)
            expected = _read_path(before, before_prefix)
            ignored = {str(item) for item in relation.get("ignored_suffixes") or ()}
            if isinstance(actual, Mapping) and isinstance(expected, Mapping):
                actual = {key: value for key, value in actual.items() if str(key) not in ignored}
                expected = {key: value for key, value in expected.items() if str(key) not in ignored}
            if actual != expected:
                raise AssertionError(f"state relation mismatch: {relation}")
            continue
        raise AssertionError(f"unsupported state relation: {relation}")


def _verify_rejection(
    edge: Mapping[str, Any],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> None:
    """Prove invalid input was rejected without consuming or mutating the field."""

    before_question = dict(before.get("pending_question") or {})
    after_question = dict(after.get("pending_question") or {})
    if not before_question or after_question.get("id") != before_question.get("id"):
        raise AssertionError("rejected input did not preserve the pending question")
    expected = dict(edge.get("expected_postcondition") or {})
    path = str(expected.get("path") or "")
    if path and _read_path(before, path) != _read_path(after, path):
        raise AssertionError(f"rejected input mutated protected path: {path}")
    responses = [str(item).strip() for item in after.get("visible_response") or []]
    if not any(responses):
        raise AssertionError("rejected input produced no explicit user-visible error")


def _read_path(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in str(path).split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _path_exists(value: Mapping[str, Any], path: str) -> bool:
    if not path:
        return False
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False
        current = current[part]
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=".agent/evidence/harness-coverage-ledger.json")
    parser.add_argument("--artifact-dir", default=".agent/evidence/harness-contract-artifacts")
    args = parser.parse_args()
    path = Path(args.output)
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    ledger = execute_ledger(existing, artifact_dir=args.artifact_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(ledger, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(ledger["summary"], ensure_ascii=False, sort_keys=True))
    return execution_exit_code(ledger, "deterministic")


if __name__ == "__main__":
    raise SystemExit(main())

"""Reproducible constrained covering arrays with explicit denominators."""

from __future__ import annotations

import itertools
import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


FACTOR_MODEL_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Factor:
    name: str
    values: tuple[str, ...]


@dataclass(frozen=True)
class ForbiddenCombination:
    constraint_id: str
    values: tuple[tuple[str, str], ...]

    @classmethod
    def from_mapping(
        cls,
        constraint_id: str,
        values: Mapping[str, str],
    ) -> "ForbiddenCombination":
        return cls(
            constraint_id=str(constraint_id),
            values=tuple(
                sorted((str(name), str(value)) for name, value in values.items())
            ),
        )


@dataclass(frozen=True)
class FactorModel:
    model_id: str
    version: int
    factors: tuple[Factor, ...]
    forbidden: tuple[ForbiddenCombination, ...] = ()


@dataclass(frozen=True)
class CoveringArrayResult:
    model_id: str
    model_version: int
    strength: int
    seed: int
    rows: tuple[dict[str, str], ...]
    row_ids: tuple[str, ...]
    report: dict[str, Any]


@dataclass(frozen=True)
class CoveringRowObservation:
    row_id: str
    revision: Mapping[str, str]
    outcome: str
    evidence_ids: tuple[str, ...]
    observed_row: Mapping[str, str]


TupleKey = tuple[tuple[str, str], ...]


def generate_covering_array(
    model: FactorModel,
    *,
    strength: int = 2,
    seed: int = 0,
    max_rows: int | None = None,
) -> CoveringArrayResult:
    """Greedily cover feasible tuples with deterministic seeded tie-breaking."""

    _validate_model(model)
    _validate_strength(model, strength)
    if max_rows is not None and max_rows < 0:
        raise ValueError("max_rows must be non-negative")

    candidates = list(feasible_assignments(model))
    feasible = _feasible_tuples(model, candidates, strength)
    uncovered = set(feasible)
    selected: list[dict[str, str]] = []
    rng = random.Random(seed)

    while uncovered and candidates and (max_rows is None or len(selected) < max_rows):
        scored = [
            (len(_row_tuples(model, row, strength) & uncovered), index, row)
            for index, row in enumerate(candidates)
        ]
        best_score = max(score for score, _, _ in scored)
        if best_score <= 0:
            break
        best = [(index, row) for score, index, row in scored if score == best_score]
        index, row = best[rng.randrange(len(best))]
        selected.append(dict(row))
        uncovered.difference_update(_row_tuples(model, row, strength))
        candidates.pop(index)

    row_ids = tuple(
        covering_row_id(model, row, strength=strength, seed=seed)
        for row in selected
    )
    report = covering_array_report(model, selected, strength=strength)
    report.update({
        "seed": seed,
        "generated_row_ids": list(row_ids),
        "execution_status": "not_run",
        "observed_row_count": 0,
        "observed_pass_row_count": 0,
        "observed_fail_row_count": 0,
        "observed_tuple_count": 0,
        "observed_uncovered_tuple_count": report["feasible_tuple_denominator"],
    })
    return CoveringArrayResult(
        model_id=model.model_id,
        model_version=model.version,
        strength=strength,
        seed=seed,
        rows=tuple(selected),
        row_ids=row_ids,
        report=report,
    )


def covering_array_report(
    model: FactorModel,
    rows: Iterable[Mapping[str, str]],
    *,
    strength: int = 2,
) -> dict[str, Any]:
    """Report theoretical, infeasible, covered, and uncovered tuple sets."""

    _validate_model(model)
    _validate_strength(model, strength)
    feasible_rows = list(feasible_assignments(model))
    theoretical = _theoretical_tuples(model, strength)
    feasible = _feasible_tuples(model, feasible_rows, strength)
    infeasible = theoretical - feasible

    valid_rows: list[dict[str, str]] = []
    invalid_rows: list[dict[str, Any]] = []
    for index, raw_row in enumerate(rows):
        row = {str(name): str(value) for name, value in raw_row.items()}
        reason = _row_invalid_reason(model, row)
        if reason:
            invalid_rows.append({"row_index": index, "row": row, "reason": reason})
        else:
            valid_rows.append(row)

    covered: set[TupleKey] = set()
    for row in valid_rows:
        covered.update(_row_tuples(model, row, strength) & feasible)
    uncovered = feasible - covered
    denominator = len(feasible)
    return {
        "schema_version": FACTOR_MODEL_SCHEMA_VERSION,
        "model_id": model.model_id,
        "model_version": model.version,
        "strength": strength,
        "generated_row_count": len(valid_rows),
        "invalid_rows": invalid_rows,
        "theoretical_tuple_count": len(theoretical),
        "feasible_tuple_denominator": denominator,
        "infeasible_tuple_count": len(infeasible),
        "generated_tuple_count": len(covered),
        "generated_uncovered_tuple_count": len(uncovered),
        "generation_coverage_ratio": (len(covered) / denominator) if denominator else 1.0,
        "infeasible_tuples": [_tuple_payload(item) for item in sorted(infeasible)],
        "generated_tuples": [_tuple_payload(item) for item in sorted(covered)],
        "generated_uncovered_tuples": [_tuple_payload(item) for item in sorted(uncovered)],
    }


def covering_row_id(
    model: FactorModel,
    row: Mapping[str, str],
    *,
    strength: int,
    seed: int,
) -> str:
    payload = json.dumps({
        "model_id": model.model_id,
        "model_version": model.version,
        "strength": strength,
        "seed": seed,
        "row": {factor.name: str(row.get(factor.name) or "") for factor in model.factors},
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def covering_array_execution_report(
    model: FactorModel,
    generated: CoveringArrayResult,
    observations: Iterable[CoveringRowObservation],
    *,
    revision: Mapping[str, str],
    valid_evidence_ids: Iterable[str],
) -> dict[str, Any]:
    """Report observed rows/tuples separately from generated coverage intent."""

    if generated.model_id != model.model_id or generated.model_version != model.version:
        raise ValueError("generated covering array does not match the factor model")
    generated_by_id = dict(zip(generated.row_ids, generated.rows))
    valid_evidence = set(str(item) for item in valid_evidence_ids)
    observed_by_id: dict[str, CoveringRowObservation] = {}
    invalid: list[dict[str, str]] = []
    for observation in observations:
        reason = ""
        if observation.row_id not in generated_by_id:
            reason = "unknown generated row id"
        elif dict(observation.revision) != dict(revision):
            reason = "repository revision mismatch"
        elif observation.outcome not in {"passed", "failed", "externally_blocked"}:
            reason = "invalid observation outcome"
        elif observation.outcome in {"passed", "failed"} and not observation.evidence_ids:
            reason = "observed row has no evidence ids"
        elif any(item not in valid_evidence for item in observation.evidence_ids):
            reason = "observation references unknown evidence ids"
        elif dict(observation.observed_row) != dict(generated_by_id[observation.row_id]):
            reason = "observed row differs from generated row"
        elif observation.row_id in observed_by_id:
            reason = "duplicate row observation"
        if reason:
            invalid.append({"row_id": observation.row_id, "reason": reason})
        else:
            observed_by_id[observation.row_id] = observation

    passed_rows = [
        dict(generated_by_id[row_id])
        for row_id, observation in observed_by_id.items()
        if observation.outcome == "passed"
    ]
    feasible = _feasible_tuples(model, list(feasible_assignments(model)), generated.strength)
    observed_tuples: set[TupleKey] = set()
    for row in passed_rows:
        observed_tuples.update(_row_tuples(model, row, generated.strength) & feasible)
    uncovered = feasible - observed_tuples
    outcomes = [observation.outcome for observation in observed_by_id.values()]
    if invalid or "failed" in outcomes:
        execution_status = "failed"
    elif len(observed_by_id) < len(generated.rows) or "externally_blocked" in outcomes:
        execution_status = "incomplete"
    else:
        execution_status = "complete"
    return {
        **generated.report,
        "revision": dict(revision),
        "execution_status": execution_status,
        "observed_row_count": len(observed_by_id),
        "observed_pass_row_count": outcomes.count("passed"),
        "observed_fail_row_count": outcomes.count("failed"),
        "externally_blocked_row_count": outcomes.count("externally_blocked"),
        "not_run_row_count": len(generated.rows) - len(observed_by_id),
        "observed_tuple_count": len(observed_tuples),
        "observed_uncovered_tuple_count": len(uncovered),
        "observed_coverage_ratio": len(observed_tuples) / len(feasible) if feasible else 1.0,
        "observed_tuples": [_tuple_payload(item) for item in sorted(observed_tuples)],
        "observed_uncovered_tuples": [_tuple_payload(item) for item in sorted(uncovered)],
        "invalid_observations": invalid,
    }


def feasible_assignments(model: FactorModel) -> tuple[dict[str, str], ...]:
    _validate_model(model)
    rows: list[dict[str, str]] = []
    names = [factor.name for factor in model.factors]
    for values in itertools.product(*(factor.values for factor in model.factors)):
        row = dict(zip(names, values))
        if not _matching_constraints(model, row):
            rows.append(row)
    return tuple(rows)


def _validate_model(model: FactorModel) -> None:
    if model.version != FACTOR_MODEL_SCHEMA_VERSION:
        raise ValueError(f"unsupported factor model version: {model.version}")
    if not model.model_id.strip():
        raise ValueError("factor model_id is required")
    names = [factor.name for factor in model.factors]
    if len(names) != len(set(names)) or any(not name.strip() for name in names):
        raise ValueError("factor names must be unique and non-empty")
    if not names:
        raise ValueError("factor model must contain factors")
    domains: dict[str, set[str]] = {}
    for factor in model.factors:
        if not factor.values or len(factor.values) != len(set(factor.values)):
            raise ValueError(f"factor values must be non-empty and unique: {factor.name}")
        if any(not str(value).strip() for value in factor.values):
            raise ValueError(f"factor values must be non-empty: {factor.name}")
        domains[factor.name] = set(factor.values)
    constraint_ids: set[str] = set()
    for constraint in model.forbidden:
        if not constraint.constraint_id or constraint.constraint_id in constraint_ids:
            raise ValueError("constraint ids must be unique and non-empty")
        constraint_ids.add(constraint.constraint_id)
        if not constraint.values:
            raise ValueError(f"constraint has no values: {constraint.constraint_id}")
        keys = [name for name, _ in constraint.values]
        if len(keys) != len(set(keys)):
            raise ValueError(f"constraint repeats a factor: {constraint.constraint_id}")
        for name, value in constraint.values:
            if name not in domains or value not in domains[name]:
                raise ValueError(
                    f"constraint references an unknown factor value: {name}={value}"
                )


def _validate_strength(model: FactorModel, strength: int) -> None:
    if not 1 <= strength <= len(model.factors):
        raise ValueError("strength must be between 1 and the factor count")


def _matching_constraints(
    model: FactorModel,
    row: Mapping[str, str],
) -> tuple[str, ...]:
    return tuple(
        constraint.constraint_id
        for constraint in model.forbidden
        if all(row.get(name) == value for name, value in constraint.values)
    )


def _row_invalid_reason(model: FactorModel, row: Mapping[str, str]) -> str:
    domains = {factor.name: set(factor.values) for factor in model.factors}
    if set(row) != set(domains):
        return "row does not contain exactly the model factors"
    invalid = [
        f"{name}={value}"
        for name, value in row.items()
        if value not in domains[name]
    ]
    if invalid:
        return "row contains unknown values: " + ", ".join(sorted(invalid))
    constraints = _matching_constraints(model, row)
    if constraints:
        return "row violates constraints: " + ", ".join(constraints)
    return ""


def _theoretical_tuples(model: FactorModel, strength: int) -> set[TupleKey]:
    result: set[TupleKey] = set()
    for factors in itertools.combinations(model.factors, strength):
        for values in itertools.product(*(factor.values for factor in factors)):
            result.add(tuple(
                (factor.name, value)
                for factor, value in zip(factors, values)
            ))
    return result


def _feasible_tuples(
    model: FactorModel,
    rows: Sequence[Mapping[str, str]],
    strength: int,
) -> set[TupleKey]:
    result: set[TupleKey] = set()
    for row in rows:
        result.update(_row_tuples(model, row, strength))
    return result


def _row_tuples(
    model: FactorModel,
    row: Mapping[str, str],
    strength: int,
) -> set[TupleKey]:
    factor_names = tuple(factor.name for factor in model.factors)
    return {
        tuple((name, row[name]) for name in names)
        for names in itertools.combinations(factor_names, strength)
    }


def _tuple_payload(item: TupleKey) -> dict[str, Any]:
    return {
        "tuple_id": "|".join(f"{name}={value}" for name, value in item),
        "values": dict(item),
    }

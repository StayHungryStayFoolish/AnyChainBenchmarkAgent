"""Product-level constrained factors for AnyChain Agent Chaos coverage.

This module defines coverage intent only. Generated rows are test inputs; they
are not runtime evidence and must never be reported as executed Chaos rounds.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from agent.workflows.group_registry import (
    GROUP_ORDER,
    GROUP_SPEC_BY_NAME,
    group_applicable,
)
from tests.agent_live.covering_arrays import (
    FACTOR_MODEL_SCHEMA_VERSION,
    Factor,
    FactorModel,
    ForbiddenCombination,
    covering_row_id,
)


PRODUCT_CHAOS_MODEL_ID = "anychain-agent-product-chaos"
PRODUCT_CHAOS_MODEL_VERSION = FACTOR_MODEL_SCHEMA_VERSION

STATE_CONTROL_FACTOR_NAMES = (
    "workflow_mode",
    "subject_group",
    "language",
    "pending_state",
    "group_state",
    "interruption_depth",
    "evidence_shape",
    "chain_case",
)
_WORKFLOW_STATE_BY_FACTOR = {
    "fake": ("rpc_benchmark", "fake-node"),
    "real": ("rpc_benchmark", "real-node"),
    "sync": ("sync_observe", "sync-observe"),
}
_CHAIN_EXTENSION_STATE_BY_FACTOR = {
    "known": ({}, {}),
    "case1": ({}, {"status": "needs_endpoint"}),
    "case2": ({"status": "existing_family_needs_endpoint"}, {}),
    "case3": ({"status": "unsupported_adapter_family"}, {}),
}


@dataclass(frozen=True)
class ProductCoveringArrayResult:
    model_id: str
    model_version: int
    strength: int
    seed: int
    rows: tuple[dict[str, str], ...]
    row_ids: tuple[str, ...]
    report: dict[str, Any]


def product_state_for_factor_row(
    factors: Mapping[str, str],
) -> dict[str, Any]:
    """Project one test-factor row into the product's applicability state."""

    workflow_factor = str(factors.get("workflow_mode") or "")
    chain_factor = str(factors.get("chain_case") or "")
    if workflow_factor not in _WORKFLOW_STATE_BY_FACTOR:
        raise ValueError(f"unknown workflow_mode factor: {workflow_factor}")
    if chain_factor not in _CHAIN_EXTENSION_STATE_BY_FACTOR:
        raise ValueError(f"unknown chain_case factor: {chain_factor}")
    workflow_mode, target_mode = _WORKFLOW_STATE_BY_FACTOR[workflow_factor]
    chain_identity, custom_rpc = _CHAIN_EXTENSION_STATE_BY_FACTOR[chain_factor]
    return {
        "workflow_mode": workflow_mode,
        "target_mode": target_mode,
        "chain_identity": dict(chain_identity),
        "custom_rpc": dict(custom_rpc),
    }


def factor_row_group_applicable(factors: Mapping[str, str]) -> bool:
    """Return whether the row's subject group can exist on its product path."""

    subject_group = str(factors.get("subject_group") or "")
    spec = GROUP_SPEC_BY_NAME.get(subject_group)
    return bool(
        spec is not None
        and group_applicable(product_state_for_factor_row(factors), spec)
    )


def build_product_factor_model() -> FactorModel:
    """Return the versioned product factor model and impossible combinations."""

    factors = (
        Factor("workflow_mode", ("fake", "real", "sync")),
        Factor("subject_group", GROUP_ORDER),
        Factor("language", ("en", "zh")),
        Factor("session_state", ("fresh", "partial", "complete", "quarantine")),
        Factor("pending_state", ("none", "manual", "choice")),
        Factor("group_state", ("partial", "completed", "invalidated")),
        Factor("interruption_depth", ("0", "1", "2+")),
        Factor(
            "input_shape",
            ("exact", "natural", "multiline", "structured", "contradictory"),
        ),
        Factor("chain_case", ("known", "case1", "case2", "case3")),
        Factor(
            "workload",
            (
                "default_single",
                "default_mixed",
                "custom_single",
                "custom_mixed",
                "not_applicable",
            ),
        ),
        Factor("evidence_shape", ("none", "request", "response", "split", "docs")),
        Factor("recovery", ("none", "back", "jump", "correct", "retry", "reset")),
    )

    forbidden: list[ForbiddenCombination] = []

    def forbid(constraint_id: str, **values: str) -> None:
        forbidden.append(ForbiddenCombination.from_mapping(constraint_id, values))

    for workflow_mode in _WORKFLOW_STATE_BY_FACTOR:
        for subject_group in GROUP_ORDER:
            applicable_cases = {
                chain_case
                for chain_case in _CHAIN_EXTENSION_STATE_BY_FACTOR
                if factor_row_group_applicable({
                    "workflow_mode": workflow_mode,
                    "chain_case": chain_case,
                    "subject_group": subject_group,
                })
            }
            if not applicable_cases:
                forbid(
                    f"registry-inapplicable-{workflow_mode}-{subject_group}",
                    workflow_mode=workflow_mode,
                    subject_group=subject_group,
                )
                continue
            for chain_case in sorted(
                set(_CHAIN_EXTENSION_STATE_BY_FACTOR) - applicable_cases
            ):
                forbid(
                    "registry-conditionally-inapplicable-"
                    f"{workflow_mode}-{chain_case}-{subject_group}",
                    workflow_mode=workflow_mode,
                    chain_case=chain_case,
                    subject_group=subject_group,
                )

    # Sync observation has no RPC workload. RPC workflows require a workload,
    # except Case 3, which exits to secondary-development handoff.
    for workload in (
        "default_single", "default_mixed", "custom_single", "custom_mixed"
    ):
        forbid(f"sync-no-{workload}", workflow_mode="sync", workload=workload)
    for workflow_mode in ("fake", "real"):
        for chain_case in ("known", "case1", "case2"):
            forbid(
                f"{workflow_mode}-{chain_case}-requires-workload",
                workflow_mode=workflow_mode,
                chain_case=chain_case,
                workload="not_applicable",
            )

    # Case 1/2 are custom-RPC paths. Case 3 is a development handoff and has no
    # executable workload until the user returns to Case 1, Case 2, or known.
    for chain_case in ("case1", "case2"):
        for workload in ("default_single", "default_mixed"):
            forbid(
                f"{chain_case}-no-{workload}",
                chain_case=chain_case,
                workload=workload,
            )
    for workload in ("custom_single", "custom_mixed"):
        forbid(
            f"known-no-{workload}",
            chain_case="known",
            workload=workload,
        )
    for workload in (
        "default_single", "default_mixed", "custom_single", "custom_mixed"
    ):
        forbid(f"case3-no-{workload}", chain_case="case3", workload=workload)

    # Evidence shape describes the receipts required by each chain lifecycle,
    # not arbitrary text pasted elsewhere in the same conversation. Template
    # workloads need no custom schema evidence; Case 1/2 require independently
    # sourced request and observed-response provenance; Case 3 requires official
    # protocol documentation for its development handoff.
    for evidence_shape in ("request", "response", "split", "docs"):
        forbid(
            f"known-no-{evidence_shape}-evidence",
            chain_case="known",
            evidence_shape=evidence_shape,
        )
    for chain_case in ("case1", "case2"):
        for evidence_shape in ("none", "request", "response", "docs"):
            forbid(
                f"{chain_case}-requires-split-not-{evidence_shape}",
                chain_case=chain_case,
                evidence_shape=evidence_shape,
            )
    for evidence_shape in ("none", "request", "response", "split"):
        forbid(
            f"case3-requires-docs-not-{evidence_shape}",
            chain_case="case3",
            evidence_shape=evidence_shape,
        )

    # Back requires a real interruption frame. A quarantined checkpoint is
    # presented through the resume selector, so it has a choice contract and
    # can only be reset. Fresh and complete checkpoints also enter through a
    # concrete question; a no-pending baseline is only a valid partial-session
    # control state.
    forbid("back-requires-interruption", interruption_depth="0", recovery="back")
    for pending_state in ("none", "manual"):
        forbid(
            f"quarantine-no-{pending_state}-question",
            session_state="quarantine",
            pending_state=pending_state,
        )
    forbid(
        "fresh-session-requires-question",
        session_state="fresh",
        pending_state="none",
    )
    forbid(
        "complete-session-requires-question",
        session_state="complete",
        pending_state="none",
    )
    for group_state in ("partial", "completed"):
        forbid(
            f"quarantine-no-{group_state}-group",
            session_state="quarantine",
            group_state=group_state,
        )
    for recovery in ("none", "back", "jump", "correct", "retry"):
        forbid(
            f"quarantine-requires-reset-not-{recovery}",
            session_state="quarantine",
            recovery=recovery,
        )
    return FactorModel(
        model_id=PRODUCT_CHAOS_MODEL_ID,
        version=PRODUCT_CHAOS_MODEL_VERSION,
        factors=factors,
        forbidden=tuple(forbidden),
    )


def build_state_control_factor_model() -> FactorModel:
    """Project the product model to the factors requiring 3-way coverage."""

    source = build_product_factor_model()
    names = set(STATE_CONTROL_FACTOR_NAMES)
    factors = tuple(factor for factor in source.factors if factor.name in names)
    forbidden = tuple(
        constraint
        for constraint in source.forbidden
        if all(name in names for name, _ in constraint.values)
    )
    return FactorModel(
        model_id=f"{source.model_id}-state-control",
        version=source.version,
        factors=factors,
        forbidden=forbidden,
    )


def generate_product_covering_array(
    model: FactorModel,
    *,
    strength: int,
    seed: int,
) -> ProductCoveringArrayResult:
    """Generate full feasible tuple coverage without full Cartesian expansion.

    One deterministic feasible completion is produced for each feasible target
    tuple, then duplicate rows are removed. This favors auditable completeness
    over a minimum-row optimization.
    """

    if not 1 <= strength <= len(model.factors):
        raise ValueError("strength must be between 1 and the factor count")

    theoretical = _theoretical_tuples(model, strength)
    feasible: set[TupleKey] = set()
    infeasible: set[TupleKey] = set()
    candidate_rows: dict[tuple[tuple[str, str], ...], dict[str, str]] = {}
    for target in sorted(theoretical):
        completion = _complete_partial(model, dict(target), seed=seed)
        if completion is None:
            infeasible.add(target)
            continue
        feasible.add(target)
        key = tuple((factor.name, completion[factor.name]) for factor in model.factors)
        candidate_rows[key] = completion

    rows = _reduce_redundant_rows(
        model,
        tuple(candidate_rows[key] for key in sorted(candidate_rows)),
        feasible,
        strength=strength,
        seed=seed,
    )
    covered = _covered_tuples(model, rows, strength) & feasible
    uncovered = feasible - covered
    report = {
        "schema_version": FACTOR_MODEL_SCHEMA_VERSION,
        "model_id": model.model_id,
        "model_version": model.version,
        "strength": strength,
        "seed": seed,
        "factors": {
            factor.name: list(factor.values) for factor in model.factors
        },
        "constraints": [
            {
                "constraint_id": constraint.constraint_id,
                "values": dict(constraint.values),
            }
            for constraint in model.forbidden
        ],
        "generated_row_count": len(rows),
        "invalid_rows": [],
        "theoretical_tuple_count": len(theoretical),
        "feasible_tuple_denominator": len(feasible),
        "infeasible_tuple_count": len(infeasible),
        "generated_tuple_count": len(covered),
        "generated_uncovered_tuple_count": len(uncovered),
        "generation_coverage_ratio": len(covered) / len(feasible) if feasible else 1.0,
        "infeasible_tuples": [_tuple_payload(item) for item in sorted(infeasible)],
        "generated_tuples": [_tuple_payload(item) for item in sorted(covered)],
        "generated_uncovered_tuples": [_tuple_payload(item) for item in sorted(uncovered)],
        "execution_status": "not_run",
        "observed_row_count": 0,
        "observed_tuple_count": 0,
        "observed_uncovered_tuple_count": len(feasible),
    }
    row_ids = tuple(
        covering_row_id(model, row, strength=strength, seed=seed)
        for row in rows
    )
    report["generated_row_ids"] = list(row_ids)
    return ProductCoveringArrayResult(
        model_id=model.model_id,
        model_version=model.version,
        strength=strength,
        seed=seed,
        rows=rows,
        row_ids=row_ids,
        report=report,
    )


TupleKey = tuple[tuple[str, str], ...]


def _theoretical_tuples(model: FactorModel, strength: int) -> set[TupleKey]:
    tuples: set[TupleKey] = set()
    for factors in itertools.combinations(model.factors, strength):
        for values in itertools.product(*(factor.values for factor in factors)):
            tuples.add(tuple(
                (factor.name, value) for factor, value in zip(factors, values)
            ))
    return tuples


def _complete_partial(
    model: FactorModel,
    partial: Mapping[str, str],
    *,
    seed: int,
) -> dict[str, str] | None:
    domains = {factor.name: factor.values for factor in model.factors}
    if any(name not in domains or value not in domains[name] for name, value in partial.items()):
        return None
    row = dict(partial)
    if not _partial_allowed(model, row):
        return None
    rng = random.Random(f"{seed}:{tuple(sorted(partial.items()))}")

    def visit(index: int) -> bool:
        if index == len(model.factors):
            return True
        factor = model.factors[index]
        if factor.name in row:
            return visit(index + 1)
        values = list(factor.values)
        rng.shuffle(values)
        for value in values:
            row[factor.name] = value
            if _partial_allowed(model, row) and visit(index + 1):
                return True
            row.pop(factor.name, None)
        return False

    return dict(row) if visit(0) else None


def _partial_allowed(model: FactorModel, row: Mapping[str, str]) -> bool:
    return not any(
        all(name in row and row[name] == value for name, value in constraint.values)
        for constraint in model.forbidden
    )


def _covered_tuples(
    model: FactorModel,
    rows: Iterable[Mapping[str, str]],
    strength: int,
) -> set[TupleKey]:
    names = tuple(factor.name for factor in model.factors)
    return {
        tuple((name, row[name]) for name in selected)
        for row in rows
        for selected in itertools.combinations(names, strength)
    }


def _reduce_redundant_rows(
    model: FactorModel,
    rows: Sequence[Mapping[str, str]],
    feasible: set[TupleKey],
    *,
    strength: int,
    seed: int,
) -> tuple[dict[str, str], ...]:
    """Remove rows whose feasible tuples are all covered elsewhere."""

    row_tuples = [
        _covered_tuples(model, (row,), strength) & feasible for row in rows
    ]
    counts: dict[TupleKey, int] = {item: 0 for item in feasible}
    for tuples in row_tuples:
        for item in tuples:
            counts[item] += 1

    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    retained = [True] * len(rows)
    for index in order:
        tuples = row_tuples[index]
        if tuples and all(counts[item] > 1 for item in tuples):
            retained[index] = False
            for item in tuples:
                counts[item] -= 1
    return tuple(dict(row) for index, row in enumerate(rows) if retained[index])


def _tuple_payload(item: Sequence[tuple[str, str]]) -> dict[str, str]:
    return dict(item)

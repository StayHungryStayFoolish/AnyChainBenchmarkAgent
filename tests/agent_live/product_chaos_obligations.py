"""Finite Phase 8C obligation catalog for product-level dynamic Chaos.

Generation is not execution.  Every row produced here remains ``not_run``
until a separate Phase 8 runner observes the required verifier contract in a
response-driven real PTY session.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from tests.agent_live.coverage_evidence import content_hash
from tests.agent_live.formal_journey_catalog import (
    FORMAL_JOURNEY_VERIFIER_REGISTRY,
    REGISTRY_IMPORT,
)
from tests.agent_live.product_chaos_factors import (
    ProductCoveringArrayResult,
    build_product_factor_model,
    build_state_control_factor_model,
    generate_product_covering_array,
)
from tests.agent_live.runtime_checkpoint import reviewed_scenario


PRODUCT_CHAOS_OBLIGATION_SCHEMA_VERSION = 1
PRODUCT_CHAOS_SEED = 20260724
PRODUCT_CHAOS_STATUS = "not_run"


@dataclass(frozen=True)
class _ModelContract:
    lane: str
    strength: int


_MODEL_CONTRACTS = (
    _ModelContract("pairwise", 2),
    _ModelContract("state_control_3way", 3),
)

_MANUAL_KINDS = frozenset({
    "chain",
    "confirm_or_value",
    "device",
    "evidence",
    "manual_value",
    "positive_integer",
    "url",
})
_CHOICE_KINDS = frozenset({"numbered_choice", "yes_no"})
_NO_PENDING_SCENARIOS = frozenset({
    "action_activate_next_workflow_goal",
    "action_change_group",
    "action_discard_next_workflow_goal",
    "action_go_back",
    "action_queue_workflow_goal",
})


def build_product_chaos_obligations(
    *,
    revision: Mapping[str, str],
) -> tuple[dict[str, Any], ...]:
    """Build the sole finite G4 row ledger from the two reviewed models."""

    active_revision = _validated_revision(revision)
    generated = _generated_models()
    obligations: list[dict[str, Any]] = []
    for contract, result in generated:
        if result.report.get("generated_uncovered_tuple_count") != 0:
            raise ValueError(f"incomplete covering model: {result.model_id}")
        for source_row_id, factors in zip(result.row_ids, result.rows, strict=True):
            obligations.append(
                _build_obligation(
                    contract=contract,
                    result=result,
                    source_row_id=source_row_id,
                    factors=factors,
                    revision=active_revision,
                )
            )
    rows = tuple(obligations)
    validate_product_chaos_obligations(rows, revision=active_revision)
    return rows


def product_chaos_obligation_report(
    obligations: Sequence[Mapping[str, Any]] | None = None,
    *,
    revision: Mapping[str, str],
) -> dict[str, Any]:
    """Summarize generated intent without converting it into observed evidence."""

    active_revision = _validated_revision(revision)
    rows = tuple(
        obligations
        or build_product_chaos_obligations(revision=active_revision)
    )
    validate_product_chaos_obligations(rows, revision=active_revision)
    by_model: dict[str, int] = {}
    for row in rows:
        model_id = str((row.get("model") or {}).get("model_id") or "")
        by_model[model_id] = by_model.get(model_id, 0) + 1
    return {
        "schema_version": PRODUCT_CHAOS_OBLIGATION_SCHEMA_VERSION,
        "seed": PRODUCT_CHAOS_SEED,
        "required_denominator": len(rows),
        "generated_count": len(rows),
        "observed_pass_count": 0,
        "observed_fail_count": 0,
        "not_run_count": len(rows),
        "execution_status": PRODUCT_CHAOS_STATUS,
        "by_model": dict(sorted(by_model.items())),
        "generation_is_execution": False,
    }


def validate_product_chaos_obligations(
    obligations: Sequence[Mapping[str, Any]],
    *,
    revision: Mapping[str, str],
) -> None:
    """Fail closed on missing rows, stale mappings, or non-executable contracts."""

    active_revision = _validated_revision(revision)
    expected = _expected_source_rows()
    rows = tuple(dict(row) for row in obligations)
    if len(rows) != len(expected):
        raise ValueError(
            f"product Chaos catalog denominator changed: {len(rows)} != {len(expected)}"
        )

    seen_ids: set[str] = set()
    seen_sources: set[tuple[str, str]] = set()
    for row in rows:
        obligation_id = _required_text(row, "obligation_id")
        if obligation_id in seen_ids:
            raise ValueError(f"duplicate product Chaos obligation id: {obligation_id}")
        seen_ids.add(obligation_id)
        if row.get("schema_version") != PRODUCT_CHAOS_OBLIGATION_SCHEMA_VERSION:
            raise ValueError(f"unsupported product Chaos schema: {obligation_id}")
        if row.get("status") != PRODUCT_CHAOS_STATUS:
            raise ValueError(f"generated product Chaos row claimed execution: {obligation_id}")

        model = dict(row.get("model") or {})
        model_id = _required_text(model, "model_id")
        source_row_id = _required_text(row, "source_row_id")
        source_key = (model_id, source_row_id)
        if source_key in seen_sources:
            raise ValueError(f"duplicate product Chaos source row: {source_key}")
        seen_sources.add(source_key)
        expected_row = expected.get(source_key)
        if expected_row is None:
            raise ValueError(f"unknown product Chaos source row: {source_key}")
        if model != expected_row["model"] or dict(row.get("factors") or {}) != expected_row["factors"]:
            raise ValueError(f"product Chaos source row drifted: {obligation_id}")
        if row.get("seed") != PRODUCT_CHAOS_SEED:
            raise ValueError(f"product Chaos seed drifted: {obligation_id}")
        if row.get("revision_binding") != active_revision:
            raise ValueError(f"stale product Chaos revision: {obligation_id}")
        if obligation_id != _obligation_id(model, source_row_id, row["factors"]):
            raise ValueError(f"unstable product Chaos obligation id: {obligation_id}")

        _validate_start_contract(row)
        _validate_simulator_contract(row)
        _validate_verifier_contract(row)

        unsigned = dict(row)
        recorded_hash = str(unsigned.pop("contract_hash", "") or "")
        if not recorded_hash or recorded_hash != content_hash(unsigned):
            raise ValueError(f"stale product Chaos contract hash: {obligation_id}")

    if seen_sources != set(expected):
        missing = sorted(set(expected) - seen_sources)
        raise ValueError(f"product Chaos catalog is missing source rows: {missing[:5]}")


def _generated_models(
) -> tuple[tuple[_ModelContract, ProductCoveringArrayResult], ...]:
    models = (
        build_product_factor_model(),
        build_state_control_factor_model(),
    )
    return tuple(
        (
            contract,
            generate_product_covering_array(
                model,
                strength=contract.strength,
                seed=PRODUCT_CHAOS_SEED,
            ),
        )
        for contract, model in zip(_MODEL_CONTRACTS, models, strict=True)
    )


def _expected_source_rows() -> dict[tuple[str, str], dict[str, Any]]:
    expected: dict[tuple[str, str], dict[str, Any]] = {}
    for contract, result in _generated_models():
        model = {
            "lane": contract.lane,
            "model_id": result.model_id,
            "model_version": result.model_version,
            "strength": result.strength,
        }
        for row_id, factors in zip(result.row_ids, result.rows, strict=True):
            key = (result.model_id, row_id)
            if key in expected:
                raise ValueError(f"covering generator produced duplicate source row: {key}")
            expected[key] = {"model": model, "factors": dict(factors)}
    return expected


def _build_obligation(
    *,
    contract: _ModelContract,
    result: ProductCoveringArrayResult,
    source_row_id: str,
    factors: Mapping[str, str],
    revision: Mapping[str, str],
) -> dict[str, Any]:
    factor_values = dict(factors)
    scenario_id = _start_scenario_for(factor_values)
    scenario = reviewed_scenario(scenario_id)
    state_fingerprint = str(getattr(scenario, "state_fingerprint", "") or "").strip()
    if not state_fingerprint or not getattr(scenario, "seed_state", None):
        raise ValueError(f"product Chaos start scenario is not executable: {scenario_id}")
    model = {
        "lane": contract.lane,
        "model_id": result.model_id,
        "model_version": result.model_version,
        "strength": result.strength,
    }
    required = _required_postconditions(factor_values)
    definitions = FORMAL_JOURNEY_VERIFIER_REGISTRY.definitions
    bindings = [asdict(definitions[item]) for item in required]
    for binding in bindings:
        binding.pop("verifier", None)

    unsigned: dict[str, Any] = {
        "schema_version": PRODUCT_CHAOS_OBLIGATION_SCHEMA_VERSION,
        "obligation_id": _obligation_id(model, source_row_id, factor_values),
        "source_row_id": source_row_id,
        "model": model,
        "seed": PRODUCT_CHAOS_SEED,
        "factors": factor_values,
        "start_contract": {
            "provider": "tests.agent_live.runtime_checkpoint:reviewed_scenario",
            "scenario_id": scenario_id,
            "scenario_state_fingerprint": state_fingerprint,
        },
        "simulator_contract": {
            "selection_mode": "response_driven",
            "actor": "codex_as_user",
            "persona": _persona_for(factor_values),
            "mission": _mission_for(factor_values),
            "max_turns": _max_turns_for(factor_values),
            "prewritten_future_turns_forbidden": True,
            "read_complete_agent_response_before_each_turn": True,
        },
        "verifier_contract": {
            "registry_import": REGISTRY_IMPORT,
            "registry_id": FORMAL_JOURNEY_VERIFIER_REGISTRY.registry_id,
            "required_postcondition_ids": list(required),
            "required_bindings": bindings,
            "forbidden_postcondition_ids": [
                "mechanical_response_loop",
                "state_regressed",
            ],
            "required_evidence": [
                "revision_bound_schedule",
                "response_bound_codex_decisions",
                "real_pty_transcript",
                "runtime_turn_events",
                "checkpoint_state_diffs",
            ],
        },
        "revision_binding": dict(revision),
        "status": PRODUCT_CHAOS_STATUS,
    }
    return {**unsigned, "contract_hash": content_hash(unsigned)}


def _start_scenario_for(factors: Mapping[str, str]) -> str:
    session_state = str(factors.get("session_state") or "unspecified")
    pending_state = str(factors.get("pending_state") or "")
    workflow_mode = str(factors.get("workflow_mode") or "")
    recovery = str(factors.get("recovery") or "")
    chain_case = str(factors.get("chain_case") or "")
    workload = str(factors.get("workload") or "")
    evidence_shape = str(factors.get("evidence_shape") or "")
    interruption_depth = str(factors.get("interruption_depth") or "")

    if session_state == "quarantine":
        if pending_state != "none" or recovery != "reset":
            raise ValueError("quarantine row has no reliable reviewed start mapping")
        return "resume_quarantine"

    if pending_state == "none":
        if recovery == "back" or interruption_depth in {"1", "2+"}:
            return "action_go_back"
        return {
            "fresh": "action_queue_workflow_goal",
            "partial": "action_change_group",
            "complete": "action_activate_next_workflow_goal",
            "unspecified": "action_change_group",
        }.get(session_state) or _mapping_gap(factors, "session_state without pending")

    if pending_state == "manual":
        if workflow_mode == "sync":
            return (
                "endpoint_sync_observe_rpc_url"
                if evidence_shape in {"request", "response", "split", "docs"}
                else "sync_duration"
            )
        if workflow_mode == "real":
            return "endpoint_local_rpc_url"
        if chain_case == "case3":
            return "case3_evidence"
        if chain_case == "case2":
            return (
                "new_chain_existing_family_needs_schema_evidence"
                if evidence_shape != "none"
                else "new_chain_existing_family_needs_method"
            )
        if chain_case == "case1" or workload in {"custom_single", "custom_mixed"}:
            return (
                "custom_needs_schema_evidence"
                if evidence_shape != "none"
                else "custom_needs_method"
            )
        return {
            "fresh": "chain_manual",
            "partial": "provider_zone",
            "complete": "advanced_adjust_value",
            "unspecified": "provider_zone",
        }.get(session_state) or _mapping_gap(factors, "manual pending")

    if pending_state == "choice":
        if session_state == "fresh":
            return "opening"
        if recovery in {"correct", "retry"}:
            return "failure_recovery"
        if session_state == "complete":
            return {
                "fake": "execution",
                "real": "runtime_real_node_final",
                "sync": "sync_after_setup",
            }.get(workflow_mode) or _mapping_gap(factors, "complete choice workflow")
        if workflow_mode == "sync":
            return "sync_stop"
        if workflow_mode == "real":
            return "runtime_real_node_smoke"
        if chain_case == "case3":
            return "case3_next"
        if chain_case == "case2":
            return "new_chain_continue"
        if chain_case == "case1":
            return "custom_continue"
        return "workload_mixed" if workload == "default_mixed" else "qps_mode"

    return _mapping_gap(factors, "pending_state")


def _required_postconditions(factors: Mapping[str, str]) -> tuple[str, ...]:
    required = ["committed_state"]
    if factors.get("pending_state") in {"manual", "choice"}:
        required.append("pending_advanced")
    if factors.get("recovery") in {"back", "jump"}:
        required.append("group_changed")
    if (
        factors.get("workflow_mode") in {"fake", "real"}
        and (
            factors.get("chain_case") in {"case1", "case2"}
            or factors.get("workload") in {"custom_single", "custom_mixed"}
        )
    ):
        required.append("rpc_action_admitted")
    return tuple(dict.fromkeys(required))


def _persona_for(factors: Mapping[str, str]) -> str:
    language = "Chinese-speaking" if factors.get("language") == "zh" else "English-speaking"
    session = str(factors.get("session_state") or "unspecified")
    behavior = {
        "back": "backtracking",
        "jump": "task-switching",
        "correct": "self-correcting",
        "retry": "recovery-focused",
        "reset": "reset-requesting",
    }.get(str(factors.get("recovery") or ""), "goal-directed")
    return f"{language} {session}-session {behavior} blockchain operator"


def _mission_for(factors: Mapping[str, str]) -> str:
    ordered = ", ".join(f"{name}={value}" for name, value in factors.items())
    return (
        "Exercise this exact generated product-chaos factor row through natural, "
        "response-driven interaction. Preserve confirmed compatible state, expose "
        "clarifications instead of guessing, and reach the bound verifier outcome. "
        f"Required factors: {ordered}."
    )


def _max_turns_for(factors: Mapping[str, str]) -> int:
    depth = {"0": 0, "1": 2, "2+": 4}.get(str(factors.get("interruption_depth") or ""))
    if depth is None:
        return _mapping_gap(factors, "interruption_depth")
    return 12 + depth + (2 if factors.get("input_shape") == "contradictory" else 0)


def _validate_start_contract(row: Mapping[str, Any]) -> None:
    obligation_id = _required_text(row, "obligation_id")
    factors = dict(row.get("factors") or {})
    start = dict(row.get("start_contract") or {})
    scenario_id = _required_text(start, "scenario_id")
    if scenario_id != _start_scenario_for(factors):
        raise ValueError(f"product Chaos start mapping drifted: {obligation_id}")
    try:
        scenario = reviewed_scenario(scenario_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"product Chaos start is not reviewed: {obligation_id}") from exc
    if start.get("provider") != "tests.agent_live.runtime_checkpoint:reviewed_scenario":
        raise ValueError(f"unknown product Chaos start provider: {obligation_id}")
    if start.get("scenario_state_fingerprint") != scenario.state_fingerprint:
        raise ValueError(f"stale product Chaos start fingerprint: {obligation_id}")

    question = dict((scenario.seed_state or {}).get("pending_question") or {})
    pending_state = str(factors.get("pending_state") or "")
    if pending_state == "none":
        if scenario_id == "resume_quarantine":
            if factors.get("session_state") != "quarantine":
                raise ValueError(f"invalid quarantine control overlay: {obligation_id}")
        elif scenario_id not in _NO_PENDING_SCENARIOS or question:
            raise ValueError(f"pending-none row has a pending start: {obligation_id}")
    elif pending_state == "manual":
        if str(question.get("kind") or "") not in _MANUAL_KINDS:
            raise ValueError(f"manual-pending row lacks a manual start: {obligation_id}")
    elif pending_state == "choice":
        if str(question.get("kind") or "") not in _CHOICE_KINDS:
            raise ValueError(f"choice-pending row lacks a choice start: {obligation_id}")
    else:
        raise ValueError(f"unknown pending_state: {obligation_id}")
    if factors.get("session_state") not in {None, "fresh"} and scenario_id == "opening":
        raise ValueError(f"opening masked a non-fresh session: {obligation_id}")


def _validate_simulator_contract(row: Mapping[str, Any]) -> None:
    obligation_id = _required_text(row, "obligation_id")
    simulator = dict(row.get("simulator_contract") or {})
    if (
        simulator.get("selection_mode") != "response_driven"
        or simulator.get("actor") != "codex_as_user"
        or simulator.get("prewritten_future_turns_forbidden") is not True
        or simulator.get("read_complete_agent_response_before_each_turn") is not True
    ):
        raise ValueError(f"product Chaos simulator contract is not dynamic: {obligation_id}")
    if not _required_text(simulator, "persona") or not _required_text(simulator, "mission"):
        raise ValueError(f"product Chaos simulator identity is incomplete: {obligation_id}")
    max_turns = simulator.get("max_turns")
    if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns <= 0:
        raise ValueError(f"product Chaos max_turns is invalid: {obligation_id}")


def _validate_verifier_contract(row: Mapping[str, Any]) -> None:
    obligation_id = _required_text(row, "obligation_id")
    verifier = dict(row.get("verifier_contract") or {})
    registry = FORMAL_JOURNEY_VERIFIER_REGISTRY
    if verifier.get("registry_import") != REGISTRY_IMPORT:
        raise ValueError(f"unknown product Chaos verifier import: {obligation_id}")
    if verifier.get("registry_id") != registry.registry_id:
        raise ValueError(f"stale product Chaos verifier registry: {obligation_id}")
    required = tuple(verifier.get("required_postcondition_ids") or ())
    forbidden = tuple(verifier.get("forbidden_postcondition_ids") or ())
    if not required or len(required) != len(set(required)):
        raise ValueError(f"invalid product Chaos required verifiers: {obligation_id}")
    if not forbidden or len(forbidden) != len(set(forbidden)):
        raise ValueError(f"invalid product Chaos forbidden verifiers: {obligation_id}")
    unknown = (set(required) | set(forbidden)) - set(registry.definitions)
    if unknown:
        raise ValueError(f"unknown product Chaos verifiers: {sorted(unknown)}")
    expected_required = _required_postconditions(dict(row.get("factors") or {}))
    if required != expected_required:
        raise ValueError(f"product Chaos verifier mapping drifted: {obligation_id}")
    expected_bindings = []
    for postcondition_id in required:
        binding = asdict(registry.definitions[postcondition_id])
        binding.pop("verifier", None)
        expected_bindings.append(binding)
    if verifier.get("required_bindings") != expected_bindings:
        raise ValueError(f"product Chaos verifier bindings drifted: {obligation_id}")
    if not all(str(item).strip() for item in verifier.get("required_evidence") or ()):
        raise ValueError(f"product Chaos evidence contract is incomplete: {obligation_id}")


def _obligation_id(
    model: Mapping[str, Any],
    source_row_id: str,
    factors: Mapping[str, str],
) -> str:
    identity = {
        "schema_version": PRODUCT_CHAOS_OBLIGATION_SCHEMA_VERSION,
        "model": dict(model),
        "seed": PRODUCT_CHAOS_SEED,
        "source_row_id": source_row_id,
        "factors": dict(factors),
    }
    return "g4-chaos-" + content_hash(identity)[:24]


def _required_text(values: Mapping[str, Any], field: str) -> str:
    value = str(values.get(field) or "").strip()
    if not value:
        raise ValueError(f"product Chaos contract requires {field}")
    return value


def _validated_revision(revision: Mapping[str, str]) -> dict[str, str]:
    normalized = {
        "commit": str(revision.get("commit") or "").strip(),
        "worktree_hash": str(revision.get("worktree_hash") or "").strip(),
    }
    if not normalized["commit"] or not normalized["worktree_hash"]:
        raise ValueError("product Chaos obligations require commit and worktree_hash")
    return normalized


def _mapping_gap(factors: Mapping[str, str], reason: str) -> str:
    raise ValueError(
        f"no reviewed product Chaos start mapping for {reason}: "
        + ", ".join(f"{name}={value}" for name, value in factors.items())
    )

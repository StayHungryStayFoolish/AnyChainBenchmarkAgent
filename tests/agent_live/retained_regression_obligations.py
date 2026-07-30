"""Revision-bound Phase 8B obligations for retained real-user regressions.

This module is a catalog and validator only.  It does not generate future user
turns, run the product CLI, ingest evidence, or claim that an obligation
passed.  Phase 8 orchestration must execute these contracts and attach
independently verified evidence later.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from tests.agent_live.coverage_evidence import content_hash
from tests.agent_live.retained_regression_attestations import (
    RELATION_BY_VARIANT,
    build_source_contract,
    build_variant_contract,
    validate_source_contract,
    validate_variant_contract,
)
from tests.agent_live.runtime_checkpoint import reviewed_scenario
from tests.agent_live.retained_regression_predicates import (
    POSTCONDITION_EVALUATORS,
)


RETAINED_REGRESSION_SCHEMA_VERSION = 2
RETAINED_REGRESSION_CASE_COUNT = 15
RETAINED_REGRESSION_OBLIGATION_COUNT = 60
RETAINED_REGRESSION_VARIANTS = (
    "exact",
    "isomorphic",
    "negative",
    "neighboring",
)


def _verifier_binding(postcondition_id: str) -> dict[str, Any]:
    evaluator = POSTCONDITION_EVALUATORS[postcondition_id]
    identity = {
        "module": evaluator.__module__,
        "qualname": evaluator.__qualname__,
        "source": inspect.getsource(evaluator),
    }
    return {
        "postcondition_id": postcondition_id,
        "verifier_version": 1,
        "implementation_hash": content_hash(identity),
    }
RETAINED_REGRESSION_FIXTURE = Path(
    "tests/agent_live/fixtures/real_user_regressions/cases.json"
)
RETAINED_REGRESSION_VARIANT_CONTRACTS = Path(
    "tests/agent_live/fixtures/real_user_regressions/variant_contracts.json"
)
RETAINED_REGRESSION_MANIFEST = Path(
    "tests/agent_live/fixtures/real_user_regressions/manifest.json"
)
SEED_PROVIDER = "tests.agent_live.runtime_checkpoint:reviewed_scenario"
POSTCONDITION_REGISTRY_ID = "retained-regression-postconditions-v1"


@dataclass(frozen=True)
class _CaseContract:
    scenario_id: str
    severity: str
    required_postcondition_ids: tuple[str, ...]
    forbidden_postcondition_ids: tuple[str, ...]


_CASE_CONTRACTS: Mapping[str, _CaseContract] = {
    "RR-001": _CaseContract(
        "opening",
        "S1",
        ("current_turn_language_preserved", "orientation_answered_read_only"),
        ("workflow_state_mutated_by_consultation", "stale_fallback_emitted"),
    ),
    "RR-002": _CaseContract(
        "provider_zone",
        "S1",
        ("blocking_question_preserved", "consultation_answered_read_only"),
        ("consultation_consumed_as_pending_value",),
    ),
    "RR-003": _CaseContract(
        "opening",
        "S1",
        (
            "visible_option_action_executed",
            "current_menu_binding_preserved",
            "real_node_selection_executed_at_source",
            "mode_consultation_preserves_chain_pending",
            "mode_change_request_routed_from_chain_pending",
            "declined_mode_change_resumes_chain_pending",
        ),
        (
            "unhandled_visible_option",
            "stale_menu_choice_applied",
            "mode_request_consumed_as_chain_identity",
        ),
    ),
    "RR-004": _CaseContract(
        "chain_manual",
        "S1",
        ("unknown_chain_identity_resolution_started", "chain_confirmation_required"),
        ("unknown_chain_silently_configured",),
    ),
    "RR-005": _CaseContract(
        "provider_region_with_chain",
        "S1",
        ("chain_mode_change_confirmed", "incompatible_state_invalidated", "fallback_resumed"),
        ("environment_text_consumed_as_region",),
    ),
    "RR-006": _CaseContract(
        "ledger_ledger_device",
        "S2",
        ("copied_scalar_normalized", "disk_size_resolved", "disk_limits_collected_once"),
        ("disk_subgroup_repeated",),
    ),
    "RR-007": _CaseContract(
        "network_interface",
        "S1",
        ("typed_detected_value_confirmed",),
        ("typed_confirmation_rejected_by_side_channel",),
    ),
    "RR-008": _CaseContract(
        "qps_mode",
        "S1",
        ("owned_group_backtrack_completed", "consultation_preserved_pending_work"),
        ("backtrack_lost_configuration_state",),
    ),
    "RR-009": _CaseContract(
        "resume",
        "S1",
        ("retained_state_described", "resume_action_contract_exposed", "new_chain_request_routed"),
        ("stale_preflight_executed",),
    ),
    "RR-010": _CaseContract(
        "new_chain_response",
        "S1",
        (
            "custom_method_collection_exited",
            "effective_workload_commit_replaces_defaults",
        ),
        (
            "removed_default_method_committed",
            "custom_method_collection_looped",
        ),
    ),
    "RR-011": _CaseContract(
        "custom_needs_schema_evidence",
        "S1",
        ("example_endpoint_scope_preserved", "rpc_schema_evidence_extracted"),
        ("example_endpoint_replaced_runtime_endpoint", "schema_intake_looped"),
    ),
    "RR-012": _CaseContract(
        "opening",
        "S1",
        ("multiline_evidence_block_collected_once", "evidence_analysis_returned"),
        ("ordinary_question_counted_as_evidence", "blank_prompt_counted_as_evidence"),
    ),
    "RR-013": _CaseContract(
        "execution",
        "S0",
        ("execution_stage_explained", "approved_execution_submitted_once", "status_uses_job_evidence"),
        ("duplicate_job_submission", "invented_job_status"),
    ),
    "RR-014": _CaseContract(
        "qps_mode",
        "S1",
        ("compatible_environment_retained", "mode_specific_state_invalidated", "rpc_groups_rerequired"),
        ("sync_observe_ran_rpc_load",),
    ),
    "RR-015": _CaseContract(
        "action_change_group",
        "S1",
        ("semantic_units_partitioned_in_order", "admitted_mutations_only", "unresolved_units_preserved"),
        ("ambiguous_change_silently_committed", "semantic_unit_dropped"),
    ),
}


# These are observable contract identifiers, not natural-language expected
# strings from the fixture.  A future Phase 8 runner must bind each identifier
# to an independent verifier before it may ingest evidence for this catalog.
_GENERIC_POSTCONDITION_IDS = frozenset({
    "exact_fixture_turns_observed",
    "isomorphic_meaning_attested",
    "response_driven_selection_observed",
    "adjacent_non_trigger_attested",
    "neighboring_transition_attested",
})
_CASE_POSTCONDITION_IDS = frozenset(
    postcondition_id
    for contract in _CASE_CONTRACTS.values()
    for postcondition_id in (
        *contract.required_postcondition_ids,
        *contract.forbidden_postcondition_ids,
    )
)
KNOWN_POSTCONDITION_IDS = _GENERIC_POSTCONDITION_IDS | _CASE_POSTCONDITION_IDS


def build_retained_regression_obligations(
    *,
    repo_root: str | Path,
    revision: Mapping[str, str],
) -> tuple[dict[str, Any], ...]:
    """Build exactly four stable, revision-bound obligations per fixture case."""

    root = Path(repo_root).resolve()
    active_revision = _validated_revision(revision)
    cases_path = root / RETAINED_REGRESSION_FIXTURE
    variant_contracts_path = root / RETAINED_REGRESSION_VARIANT_CONTRACTS
    manifest_path = root / RETAINED_REGRESSION_MANIFEST
    cases_bytes = cases_path.read_bytes()
    variant_contracts_bytes = variant_contracts_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    cases_payload = json.loads(cases_bytes)
    variant_contracts_payload = json.loads(variant_contracts_bytes)
    manifest_payload = json.loads(manifest_bytes)
    source_binding = _validate_fixture_source(
        cases_payload=cases_payload,
        cases_bytes=cases_bytes,
        variant_contracts_payload=variant_contracts_payload,
        variant_contracts_bytes=variant_contracts_bytes,
        manifest_payload=manifest_payload,
        manifest_bytes=manifest_bytes,
    )
    variant_cases = {
        str(item.get("case_id") or ""): dict(item)
        for item in variant_contracts_payload.get("cases") or ()
        if isinstance(item, Mapping)
    }
    variant_declarations = dict(
        variant_contracts_payload.get("variant_contracts") or {}
    )

    cases = tuple(
        sorted(
            cases_payload.get("cases") or (),
            key=lambda item: str(item.get("id") or ""),
        )
    )
    obligations: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case.get("id") or "").strip()
        contract = _CASE_CONTRACTS.get(case_id)
        if contract is None:
            raise ValueError(f"retained regression has no reviewed contract: {case_id}")
        scenario = reviewed_scenario(contract.scenario_id)
        if not _scenario_is_executable(scenario):
            raise ValueError(
                f"retained regression seed is not executable: {contract.scenario_id}"
            )
        state_fingerprint = str(getattr(scenario, "state_fingerprint", "") or "").strip()
        if not state_fingerprint:
            raise ValueError(
                f"retained regression seed has no fingerprint: {contract.scenario_id}"
            )
        for variant in RETAINED_REGRESSION_VARIANTS:
            obligation = _build_obligation(
                case=case,
                contract=contract,
                variant=variant,
                state_fingerprint=state_fingerprint,
                revision=active_revision,
                source_binding=source_binding,
                immutable_case_contract=variant_cases[case_id],
                variant_declaration=dict(variant_declarations[variant]),
            )
            obligations.append(obligation)

    result = tuple(obligations)
    validate_retained_regression_obligations(
        result,
        revision=active_revision,
        source_binding=source_binding,
    )
    return result


def validate_retained_regression_obligations(
    obligations: Sequence[Mapping[str, Any]],
    *,
    revision: Mapping[str, str],
    source_binding: Mapping[str, Any] | None = None,
) -> None:
    """Fail closed unless the catalog is complete, unique, and executable."""

    active_revision = _validated_revision(revision)
    rows = tuple(dict(row) for row in obligations)
    if len(rows) != RETAINED_REGRESSION_OBLIGATION_COUNT:
        raise ValueError(
            "retained regression catalog must contain exactly "
            f"{RETAINED_REGRESSION_OBLIGATION_COUNT} obligations"
        )

    expected_case_ids = set(_CASE_CONTRACTS)
    observed_ids: set[str] = set()
    observed_pairs: set[tuple[str, str]] = set()
    observed_case_ids: set[str] = set()
    for row in rows:
        allowed_fields = {
            "schema_version",
            "obligation_id",
            "case_id",
            "variant",
            "severity",
            "seed_contract",
            "stimulus_contract",
            "verifier_contract",
            "revision_binding",
            "execution_status",
            "contract_hash",
        }
        unknown_fields = sorted(set(row) - allowed_fields)
        if unknown_fields:
            raise ValueError(
                "retained regression obligation contains unsupported fields: "
                + ", ".join(unknown_fields)
            )
        case_id = str(row.get("case_id") or "").strip()
        variant = str(row.get("variant") or "").strip()
        obligation_id = str(row.get("obligation_id") or "").strip()
        if case_id not in expected_case_ids:
            raise ValueError(f"unknown retained regression case: {case_id}")
        if variant not in RETAINED_REGRESSION_VARIANTS:
            raise ValueError(f"unknown retained regression variant: {variant}")
        expected_id = _obligation_id(case_id, variant)
        if obligation_id != expected_id:
            raise ValueError(
                f"unstable retained regression obligation id: {obligation_id!r}"
            )
        pair = (case_id, variant)
        if obligation_id in observed_ids or pair in observed_pairs:
            raise ValueError(f"duplicate retained regression obligation: {obligation_id}")
        observed_ids.add(obligation_id)
        observed_pairs.add(pair)
        observed_case_ids.add(case_id)

        if row.get("schema_version") != RETAINED_REGRESSION_SCHEMA_VERSION:
            raise ValueError(f"unsupported retained regression schema: {obligation_id}")
        if row.get("execution_status") != "not_run":
            raise ValueError(
                f"retained regression catalog cannot claim execution: {obligation_id}"
            )
        contract = _CASE_CONTRACTS[case_id]
        if str(row.get("severity") or "") != contract.severity:
            raise ValueError(f"invalid retained regression severity: {obligation_id}")
        revision_binding = dict(row.get("revision_binding") or {})
        if revision_binding.get("revision") != active_revision:
            raise ValueError(f"stale retained regression revision: {obligation_id}")
        _validate_source_binding_shape(
            revision_binding.get("source_fixture"),
            obligation_id=obligation_id,
        )
        if source_binding is not None:
            actual_source = revision_binding.get("source_fixture")
            if actual_source != dict(source_binding):
                raise ValueError(f"stale retained regression fixture binding: {obligation_id}")

        _validate_seed_contract(row, contract=contract, obligation_id=obligation_id)
        _validate_stimulus_contract(row, variant=variant, obligation_id=obligation_id)
        _validate_verifier_contract(row, contract=contract, variant=variant)

        unsigned = dict(row)
        recorded_hash = str(unsigned.pop("contract_hash", "") or "")
        if not recorded_hash or recorded_hash != content_hash(unsigned):
            raise ValueError(f"stale retained regression contract hash: {obligation_id}")

    if observed_case_ids != expected_case_ids:
        missing = sorted(expected_case_ids - observed_case_ids)
        raise ValueError(f"retained regression catalog is missing cases: {missing}")
    expected_pairs = {
        (case_id, variant)
        for case_id in expected_case_ids
        for variant in RETAINED_REGRESSION_VARIANTS
    }
    if observed_pairs != expected_pairs:
        missing = sorted(expected_pairs - observed_pairs)
        raise ValueError(f"retained regression catalog is missing variants: {missing}")


def _build_obligation(
    *,
    case: Mapping[str, Any],
    contract: _CaseContract,
    variant: str,
    state_fingerprint: str,
    revision: Mapping[str, str],
    source_binding: Mapping[str, Any],
    immutable_case_contract: Mapping[str, Any],
    variant_declaration: Mapping[str, Any],
) -> dict[str, Any]:
    case_id = str(case["id"])
    source_turns = tuple(str(turn) for turn in case.get("turns") or ())
    expected_text = str(case.get("expected") or "")
    precondition = str(case.get("precondition") or "")
    required = list(contract.required_postcondition_ids)
    source_contract = build_source_contract(immutable_case_contract)
    variant_contract = build_variant_contract(
        variant=variant,
        source_contract=source_contract,
        declaration=variant_declaration,
    )
    common_stimulus = {
        "source_contract": source_contract,
        "source_contract_hash": content_hash(source_contract),
        "variant_contract": variant_contract,
        "variant_contract_hash": content_hash(variant_contract),
    }
    if variant == "exact":
        required.insert(0, "exact_fixture_turns_observed")
        stimulus = {
            **common_stimulus,
            "mode": "exact_fixture_replay",
            "turns": list(source_turns),
            "turns_hash": content_hash(source_turns),
        }
    elif variant == "isomorphic":
        required[:0] = [
            "response_driven_selection_observed",
            "isomorphic_meaning_attested",
        ]
        stimulus = {
            **common_stimulus,
            "mode": "response_driven_isomorphic",
            "source_turns_hash": content_hash(source_turns),
            "constraints": {
                "preserve_case_intent": True,
                "change_surface_form": True,
                "read_each_complete_agent_response": True,
                "prewritten_future_turns_forbidden": True,
            },
        }
    elif variant == "negative":
        required[:0] = [
            "response_driven_selection_observed",
            "adjacent_non_trigger_attested",
        ]
        stimulus = {
            **common_stimulus,
            "mode": "response_driven_negative",
            "source_turns_hash": content_hash(source_turns),
            "constraints": {
                "exercise_adjacent_but_non_triggering_intent": True,
                "read_each_complete_agent_response": True,
                "prewritten_future_turns_forbidden": True,
            },
        }
    elif variant == "neighboring":
        required[:0] = [
            "response_driven_selection_observed",
            "neighboring_transition_attested",
        ]
        stimulus = {
            **common_stimulus,
            "mode": "response_driven_neighboring",
            "source_turns_hash": content_hash(source_turns),
            "constraints": {
                "exercise_neighboring_authoritative_transition": True,
                "read_each_complete_agent_response": True,
                "prewritten_future_turns_forbidden": True,
            },
        }
    else:
        raise ValueError(f"unknown retained regression variant: {variant}")

    unsigned: dict[str, Any] = {
        "schema_version": RETAINED_REGRESSION_SCHEMA_VERSION,
        "obligation_id": _obligation_id(case_id, variant),
        "case_id": case_id,
        "variant": variant,
        "severity": contract.severity,
        "seed_contract": {
            "provider": SEED_PROVIDER,
            "scenario_id": contract.scenario_id,
            "scenario_state_fingerprint": state_fingerprint,
            "precondition_hash": content_hash(precondition),
        },
        "stimulus_contract": stimulus,
        "verifier_contract": {
            "registry_id": POSTCONDITION_REGISTRY_ID,
            "required_postcondition_ids": required,
            "forbidden_postcondition_ids": list(contract.forbidden_postcondition_ids),
            "required_bindings": [
                _verifier_binding(postcondition_id)
                for postcondition_id in required
            ],
            "forbidden_bindings": [
                _verifier_binding(postcondition_id)
                for postcondition_id in contract.forbidden_postcondition_ids
            ],
            "required_evidence": [
                "runtime_turn_events",
                "checkpoint_state_diff",
                "complete_agent_responses",
            ],
            "fixture_expected_text_hash": content_hash(expected_text),
            "fixture_expected_text_is_verifier": False,
        },
        "revision_binding": {
            "revision": dict(revision),
            "source_fixture": dict(source_binding),
        },
        "execution_status": "not_run",
    }
    return {**unsigned, "contract_hash": content_hash(unsigned)}


def _validate_seed_contract(
    row: Mapping[str, Any],
    *,
    contract: _CaseContract,
    obligation_id: str,
) -> None:
    seed = dict(row.get("seed_contract") or {})
    if seed.get("provider") != SEED_PROVIDER:
        raise ValueError(f"unknown retained regression seed provider: {obligation_id}")
    if seed.get("scenario_id") != contract.scenario_id:
        raise ValueError(f"retained regression seed scenario mismatch: {obligation_id}")
    try:
        scenario = reviewed_scenario(str(seed.get("scenario_id") or ""))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"retained regression seed is not executable: {obligation_id}"
        ) from exc
    if not _scenario_is_executable(scenario):
        raise ValueError(f"retained regression seed is not executable: {obligation_id}")
    fingerprint = str(getattr(scenario, "state_fingerprint", "") or "")
    if not fingerprint or seed.get("scenario_state_fingerprint") != fingerprint:
        raise ValueError(f"retained regression seed fingerprint is stale: {obligation_id}")
    if not str(seed.get("precondition_hash") or ""):
        raise ValueError(f"retained regression seed has no precondition hash: {obligation_id}")


def _validate_stimulus_contract(
    row: Mapping[str, Any],
    *,
    variant: str,
    obligation_id: str,
) -> None:
    stimulus = dict(row.get("stimulus_contract") or {})
    source_contract = validate_source_contract(
        dict(stimulus.get("source_contract") or {})
    )
    variant_contract = validate_variant_contract(
        dict(stimulus.get("variant_contract") or {}),
        source_contract=source_contract,
    )
    if (
        source_contract.get("case_id") != row.get("case_id")
        or source_contract.get("source_turns_hash")
        != stimulus.get("source_turns_hash", stimulus.get("turns_hash"))
        or stimulus.get("source_contract_hash") != content_hash(source_contract)
        or stimulus.get("variant_contract_hash") != content_hash(variant_contract)
        or variant_contract.get("variant") != variant
        or variant_contract.get("relation") != RELATION_BY_VARIANT[variant]
    ):
        raise ValueError(
            f"retained regression immutable stimulus binding is stale: {obligation_id}"
        )
    expected_modes = {
        "exact": "exact_fixture_replay",
        "isomorphic": "response_driven_isomorphic",
        "negative": "response_driven_negative",
        "neighboring": "response_driven_neighboring",
    }
    if stimulus.get("mode") != expected_modes[variant]:
        raise ValueError(f"retained regression stimulus mode mismatch: {obligation_id}")
    if variant == "exact":
        turns = stimulus.get("turns")
        if (
            not isinstance(turns, Sequence)
            or isinstance(turns, (str, bytes))
            or not turns
            or any(not isinstance(turn, str) or not turn for turn in turns)
        ):
            raise ValueError(f"exact retained regression has no executable turns: {obligation_id}")
        if stimulus.get("turns_hash") != content_hash(tuple(turns)):
            raise ValueError(f"exact retained regression turns are stale: {obligation_id}")
        if source_contract["source_turns_hash"] != stimulus["turns_hash"]:
            raise ValueError(
                f"exact retained regression source contract is stale: {obligation_id}"
            )
        return
    if "turns" in stimulus or "messages" in stimulus:
        raise ValueError(
            f"response-driven retained regression contains prewritten turns: {obligation_id}"
        )
    constraints = dict(stimulus.get("constraints") or {})
    if (
        not stimulus.get("source_turns_hash")
        or constraints.get("read_each_complete_agent_response") is not True
        or constraints.get("prewritten_future_turns_forbidden") is not True
    ):
        raise ValueError(
            f"response-driven retained regression is not executable: {obligation_id}"
        )


def _validate_verifier_contract(
    row: Mapping[str, Any],
    *,
    contract: _CaseContract,
    variant: str,
) -> None:
    obligation_id = str(row.get("obligation_id") or "")
    verifier = dict(row.get("verifier_contract") or {})
    if verifier.get("registry_id") != POSTCONDITION_REGISTRY_ID:
        raise ValueError(f"unknown retained regression verifier registry: {obligation_id}")
    required = tuple(verifier.get("required_postcondition_ids") or ())
    forbidden = tuple(verifier.get("forbidden_postcondition_ids") or ())
    if not required or len(required) != len(set(required)):
        raise ValueError(f"invalid required postconditions: {obligation_id}")
    if len(forbidden) != len(set(forbidden)):
        raise ValueError(f"invalid forbidden postconditions: {obligation_id}")
    unknown = (set(required) | set(forbidden)) - KNOWN_POSTCONDITION_IDS
    if unknown:
        raise ValueError(
            f"unknown retained regression postconditions: {sorted(unknown)}"
        )
    if not set(contract.required_postcondition_ids).issubset(required):
        raise ValueError(f"case postconditions are missing: {obligation_id}")
    if tuple(contract.forbidden_postcondition_ids) != forbidden:
        raise ValueError(f"case forbidden postconditions changed: {obligation_id}")
    if verifier.get("required_bindings") != [
        _verifier_binding(postcondition_id)
        for postcondition_id in required
    ]:
        raise ValueError(f"required verifier bindings changed: {obligation_id}")
    if verifier.get("forbidden_bindings") != [
        _verifier_binding(postcondition_id)
        for postcondition_id in forbidden
    ]:
        raise ValueError(f"forbidden verifier bindings changed: {obligation_id}")
    if variant != "exact" and "response_driven_selection_observed" not in required:
        raise ValueError(f"response-driven verifier is missing: {obligation_id}")
    if verifier.get("fixture_expected_text_is_verifier") is not False:
        raise ValueError(f"fixture expected text cannot be a verifier: {obligation_id}")
    if not str(verifier.get("fixture_expected_text_hash") or ""):
        raise ValueError(f"fixture expected traceability hash is missing: {obligation_id}")
    if "expected" in verifier or "expected_text" in verifier:
        raise ValueError(f"natural-language expected text leaked into verifier: {obligation_id}")


def _validate_fixture_source(
    *,
    cases_payload: Mapping[str, Any],
    cases_bytes: bytes,
    variant_contracts_payload: Mapping[str, Any],
    variant_contracts_bytes: bytes,
    manifest_payload: Mapping[str, Any],
    manifest_bytes: bytes,
) -> dict[str, Any]:
    if cases_payload.get("schema_version") != 1:
        raise ValueError("unsupported retained regression fixture schema")
    cases = cases_payload.get("cases")
    if not isinstance(cases, list) or len(cases) != RETAINED_REGRESSION_CASE_COUNT:
        raise ValueError(
            "retained regression fixture must contain exactly "
            f"{RETAINED_REGRESSION_CASE_COUNT} cases"
        )
    case_ids = [str(case.get("id") or "").strip() for case in cases]
    if any(not case_id for case_id in case_ids) or len(case_ids) != len(set(case_ids)):
        raise ValueError("retained regression fixture has missing or duplicate case ids")
    if set(case_ids) != set(_CASE_CONTRACTS):
        raise ValueError("retained regression fixture case set is not reviewed")

    if variant_contracts_payload.get("schema_version") != 1:
        raise ValueError("unsupported retained regression variant contract schema")
    raw_variant_cases = variant_contracts_payload.get("cases")
    if (
        not isinstance(raw_variant_cases, list)
        or len(raw_variant_cases) != RETAINED_REGRESSION_CASE_COUNT
    ):
        raise ValueError(
            "retained regression variant contracts must cover every case"
        )
    variant_cases: dict[str, Mapping[str, Any]] = {}
    case_turn_hashes = {
        str(case["id"]): content_hash(tuple(case.get("turns") or ()))
        for case in cases
    }
    for raw in raw_variant_cases:
        if not isinstance(raw, Mapping):
            raise ValueError("retained regression variant case is invalid")
        case_contract = build_source_contract(raw)
        case_id = str(case_contract["case_id"])
        if (
            case_id in variant_cases
            or case_turn_hashes.get(case_id)
            != case_contract["source_turns_hash"]
        ):
            raise ValueError(
                "retained regression variant source binding is stale"
            )
        variant_cases[case_id] = case_contract
    if set(variant_cases) != set(case_ids):
        raise ValueError("retained regression variant case set is incomplete")
    declarations = variant_contracts_payload.get("variant_contracts")
    if (
        not isinstance(declarations, Mapping)
        or set(declarations) != set(RETAINED_REGRESSION_VARIANTS)
    ):
        raise ValueError("retained regression variant declarations are incomplete")
    exemplar = next(iter(variant_cases.values()))
    for variant, declaration in declarations.items():
        if not isinstance(declaration, Mapping):
            raise ValueError("retained regression variant declaration is invalid")
        build_variant_contract(
            variant=str(variant),
            source_contract=exemplar,
            declaration=declaration,
        )

    if manifest_payload.get("schema_version") != 1:
        raise ValueError("unsupported retained regression manifest schema")
    files = manifest_payload.get("files")
    if not isinstance(files, list) or len(files) != 2:
        raise ValueError("retained regression manifest must declare two fixtures")
    file_index = {
        str(item.get("path") or ""): dict(item)
        for item in files
        if isinstance(item, Mapping)
    }
    if set(file_index) != {
        RETAINED_REGRESSION_FIXTURE.as_posix(),
        RETAINED_REGRESSION_VARIANT_CONTRACTS.as_posix(),
    }:
        raise ValueError("retained regression manifest paths are not authoritative")
    if any(
        item.get("sanitized") is not True
        or item.get("case_count") != RETAINED_REGRESSION_CASE_COUNT
        for item in file_index.values()
    ):
        raise ValueError("retained regression manifest contract is stale")
    cases_sha256 = hashlib.sha256(cases_bytes).hexdigest()
    variants_sha256 = hashlib.sha256(variant_contracts_bytes).hexdigest()
    if (
        file_index[RETAINED_REGRESSION_FIXTURE.as_posix()].get("sha256")
        != cases_sha256
        or file_index[RETAINED_REGRESSION_VARIANT_CONTRACTS.as_posix()].get(
            "sha256"
        )
        != variants_sha256
    ):
        raise ValueError("retained regression fixture hash does not match manifest")
    source_revision = str(manifest_payload.get("source_revision") or "").strip()
    if not source_revision:
        raise ValueError("retained regression manifest has no source revision")
    return {
        "fixture_path": RETAINED_REGRESSION_FIXTURE.as_posix(),
        "fixture_sha256": cases_sha256,
        "variant_contract_path": (
            RETAINED_REGRESSION_VARIANT_CONTRACTS.as_posix()
        ),
        "variant_contract_sha256": variants_sha256,
        "manifest_path": RETAINED_REGRESSION_MANIFEST.as_posix(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_revision": source_revision,
        "case_count": RETAINED_REGRESSION_CASE_COUNT,
        "sanitized": True,
    }


def _validate_source_binding_shape(value: Any, *, obligation_id: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"retained regression source binding is missing: {obligation_id}")
    expected_fields = {
        "fixture_path",
        "fixture_sha256",
        "variant_contract_path",
        "variant_contract_sha256",
        "manifest_path",
        "manifest_sha256",
        "source_revision",
        "case_count",
        "sanitized",
    }
    if set(value) != expected_fields:
        raise ValueError(f"retained regression source binding is incomplete: {obligation_id}")
    if (
        value.get("fixture_path") != RETAINED_REGRESSION_FIXTURE.as_posix()
        or value.get("variant_contract_path")
        != RETAINED_REGRESSION_VARIANT_CONTRACTS.as_posix()
        or value.get("manifest_path") != RETAINED_REGRESSION_MANIFEST.as_posix()
        or len(str(value.get("fixture_sha256") or "")) != 64
        or len(str(value.get("variant_contract_sha256") or "")) != 64
        or len(str(value.get("manifest_sha256") or "")) != 64
        or not str(value.get("source_revision") or "").strip()
        or value.get("case_count") != RETAINED_REGRESSION_CASE_COUNT
        or value.get("sanitized") is not True
    ):
        raise ValueError(f"retained regression source binding is invalid: {obligation_id}")


def _validated_revision(revision: Mapping[str, str]) -> dict[str, str]:
    commit = str(revision.get("commit") or "").strip()
    worktree_hash = str(revision.get("worktree_hash") or "").strip()
    if not commit or not worktree_hash:
        raise ValueError("retained regression obligations require a revision binding")
    return {"commit": commit, "worktree_hash": worktree_hash}


def _scenario_is_executable(scenario: Any) -> bool:
    return bool(
        getattr(scenario, "seed_state", None)
        and str(getattr(scenario, "state_fingerprint", "") or "").strip()
    )


def _obligation_id(case_id: str, variant: str) -> str:
    return f"g3-retained-{case_id.lower()}-{variant}-v1"

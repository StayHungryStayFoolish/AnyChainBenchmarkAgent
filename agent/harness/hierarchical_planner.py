"""Hierarchical semantic planning for current product turns.

Stage A partitions the complete turn and routes exact source units to domain
owners. Stage B compiles only the actions owned by each selected owner. The
result continues through the immutable whole-plan admission boundary until
the graph owns the complete turn receipt in Phase 3.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..llm.providers import provider_from_config
from ..llm.types import (
    LLMProviderError,
    LLMTurnTimeoutError,
    run_independent_llm_tasks,
)
from .action_registry import (
    ACTION_ARGUMENT_SCHEMAS,
    ACTION_BY_TYPE,
    ACTION_SPECS,
    SEMANTIC_OPERATION_PURPOSES,
    SEMANTIC_OPERATIONS,
    UNIVERSAL_SEMANTIC_OPERATION_OWNERS,
    pending_barrier_semantics,
    registered_semantic_value_domains,
    semantic_grounding_arguments,
    semantic_grounding_values,
    semantic_value_domain_conflicts,
    action_spec_serves_route,
    lower_empty_entry_action_to_registered_intake,
    validate_action_contract,
)
from .context import (
    action_schema,
    group_schema,
    owner_workflow_snapshot,
    workflow_snapshot,
)
from .domains.environment import extract_structured_input_candidates
from .semantic_admission import (
    _admitted_action_queue,
    _review_bounded_semantic_candidate,
    _semantic_fulfillment_prompt,
    _strip_candidate_admission_metadata,
    _unresolved_action_queue,
    prepare_hierarchical_candidate,
)
from .plan_coverage import (
    PlanCoverageResult,
    TurnClause,
    segment_user_turn,
    validate_semantic_partition,
)
from .questions import (
    exact_option_prefix_answer,
    manual_action_for_value,
    pending_option_value_exists,
    pending_value_identity,
    researched_identity_value_is_valid,
    semantic_pending_question,
    typed_pending_value_candidates,
    value_satisfies_pending_contract,
)
from .queue import mutation_conflict_action_groups
from .semantic_compiler import (
    STRICT_JSON_REASONING_MODE,
    closed_enum_quote_names_only_competing_values,
    is_explicit_semantic_rejection,
    request_open_identity_relation_jury,
    request_semantic_compilation,
    whole_plan_admission_prompt,
)
from .semantic_drafts import (
    build_semantic_plan_draft,
    semantic_draft_atom_resolution_binding,
    semantic_draft_question_binding,
    validate_semantic_plan_draft,
)
from .semantic_policy import (
    FRAMED_REQUEST_SEMANTIC_POLICY,
    PENDING_CANDIDATE_SEMANTIC_POLICY,
)
from .secret_refs import (
    authorize_secret_reference,
    new_secret_reference,
    resolve_secret_reference,
    secret_value_verifier,
    store_secret_reference,
)
from .state import AgentGraphState
from agent.utils.redaction import secret_values
from agent.workflows.group_registry import GROUP_SPEC_BY_NAME


_OWNERS = frozenset(spec.owner for spec in ACTION_BY_TYPE.values())
_UNIVERSAL_OPERATIONS = SEMANTIC_OPERATIONS
_UNIVERSAL_OPERATION_OWNERS = UNIVERSAL_SEMANTIC_OPERATION_OWNERS
_PARTITION_KEYS = frozenset({
    "unit_id",
    "clause_id",
    "source_text",
    "source_path",
    "operation",
    "owner_routes",
    "reason",
})


def _replace_secret_values(value: Any, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        result = value
        for secret, reference in replacements.items():
            result = result.replace(secret, reference)
        return result
    if isinstance(value, Mapping):
        return {
            key: _replace_secret_values(item, replacements)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_secret_values(item, replacements) for item in value]
    if isinstance(value, tuple):
        return tuple(_replace_secret_values(item, replacements) for item in value)
    return value


def _source_secret_projection(
    original_input: str,
    existing_bindings: Sequence[Mapping[str, Any]] = (),
) -> tuple[dict[str, str], list[dict[str, str]], dict[str, str]]:
    replacements: dict[str, str] = {}
    bindings: list[dict[str, str]] = []
    raw_by_reference: dict[str, str] = {}
    for raw in existing_bindings:
        reference = str(raw.get("reference") or "")
        if reference and reference in original_input:
            bindings.append({
                "reference": reference,
                "atom_id": str(raw.get("atom_id") or ""),
                "value_hash": str(raw.get("value_hash") or ""),
            })
    for index, secret in enumerate(secret_values(original_input)):
        reference = new_secret_reference()
        atom_id = f"source-secret-{index + 1}"
        replacements[secret] = reference
        raw_by_reference[reference] = secret
        bindings.append({
            "reference": reference,
            "atom_id": atom_id,
            "value_hash": secret_value_verifier(secret, reference),
        })
    return replacements, bindings, raw_by_reference
_ROUTE_KEYS = frozenset({"owner", "group"})
_BINDING_KEYS = frozenset({"unit_id", "action_indexes", "disposition", "reason"})


@dataclass(frozen=True)
class HierarchicalPlannerMetrics:
    stage_a_calls: int
    stage_b_calls: int
    admission_calls: int
    prompt_bytes: int
    largest_request_bytes: int
    elapsed_ms: int
    owner_count: int
    unit_count: int

    def as_dict(self) -> dict[str, int]:
        return {
            "stage_a_calls": self.stage_a_calls,
            "stage_b_calls": self.stage_b_calls,
            "admission_calls": self.admission_calls,
            "model_calls": (
                self.stage_a_calls + self.stage_b_calls + self.admission_calls
            ),
            "prompt_bytes": self.prompt_bytes,
            "largest_request_bytes": self.largest_request_bytes,
            "elapsed_ms": self.elapsed_ms,
            "owner_count": self.owner_count,
            "unit_count": self.unit_count,
        }


def _semantic_draft_clarification_prompt() -> str:
    """Return the bounded review contract for one active draft atom."""

    return (
        "You are an independent semantic-draft clarification reviewer for "
        "AnyChain Benchmark Agent. Return one strict JSON object with exactly "
        "contract_hash, verdict, evidence_quote, resolution_disposition, and "
        "reason. verdict is one of "
        "clarifies_atom, does_not_clarify, or uncertain. The active_atom is the "
        "exact unresolved demand currently shown to the user. Select "
        "clarifies_atom only when one exact contiguous span of the current user "
        "turn answers, corrects, replaces, supersedes, or explicitly declines "
        "that exact demand. Copy that complete span byte-for-byte into "
        "evidence_quote. resolution_disposition must be semantic_value when the "
        "span supplies or changes the atom's meaning, and background when the "
        "span explicitly establishes that the atom is retained context requiring "
        "no product action. For does_not_clarify or uncertain use an empty "
        "resolution_disposition. Do not include an independent sibling request in the "
        "quote; normal whole-turn routing must preserve every unquoted span. "
        "Select does_not_clarify or uncertain when no single exact source span "
        "is a complete clarification. Questions about the clarification prompt, "
        "unrelated navigation, new configuration, and new analysis requests do "
        "not answer the atom merely because they occur while it is pending. Do "
        "not infer product actions, mutate state, shorten or paraphrase evidence, "
        "or trust prior reviewer output. Copy contract_hash exactly. reason must "
        "be non-empty."
    )


def _review_bound_semantic_draft_clarification(
    provider: Any,
    state: AgentGraphState,
    text: str,
    clauses: Sequence[TurnClause],
) -> tuple[dict[str, Any], tuple[int, ...], dict[str, Any]]:
    """Prove that a complete turn is evidence for the exact active draft atom.

    The jury may authorize only the pending clarification boundary. It does not
    compile an action or decide the meaning of any other product operation.
    """

    pending = dict(state.get("pending_question") or {})
    binding = dict(pending.get("semantic_draft_binding") or {})
    draft = dict(state.get("semantic_plan_draft") or {})
    if not binding or draft.get("status") != "awaiting_clarification":
        return {}, (), {}
    try:
        draft = validate_semantic_plan_draft(draft)
        expected = semantic_draft_question_binding(draft)
    except ValueError:
        return {}, (), {}
    if binding != expected:
        return {}, (), {}
    active_atom = next(
        (
            dict(item)
            for item in draft.get("unresolved_atoms") or ()
            if isinstance(item, Mapping)
            and str(item.get("atom_id") or "")
            == str(binding.get("atom_id") or "")
        ),
        {},
    )
    user_text = str(text or "").strip()
    if not active_atom or not user_text:
        return {}, (), {}
    contract = {
        "draft_id": str(binding.get("draft_id") or ""),
        "revision": int(binding.get("revision") or 0),
        "atom_id": str(binding.get("atom_id") or ""),
        "active_atom": {
            "source_text": str(active_atom.get("source_text") or ""),
            "source_path": str(active_atom.get("source_path") or ""),
            "reason": str(active_atom.get("reason") or ""),
            "reason_detail": str(active_atom.get("reason_detail") or ""),
        },
        "user_text": user_text,
        "clauses": [clause.as_dict() for clause in clauses],
    }
    contract_hash = hashlib.sha256(
        json.dumps(
            contract,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    payload = {"contract_hash": contract_hash, **contract}
    prompt = _semantic_draft_clarification_prompt()

    def review() -> Mapping[str, Any]:
        response = request_semantic_compilation(
            provider,
            system_prompt=prompt,
            request_payload=payload,
            max_tokens=700,
            reasoning_mode=STRICT_JSON_REASONING_MODE,
        )
        try:
            document = json.loads(response)
        except json.JSONDecodeError:
            return {}
        if not isinstance(document, Mapping) or set(document) != {
            "contract_hash",
            "verdict",
            "evidence_quote",
            "resolution_disposition",
            "reason",
        }:
            return {}
        verdict = str(document.get("verdict") or "")
        quote = str(document.get("evidence_quote") or "")
        disposition = str(document.get("resolution_disposition") or "")
        if (
            str(document.get("contract_hash") or "") != contract_hash
            or verdict not in {
                "clarifies_atom",
                "does_not_clarify",
                "uncertain",
            }
            or not str(document.get("reason") or "").strip()
            or not quote
            or quote not in user_text
            or (
                verdict == "clarifies_atom"
                and disposition not in {"semantic_value", "background"}
            )
            or (
                verdict != "clarifies_atom"
                and disposition
            )
            or (
                verdict == "clarifies_atom"
                and user_text.count(quote) != 1
            )
        ):
            return {}
        return dict(document)

    reviews = run_independent_llm_tasks((review, review, review))
    request_size = len(prompt.encode("utf-8")) + len(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    valid_reviews = [dict(item) for item in reviews if item]
    clarifying_identities = [
        (
            str(item.get("evidence_quote") or ""),
            str(item.get("resolution_disposition") or ""),
        )
        for item in valid_reviews
        if str(item.get("verdict") or "") == "clarifies_atom"
    ]
    identity_votes = {
        identity: clarifying_identities.count(identity)
        for identity in set(clarifying_identities)
    }
    selected_identity = max(
        identity_votes,
        key=lambda identity: (
            identity_votes[identity],
            len(identity[0]),
            identity,
        ),
        default=("", ""),
    )
    selected_quote, selected_disposition = selected_identity
    clarifying_votes = identity_votes.get(selected_identity, 0)
    candidate = (
        {
            "contract_hash": contract_hash,
            "draft_id": contract["draft_id"],
            "revision": contract["revision"],
            "atom_id": contract["atom_id"],
            "evidence_quote": selected_quote,
            "resolution_disposition": selected_disposition,
        }
        if clarifying_votes >= 2
        else {}
    )
    receipt = {
        "contract_hash": contract_hash,
        "valid_review_count": len(valid_reviews),
        "clarifying_vote_count": clarifying_votes,
        "resolution_disposition": (
            selected_disposition if clarifying_votes >= 2 else ""
        ),
        "verdict_hashes": [
            hashlib.sha256(
                json.dumps(
                    item,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            for item in valid_reviews
        ],
        "candidate_authorized": bool(candidate),
    }
    return candidate, (request_size, request_size, request_size), receipt


def _bound_semantic_draft_resolution_partition(
    state: AgentGraphState,
    text: str,
    candidate: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Project one jury-authorized clarification onto the signed owner route."""

    pending = dict(state.get("pending_question") or {})
    binding = dict(pending.get("semantic_draft_binding") or {})
    user_text = str(text or "").strip()
    evidence = str(candidate.get("evidence_quote") or "")
    if (
        not binding
        or str(candidate.get("draft_id") or "")
        != str(binding.get("draft_id") or "")
        or int(candidate.get("revision") or 0)
        != int(binding.get("revision") or 0)
        or str(candidate.get("atom_id") or "")
        != str(binding.get("atom_id") or "")
        or not evidence
        or evidence not in user_text
    ):
        raise ValueError("semantic draft clarification candidate is not bound")
    return [{
        "unit_id": "__harness_semantic_draft_resolution",
        "clause_id": "clause-1",
        "source_text": evidence,
        "operation": "pending_answer",
        "owner_routes": [{
            "owner": "coordinator",
            "group": str(
                pending.get("group")
                or state.get("active_group")
                or "opening"
            ),
        }],
        "reason": (
            "Independent semantic-draft jury bound the complete turn to the "
            "exact active atom."
        ),
    }]


def _bind_active_semantic_draft_disposition(
    partition: Sequence[Mapping[str, Any]],
    stage_a_payload: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    """Attach the jury disposition to its exact admitted source unit."""

    contract = dict(
        stage_a_payload.get("contract_proven_semantic_draft_resolution") or {}
    )
    if not contract:
        return [dict(unit) for unit in partition], ()
    evidence = str(contract.get("evidence_quote") or "")
    disposition = str(contract.get("resolution_disposition") or "")
    if disposition not in {"semantic_value", "background"}:
        return [dict(unit) for unit in partition], (
            "semantic draft clarification disposition is invalid",
        )
    matches = [
        index
        for index, unit in enumerate(partition)
        if str(unit.get("operation") or "") == "pending_answer"
        and str(unit.get("source_text") or "") == evidence
        and any(
            str(route.get("owner") or "") == "coordinator"
            for route in unit.get("owner_routes") or ()
            if isinstance(route, Mapping)
        )
    ]
    if len(matches) != 1:
        return [dict(unit) for unit in partition], (
            "semantic draft clarification disposition has no unique source unit",
        )
    output = [dict(unit) for unit in partition]
    output[matches[0]]["_semantic_draft_resolution_disposition"] = disposition
    return output, ()


def begin_semantic_partition(
    state: AgentGraphState,
    text: str,
) -> dict[str, Any]:
    """Run Stage A and persist an immutable owner-compilation schedule."""

    started = time.monotonic()
    clauses = _turn_clauses(state, text)
    document: dict[str, Any] = {
        "contract_version": 1,
        "status": "partition",
        "started_monotonic": started,
        "clauses": [clause.as_dict() for clause in clauses],
        "source_partition": [],
        "routed_partition": [],
        "owner_requests": [],
        "owner_cursor": 0,
        "owner_documents": {},
        "request_sizes": [],
        "stage_a_calls": 0,
        "stage_b_calls": 0,
        "admission_calls": 0,
        "stage_a_relation_reviews": [],
        "pending_entailment_reviews": [],
        "errors": [],
    }
    if not clauses:
        document["status"] = "failed"
        document["errors"] = ["empty semantic turn"]
        return document
    try:
        provider = provider_from_config()
        stage_a_payload = _stage_a_payload(state, text, clauses)
        (
            clarification_candidate,
            clarification_sizes,
            clarification_receipt,
        ) = _review_bound_semantic_draft_clarification(
            provider,
            state,
            text,
            clauses,
        )
        document["request_sizes"].extend(clarification_sizes)
        document["admission_calls"] += len(clarification_sizes)
        if clarification_receipt:
            document["semantic_draft_clarification_review"] = (
                clarification_receipt
            )
        if clarification_candidate:
            clarification_evidence = str(
                clarification_candidate.get("evidence_quote") or ""
            )
            stage_a_payload["contract_proven_semantic_draft_resolution"] = (
                clarification_candidate
            )
        else:
            clarification_evidence = ""
        if clarification_candidate and clarification_evidence == str(
            text or ""
        ).strip():
            clauses = (TurnClause("clause-1", str(text or "").strip(), "prose"),)
            document["clauses"] = [clause.as_dict() for clause in clauses]
            stage_a_payload = _stage_a_payload(state, text, clauses)
            stage_a_payload["contract_proven_semantic_draft_resolution"] = (
                clarification_candidate
            )
        stage_a_prompt = _stage_a_prompt()
        if clarification_candidate and clarification_evidence == str(
            text or ""
        ).strip():
            partition = _bound_semantic_draft_resolution_partition(
                state,
                text,
                clarification_candidate,
            )
            partition_errors = ()
            proposal_sizes = ()
        else:
            partition, partition_errors, proposal_sizes = _request_stage_a_proposal(
                provider,
                stage_a_payload,
                clauses,
                state,
                stage_a_prompt,
            )
        document["request_sizes"].extend(proposal_sizes)
        document["stage_a_calls"] += len(proposal_sizes)
        if partition_errors:
            document["status"] = "failed"
            document["errors"] = list(partition_errors)
            document["unit_count"] = len(partition)
            return document
        (
            partition,
            _primary_relation_errors,
            primary_relation_sizes,
            primary_relation_receipt,
        ) = _review_competing_open_identity_relations(
            provider,
            partition,
            stage_a_payload,
        )
        document["request_sizes"].extend(primary_relation_sizes)
        document["admission_calls"] += len(primary_relation_sizes)
        if primary_relation_receipt:
            document["stage_a_relation_reviews"].append(
                primary_relation_receipt
            )
        (
            admission_errors,
            stage_a_admission_sizes,
            redundant_unit_ids,
            primary_review_contract_valid,
            _primary_scope_rejections,
        ) = _review_stage_a_candidate(
            provider,
            stage_a_payload,
            partition,
            receipt_sink=document["pending_entailment_reviews"],
        )
        document["request_sizes"].extend(stage_a_admission_sizes)
        document["admission_calls"] += len(stage_a_admission_sizes)
        primary_review_valid = not admission_errors
        if primary_review_valid:
            source_partition, compilation_partition = (
                _partition_after_stage_a_admission(
                    partition,
                    redundant_unit_ids,
                )
            )
        else:
            source_partition = [dict(unit) for unit in partition]
            compilation_partition = []
        recoverable_primary_rejection = bool(
            admission_errors and primary_review_contract_valid
        )
        if recoverable_primary_rejection or _partition_requires_independent_proposal(
            source_partition,
            stage_a_payload,
        ):
            (
                independent_partition,
                independent_errors,
                independent_sizes,
            ) = _request_stage_a_proposal(
                provider,
                stage_a_payload,
                clauses,
                state,
                stage_a_prompt,
                independent=True,
            )
            document["request_sizes"].extend(independent_sizes)
            document["stage_a_calls"] += len(independent_sizes)
            if independent_errors:
                document["status"] = "failed"
                document["errors"] = list(independent_errors)
                document["unit_count"] = len(source_partition)
                return document
            (
                independent_partition,
                _independent_relation_errors,
                independent_relation_sizes,
                independent_relation_receipt,
            ) = _review_competing_open_identity_relations(
                provider,
                independent_partition,
                stage_a_payload,
            )
            document["request_sizes"].extend(independent_relation_sizes)
            document["admission_calls"] += len(independent_relation_sizes)
            if independent_relation_receipt:
                document["stage_a_relation_reviews"].append(
                    independent_relation_receipt
                )
            (
                independent_admission_errors,
                independent_admission_sizes,
                independent_redundant_unit_ids,
                _independent_review_contract_valid,
                independent_scope_rejections,
            ) = _review_stage_a_candidate(
                provider,
                stage_a_payload,
                independent_partition,
                receipt_sink=document["pending_entailment_reviews"],
            )
            document["request_sizes"].extend(independent_admission_sizes)
            document["admission_calls"] += len(independent_admission_sizes)
            if independent_admission_errors and independent_scope_rejections:
                (
                    replacement_partition,
                    replacement_errors,
                    replacement_sizes,
                ) = _request_stage_a_proposal(
                    provider,
                    stage_a_payload,
                    clauses,
                    state,
                    stage_a_prompt,
                    independent=True,
                    rejected_semantic_claims=independent_scope_rejections,
                )
                document["request_sizes"].extend(replacement_sizes)
                document["stage_a_calls"] += len(replacement_sizes)
                if not replacement_errors:
                    (
                        replacement_partition,
                        _replacement_relation_errors,
                        replacement_relation_sizes,
                        replacement_relation_receipt,
                    ) = _review_competing_open_identity_relations(
                        provider,
                        replacement_partition,
                        stage_a_payload,
                    )
                    document["request_sizes"].extend(replacement_relation_sizes)
                    document["admission_calls"] += len(replacement_relation_sizes)
                    if replacement_relation_receipt:
                        document["stage_a_relation_reviews"].append(
                            replacement_relation_receipt
                        )
                    (
                        replacement_admission_errors,
                        replacement_admission_sizes,
                        replacement_redundant_unit_ids,
                        _replacement_review_contract_valid,
                        _replacement_scope_rejections,
                    ) = _review_stage_a_candidate(
                        provider,
                        stage_a_payload,
                        replacement_partition,
                        receipt_sink=document["pending_entailment_reviews"],
                    )
                    document["request_sizes"].extend(replacement_admission_sizes)
                    document["admission_calls"] += len(replacement_admission_sizes)
                    if not replacement_admission_errors:
                        independent_partition = replacement_partition
                        independent_redundant_unit_ids = (
                            replacement_redundant_unit_ids
                        )
                        independent_admission_errors = ()
                    else:
                        independent_admission_errors = tuple(dict.fromkeys((
                            *independent_admission_errors,
                            *replacement_admission_errors,
                        )))
                else:
                    independent_admission_errors = tuple(dict.fromkeys((
                        *independent_admission_errors,
                        *replacement_errors,
                    )))
            if independent_admission_errors:
                document["status"] = "failed"
                document["errors"] = list(dict.fromkeys((
                    *admission_errors,
                    *independent_admission_errors,
                )))
                document["unit_count"] = len(source_partition)
                return document
            (
                independent_source_partition,
                independent_compilation_partition,
            ) = _partition_after_stage_a_admission(
                independent_partition,
                independent_redundant_unit_ids,
            )
            primary_eligible = (
                primary_review_valid
                and _proposal_selection_eligible(source_partition)
            )
            secondary_eligible = _proposal_selection_eligible(
                independent_source_partition
            )
            if (
                primary_eligible
                and secondary_eligible
                and _partition_semantic_hash(source_partition)
                != _partition_semantic_hash(independent_source_partition)
            ):
                compilation_reviews = run_independent_llm_tasks((
                    lambda: _proposal_compilation_eligible(
                        state,
                        compilation_partition,
                    ),
                    lambda: _proposal_compilation_eligible(
                        state,
                        independent_compilation_partition,
                    ),
                ))
                primary_eligible, primary_compilation_sizes = (
                    compilation_reviews[0]
                )
                secondary_eligible, secondary_compilation_sizes = (
                    compilation_reviews[1]
                )
                compilation_sizes = (
                    *primary_compilation_sizes,
                    *secondary_compilation_sizes,
                )
                document["request_sizes"].extend(compilation_sizes)
                document["stage_b_calls"] += len(compilation_sizes)
            (
                selected_partition,
                convergence_errors,
                convergence_sizes,
                convergence_receipt,
            ) = _select_stage_a_proposal(
                provider,
                stage_a_payload,
                source_partition,
                independent_source_partition,
                primary_eligible=primary_eligible,
                secondary_eligible=secondary_eligible,
            )
            document["request_sizes"].extend(convergence_sizes)
            document["admission_calls"] += len(convergence_sizes)
            document["stage_a_convergence"] = convergence_receipt
            if convergence_errors:
                document["status"] = "failed"
                document["errors"] = list(convergence_errors)
                document["unit_count"] = len(source_partition)
                return document
            if convergence_receipt["selected_proposal"] == "secondary":
                source_partition = independent_source_partition
                compilation_partition = independent_compilation_partition
        elif admission_errors:
            document["status"] = "failed"
            document["errors"] = list(admission_errors)
            document["unit_count"] = len(partition)
            return document
        source_partition, compilation_partition = (
            _canonicalize_unique_manual_pending_partition(
                state, clauses, source_partition, compilation_partition
            )
        )
        source_partition, source_disposition_errors = (
            _bind_active_semantic_draft_disposition(
                source_partition,
                stage_a_payload,
            )
        )
        compilation_partition, compilation_disposition_errors = (
            _bind_active_semantic_draft_disposition(
                compilation_partition,
                stage_a_payload,
            )
        )
        if source_disposition_errors or compilation_disposition_errors:
            document["status"] = "failed"
            document["errors"] = list(dict.fromkeys((
                *source_disposition_errors,
                *compilation_disposition_errors,
            )))
            document["unit_count"] = len(source_partition)
            return document
        source_partition = _expand_partition_routes(source_partition)
        compilation_partition = _expand_partition_routes(
            compilation_partition
        )
        (
            source_partition,
            compilation_partition,
            resolution_errors,
        ) = _bind_semantic_draft_resolution_evidence(
            state,
            source_partition,
            compilation_partition,
        )
        if resolution_errors:
            document["status"] = "failed"
            document["errors"] = list(resolution_errors)
            document["unit_count"] = len(source_partition)
            return document
        routed_partition = compilation_partition
        owner_requests = []
        for owner, unit_ids in _owner_requests(routed_partition).items():
            groups = sorted({
                str(route["group"])
                for unit in routed_partition
                if unit["unit_id"] in unit_ids
                for route in unit["owner_routes"]
                if route["owner"] == owner and route["group"]
            })
            owner_requests.append({
                "owner": owner,
                "unit_ids": list(unit_ids),
                "groups": groups,
            })
        document.update({
            "status": "compile_owner" if owner_requests else "review_plan",
            "source_partition": source_partition,
            "routed_partition": routed_partition,
            "owner_requests": owner_requests,
            "owner_count": len(owner_requests),
            "unit_count": len(routed_partition),
        })
        return document
    except (LLMTurnTimeoutError, LLMProviderError):
        raise
    except Exception as exc:
        document["status"] = "failed"
        document["errors"] = [
            f"hierarchical planner failed: {type(exc).__name__}"
        ]
        return document


def _request_stage_a_proposal(
    provider: Any,
    stage_a_payload: Mapping[str, Any],
    clauses: Sequence[TurnClause],
    state: AgentGraphState,
    prompt: str,
    *,
    independent: bool = False,
    rejected_semantic_claims: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[dict[str, Any]], tuple[str, ...], tuple[int, ...]]:
    """Build one structurally valid proposal without seeing another proposal."""

    partition: list[dict[str, Any]] = []
    errors: tuple[str, ...] = ()
    previous_output = ""
    request_sizes: list[int] = []
    role_prompt = prompt
    if independent:
        role_prompt = (
            f"{prompt} Produce an independent semantic partition from the immutable "
            "source and registries. You have not seen and must not assume another "
            "partition. Use a counterfactual source-first decomposition: first "
            "identify every standalone present request without assuming that the "
            "active pending question was answered. Classify a span as "
            "pending_answer only when the source itself commits to the pending "
            "question's effect; otherwise preserve the standalone request under "
            "its registered operation and owner."
        )
    if rejected_semantic_claims:
        role_prompt = (
            f"{role_prompt} The Harness rejected the semantic claims listed in "
            "rejected_semantic_claims because their registered operation purpose "
            "and complete scope were not explicitly authorized by the immutable "
            "source. Return a complete replacement partition of the original "
            "source. Do not repeat an exact rejected operation/owner/group claim. "
            "Use another registered purpose only when the source supports it; "
            "otherwise emit unresolved. The rejection does not authorize you to "
            "invent a value, route, action, or omitted demand."
        )
    for attempt in range(2):
        request_payload = dict(stage_a_payload)
        if rejected_semantic_claims:
            request_payload["rejected_semantic_claims"] = [
                dict(claim) for claim in rejected_semantic_claims
            ]
        request_prompt = role_prompt
        if attempt:
            request_payload["contract_repair"] = {
                "prior_invalid_output": previous_output,
                "validation_errors": list(errors),
                "instruction": (
                    "Return a complete replacement document. Correct every "
                    "validation error without dropping or paraphrasing source text."
                ),
            }
            request_prompt = (
                f"{role_prompt} This is a contract-repair attempt. The prior "
                f"document was rejected for: {'; '.join(errors)}. Return a new full "
                "document that satisfies those errors exactly."
            )
        request_sizes.append(_wire_size(request_prompt, request_payload))
        previous_output = request_semantic_compilation(
            provider,
            system_prompt=request_prompt,
            request_payload=request_payload,
            max_tokens=_stage_a_output_token_budget(stage_a_payload),
            reasoning_mode=STRICT_JSON_REASONING_MODE,
        )
        partition, errors = _validate_partition_document(
            previous_output,
            clauses,
            state=state,
        )
        partition, background_errors = (
            _project_ready_semantic_draft_backgrounds(
                partition,
                stage_a_payload,
            )
        )
        errors = tuple(dict.fromkeys((
            *errors,
            *background_errors,
            *_pending_answer_contract_errors(partition, state),
            *_semantic_draft_resolution_contract_errors(
                partition,
                stage_a_payload,
            ),
            *_cross_domain_pending_errors(partition, state),
        )))
        if not errors:
            break
    return partition, errors, tuple(request_sizes)


def _project_ready_semantic_draft_backgrounds(
    partition: Sequence[Mapping[str, Any]],
    stage_a_payload: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    """Project reviewed background atoms without granting the model authority."""

    backgrounds = [
        dict(row)
        for row in stage_a_payload.get("semantic_draft_resolutions") or ()
        if isinstance(row, Mapping)
        and str(row.get("resolution_disposition") or "") == "background"
    ]
    if not backgrounds:
        return [dict(unit) for unit in partition], ()
    output = [dict(unit) for unit in partition]
    errors: list[str] = []
    for row in backgrounds:
        matches = [
            index
            for index, unit in enumerate(output)
            if str(unit.get("unit_id") or "") == str(row.get("unit_id") or "")
            and str(unit.get("clause_id") or "") == str(row.get("clause_id") or "")
            and str(unit.get("source_path") or "") == str(row.get("source_path") or "")
            and str(unit.get("source_text") or "") == str(row.get("source_text") or "")
        ]
        if len(matches) != 1:
            errors.append(
                "semantic draft background atom has no unique final source: "
                f"{str(row.get('atom_id') or '')}"
            )
            continue
        unit = output[matches[0]]
        unit["operation"] = "context"
        unit["owner_routes"] = []
        unit["reason"] = "Harness-proven semantic draft background disposition."
        unit["_admission_support"] = True
    return output, tuple(dict.fromkeys(errors))


def _stage_a_output_token_budget(
    stage_a_payload: Mapping[str, Any],
) -> int:
    """Size the bounded partition response by parser-owned atom cardinality."""

    structured_atom_count = sum(
        len(candidate.get("field_candidates") or ())
        for candidate in stage_a_payload.get("structured_candidates") or ()
        if isinstance(candidate, Mapping)
    )
    return min(12000, max(2600, 2200 + (900 * structured_atom_count)))


def _stage_a_review_output_token_budget(
    partition: Sequence[Mapping[str, Any]],
    clauses: Sequence[Mapping[str, Any]],
) -> int:
    """Bound review output by its required unit and clause verdict rows."""

    verdict_rows = len(partition) + len(clauses)
    return min(12000, max(2200, 1600 + (700 * verdict_rows)))


def _partition_requires_independent_proposal(
    partition: Sequence[Mapping[str, Any]],
    stage_a_payload: Mapping[str, Any] | None = None,
) -> bool:
    if any(
        str(unit.get("operation") or "") == "unresolved"
        for unit in partition
    ):
        return True
    if any(
        len(_UNIVERSAL_OPERATION_OWNERS.get(str(unit.get("operation") or ""), ()))
        > 1
        for unit in partition
    ):
        return True
    payload = stage_a_payload or {}
    pending_group = str(
        (payload.get("pending_question") or {}).get("group") or ""
    )
    registered_cross_group_mutation_groups = {
        str(mention.get("target_group") or "")
        for mention in payload.get("registered_value_mentions") or ()
        if isinstance(mention, Mapping)
        and str(mention.get("target_group") or "")
        and str(mention.get("target_group") or "") != pending_group
    }
    has_registered_cross_group_mutation = bool(
        registered_cross_group_mutation_groups
    ) and any(
        str(unit.get("operation") or "") == "domain_request"
        and any(
            isinstance(route, Mapping)
            and str(route.get("group") or "")
            in registered_cross_group_mutation_groups
            for route in unit.get("owner_routes") or ()
        )
        for unit in partition
    )
    has_competing_open_identity = any(
        str(unit.get("operation") or "") == "domain_request"
        and any(
            isinstance(route, Mapping)
            and str(route.get("group") or "") == pending_group
            and any(
                spec.owner == str(route.get("owner") or "")
                and spec.open_identity_grounding_arguments
                and action_spec_serves_route(
                    spec,
                    operation="domain_request",
                    group=pending_group,
                )
                for spec in ACTION_SPECS
            )
            for route in unit.get("owner_routes") or ()
        )
        for unit in partition
    )
    if has_registered_cross_group_mutation and has_competing_open_identity:
        # A registered value that interrupts another group's signed question
        # is semantically high risk: neighboring framing can otherwise be
        # mistaken for a manual value owned by the pending group. Require a
        # second complete proposal before either interpretation is executable.
        return True
    has_semantic_pending_claim = any(
        str(unit.get("operation") or "") == "pending_answer"
        for unit in partition
    )
    if not has_semantic_pending_claim:
        return False
    return not (
        payload.get("contract_proven_pending_prefixes")
        or payload.get("pending_typed_candidates")
        or payload.get("contract_proven_semantic_draft_resolution")
    )


def _semantic_draft_resolution_contract_errors(
    partition: Sequence[Mapping[str, Any]],
    stage_a_payload: Mapping[str, Any],
) -> tuple[str, ...]:
    """Require the jury-bound span to remain one exact pending answer."""

    contract = dict(
        stage_a_payload.get("contract_proven_semantic_draft_resolution") or {}
    )
    if not contract:
        return ()
    evidence = str(contract.get("evidence_quote") or "")
    matching = [
        unit
        for unit in partition
        if str(unit.get("operation") or "") == "pending_answer"
        and str(unit.get("source_text") or "") == evidence
        and [
            {
                "owner": str(route.get("owner") or ""),
                "group": str(route.get("group") or ""),
            }
            for route in unit.get("owner_routes") or ()
            if isinstance(route, Mapping)
        ]
        == [{
            "owner": "coordinator",
            "group": str(
                (stage_a_payload.get("pending_question") or {}).get("group")
                or ""
            ),
        }]
    ]
    if len(matching) != 1:
        return (
            "Stage A did not preserve the exact jury-bound semantic draft "
            "clarification as one coordinator pending answer",
        )
    if any(
        unit is not matching[0]
        and evidence
        and evidence in str(unit.get("source_text") or "")
        for unit in partition
    ):
        return (
            "Stage A duplicated the jury-bound semantic draft clarification",
        )
    return ()


def _competing_open_identity_relations(
    partition: Sequence[Mapping[str, Any]],
    stage_a_payload: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Return open-identity units competing with registered cross-group values."""

    pending_group = str(
        (stage_a_payload.get("pending_question") or {}).get("group") or ""
    )
    if not pending_group:
        return ()
    mentions = [
        dict(row)
        for row in stage_a_payload.get("registered_value_mentions") or ()
        if isinstance(row, Mapping)
        and str(row.get("target_group") or "")
        and str(row.get("target_group") or "") != pending_group
    ]
    support_units: list[dict[str, Any]] = []
    for unit in partition:
        if str(unit.get("operation") or "") != "domain_request":
            continue
        source = str(unit.get("source_text") or "")
        routes = [
            dict(route)
            for route in unit.get("owner_routes") or ()
            if isinstance(route, Mapping)
        ]
        matched_mentions = [
            mention
            for mention in mentions
            if any(
                str(route.get("group") or "")
                == str(mention.get("target_group") or "")
                for route in routes
            )
            and any(
                str(conflict.get("target_group") or "")
                == str(mention.get("target_group") or "")
                and str(conflict.get("value") or "").casefold()
                == str(mention.get("value") or "").casefold()
                for conflict in semantic_value_domain_conflicts(
                    source,
                    owning_group="",
                )
            )
        ]
        if matched_mentions:
            support_units.append({
                "unit_id": str(unit.get("unit_id") or ""),
                "clause_id": str(unit.get("clause_id") or ""),
                "source_text": source,
                "operation": "domain_request",
                "owner_routes": routes,
                "registered_mentions": matched_mentions,
            })
    if not support_units:
        return ()

    relations: list[dict[str, Any]] = []
    for unit in partition:
        if str(unit.get("operation") or "") != "domain_request":
            continue
        routes = [
            dict(route)
            for route in unit.get("owner_routes") or ()
            if isinstance(route, Mapping)
        ]
        claims_open_identity = any(
            str(route.get("group") or "") == pending_group
            and any(
                spec.owner == str(route.get("owner") or "")
                and spec.open_identity_grounding_arguments
                and action_spec_serves_route(
                    spec,
                    operation="domain_request",
                    group=pending_group,
                )
                for spec in ACTION_SPECS
            )
            for route in routes
        )
        same_clause_support = [
            row
            for row in support_units
            if row["clause_id"] == str(unit.get("clause_id") or "")
            and row["unit_id"] != str(unit.get("unit_id") or "")
        ]
        if claims_open_identity and same_clause_support:
            relations.append({
                "unit": {
                    "unit_id": str(unit.get("unit_id") or ""),
                    "clause_id": str(unit.get("clause_id") or ""),
                    "source_text": str(unit.get("source_text") or ""),
                    "operation": "domain_request",
                    "owner_routes": routes,
                },
                "possible_support_units": same_clause_support,
            })
    return tuple(relations)


def _review_competing_open_identity_relations(
    provider: Any,
    partition: Sequence[Mapping[str, Any]],
    stage_a_payload: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], tuple[str, ...], tuple[int, ...], dict[str, Any]]:
    """Normalize only relation verdicts that reach a fixed semantic quorum."""

    relations = _competing_open_identity_relations(partition, stage_a_payload)
    if not relations:
        return [dict(unit) for unit in partition], (), (), {}
    proposal_hash = _partition_hash(partition)
    request_payload = {
        "user_text": str(stage_a_payload.get("user_text") or ""),
        "clauses": list(stage_a_payload.get("clauses") or []),
        "pending_question": dict(stage_a_payload.get("pending_question") or {}),
        "relations": list(relations),
        "open_identity_action_purposes": {
            spec.action_type: spec.purpose
            for spec in ACTION_SPECS
            if spec.open_identity_grounding_arguments
        },
    }
    review = request_open_identity_relation_jury(
        provider,
        proposal_hash=proposal_hash,
        request_payload=request_payload,
        max_tokens=max(900, 500 + (450 * len(relations))),
    )
    decisions = {
        str(row.get("unit_id") or ""): str(row.get("relation") or "")
        for row in review.decisions
    }
    normalized: list[dict[str, Any]] = []
    for raw_unit in partition:
        unit = dict(raw_unit)
        decision = decisions.get(str(unit.get("unit_id") or ""))
        if decision is not None and decision != "named_identity":
            unit["operation"] = "context" if decision == "supports_unit" else "unresolved"
            unit["owner_routes"] = []
            if decision == "supports_unit":
                unit["_admission_support"] = True
        normalized.append(unit)
    return (
        normalized,
        review.errors,
        review.request_sizes,
        dict(review.receipt),
    )


def _partition_hash(partition: Sequence[Mapping[str, Any]]) -> str:
    encoded = json.dumps(
        [dict(unit) for unit in partition],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _partition_semantic_hash(
    partition: Sequence[Mapping[str, Any]],
) -> str:
    """Hash Harness-effective semantics without model-local presentation fields."""

    projection: list[dict[str, Any]] = []
    for raw in partition:
        unit = {
            key: value
            for key, value in dict(raw).items()
            if key not in {"unit_id", "parent_unit_id", "reason"}
        }
        routes = unit.get("owner_routes")
        if isinstance(routes, list):
            unit["owner_routes"] = sorted(
                (dict(route) for route in routes if isinstance(route, Mapping)),
                key=lambda route: (
                    str(route.get("owner") or ""),
                    str(route.get("group") or ""),
                ),
            )
        projection.append(unit)
    encoded = json.dumps(
        projection,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _proposal_selection_eligible(
    partition: Sequence[Mapping[str, Any]],
) -> bool:
    """Return whether a reviewed proposal is complete enough to select."""

    return bool(partition) and not any(
        str(unit.get("operation") or "") == "unresolved"
        for unit in partition
    )


def _proposal_compilation_eligible(
    state: AgentGraphState,
    partition: Sequence[Mapping[str, Any]],
) -> tuple[bool, tuple[int, ...]]:
    """Prove that every executable unit compiles under its routed owner.

    Stage A validates source coverage and route shape. This non-admitting
    preflight closes the remaining authority gap: a structurally valid owner
    route is selectable only when that owner can bind every routed unit to a
    registered action. The resulting documents are intentionally discarded;
    canonicalization and normal Stage B compilation remain authoritative.
    """

    if not _proposal_selection_eligible(partition):
        return False, ()
    requests: list[tuple[str, frozenset[str], tuple[str, ...]]] = []
    for owner, unit_ids in _owner_requests(partition).items():
        groups = frozenset(
            str(route.get("group") or "")
            for unit in partition
            if str(unit.get("unit_id") or "") in unit_ids
            for route in unit.get("owner_routes") or ()
            if (
                isinstance(route, Mapping)
                and str(route.get("owner") or "") == owner
                and str(route.get("group") or "")
            )
        )
        requests.append((owner, groups, unit_ids))
    if not requests:
        return False, ()

    compiled = run_independent_llm_tasks(tuple(
        lambda request=request: _compile_owner_document(
            state,
            request[0],
            request[1],
            partition,
            request[2],
        )
        for request in requests
    ))
    request_sizes: list[int] = []
    eligible = True
    for (_owner, _groups, unit_ids), (document, errors, sizes) in zip(
        requests,
        compiled,
    ):
        request_sizes.extend(int(value) for value in sizes)
        if errors:
            eligible = False
            continue
        bindings = {
            str(binding.get("unit_id") or ""): dict(binding)
            for binding in document.get("bindings") or ()
            if isinstance(binding, Mapping)
        }
        if any(
            str(bindings.get(unit_id, {}).get("disposition") or "") != "action"
            or not bindings.get(unit_id, {}).get("action_indexes")
            for unit_id in unit_ids
        ):
            eligible = False
    return eligible, tuple(request_sizes)


def _stage_a_convergence_prompt() -> str:
    return (
        "You are the independent Stage A proposal convergence authority for "
        "AnyChain Benchmark Agent. Compare two already validated immutable "
        "semantic partitions against the complete source clauses, operation "
        "purposes, owner/group purposes, and pending contract. You may select "
        "only primary, secondary, or none. You must not create, merge, repair, "
        "relabel, paraphrase, or synthesize units, routes, operations, or actions. "
        "Select a proposal only when it preserves every present request, question, "
        "correction, navigation, administrative instruction, evidence-analysis "
        "request, and context contribution with safe registered ownership. A "
        "proposal that calls an explicitly routable operation unresolved is not "
        "complete. When an operation has several registered owners, compare each "
        "proposal's owner with universal_owner_action_purposes and reject a route "
        "whose owner has no action purpose matching the exact source demand. A "
        "registered_value_mentions row proves the exact value and its owner, but "
        "not mutation intent. When a proposal routes that registered value as a "
        "present domain request, adjacent operation framing is support unless it "
        "independently supplies another named identity, value, question, "
        "navigation, analysis request, or mutation. Do not reinterpret generic "
        "framing as an open identity merely because an identity question is "
        "pending. Conversely, preserve a genuinely named sibling identity. A "
        "request to ingest, diagnose, or explain logs, errors, traces, diagnostics, "
        "failures, or other evidence is evidence_analysis rather than generic "
        "consultation even when the evidence has not been supplied yet. A "
        "pending_answer not proven by a signed option prefix or typed "
        "candidate must still semantically provide or authorize a value accepted by "
        "the exact active pending contract. Rejection, deferral, explanation, or "
        "navigation away from that question is not a pending answer. Genuine "
        "semantic uncertainty may remain unresolved when neither "
        "proposal has a safe source-grounded route. selection_eligible is a "
        "Harness-derived completeness fact. Never select an ineligible proposal "
        "unless both independently reviewed proposals are byte-identical; that "
        "identity represents convergence on the same clarification boundary, not "
        "authority to execute the unresolved unit. Return one strict JSON object "
        "with exactly selected_proposal, primary_hash, secondary_hash, and reason. "
        "Copy both supplied hashes exactly. "
        f"{PENDING_CANDIDATE_SEMANTIC_POLICY}"
    )


def _select_stage_a_proposal(
    provider: Any,
    stage_a_payload: Mapping[str, Any],
    primary: Sequence[Mapping[str, Any]],
    secondary: Sequence[Mapping[str, Any]],
    *,
    primary_eligible: bool | None = None,
    secondary_eligible: bool | None = None,
) -> tuple[
    list[dict[str, Any]],
    tuple[str, ...],
    tuple[int, ...],
    dict[str, Any],
]:
    primary_hash = _partition_hash(primary)
    secondary_hash = _partition_hash(secondary)
    primary_can_select = (
        _proposal_selection_eligible(primary)
        if primary_eligible is None
        else bool(primary_eligible)
    )
    secondary_can_select = (
        _proposal_selection_eligible(secondary)
        if secondary_eligible is None
        else bool(secondary_eligible)
    )
    if primary_can_select != secondary_can_select:
        selected = "primary" if primary_can_select else "secondary"
        selected_partition = primary if primary_can_select else secondary
        verdict = {
            "selected_proposal": selected,
            "primary_hash": primary_hash,
            "primary_eligible": primary_can_select,
            "secondary_hash": secondary_hash,
            "secondary_eligible": secondary_can_select,
            "reason": "Harness selected the sole independently eligible proposal.",
        }
        receipt = {
            "primary_hash": primary_hash,
            "primary_eligible": primary_can_select,
            "secondary_hash": secondary_hash,
            "secondary_eligible": secondary_can_select,
            "selected_proposal": selected,
            "selection_authority": "harness_eligibility",
            "verdict_hash": hashlib.sha256(
                json.dumps(
                    verdict,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "request_count": 0,
            "request_sizes": [],
            "valid": True,
        }
        return [dict(unit) for unit in selected_partition], (), (), receipt
    primary_semantic_hash = _partition_semantic_hash(primary)
    secondary_semantic_hash = _partition_semantic_hash(secondary)
    if (
        primary_can_select
        and secondary_can_select
        and primary_semantic_hash == secondary_semantic_hash
    ):
        verdict = {
            "selected_proposal": "primary",
            "primary_hash": primary_hash,
            "secondary_hash": secondary_hash,
            "semantic_hash": primary_semantic_hash,
            "reason": (
                "Harness selected semantically equivalent independently "
                "eligible proposals without another model decision."
            ),
        }
        receipt = {
            "primary_hash": primary_hash,
            "primary_eligible": True,
            "secondary_hash": secondary_hash,
            "secondary_eligible": True,
            "selected_proposal": "primary",
            "selection_authority": "harness_semantic_equivalence",
            "verdict_hash": hashlib.sha256(
                json.dumps(
                    verdict,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "request_count": 0,
            "request_sizes": [],
            "valid": True,
        }
        return [dict(unit) for unit in primary], (), (), receipt
    payload = {
        "user_text": stage_a_payload["user_text"],
        "clauses": stage_a_payload["clauses"],
        "pending_question": dict(stage_a_payload.get("pending_question") or {}),
        "pending_barrier_contract": dict(
            stage_a_payload.get("pending_barrier_contract") or {}
        ),
        "pending_typed_candidates": list(
            stage_a_payload.get("pending_typed_candidates") or []
        ),
        "contract_proven_pending_entailment_unit_ids": list(
            stage_a_payload.get(
                "contract_proven_pending_entailment_unit_ids"
            )
            or []
        ),
        "contract_proven_pending_prefixes": list(
            stage_a_payload.get("contract_proven_pending_prefixes") or []
        ),
        "registered_semantic_value_domains": list(
            stage_a_payload.get("registered_semantic_value_domains") or []
        ),
        "registered_value_mentions": list(
            stage_a_payload.get("registered_value_mentions") or []
        ),
        "groups": stage_a_payload["groups"],
        "routing_purposes": list(
            stage_a_payload.get("routing_purposes") or []
        ),
        "universal_operations": list(
            stage_a_payload.get("universal_operations") or []
        ),
        "universal_operation_purposes": dict(
            stage_a_payload.get("universal_operation_purposes")
            or SEMANTIC_OPERATION_PURPOSES
        ),
        "universal_owner_action_purposes": dict(
            stage_a_payload.get("universal_owner_action_purposes") or {}
        ),
        "primary": {
            "hash": primary_hash,
            "selection_eligible": primary_can_select,
            "semantic_units": [dict(unit) for unit in primary],
        },
        "secondary": {
            "hash": secondary_hash,
            "selection_eligible": secondary_can_select,
            "semantic_units": [dict(unit) for unit in secondary],
        },
    }
    prompt = _stage_a_convergence_prompt()
    response = request_semantic_compilation(
        provider,
        system_prompt=prompt,
        request_payload=payload,
        max_tokens=900,
        reasoning_mode=STRICT_JSON_REASONING_MODE,
    )
    request_sizes = (_wire_size(prompt, payload),)
    errors: list[str] = []
    try:
        verdict = json.loads(response)
    except json.JSONDecodeError:
        verdict = {}
        errors.append("Stage A convergence did not return strict JSON")
    if not isinstance(verdict, Mapping) or set(verdict) != {
        "selected_proposal",
        "primary_hash",
        "secondary_hash",
        "reason",
    }:
        errors.append("Stage A convergence returned an invalid contract")
        verdict = {}
    selected = str(verdict.get("selected_proposal") or "")
    if selected not in {"primary", "secondary", "none"}:
        errors.append("Stage A convergence selected an invalid proposal")
    if str(verdict.get("primary_hash") or "") != primary_hash:
        errors.append("Stage A convergence changed the primary proposal hash")
    if str(verdict.get("secondary_hash") or "") != secondary_hash:
        errors.append("Stage A convergence changed the secondary proposal hash")
    if not str(verdict.get("reason") or "").strip():
        errors.append("Stage A convergence has no reason")
    if selected == "none":
        errors.append("Stage A proposals did not converge on complete turn coverage")
    selected_partition = secondary if selected == "secondary" else primary
    selected_eligible = (
        secondary_can_select if selected == "secondary" else primary_can_select
    )
    if (
        selected in {"primary", "secondary"}
        and not selected_eligible
        and not (
            primary_hash == secondary_hash
            and primary_can_select == secondary_can_select
        )
    ):
        errors.append("Stage A convergence selected an incomplete proposal")
    receipt = {
        "primary_hash": primary_hash,
        "primary_eligible": primary_can_select,
        "secondary_hash": secondary_hash,
        "secondary_eligible": secondary_can_select,
        "selected_proposal": selected,
        "selection_authority": "model_convergence",
        "verdict_hash": hashlib.sha256(response.encode("utf-8")).hexdigest(),
        "request_count": len(request_sizes),
        "request_sizes": list(request_sizes),
        "valid": not errors,
    }
    chosen = selected_partition
    return (
        [dict(unit) for unit in chosen],
        tuple(dict.fromkeys(errors)),
        request_sizes,
        receipt,
    )


def compile_next_owner(
    state: AgentGraphState,
    document: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile the remaining independent owners and merge canonical order."""

    output = dict(document)
    if output.get("status") != "compile_owner":
        return output
    requests = [
        dict(item)
        for item in output.get("owner_requests") or []
        if isinstance(item, Mapping)
    ]
    cursor = int(output.get("owner_cursor") or 0)
    if cursor >= len(requests):
        output["status"] = "review_plan"
        return output
    remaining_requests = requests[cursor:]
    routed_partition = [
        dict(item)
        for item in output.get("routed_partition") or []
        if isinstance(item, Mapping)
    ]

    def compile_request(
        request: Mapping[str, Any],
    ) -> tuple[dict[str, Any], tuple[str, ...], tuple[int, ...]]:
        owner = str(request.get("owner") or "")
        try:
            return _compile_owner_document(
                state,
                owner,
                frozenset(str(item) for item in request.get("groups") or []),
                routed_partition,
                tuple(str(item) for item in request.get("unit_ids") or []),
            )
        except (LLMTurnTimeoutError, LLMProviderError):
            raise
        except Exception as exc:
            return (
                {},
                (f"Stage B {owner} failed: {type(exc).__name__}",),
                (),
            )

    compiled = run_independent_llm_tasks(tuple(
        lambda request=request: compile_request(request)
        for request in remaining_requests
    ))
    all_request_sizes = [
        int(value) for value in output.get("request_sizes") or []
    ]
    stage_b_calls = int(output.get("stage_b_calls") or 0)
    owner_documents = {
        str(key): dict(value)
        for key, value in dict(output.get("owner_documents") or {}).items()
        if isinstance(value, Mapping)
    }
    owner_failures = [
        dict(item)
        for item in output.get("owner_failures") or ()
        if isinstance(item, Mapping)
    ]
    for request, (owner_document, errors, request_sizes) in zip(
        remaining_requests,
        compiled,
    ):
        owner = str(request.get("owner") or "")
        all_request_sizes.extend(int(value) for value in request_sizes)
        stage_b_calls += len(request_sizes)
        if errors:
            semantic_review_receipts = [
                dict(receipt)
                for receipt in owner_document.get("semantic_review_receipts") or ()
                if isinstance(receipt, Mapping)
            ]
            owner_document = {
                "actions": [],
                "bindings": [
                    {
                        "unit_id": str(unit_id),
                        "action_indexes": [],
                        "disposition": "unresolved",
                        "reason": (
                            "The owning compiler could not produce a valid typed "
                            "action after bounded repair."
                        ),
                    }
                    for unit_id in request.get("unit_ids") or ()
                    if str(unit_id)
                ],
                **(
                    {"semantic_review_receipts": semantic_review_receipts}
                    if semantic_review_receipts
                    else {}
                ),
            }
            owner_failures.append({
                "owner": owner,
                "unit_ids": [
                    str(unit_id)
                    for unit_id in request.get("unit_ids") or ()
                    if str(unit_id)
                ],
                "errors": list(errors),
            })
        owner_documents[owner] = owner_document
    output["request_sizes"] = all_request_sizes
    output["stage_b_calls"] = stage_b_calls
    output["owner_documents"] = owner_documents
    output["owner_failures"] = owner_failures
    output["owner_cursor"] = len(requests)
    output["status"] = "review_plan"
    return output


def _collapse_logical_semantic_units(
    units: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Restore route-expanded rows to their immutable Stage A identity."""

    output: list[dict[str, Any]] = []
    by_identity: dict[str, dict[str, Any]] = {}
    for raw in units:
        unit_id = str(raw.get("unit_id") or "")
        logical_id = str(raw.get("parent_unit_id") or unit_id)
        if not logical_id:
            continue
        current = by_identity.get(logical_id)
        if current is None:
            current = dict(raw)
            current["unit_id"] = logical_id
            current.pop("parent_unit_id", None)
            current["owner_routes"] = []
            if "action_indexes" in raw:
                current["action_indexes"] = []
            by_identity[logical_id] = current
            output.append(current)
        for route in raw.get("owner_routes") or ():
            if not isinstance(route, Mapping):
                continue
            normalized = dict(route)
            if normalized not in current["owner_routes"]:
                current["owner_routes"].append(normalized)
        for index in raw.get("action_indexes") or ():
            if (
                isinstance(index, int)
                and not isinstance(index, bool)
                and index not in current.setdefault("action_indexes", [])
            ):
                current["action_indexes"].append(index)
    return output


def _semantic_rejection_draft_projection(
    source_partition: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
    plan: Any,
    admission: Any,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    PlanCoverageResult,
]:
    """Project one explicit rejection into a durable, non-executable draft."""

    raw_units = [
        dict(item)
        for item in candidate.get("semantic_units") or ()
        if isinstance(item, Mapping)
    ]
    actions = [
        dict(item)
        for item in candidate.get("actions") or ()
        if isinstance(item, Mapping)
    ]
    action_ids = tuple(str(item) for item in getattr(plan, "action_ids", ()))
    admitted_indexes = {
        index
        for index, action_id in enumerate(action_ids)
        if any(
            str(row.get("action_id") or "") == action_id
            and str(row.get("verdict") or "") == "admit"
            for row in getattr(admission, "action_verdicts", ())
            if isinstance(row, Mapping)
        )
    }
    unit_verdicts = {
        str(row.get("unit_id") or ""): str(row.get("verdict") or "")
        for row in getattr(admission, "unit_verdicts", ())
        if isinstance(row, Mapping)
    }
    route_rows_by_logical: dict[str, list[dict[str, Any]]] = defaultdict(list)
    action_logical_owners: dict[int, set[str]] = defaultdict(set)
    for unit in raw_units:
        logical_id = str(unit.get("parent_unit_id") or unit.get("unit_id") or "")
        if not logical_id:
            continue
        route_rows_by_logical[logical_id].append(unit)
        for index in unit.get("action_indexes") or ():
            if isinstance(index, int) and not isinstance(index, bool):
                action_logical_owners[index].add(logical_id)
    safe_logical_ids: set[str] = set()
    for logical_id, rows in route_rows_by_logical.items():
        actionable = [
            row for row in rows
            if str(row.get("disposition") or "") != "context"
        ]
        if actionable and all(
            unit_verdicts.get(str(row.get("unit_id") or ""))
            in {"complete", "support"}
            and all(
                index in admitted_indexes
                for index in row.get("action_indexes") or ()
                if isinstance(index, int) and not isinstance(index, bool)
            )
            for row in actionable
        ):
            safe_logical_ids.add(logical_id)
    retained_indexes = [
        index
        for index in sorted(admitted_indexes)
        if action_logical_owners.get(index)
        and action_logical_owners[index].issubset(safe_logical_ids)
        and index < len(actions)
    ]
    old_to_new = {
        old: new for new, old in enumerate(retained_indexes)
    }
    retained_actions = [actions[index] for index in retained_indexes]
    logical_units = _collapse_logical_semantic_units(raw_units)
    unresolved_units: list[dict[str, Any]] = []
    for unit in logical_units:
        logical_id = str(unit["unit_id"])
        if logical_id in safe_logical_ids:
            mapped = [
                old_to_new[index]
                for index in unit.get("action_indexes") or ()
                if index in old_to_new
            ]
            unit["disposition"] = "action"
            unit["action_indexes"] = list(dict.fromkeys(mapped))
            continue
        rows = route_rows_by_logical.get(logical_id, [])
        if rows and all(
            str(row.get("disposition") or "") == "context"
            for row in rows
        ):
            unit["disposition"] = "context"
            unit["action_indexes"] = []
            continue
        unit["disposition"] = "unresolved"
        unit["action_indexes"] = []
        unit["reason"] = (
            "independent whole-plan admission did not safely admit every "
            "route owned by this logical demand"
        )
        unresolved_units.append(dict(unit))
    logical_partition = _collapse_logical_semantic_units(source_partition)
    validation = PlanCoverageResult(
        valid=False,
        errors=(),
        unresolved_clauses=tuple(
            str(item.get("source_text") or "")
            for item in unresolved_units
            if str(item.get("source_text") or "")
        ),
        unresolved_units=tuple(unresolved_units),
    )
    return logical_partition, logical_units, retained_actions, validation


def _build_semantic_draft_result(
    state: AgentGraphState,
    clauses: Sequence[TurnClause],
    *,
    source_partition: Sequence[Mapping[str, Any]],
    semantic_units: Sequence[Mapping[str, Any]],
    candidate_actions: Sequence[Mapping[str, Any]],
    validation: PlanCoverageResult,
    reason: str,
) -> dict[str, Any]:
    """Persist unresolved source atoms without admitting unsafe mutations."""

    original_input = "\n".join(clause.text for clause in clauses)
    (
        source_secret_replacements,
        source_secret_bindings,
        source_secret_values_by_reference,
    ) = _source_secret_projection(
        original_input,
        [
            dict(item)
            for item in (state.get("turn_context") or {}).get(
                "input_secret_bindings"
            ) or ()
            if isinstance(item, Mapping)
        ],
    )
    draft = build_semantic_plan_draft(
        state,
        original_input=_replace_secret_values(
            original_input,
            source_secret_replacements,
        ),
        source_clauses=_replace_secret_values(
            [clause.as_dict() for clause in clauses],
            source_secret_replacements,
        ),
        source_partition=_replace_secret_values(
            list(source_partition),
            source_secret_replacements,
        ),
        semantic_units=list(semantic_units),
        candidate_actions=_replace_secret_values(
            list(candidate_actions),
            source_secret_replacements,
        ),
        validation=validation,
        source_secret_bindings=source_secret_bindings,
    )
    for binding in source_secret_bindings:
        reference = str(binding["reference"])
        if reference in source_secret_values_by_reference:
            store_secret_reference(
                source_secret_values_by_reference[reference],
                draft_id=str(draft["draft_id"]),
                atom_id=str(binding["atom_id"]),
                reference=reference,
            )
        else:
            authorize_secret_reference(
                reference,
                draft_id=str(draft["draft_id"]),
                atom_id=str(binding["atom_id"]),
                expected_hash=str(binding["value_hash"]),
            )
    return {
        "actions": [],
        "semantic_units": [dict(item) for item in semantic_units],
        "semantic_draft": draft,
        "reason": reason,
    }


def _is_finalizing_ready_semantic_draft(
    state: Mapping[str, Any],
) -> bool:
    draft = dict(state.get("semantic_plan_draft") or {})
    finalization = dict(
        (state.get("turn_context") or {}).get(
            "semantic_draft_finalization"
        )
        or {}
    )
    return bool(
        draft.get("status") == "ready_for_review"
        and str(finalization.get("draft_id") or "")
        == str(draft.get("draft_id") or "")
        and int(finalization.get("revision") or 0)
        == int(draft.get("revision") or 0)
    )


def review_semantic_plan(
    state: AgentGraphState,
    document: Mapping[str, Any],
) -> dict[str, Any]:
    """Run whole-plan admission after every owner document is checkpointed."""

    output = dict(document)
    clauses = tuple(
        TurnClause(
            str(item.get("clause_id") or ""),
            str(item.get("text") or ""),
            str(item.get("input_shape") or "prose"),
        )
        for item in output.get("clauses") or []
        if isinstance(item, Mapping)
    )
    started = float(output.get("started_monotonic") or time.monotonic())
    request_sizes = [int(value) for value in output.get("request_sizes") or []]
    stage_a_calls = int(output.get("stage_a_calls") or 0)
    stage_b_calls = int(output.get("stage_b_calls") or 0)
    admission_calls = int(output.get("admission_calls") or 0)
    owner_count = int(output.get("owner_count") or 0)
    unit_count = int(output.get("unit_count") or 0)
    if output.get("status") == "failed":
        return _with_metrics(
            _unresolved_action_queue(clauses, tuple(output.get("errors") or [])),
            started,
            request_sizes=request_sizes,
            stage_a_calls=stage_a_calls,
            stage_b_calls=stage_b_calls,
            admission_calls=admission_calls,
            owner_count=owner_count,
            unit_count=unit_count,
        )
    if output.get("status") != "review_plan":
        return _with_metrics(
            _unresolved_action_queue(
                clauses,
                ("semantic plan reached review before owner compilation completed",),
            ),
            started,
            request_sizes=request_sizes,
            stage_a_calls=stage_a_calls,
            stage_b_calls=stage_b_calls,
            admission_calls=admission_calls,
            owner_count=owner_count,
            unit_count=unit_count,
        )
    source_partition = [
        dict(item)
        for item in output.get("source_partition") or []
        if isinstance(item, Mapping)
    ]
    routed_partition = [
        dict(item)
        for item in output.get("routed_partition") or []
        if isinstance(item, Mapping)
    ]
    owner_documents = {
        str(key): dict(value)
        for key, value in dict(output.get("owner_documents") or {}).items()
        if isinstance(value, Mapping)
    }
    candidate = _merge_owner_documents(
        source_partition,
        routed_partition,
        owner_documents,
    )
    candidate = _project_unresolved_mutation_conflicts(candidate)
    authoritative_direct_unit_ids = frozenset(
        str(unit.get("unit_id") or "")
        for unit in source_partition
        if str(unit.get("operation") or "") not in {"context", "unresolved"}
    )
    authoritative_context_unit_ids = frozenset(
        str(unit_id)
        for unit_id in candidate.get("semantic_support_unit_ids") or ()
        if str(unit_id)
    )
    candidate, candidate_text, validation = (
        _prepare_candidate_with_normalized_conflict_projection(
            candidate,
            state,
            clauses,
            pending_choice_unit_ids=frozenset(
                str(unit["unit_id"])
                for unit in source_partition
                if str(unit.get("operation") or "") == "pending_answer"
            ),
            semantic_support_unit_ids=frozenset(
                str(unit_id)
                for unit_id in candidate.get("semantic_support_unit_ids") or ()
                if str(unit_id)
            ),
        )
    )
    if not validation.valid:
        if validation.unresolved_units and not validation.errors:
            if _is_finalizing_ready_semantic_draft(state):
                return _with_metrics(
                    _unresolved_action_queue(
                        clauses,
                        (
                            "semantic draft finalization remained unresolved "
                            "after bound clarification",
                        ),
                        semantic_units=[
                            dict(item)
                            for item in validation.unresolved_units
                            if isinstance(item, Mapping)
                        ],
                    ),
                    started,
                    request_sizes=request_sizes,
                    stage_a_calls=stage_a_calls,
                    stage_b_calls=stage_b_calls,
                    admission_calls=admission_calls,
                    owner_count=owner_count,
                    unit_count=unit_count,
                )
            return _with_metrics(
                _build_semantic_draft_result(
                    state,
                    clauses,
                    source_partition=source_partition,
                    semantic_units=[
                        dict(item)
                        for item in candidate.get("semantic_units") or []
                        if isinstance(item, Mapping)
                    ],
                    candidate_actions=[
                        dict(item)
                        for item in candidate.get("actions") or []
                        if isinstance(item, Mapping)
                    ],
                    validation=validation,
                    reason="semantic plan requires atom clarification",
                ),
                started,
                request_sizes=request_sizes,
                stage_a_calls=stage_a_calls,
                stage_b_calls=stage_b_calls,
                admission_calls=admission_calls,
                owner_count=owner_count,
                unit_count=unit_count,
            )
        return _with_metrics(
            _unresolved_action_queue(clauses, validation.errors),
            started,
            request_sizes=request_sizes,
            stage_a_calls=stage_a_calls,
            stage_b_calls=stage_b_calls,
            admission_calls=admission_calls,
            owner_count=owner_count,
            unit_count=unit_count,
        )
    provider = provider_from_config()
    plan, admission, admission_errors = _review_bounded_semantic_candidate(
        provider,
        candidate_text,
        validation,
        state,
        clauses,
        allowed_action_types=_allowed_action_types_for_partition(
            routed_partition
        ),
        whole_plan_contract_repair=True,
        reasoning_mode=STRICT_JSON_REASONING_MODE,
        authoritative_direct_unit_ids=authoritative_direct_unit_ids,
        authoritative_context_unit_ids=authoritative_context_unit_ids,
    )
    admission_calls += (
        int(getattr(admission, "request_count", 1))
        if admission is not None
        else 0
    )
    if admission is not None and getattr(admission, "request_sizes", ()):
        request_sizes.extend(admission.request_sizes)
    elif plan is not None:
        request_sizes.append(
            len(
                whole_plan_admission_prompt(
                    _semantic_fulfillment_prompt()
                ).encode("utf-8")
            )
            + len(plan.request_json.encode("utf-8"))
        )
    if (
        plan is not None
        and admission is not None
        and not admission.valid
    ):
        repaired_candidate = _conservative_closed_enum_intake_repair(
            candidate,
            plan,
            admission,
        )
        if repaired_candidate is not None:
            repaired_text, repaired_validation = prepare_hierarchical_candidate(
                json.dumps(
                    repaired_candidate,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                state,
                clauses,
                pending_choice_unit_ids=frozenset(
                    str(unit["unit_id"])
                    for unit in source_partition
                    if str(unit.get("operation") or "") == "pending_answer"
                ),
                semantic_support_unit_ids=frozenset(
                    str(unit_id)
                    for unit_id in repaired_candidate.get(
                        "semantic_support_unit_ids"
                    ) or ()
                    if str(unit_id)
                ),
            )
            if repaired_validation.valid:
                repaired_plan, repaired_admission, repaired_errors = (
                    _review_bounded_semantic_candidate(
                        provider,
                        repaired_text,
                        repaired_validation,
                        state,
                        clauses,
                        allowed_action_types=_allowed_action_types_for_partition(
                            routed_partition
                        ),
                        whole_plan_contract_repair=True,
                        reasoning_mode=STRICT_JSON_REASONING_MODE,
                        authoritative_direct_unit_ids=(
                            authoritative_direct_unit_ids
                        ),
                        authoritative_context_unit_ids=(
                            authoritative_context_unit_ids
                        ),
                    )
                )
                if repaired_admission is not None:
                    admission_calls += int(
                        getattr(repaired_admission, "request_count", 1)
                    )
                    request_sizes.extend(
                        tuple(
                            getattr(
                                repaired_admission,
                                "request_sizes",
                                (),
                            )
                            or ()
                        )
                    )
                plan = repaired_plan
                admission = repaired_admission
                admission_errors = repaired_errors
                candidate = repaired_candidate
    semantic_units = [
        dict(item)
        for item in candidate.get("semantic_units") or []
        if isinstance(item, Mapping)
    ]
    if admission is not None and admission.valid and plan is not None:
        try:
            result = _admitted_action_queue(plan, admission, state)
        except ValueError as exc:
            result = _unresolved_action_queue(
                clauses,
                (f"immutable semantic plan rejected: {exc}",),
                semantic_units=semantic_units,
            )
    elif (
        admission is not None
        and plan is not None
        and is_explicit_semantic_rejection(admission)
    ):
        (
            logical_partition,
            draft_units,
            safe_candidate_actions,
            draft_validation,
        ) = _semantic_rejection_draft_projection(
            source_partition,
            candidate,
            plan,
            admission,
        )
        if (
            draft_validation.unresolved_units
            and not _is_finalizing_ready_semantic_draft(state)
        ):
            result = _build_semantic_draft_result(
                state,
                clauses,
                source_partition=logical_partition,
                semantic_units=draft_units,
                candidate_actions=safe_candidate_actions,
                validation=draft_validation,
                reason=(
                    "independent admission requires durable clarification"
                ),
            )
        else:
            result = _unresolved_action_queue(
                clauses,
                admission_errors,
                semantic_units=semantic_units,
            )
    else:
        result = _unresolved_action_queue(
            clauses,
            admission_errors,
            semantic_units=semantic_units,
        )
    return _with_metrics(
        result,
        started,
        request_sizes=request_sizes,
        stage_a_calls=stage_a_calls,
        stage_b_calls=stage_b_calls,
        admission_calls=admission_calls,
        owner_count=owner_count,
        unit_count=unit_count,
    )


def _conservative_closed_enum_intake_repair(
    candidate: Mapping[str, Any],
    plan: Any,
    admission: Any,
) -> dict[str, Any] | None:
    """Replace rejected enum guesses with one registry-owned typed intake.

    Stage A already established that the source is a domain request. The
    independent admission reviewer may still reject Stage B's concrete enum
    selection. When the registry exposes exactly one intake for that same
    group, asking the user is the only lossless projection.
    """

    payload = json.loads(
        json.dumps(candidate, ensure_ascii=False, sort_keys=True)
    )
    actions = payload.get("actions")
    units = payload.get("semantic_units")
    if not isinstance(actions, list) or not isinstance(units, list):
        return None
    action_ids = tuple(getattr(plan, "action_ids", ()) or ())
    verdicts = {
        str(row.get("action_id") or ""): row
        for row in tuple(getattr(admission, "action_verdicts", ()) or ())
        if isinstance(row, Mapping)
    }
    changed = False
    for index, raw_action in enumerate(actions):
        if not isinstance(raw_action, Mapping) or index >= len(action_ids):
            continue
        verdict = verdicts.get(str(action_ids[index]) or "")
        if (
            not isinstance(verdict, Mapping)
            or str(verdict.get("verdict") or "") == "admit"
        ):
            continue
        action = dict(raw_action)
        spec = ACTION_BY_TYPE.get(str(action.get("type") or ""))
        if spec is None or not spec.target_group:
            continue
        closed_enum_arguments = [
            argument
            for argument in semantic_grounding_arguments(action)
            if isinstance(
                (ACTION_ARGUMENT_SCHEMAS.get(argument) or {}).get("enum"),
                list,
            )
            and action.get(argument) not in {"", None}
        ]
        if not closed_enum_arguments:
            continue
        intake_spec = _unique_incomplete_mutation_intake(
            owner=spec.owner,
            target_group=spec.target_group,
        )
        if intake_spec is None:
            continue
        owned_units = [
            unit
            for unit in units
            if isinstance(unit, Mapping)
            and str(unit.get("disposition") or "") == "action"
            and index in tuple(unit.get("action_indexes") or ())
            and str(unit.get("source_text") or "").strip()
        ]
        if len(owned_units) != 1:
            continue
        replacement = {
            "type": intake_spec.action_type,
            "source_evidence": str(
                owned_units[0].get("source_text") or ""
            ).strip(),
        }
        try:
            actions[index] = validate_action_contract(replacement)
        except ValueError:
            continue
        changed = True
    return payload if changed else None


def _unique_incomplete_mutation_intake(
    *,
    owner: str,
    target_group: str,
) -> Any | None:
    candidates = [
        spec
        for spec in ACTION_SPECS
        if spec.owner == owner
        and spec.incomplete_mutation_intake
        and spec.target_group == target_group
        and set(spec.required_arguments).issubset({"source_evidence"})
    ]
    return candidates[0] if len(candidates) == 1 else None


def _turn_clauses(
    state: AgentGraphState,
    text: str,
) -> tuple[TurnClause, ...]:
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    pending = dict(state.get("pending_question") or {})
    validation = dict(pending.get("validation") or {})
    segmented = segment_user_turn(raw)
    has_contract_option_prefix = bool(pending.get("options")) and any(
        exact_option_prefix_answer(clause.text, pending)[0]
        for clause in segmented
    )
    if (
        raw
        and pending.get("manual_input_allowed") is True
        and str(validation.get("value_type") or "") == "evidence_contribution"
        and not has_contract_option_prefix
    ):
        return (TurnClause("clause-1", raw, "structured"),)
    return segmented


def _stage_a_prompt() -> str:
    return (
        "You are the Stage A semantic partitioner for AnyChain Benchmark Agent. "
        "Return one strict JSON object with semantic_units and reason only. "
        "Identify ordered minimal exact source anchors for every semantic operation. The "
        "Harness expands the intervals between prose anchors into the lossless partition, so "
        "do not repeat surrounding prose merely to cover conjunctions or punctuation. "
        "Do not choose product actions, mutate state, answer the user, or copy values from state. "
        "Each prose semantic unit must contain exactly: unit_id, clause_id, source_text, "
        "operation, owner_routes, reason. Each structured DemandAtom must contain exactly: "
        "unit_id, operation, owner_routes, reason. The Harness binds its immutable clause_id, "
        "source_text, and source_path from atom_id after your classification; do not repeat "
        "those fields in structured output. operation must be one of the supplied "
        "universal_operations. "
        "Prose unit_id values are turn-local labels and must be non-empty and unique. "
        "The Harness may deterministically rebind missing or colliding prose labels; they "
        "carry no semantic or execution authority. "
        "Treat universal_operation_purposes as authoritative. pending_answer requires a present "
        "commitment to the active question; a hypothetical, counterfactual, consequence, or "
        "explanation question is consultation and never authorizes the pending action. A request "
        "to analyze logs, errors, traces, or other evidence is evidence_analysis even when the "
        "user has not pasted the evidence yet. "
        f"{PENDING_CANDIDATE_SEMANTIC_POLICY}"
        "A source-grounded instruction to keep, reuse, or reconfirm one concrete registered "
        "value is a present idempotent domain_request even when owner state already contains "
        "that value. A statement that a value or evidence will be supplied later is temporal "
        "context, not a present value selection or mutation. Deferral, delay, absence, or later "
        "submission by itself never requests the workflow to continue without the value and "
        "must not be marked unresolved merely because the future value is absent. If that "
        "deferral also explicitly "
        "asks to enter or continue the owning workflow, route only that workflow request and "
        "do not invent the missing value. A question about requirements, format, meaning, or "
        "validity is consultation rather than a mutation merely because it names a group or "
        "field. "
        "contract_proven_pending_prefixes contains Harness-proven exact option prefixes from "
        "the active signed question. Emit each declared prefix as its own coordinator-owned "
        "pending_answer unit and independently classify every remaining source character. "
        "Never absorb a remaining mutation, consultation, navigation, analysis, or context "
        "span into that pending answer. "
        "contract_proven_semantic_draft_resolution is a quorum-reviewed Harness fact "
        "bound to the exact active semantic draft atom. When present, its evidence_quote "
        "is one exact immutable source span and must remain one coordinator-owned "
        "pending_answer; every source span outside that quote must still be partitioned "
        "under its own registered purpose. The quote is not a new unresolved demand and "
        "cannot be relabeled as "
        "consultation, clarification, navigation, or another group operation. "
        "owner_routes is an ordered list of {owner,group}; use an empty list only for context "
        "or unresolved. A prose unit may route to several owners when one indivisible excerpt "
        "contains independently owned values. A structured clause is one immutable SourceClause; "
        "represent each semantic field demand as a separate DemandAtom whose unit_id exactly "
        "reuses the supplied atom_id. Return exactly one "
        "semantic unit for every supplied structured atom, in supplied order. The atom_id, "
        "clause_id, source_text, and source_path are immutable Harness provenance; classify only "
        "operation, owner_routes, and reason, and omit the provenance fields from your output. "
        "Never omit, combine, duplicate, or invent an atom. "
        "Preserve questions, corrections, contradictions, "
        "pending answers, navigation, multiline evidence, and every sibling demand separately. "
        "input_shape and structured_candidates describe terminal syntax only; they do not classify "
        "intent, authorize an action, or assign an owner. Treat each parser candidate as a read-only "
        "structural fact. structured_intake_contracts are registry-owned action-intake semantics; "
        "when a field candidate exactly matches an enabled contract, route that DemandAtom to the "
        "declared owner/group and let Stage B compile only the declared action. "
        "Route the user's actual semantic request using the supplied owner/group "
        "registries and routing purposes, including several immutable owner routes when one structured "
        "block supplies independently owned values. Never route every structured value to one default "
        "owner. If the semantic operation or any required owner/group route is uncertain, mark the "
        "unit unresolved; do not guess, omit, or repair a route. "
        "When a turn answers the active pending question and also supplies sibling configuration, "
        "emit the exact answer excerpt as one pending_answer unit routed only to coordinator, using "
        "the active pending_question.group as that route's group, and "
        "emit every sibling as separate domain_request units. Never label a normal domain_request "
        "as a pending answer, and never combine a pending answer with a sibling mutation. "
        "Respect pending_question.value_domain and registered_semantic_value_domains. A known "
        "identity or closed value owned by another registered dimension is routed to that "
        "dimension's owner; it is not consumed by an unrelated manual-text question merely "
        "because that question is pending. If the source explicitly claims that a registered "
        "value is instead a new identity, "
        "mark that unit unresolved so the user can disambiguate. "
        "registered_value_mentions contains only registry values that occur in this turn. "
        "It proves spelling and ownership, not intent. Use the complete source meaning to "
        "decide whether each mention participates in a mutation, consultation, hypothetical, "
        "comparison, or context. When one compact phrase uses a registered value plus adjacent "
        "operation framing to express one domain request, keep the complete phrase in one "
        "domain_request unit. Do not split its framing into a second pending_answer or "
        "unresolved demand unless that framing independently answers the active question or "
        "requests another operation. "
        f"{FRAMED_REQUEST_SEMANTIC_POLICY}"
        "A semantic option selection and adjacent prose that only explains the reason, uncertainty, "
        "basis, referential application, or declared completion effect for that same selection form "
        "one pending_answer operation even when punctuation or line breaks create several clauses. "
        "When the source rejects one or more pending options and affirmatively requests another "
        "declared option's meaning or action, form one pending_answer for the affirmed option. "
        "The rejected alternative is contrast evidence, not a pending answer of its own, and the "
        "affirmed option action is not also a sibling domain request. "
        "pending_barrier_contract governs pending-answer ownership and queue scheduling only. "
        "It never forbids semantic routing to another registered group. In particular, when "
        "registered_cross_group_values is route_to_registered_owner, route an independent "
        "registered value to that owner even under exclusive_owner; do not mark it unresolved "
        "merely because another group has a pending question. "
        "The supporting clause may be context, but it is not a separate demand. The explanation is not a "
        "domain request merely because it discusses the option's subject. Split it only when the "
        "user independently asks for research, explanation, navigation, or a mutation. "
        "Apply the same transaction boundary to one parser-proven manual value for the active "
        "pending question: adjacent prose that only states the value's purpose, scope, exclusion, "
        "or non-application is support for that pending answer, not a sibling domain request. "
        "Keep a separate operation only when the prose independently requests another value, "
        "mutation, consultation, navigation, or analysis. "
        "A semantic unit represents one indivisible prose excerpt or one structured DemandAtom. "
        "Split independent demands into separate minimal exact source anchors whenever each "
        "demand has its own exact substring, even when they occur in one grammatical sentence, "
        "share an owner, or declare the same operation. Use several owner routes only when the "
        "same indivisible exact words require several owners and cannot be separated without "
        "paraphrasing or losing meaning. A unit has exactly one operation: "
        "when a compact excerpt combines mutation, consultation, navigation, analysis, or "
        "another different operation, split minimal exact source anchors by operation even "
        "when they occur in one grammatical sentence. For structured input, use separate "
        "source_path DemandAtoms "
        "instead of assigning one field to competing semantic owners. "
        "Universal operation ownership is constrained by universal_operation_owners. Select "
        "exactly one owner from the declared list for that operation. The group on a "
        "consultation route identifies its subject but never transfers read-only consultation "
        "ownership away from orientation. "
        "When one universal operation declares several owners, use "
        "universal_owner_action_purposes to select the owner whose registered action purpose "
        "can represent the exact source demand. Do not select an owner merely because it owns "
        "another action with the same broad operation label. If no owner purpose safely fits, "
        "mark the unit unresolved. "
        "For a consultation action with consultation_topic_purposes, those closed topic "
        "purposes are the authoritative read-only capabilities of that action. Route a "
        "question when exactly one registered topic purpose represents it; do not require "
        "the user to name the topic identifier and do not convert the question into a "
        "mutation or pending answer. "
        "Use exact owner and group identifiers from the supplied registries. "
        "Do not infer a mutation from examples, hypothetical values, logs, or current state."
        " When semantic_draft_resolutions is present, each row is an exact "
        "user-confirmed interpretation of the identified source DemandAtom. "
        "Use it to resolve that atom while still partitioning and routing the "
        "complete original source. Do not treat the resolution rows as new "
        "sibling demands and do not omit any original source atom. A background "
        "disposition is Harness authority and will be projected to context after "
        "your source-complete partition; do not invent or alter dispositions."
    )


def _stage_a_payload(
    state: AgentGraphState,
    text: str,
    clauses: Sequence[TurnClause],
) -> dict[str, Any]:
    pending = dict(state.get("pending_question") or {})
    structured_intake_contracts = _structured_intake_contracts()
    structured_candidates = []
    for clause in clauses:
        if clause.input_shape != "structured":
            continue
        candidates = _semantic_structured_input_candidates(clause.text)
        if candidates:
            candidates = dict(candidates)
            candidates["field_candidates"] = [
                {
                    **dict(candidate),
                    "atom_id": _structured_atom_id(clause.clause_id, index),
                    **(
                        {"registered_intake": matching}
                        if (
                            matching := _matching_structured_intake(
                                candidate,
                                structured_intake_contracts,
                            )
                        )
                        else {}
                    ),
                }
                for index, candidate in enumerate(
                    candidates.get("field_candidates") or [],
                    start=1,
                )
                if isinstance(candidate, Mapping)
            ]
            structured_candidates.append({
                "clause_id": clause.clause_id,
                **candidates,
            })
    draft = dict(state.get("semantic_plan_draft") or {})
    draft_resolutions = [
        {
            "atom_id": str(item.get("atom_id") or ""),
            "unit_id": str(item.get("unit_id") or ""),
            "clause_id": str(item.get("clause_id") or ""),
            "source_path": str(item.get("source_path") or ""),
            "source_text": str(item.get("source_text") or ""),
            "user_resolution": str(item.get("resolution") or ""),
            "resolution_disposition": str(
                item.get("resolution_disposition") or ""
            ),
        }
        for item in draft.get("unresolved_atoms") or ()
        if isinstance(item, Mapping)
        and str(item.get("resolution") or "").strip()
    ] if draft.get("status") == "ready_for_review" else []
    option_prefix = _unique_option_prefix_candidate(state, clauses)
    projected_actions = action_schema()

    def semantic_purpose(row: Mapping[str, Any]) -> dict[str, Any]:
        purpose = {
            "action_type": str(row["type"]),
            "purpose": str(row["purpose"]),
        }
        topic_purposes = row.get("topic_purposes")
        if isinstance(topic_purposes, Mapping) and topic_purposes:
            purpose["consultation_topic_purposes"] = dict(topic_purposes)
        return purpose

    return {
        "product": "AnyChain Benchmark Agent",
        "user_text": text,
        "clauses": [clause.as_dict() for clause in clauses],
        "language": state.get("language") or "en",
        "active_group": state.get("active_group") or "opening",
        "pending_question": semantic_pending_question(pending),
        "pending_barrier_contract": pending_barrier_semantics(pending),
        "registered_semantic_value_domains": list(
            registered_semantic_value_domains()
        ),
        "registered_value_mentions": list(
            _registered_value_mentions(text)
        ),
        "pending_typed_candidates": [
            value
            for clause in clauses
            for value in _turn_bound_pending_typed_candidates(
                state,
                clause.text,
            )
        ],
        "contract_proven_pending_prefixes": [
            {
                "clause_id": clause.clause_id,
                "start": 0,
                "end": end,
                "source_text": clause.text[:end],
                "selected_value": value,
            }
            for clause, value, end in ([option_prefix] if option_prefix else [])
        ],
        "group_readiness": workflow_snapshot(state).get("group_states") or {},
        "interruption_stack": state.get("interruption_stack") or [],
        "workflow_goals": state.get("workflow_goals") or [],
        "structured_candidates": structured_candidates,
        "structured_intake_contracts": structured_intake_contracts,
        **(
            {
                "semantic_draft_resolutions": draft_resolutions,
                "semantic_draft_id": str(draft.get("draft_id") or ""),
                "semantic_draft_revision": int(draft.get("revision") or 0),
            }
            if draft_resolutions
            else {}
        ),
        "universal_operations": sorted(_UNIVERSAL_OPERATIONS),
        "universal_operation_purposes": dict(SEMANTIC_OPERATION_PURPOSES),
        "universal_operation_owners": {
            operation: list(owners)
            for operation, owners in _UNIVERSAL_OPERATION_OWNERS.items()
        },
        "universal_owner_action_purposes": {
            operation: {
                owner: [
                    semantic_purpose(row)
                    for row in projected_actions
                    if str(row["owner"]) == owner
                    and operation in row["semantic_operations"]
                ]
                for owner in owners
            }
            for operation, owners in _UNIVERSAL_OPERATION_OWNERS.items()
        },
        "owners": sorted(_OWNERS),
        "routing_purposes": [
            {
                "action_type": action["type"],
                "owner": action["owner"],
                "purpose": action["purpose"],
                "semantic_operations": action["semantic_operations"],
                "route_groups": action["route_groups"],
            }
            for action in projected_actions
        ],
        "groups": [
            {
                "name": row["name"],
                "owner": row["owner"],
                "category": row["category"],
                "fields": row["fields"],
                "depends_on": row["depends_on"],
                "entry_actions": row["entry_actions"],
            }
            for row in group_schema()
        ],
    }


def _structured_intake_contracts() -> list[dict[str, Any]]:
    """Project model-reachable structured entry semantics from ActionSpec."""

    return [
        {
            "action_type": row["type"],
            "owner": row["owner"],
            "target_group": row["target_group"],
            **dict(intake),
        }
        for row in action_schema()
        for intake in row.get("structured_intake") or []
    ]


def _structured_atom_id(clause_id: str, index: int) -> str:
    """Return the Harness-owned identity for one parser-proven field atom."""

    return f"__harness_structured_{clause_id}_{int(index)}"


def _bind_structured_partition_provenance(
    raw_units: Sequence[Any],
    clauses: Sequence[TurnClause],
) -> list[Any]:
    """Bind semantic classifications to immutable parser-owned evidence."""

    atoms: dict[str, tuple[str, str, str]] = {}
    for clause in clauses:
        if clause.input_shape != "structured":
            continue
        candidates = _semantic_structured_input_candidates(clause.text)
        for index, candidate in enumerate(
            candidates.get("field_candidates") or (),
            start=1,
        ):
            if not isinstance(candidate, Mapping):
                continue
            atoms[_structured_atom_id(clause.clause_id, index)] = (
                clause.clause_id,
                str(candidate.get("source_path") or ""),
                clause.text,
            )

    output: list[Any] = []
    for raw in raw_units:
        if not isinstance(raw, Mapping):
            output.append(raw)
            continue
        unit = dict(raw)
        provenance = atoms.get(str(unit.get("unit_id") or ""))
        if provenance is not None:
            clause_id, source_path, source_text = provenance
            unit["clause_id"] = clause_id
            unit["source_path"] = source_path
            unit["source_text"] = source_text
        output.append(unit)
    return output


def _bind_ambiguous_prose_partition_identities(
    raw_units: Sequence[Any],
    clauses: Sequence[TurnClause],
) -> list[Any]:
    """Replace only missing or colliding model prose labels deterministically.

    Structured atom identities are parser-owned and remain untouched. Prose
    labels carry no semantic authority, so an identity collision must not hide
    otherwise reviewable semantic units from the admission boundary.
    """

    clause_shapes = {clause.clause_id: clause.input_shape for clause in clauses}
    identities = [
        str(raw.get("unit_id") or "").strip()
        if isinstance(raw, Mapping)
        else ""
        for raw in raw_units
    ]
    identity_counts: dict[str, int] = defaultdict(int)
    for identity in identities:
        if identity:
            identity_counts[identity] += 1
    preserved = {
        identity
        for index, identity in enumerate(identities)
        if identity
        and identity_counts[identity] == 1
        and isinstance(raw_units[index], Mapping)
    }
    generated = set(preserved)
    clause_sequences: dict[str, int] = defaultdict(int)
    output: list[Any] = []
    for index, raw in enumerate(raw_units):
        if not isinstance(raw, Mapping):
            output.append(raw)
            continue
        unit = dict(raw)
        clause_id = str(unit.get("clause_id") or "").strip()
        identity = identities[index]
        if (
            clause_shapes.get(clause_id) != "prose"
            or (identity and identity_counts[identity] == 1)
        ):
            output.append(unit)
            continue
        while True:
            clause_sequences[clause_id] += 1
            candidate = (
                f"__harness_prose_{clause_id or 'unknown'}_"
                f"{clause_sequences[clause_id]}"
            )
            if candidate not in generated:
                break
        generated.add(candidate)
        unit["unit_id"] = candidate
        output.append(unit)
    return output


def _matching_structured_intake(
    candidate: Mapping[str, Any],
    contracts: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    leaf = str(candidate.get("source_path") or "").rsplit(".", 1)[-1].casefold()
    matches = [
        dict(contract)
        for contract in contracts
        if str(contract.get("alias") or "").casefold() == leaf
        and _structured_intake_value_enabled(
            candidate.get("raw_value"),
            str(contract.get("value_semantics") or ""),
        )
    ]
    return matches[0] if len(matches) == 1 else None


def _semantic_structured_input_candidates(text: str) -> dict[str, Any]:
    """Return one registry-aware, non-overlapping structured atom set."""

    parsed = extract_structured_input_candidates(text) or {}
    candidates = [
        dict(candidate)
        for candidate in parsed.get("field_candidates") or ()
        if isinstance(candidate, Mapping)
    ]
    contracts = _structured_intake_contracts()
    composite_paths = {
        str(candidate.get("source_path") or "")
        for candidate in candidates
        if isinstance(candidate.get("raw_value"), (Mapping, list))
        and _matching_structured_intake(candidate, contracts)
    }
    filtered = [
        candidate
        for candidate in candidates
        if not any(
            str(candidate.get("source_path") or "").startswith(f"{path}.")
            for path in composite_paths
        )
    ]
    return {**dict(parsed), "field_candidates": filtered}


def _structured_intake_value_enabled(value: Any, semantics: str) -> bool:
    if semantics == "boolean_true":
        return value is True or str(value).strip().casefold() == "true"
    if semantics == "direct_value":
        return value is not None and value != ""
    return False


def _registered_value_mentions(text: str) -> tuple[dict[str, Any], ...]:
    """Project only registry values that occur in the complete source turn."""

    records = semantic_value_domain_conflicts(
        text,
        owning_group="",
    )
    allowed_keys = (
        "action_type",
        "argument",
        "target_group",
        "semantic_owner",
        "value",
        "canonical_value",
        "domain_kind",
    )
    return tuple({
        key: record.get(key)
        for key in allowed_keys
    } for record in records)


def _cross_domain_pending_errors(
    partition: Sequence[Mapping[str, Any]],
    state: AgentGraphState,
) -> tuple[str, ...]:
    """Enforce registered value ownership while another group is pending."""

    pending = dict(state.get("pending_question") or {})
    if not pending:
        return ()
    pending_group = str(pending.get("group") or "")
    errors: list[str] = []
    for unit in partition:
        operation = str(unit.get("operation") or "")
        if _is_bound_semantic_draft_pending_answer(unit, state):
            continue
        source = _semantic_source_for_unit(unit)
        conflicts = semantic_value_domain_conflicts(
            source,
            owning_group=pending_group,
            pending_question=pending,
        )
        if not conflicts:
            continue
        if operation == "pending_answer":
            for record in conflicts:
                errors.append(
                    "Stage A assigned a registered semantic value to the wrong "
                    "pending owner: "
                    f"{unit.get('unit_id')}/{record.get('semantic_owner')}/"
                    f"{record.get('action_type')}/{record.get('value')}"
                )
            continue
        if operation == "unresolved":
            for record in conflicts:
                errors.append(
                    "Stage A left a registered cross-group semantic value "
                    "unresolved while another group was pending; route an "
                    "independent request to its registered owner instead of "
                    "treating the pending barrier as a routing barrier: "
                    f"{unit.get('unit_id')}/{record.get('semantic_owner')}/"
                    f"{record.get('action_type')}/{record.get('value')}"
                )
            continue
        if operation != "domain_request":
            continue
        routed_groups = {
            str(route.get("group") or "")
            for route in unit.get("owner_routes") or []
            if isinstance(route, Mapping)
        }
        for record in conflicts:
            target_group = str(record.get("target_group") or "")
            if target_group and target_group not in routed_groups:
                errors.append(
                    "Stage A routed a registered cross-group semantic value "
                    "to the wrong domain owner: "
                    f"{unit.get('unit_id')}/{target_group}/"
                    f"{record.get('action_type')}/{record.get('value')}"
                )
    return tuple(dict.fromkeys(errors))


def _pending_answer_contract_errors(
    partition: Sequence[Mapping[str, Any]],
    state: AgentGraphState,
) -> tuple[str, ...]:
    """Reject parser-bound pending values that violate the signed contract.

    A prose unit is only a semantic claim that the source contains an answer;
    Stage B still owns value extraction.  Applying the scalar validator to the
    complete prose span would incorrectly reject labelled values and sibling
    demands before semantic compilation can isolate them.
    """

    pending = dict(state.get("pending_question") or {})
    if not pending or pending.get("options"):
        return ()
    errors: list[str] = []
    for unit in partition:
        if str(unit.get("operation") or "") != "pending_answer":
            continue
        if not str(unit.get("source_path") or "").strip():
            continue
        value = _semantic_value_for_unit(unit)
        if not value_satisfies_pending_contract(value, pending):
            errors.append(
                "Stage A parser-bound pending value does not satisfy the active "
                f"signed manual contract: {unit.get('unit_id')}/{pending.get('id')}"
            )
    return tuple(dict.fromkeys(errors))


def _is_bound_semantic_draft_pending_answer(
    unit: Mapping[str, Any],
    state: Mapping[str, Any],
) -> bool:
    """Recognize an answer owned by the exact active semantic draft atom."""

    if str(unit.get("operation") or "") != "pending_answer":
        return False
    pending = dict(state.get("pending_question") or {})
    binding = dict(pending.get("semantic_draft_binding") or {})
    draft = dict(state.get("semantic_plan_draft") or {})
    if (
        draft.get("status") != "awaiting_clarification"
        or not binding
    ):
        return False
    try:
        expected = semantic_draft_question_binding(draft)
    except ValueError:
        return False
    if binding != expected:
        return False
    routes = [
        {
            "owner": str(route.get("owner") or ""),
            "group": str(route.get("group") or ""),
        }
        for route in unit.get("owner_routes") or []
        if isinstance(route, Mapping)
    ]
    return routes == [{
        "owner": "coordinator",
        "group": str(pending.get("group") or ""),
    }]


def _semantic_source_for_unit(unit: Mapping[str, Any]) -> str:
    """Return the exact semantic field evidence represented by one unit."""

    source = str(unit.get("source_text") or "")
    source_path = str(unit.get("source_path") or "").strip()
    if not source_path:
        return source
    candidates = _semantic_structured_input_candidates(source)
    matching = [
        candidate
        for candidate in candidates.get("field_candidates") or []
        if isinstance(candidate, Mapping)
        and str(candidate.get("source_path") or "") == source_path
    ]
    if len(matching) != 1:
        return source
    exact_source = str(matching[0].get("source_evidence") or "")
    if exact_source:
        return exact_source
    value = matching[0].get("raw_value")
    return json.dumps(
        {source_path: value},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _semantic_value_for_unit(unit: Mapping[str, Any]) -> Any:
    """Return the parser-proven typed value for one structured DemandAtom."""

    source_path = str(unit.get("source_path") or "").strip()
    if not source_path:
        return None
    candidates = _semantic_structured_input_candidates(
        str(unit.get("source_text") or "")
    )
    matching = [
        candidate
        for candidate in candidates.get("field_candidates") or []
        if isinstance(candidate, Mapping)
        and str(candidate.get("source_path") or "") == source_path
    ]
    if len(matching) != 1:
        return None
    return matching[0].get("raw_value")


def _turn_bound_pending_typed_candidates(
    state: Mapping[str, Any],
    text: str,
) -> tuple[str, ...]:
    """Return syntax candidates without exposing ingress secret material.

    Sensitive manual values are projected before LangGraph sees the turn.  The
    opaque reference remains a typed candidate only when its process-local
    binding resolves and the resolved value satisfies the active signed
    question contract.  This supplies syntax, not intent; Stage A still decides
    whether the surrounding source selects, rejects, or merely mentions it.
    """

    pending = dict(state.get("pending_question") or {})
    candidates = list(typed_pending_value_candidates(text, pending))
    bindings = (
        (state.get("turn_context") or {}).get("input_secret_bindings") or ()
    )
    for raw in bindings:
        if not isinstance(raw, Mapping):
            continue
        reference = str(raw.get("reference") or "")
        if not reference or reference not in text:
            continue
        resolved = resolve_secret_reference(
            reference,
            draft_id=str(raw.get("draft_id") or ""),
            atom_id=str(raw.get("atom_id") or ""),
            expected_hash=str(raw.get("value_hash") or ""),
        )
        if (
            resolved is not None
            and value_satisfies_pending_contract(resolved, pending)
        ):
            candidates.append(reference)
    return tuple(dict.fromkeys(candidates))


def _stage_b_semantics_for_unit(
    state: Mapping[str, Any],
    unit: Mapping[str, Any],
) -> tuple[str, Any]:
    """Bind one parser-proven pending candidate to owner compilation."""

    source = _semantic_source_for_unit(unit)
    value = _semantic_value_for_unit(unit)
    if str(unit.get("operation") or "") != "pending_answer":
        return source, value
    candidates = _turn_bound_pending_typed_candidates(state, source)
    if len(candidates) != 1:
        return source, value
    return candidates[0], candidates[0]


def _validate_partition_document(
    text: str,
    clauses: Sequence[TurnClause],
    *,
    state: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [], ("Stage A did not return strict JSON",)
    if not isinstance(payload, dict) or set(payload) - {"semantic_units", "reason"}:
        return [], ("Stage A returned undeclared top-level fields",)
    raw_units = payload.get("semantic_units")
    if not isinstance(raw_units, list):
        return [], ("Stage A semantic_units is not a list",)
    raw_units = _normalize_contract_bound_pending_routes(
        raw_units,
        state or {},
    )
    raw_units, _ = _canonicalize_atomic_evidence_partition(
        dict(state or {}),
        clauses,
        raw_units,
        raw_units,
    )
    raw_units = _bind_structured_partition_provenance(raw_units, clauses)
    raw_units = _bind_ambiguous_prose_partition_identities(raw_units, clauses)
    raw_units = _retain_unclaimed_prose_clauses(raw_units, clauses)
    placed_partition = validate_semantic_partition(raw_units, clauses)
    raw_units = list(placed_partition.units)
    raw_units, _ = _canonicalize_unique_option_pending_partition(
        dict(state or {}),
        clauses,
        raw_units,
        raw_units,
    )
    raw_units = _project_uncompilable_routes_to_unresolved(raw_units)
    partition = validate_semantic_partition(raw_units, clauses)
    errors = list(dict.fromkeys((
        *placed_partition.errors,
        *partition.errors,
    )))
    output: list[dict[str, Any]] = []
    clauses_by_id = {clause.clause_id: clause for clause in clauses}
    structured_fields = {
        clause.clause_id: {
            str(candidate.get("source_path") or ""): dict(candidate)
            for candidate in (
                _semantic_structured_input_candidates(clause.text).get(
                    "field_candidates"
                )
                or []
            )
            if isinstance(candidate, Mapping)
            and str(candidate.get("source_path") or "")
        }
        for clause in clauses
        if clause.input_shape == "structured"
    }
    structured_paths = {
        clause_id: set(fields)
        for clause_id, fields in structured_fields.items()
    }
    expected_structured_atom_ids = {
        clause.clause_id: {
            _structured_atom_id(clause.clause_id, index)
            for index, _candidate in enumerate(fields.values(), start=1)
        }
        for clause in clauses
        if (fields := structured_fields.get(clause.clause_id))
    }
    structured_intake_contracts = _structured_intake_contracts()
    observed_structured_paths: dict[str, set[str]] = defaultdict(set)
    observed_structured_atom_ids: dict[str, set[str]] = defaultdict(set)
    atomic_evidence_contract = _atomic_evidence_contract_active(
        state or {},
        clauses,
    )
    atomic_evidence_answer_clauses: set[str] = set()
    semantic_draft_answer_clauses: set[str] = set()
    for raw in partition.units:
        unit = dict(raw)
        unit_id = str(unit.get("unit_id") or "")
        unit["unit_id"] = unit_id
        unit["clause_id"] = str(unit.get("clause_id") or "")
        if set(unit) - (_PARTITION_KEYS | {"start", "end"}):
            errors.append(f"Stage A unit has undeclared fields: {unit_id}")
        operation = str(unit.get("operation") or "")
        if operation not in _UNIVERSAL_OPERATIONS:
            errors.append(f"Stage A unit has invalid operation: {unit_id}")
        routes = unit.get("owner_routes")
        if not isinstance(routes, list):
            errors.append(f"Stage A owner_routes is not a list: {unit_id}")
            routes = []
        normalized_routes: list[dict[str, str]] = []
        route_identities: set[tuple[str, str]] = set()
        for raw_route in routes:
            if not isinstance(raw_route, Mapping) or set(raw_route) != _ROUTE_KEYS:
                errors.append(f"Stage A route contract is invalid: {unit_id}")
                continue
            owner = str(raw_route.get("owner") or "")
            group = str(raw_route.get("group") or "")
            if owner not in _OWNERS:
                errors.append(f"Stage A route owner is invalid: {unit_id}/{owner}")
            group_spec = GROUP_SPEC_BY_NAME.get(group) if group else None
            if operation == "domain_request" and not group:
                errors.append(
                    f"Stage A domain request route has no group: {unit_id}/{owner}"
                )
            if group and group_spec is None:
                errors.append(f"Stage A route group is invalid: {unit_id}/{group}")
            if (
                group_spec is not None
                and owner not in {"coordinator", "orientation", "analysis"}
                and group_spec.owner != owner
            ):
                errors.append(
                    f"Stage A route owner/group mismatch: {unit_id}/{owner}/{group}"
                )
            identity = (owner, group)
            if identity in route_identities:
                errors.append(
                    f"Stage A route is duplicated: {unit_id}/{owner}/{group}"
                )
            route_identities.add(identity)
            normalized_routes.append({"owner": owner, "group": group})
            if operation not in {"context", "unresolved"} and not any(
                not spec.internal_only
                and spec.owner == owner
                and _action_spec_applies(
                    spec,
                    groups=frozenset({group}) if group else frozenset(),
                    operations=frozenset({operation}),
                )
                for spec in ACTION_SPECS
            ):
                errors.append(
                    "Stage A route has no registered compiler action: "
                    f"{unit_id}/{operation}/{owner}/{group}"
                )
        allowed_owners = _UNIVERSAL_OPERATION_OWNERS.get(operation)
        if allowed_owners is not None and (
            len(normalized_routes) != 1
            or normalized_routes[0]["owner"] not in allowed_owners
        ):
            errors.append(
                "Stage A universal operation owner mismatch: "
                f"{unit_id}/{operation}/{list(allowed_owners)}"
            )
        if operation in {"context", "unresolved"} and normalized_routes:
            errors.append(f"Stage A non-action unit declares routes: {unit_id}")
        if operation not in {"context", "unresolved"} and not normalized_routes:
            errors.append(f"Stage A actionable unit has no route: {unit_id}")
        clause_id = str(unit.get("clause_id") or "")
        source_path = str(unit.get("source_path") or "").strip()
        bound_draft_answer = _is_bound_semantic_draft_pending_answer(
            unit,
            state or {},
        )
        bound_atomic_evidence = (
            atomic_evidence_contract
            and operation == "pending_answer"
            and unit_id == f"__harness_atomic_evidence_{clause_id}"
            and clause_id in clauses_by_id
        )
        if bound_atomic_evidence:
            atomic_evidence_answer_clauses.add(clause_id)
            if source_path:
                errors.append(
                    "Stage A atomic evidence answer declares source_path: "
                    f"{unit_id}"
                )
            if str(unit.get("source_text") or "") != clauses_by_id[clause_id].text:
                errors.append(
                    "Stage A atomic evidence answer must preserve its "
                    f"SourceClause: {unit_id}"
                )
        elif bound_draft_answer:
            semantic_draft_answer_clauses.add(clause_id)
            if source_path:
                errors.append(
                    "Stage A semantic draft answer declares source_path: "
                    f"{unit_id}"
                )
            if str(unit.get("source_text") or "") != clauses_by_id[clause_id].text:
                errors.append(
                    "Stage A semantic draft answer must preserve its "
                    f"SourceClause: {unit_id}"
                )
        elif clause_id in structured_paths:
            expected_atom_ids = expected_structured_atom_ids.get(clause_id, set())
            if unit_id not in expected_atom_ids:
                errors.append(
                    "Stage A structured DemandAtom has unknown atom_id: "
                    f"{unit_id}"
                )
            elif unit_id in observed_structured_atom_ids[clause_id]:
                errors.append(
                    "Stage A structured DemandAtom duplicates atom_id: "
                    f"{unit_id}"
                )
            else:
                observed_structured_atom_ids[clause_id].add(unit_id)
            if not source_path and len(structured_paths[clause_id]) == 1:
                source_path = next(iter(structured_paths[clause_id]))
                unit["source_path"] = source_path
            if not source_path:
                errors.append(
                    f"Stage A structured DemandAtom has no source_path: {unit_id}"
                )
            elif source_path not in structured_paths[clause_id]:
                errors.append(
                    "Stage A structured DemandAtom has unknown source_path: "
                    f"{unit_id}/{source_path}"
                )
            elif source_path in observed_structured_paths[clause_id]:
                errors.append(
                    "Stage A structured DemandAtom duplicates source_path: "
                    f"{unit_id}/{source_path}"
                )
            else:
                observed_structured_paths[clause_id].add(source_path)
                matched_intake = _matching_structured_intake(
                    structured_fields[clause_id][source_path],
                    structured_intake_contracts,
                )
                if matched_intake:
                    expected_route = (
                        str(matched_intake.get("owner") or ""),
                        str(matched_intake.get("target_group") or ""),
                    )
                    actual_routes = {
                        (
                            str(route.get("owner") or ""),
                            str(route.get("group") or ""),
                        )
                        for route in normalized_routes
                    }
                    if operation != "domain_request" or expected_route not in actual_routes:
                        errors.append(
                            "Stage A structured intake route mismatch: "
                            f"{unit_id}/{source_path}/{expected_route[0]}/"
                            f"{expected_route[1]}"
                        )
                    # This contract is derived from the action registry and the
                    # parser-proven source path. The model may classify and route
                    # the atom, but it does not author its executable intake.
                    unit["registered_intake"] = matched_intake
            if str(unit.get("source_text") or "") != clauses_by_id[clause_id].text:
                errors.append(
                    "Stage A structured DemandAtom must preserve its SourceClause: "
                    f"{unit_id}"
                )
        elif source_path:
            errors.append(f"Stage A prose unit declares source_path: {unit_id}")
        unit["owner_routes"] = normalized_routes
        output.append(unit)
    for clause_id, expected_paths in structured_paths.items():
        if (
            clause_id in semantic_draft_answer_clauses
            or clause_id in atomic_evidence_answer_clauses
        ):
            continue
        missing = sorted(expected_paths - observed_structured_paths[clause_id])
        if missing:
            errors.append(
                f"Stage A structured DemandAtoms omit source paths in {clause_id}: "
                + ", ".join(missing)
            )
        missing_atom_ids = sorted(
            expected_structured_atom_ids.get(clause_id, set())
            - observed_structured_atom_ids[clause_id]
        )
        if missing_atom_ids:
            errors.append(
                f"Stage A structured DemandAtoms omit atom ids in {clause_id}: "
                + ", ".join(missing_atom_ids)
            )
    return output, tuple(dict.fromkeys(errors))


def _project_uncompilable_routes_to_unresolved(
    units: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Fail closed per source unit when no registered action can compile it.

    Stage A may recognize a real demand whose owner currently has no action for
    the selected group.  That source span is unresolved product work, not an
    invalid replacement for otherwise compilable sibling units.  Structurally
    invalid or partly compilable route sets remain untouched so normal
    validation still rejects them.
    """

    projected: list[dict[str, Any]] = []
    for raw in units:
        unit = dict(raw)
        operation = str(unit.get("operation") or "")
        routes = unit.get("owner_routes")
        if (
            operation in {"context", "unresolved"}
            or operation not in _UNIVERSAL_OPERATIONS
            or not isinstance(routes, list)
            or not routes
        ):
            projected.append(unit)
            continue
        structurally_valid = all(
            isinstance(route, Mapping)
            and set(route) == _ROUTE_KEYS
            and str(route.get("owner") or "") in _OWNERS
            and (
                not str(route.get("group") or "")
                or str(route.get("group") or "") in GROUP_SPEC_BY_NAME
            )
            and (
                not str(route.get("group") or "")
                or str(route.get("owner") or "")
                in {"coordinator", "orientation", "analysis"}
                or GROUP_SPEC_BY_NAME[str(route.get("group") or "")].owner
                == str(route.get("owner") or "")
            )
            for route in routes
        )
        if not structurally_valid:
            projected.append(unit)
            continue
        route_support = [
            any(
                not spec.internal_only
                and spec.owner == str(route.get("owner") or "")
                and _action_spec_applies(
                    spec,
                    groups=(
                        frozenset({str(route.get("group") or "")})
                        if str(route.get("group") or "")
                        else frozenset()
                    ),
                    operations=frozenset({operation}),
                )
                for spec in ACTION_SPECS
            )
            for route in routes
        ]
        if any(route_support):
            projected.append(unit)
            continue
        unit["operation"] = "unresolved"
        unit["owner_routes"] = []
        unit["reason"] = (
            "The Harness has no registered compiler action for this exact "
            "source demand; preserve it as an unresolved atom."
        )
        projected.append(unit)
    return projected


def _normalize_contract_bound_pending_routes(
    raw_units: Sequence[Any],
    state: Mapping[str, Any],
) -> list[Any]:
    """Derive a pending-answer group only from the active signed contract."""

    pending = dict(state.get("pending_question") or {})
    pending_group = str(pending.get("group") or "")
    pending_owners = _UNIVERSAL_OPERATION_OWNERS.get("pending_answer") or ()
    if (
        not pending_group
        or pending_group not in GROUP_SPEC_BY_NAME
        or len(pending_owners) != 1
    ):
        return list(raw_units)
    pending_owner = pending_owners[0]
    output: list[Any] = []
    for raw in raw_units:
        if not isinstance(raw, Mapping):
            output.append(raw)
            continue
        unit = dict(raw)
        routes = unit.get("owner_routes")
        if (
            str(unit.get("operation") or "") != "pending_answer"
            or not isinstance(routes, list)
            or len(routes) != 1
            or not isinstance(routes[0], Mapping)
            or set(routes[0]) != _ROUTE_KEYS
            or str(routes[0].get("owner") or "") != pending_owner
        ):
            output.append(unit)
            continue
        unit["owner_routes"] = [{
            "owner": pending_owner,
            "group": pending_group,
        }]
        output.append(unit)
    return output


def _retain_unclaimed_prose_clauses(
    raw_units: Sequence[Any],
    clauses: Sequence[TurnClause],
) -> list[Any]:
    """Keep unclaimed prose visible to the independent omission reviewer."""

    clause_order = {
        clause.clause_id: index
        for index, clause in enumerate(clauses)
    }
    rows_by_clause: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    observed_order: list[int] = []
    used_ids: set[str] = set()
    for raw in raw_units:
        if not isinstance(raw, Mapping):
            return list(raw_units)
        clause_id = str(raw.get("clause_id") or "")
        if clause_id not in clause_order:
            return list(raw_units)
        observed_order.append(clause_order[clause_id])
        rows_by_clause[clause_id].append(raw)
        used_ids.add(str(raw.get("unit_id") or ""))
    if observed_order != sorted(observed_order):
        return list(raw_units)

    output: list[Any] = []
    for clause in clauses:
        rows = rows_by_clause.get(clause.clause_id, [])
        if rows:
            output.extend(rows)
            continue
        if clause.input_shape != "prose":
            continue
        sequence = 1
        while True:
            unit_id = f"__harness_context_{clause.clause_id}_{sequence}"
            if unit_id not in used_ids:
                used_ids.add(unit_id)
                break
            sequence += 1
        output.append({
            "unit_id": unit_id,
            "clause_id": clause.clause_id,
            "source_text": clause.text,
            "operation": "context",
            "owner_routes": [],
            "reason": "unclaimed prose retained for independent coverage review",
        })
    return output


def _partition_after_stage_a_admission(
    partition: Sequence[Mapping[str, Any]],
    redundant_unit_ids: frozenset[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate immutable source coverage from executable owner compilation."""

    source_partition: list[dict[str, Any]] = []
    compilation_partition: list[dict[str, Any]] = []
    for raw in partition:
        unit = dict(raw)
        if str(unit.get("unit_id") or "") in redundant_unit_ids:
            unit["operation"] = "context"
            unit["owner_routes"] = []
            unit["_admission_support"] = True
            unit["reason"] = (
                f"{str(unit.get('reason') or '').strip()} "
                "Stage A admission classified this as a non-executable duplicate."
            ).strip()
            source_partition.append(unit)
            continue
        source_partition.append(unit)
        compilation_partition.append(dict(unit))
    return source_partition, compilation_partition


def _canonicalize_atomic_evidence_partition(
    state: AgentGraphState,
    clauses: Sequence[TurnClause],
    source_partition: Sequence[Mapping[str, Any]],
    compilation_partition: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Preserve one typed evidence contribution as one lossless owner unit."""

    pending = dict(state.get("pending_question") or {})
    if not _atomic_evidence_contract_active(state, clauses):
        return (
            [dict(unit) for unit in source_partition],
            [dict(unit) for unit in compilation_partition],
        )
    executable = [
        unit
        for unit in compilation_partition
        if str(unit.get("operation") or "") != "context"
    ]
    if (
        not executable
        or any(
            str(unit.get("operation") or "") != "pending_answer"
            for unit in executable
        )
    ):
        return (
            [dict(unit) for unit in source_partition],
            [dict(unit) for unit in compilation_partition],
        )
    atomic = {
        "unit_id": f"__harness_atomic_evidence_{clauses[0].clause_id}",
        "clause_id": clauses[0].clause_id,
        "source_text": clauses[0].text,
        "operation": "pending_answer",
        "owner_routes": [{
            "owner": "coordinator",
            "group": str(pending.get("group") or state.get("active_group") or ""),
        }],
        "reason": (
            "Typed evidence_contribution remains one lossless pending-owner "
            "transaction."
        ),
    }
    return [atomic], [dict(atomic)]


def _atomic_evidence_contract_active(
    state: Mapping[str, Any],
    clauses: Sequence[TurnClause],
) -> bool:
    """Return whether one exact SourceClause owns the signed evidence value."""

    pending = dict(state.get("pending_question") or {})
    return bool(
        str((pending.get("validation") or {}).get("value_type") or "")
        == "evidence_contribution"
        and len(clauses) == 1
        and clauses[0].input_shape == "structured"
    )


def _canonicalize_unique_manual_pending_partition(
    state: AgentGraphState,
    clauses: Sequence[TurnClause],
    source_partition: Sequence[Mapping[str, Any]],
    compilation_partition: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep one typed manual value executable and retain its wrapper as context."""

    pending = dict(state.get("pending_question") or {})
    if (
        pending.get("manual_input_allowed") is not True
        or pending.get("options")
    ):
        return (
            [dict(unit) for unit in source_partition],
            [dict(unit) for unit in compilation_partition],
        )
    pending_units = [
        unit
        for unit in compilation_partition
        if str(unit.get("operation") or "") == "pending_answer"
    ]
    if len(pending_units) <= 1:
        return (
            [dict(unit) for unit in source_partition],
            [dict(unit) for unit in compilation_partition],
        )
    candidates: list[tuple[str, str]] = []
    for clause in clauses:
        for value in typed_pending_value_candidates(clause.text, pending):
            identity = pending_value_identity(value, pending)
            if identity:
                candidates.append((clause.clause_id, identity))
    identities = {identity for _clause_id, identity in candidates}
    candidate_clause_ids = {
        clause_id for clause_id, _identity in candidates
    }
    if len(identities) != 1 or len(candidate_clause_ids) != 1:
        return (
            [dict(unit) for unit in source_partition],
            [dict(unit) for unit in compilation_partition],
        )
    candidate_clause_id = next(iter(candidate_clause_ids))
    if not any(
        str(unit.get("clause_id") or "") == candidate_clause_id
        for unit in pending_units
    ):
        return (
            [dict(unit) for unit in source_partition],
            [dict(unit) for unit in compilation_partition],
        )
    clause_by_id = {clause.clause_id: clause for clause in clauses}
    candidate_clause = clause_by_id[candidate_clause_id]
    candidate_unit = next(
        unit
        for unit in pending_units
        if str(unit.get("clause_id") or "") == candidate_clause_id
    )
    atomic = {
        "unit_id": str(
            candidate_unit.get("unit_id")
            or f"manual-{candidate_clause_id}"
        ),
        "clause_id": candidate_clause_id,
        "source_text": candidate_clause.text,
        "operation": "pending_answer",
        "owner_routes": [{
            "owner": "coordinator",
            "group": str(
                pending.get("group")
                or state.get("active_group")
                or ""
            ),
        }],
        "reason": (
            "The complete turn contains one typed manual candidate; wrapper "
            "clauses remain visible as non-executable source context."
        ),
    }

    def canonicalize(
        partition: Sequence[Mapping[str, Any]],
        *,
        retain_context: bool,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        emitted = False
        for raw in partition:
            unit = dict(raw)
            if str(unit.get("operation") or "") != "pending_answer":
                result.append(unit)
                continue
            if (
                str(unit.get("clause_id") or "") == candidate_clause_id
                and not emitted
            ):
                result.append(dict(atomic))
                emitted = True
                continue
            if retain_context:
                unit["operation"] = "context"
                unit["owner_routes"] = []
                unit["reason"] = (
                    "Stage A classified this complete wrapper as part of the "
                    "same unique manual transaction."
                )
                result.append(unit)
        if not emitted:
            result.append(dict(atomic))
        return result

    return (
        canonicalize(source_partition, retain_context=True),
        canonicalize(compilation_partition, retain_context=False),
    )


def _canonicalize_unique_option_pending_partition(
    state: AgentGraphState,
    clauses: Sequence[TurnClause],
    source_partition: Sequence[Mapping[str, Any]],
    compilation_partition: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Bind one contract-exact option while retaining every sibling source span."""

    pending = dict(state.get("pending_question") or {})
    candidate = _unique_option_prefix_candidate(state, clauses)
    if candidate is None:
        return (
            [dict(unit) for unit in source_partition],
            [dict(unit) for unit in compilation_partition],
        )
    clause, _selected_value, answer_end = candidate
    if answer_end <= 0 or answer_end > len(clause.text):
        return (
            [dict(unit) for unit in source_partition],
            [dict(unit) for unit in compilation_partition],
        )
    used_ids = {
        str(unit.get("unit_id") or "")
        for unit in (*source_partition, *compilation_partition)
    }
    sequence = 1
    while True:
        pending_unit_id = (
            f"__harness_pending_option_{clause.clause_id}_{sequence}"
        )
        if pending_unit_id not in used_ids:
            break
        sequence += 1
    atomic = {
        "unit_id": pending_unit_id,
        "clause_id": clause.clause_id,
        "start": 0,
        "end": answer_end,
        "source_text": clause.text[:answer_end],
        "operation": "pending_answer",
        "owner_routes": [{
            "owner": "coordinator",
            "group": str(
                pending.get("group")
                or state.get("active_group")
                or ""
            ),
        }],
        "reason": (
            "The active signed question contract proves one structurally "
            "delimited option selection."
        ),
    }

    def canonicalize(
        partition: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        emitted = False
        for raw in partition:
            unit = dict(raw)
            if str(unit.get("clause_id") or "") != clause.clause_id:
                output.append(unit)
                continue
            if not emitted:
                output.append(dict(atomic))
                emitted = True
            start = unit.get("start")
            end = unit.get("end")
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
            ):
                continue
            if end <= answer_end:
                continue
            if start < answer_end:
                unit["start"] = answer_end
                unit["source_text"] = clause.text[answer_end:end]
            if str(unit.get("operation") or "") == "pending_answer":
                unit["operation"] = "unresolved"
                unit["owner_routes"] = []
                unit["reason"] = (
                    "Stage A absorbed source after the contract-proven option "
                    "prefix instead of classifying the remaining demand."
                )
            if str(unit.get("source_text") or ""):
                output.append(unit)
        if not emitted:
            output.append(dict(atomic))
        return output

    return (
        canonicalize(source_partition),
        canonicalize(compilation_partition),
    )


def _unique_option_prefix_candidate(
    state: Mapping[str, Any],
    clauses: Sequence[TurnClause],
) -> tuple[TurnClause, Any, int] | None:
    """Return one uniquely contract-proven option prefix for this turn."""

    pending = dict(state.get("pending_question") or {})
    if not pending.get("options"):
        return None
    candidates = [
        (clause, value, end)
        for clause in clauses
        for matched, value, end in (
            exact_option_prefix_answer(clause.text, pending),
        )
        if matched
    ]
    return candidates[0] if len(candidates) == 1 else None


def _stage_a_admission_prompt() -> str:
    return (
        "You are the independent Stage A coverage authority for AnyChain Benchmark Agent. "
        "You are not a planner and must not create product actions. Compare the complete user "
        "clauses with the immutable semantic units and the supplied owner/group purposes. "
        "Treat universal_operation_purposes as authoritative. Reject pending_answer for a "
        "hypothetical, counterfactual, consequence, explanation, or capability question because "
        "it contains no present authorization. Require evidence_analysis for a request to ingest "
        "or analyze logs, errors, traces, or evidence even when the evidence will be pasted later. "
        "When pending_question accepts manual input, a prose pending_answer is complete only "
        "when its exact source supplies one concrete value for that active typed question. A "
        "field label may surround the value, but another operation, a refusal, a question, or "
        "unrelated prose is not a manual answer. Stage B owns extraction and final admission "
        "owns value validation; this authority only accepts or rejects the semantic partition. "
        f"{PENDING_CANDIDATE_SEMANTIC_POLICY}"
        "For an operation with several registered owners, a unit is complete only "
        "when universal_owner_action_purposes proves that the selected owner has an "
        "action purpose matching the exact source demand. "
        "When that purpose declares consultation_topic_purposes, treat the closed topic "
        "map as authoritative: a read-only question is safely represented only when one "
        "topic purpose matches its complete subject. The topic identifier need not appear "
        "verbatim in user text. "
        "Return one strict JSON object with exactly unit_verdicts, clause_verdicts, and reason. "
        "unit_verdicts contains exactly one row per supplied unit in order: "
        "{unit_id,verdict:'complete'|'redundant'|'unresolved',supports_unit_id,reason}. "
        "supports_unit_id must be empty for complete/unresolved. Use redundant only when the "
        "unit duplicates, contrasts with, or merely restates one other complete unit in this "
        "turn; set supports_unit_id to that exact distinct complete unit id regardless of whether "
        "the supporting prose appears before or after it, and require the redundant unit's clause "
        "to contain no omitted demand. When a later complete unit explicitly corrects, replaces, "
        "or supersedes an earlier mutation of the same owner/group decision, classify the earlier "
        "unit as redundant and set supports_unit_id to the later correction. Do this only when the "
        "source itself makes supersession explicit. Alternatives, comparisons, conjunctions, "
        "uncertain candidates, and merely different values are not supersession and must remain "
        "unresolved rather than becoming last-value-wins. The corrected unit remains complete. "
        "registered_value_mentions proves only spelling and owner identity. When one complete "
        "unit already represents a compact registered-domain request, adjacent operation framing "
        "that adds no independent value, question, navigation, analysis, or mutation supports "
        "that unit; it is not a second pending answer or unresolved demand. "
        f"{FRAMED_REQUEST_SEMANTIC_POLICY}"
        "An explicit keep, reuse, or reconfirm instruction for one concrete registered value "
        "is complete only when routed as the owning idempotent domain request; equality with "
        "current state does not make it context or redundant. A statement that a value or "
        "evidence will be supplied later adds no present value mutation. A question about a "
        "field's requirements, format, meaning, or validity is an independent consultation, "
        "not a domain mutation. Because every semantic unit declares exactly one operation, "
        "a source span combining independently actionable operations must be split into "
        "minimal exact operation-specific units; several owner routes on one unit are valid "
        "only when every route compiles that same declared operation and the same exact words "
        "cannot be divided into independent source-grounded demands. Distinct exact substrings "
        "must remain distinct units even when they share an owner or operation. "
        "A temporal deferral, delay, absence, or later-submission statement does not become a "
        "domain request without exact source words that separately request entry, continuation, "
        "or skipping a required value. Reject a planner reason that invents such a request. "
        "contract_proven_pending_prefixes are signed Harness facts: each prefix must remain one "
        "complete pending_answer and all source after it must be independently represented. "
        "clause_verdicts contains exactly one "
        "row per supplied clause in order: {clause_id,verdict:'complete'|'omitted'|'unresolved',"
        "omitted_owner_routes:[{owner,group}],reason}. A clause is complete only when every "
        "independent present request, question, correction, navigation, value, and evidence "
        "contribution appears in an appropriate semantic unit. Connective or framing prose is "
        "not an omitted demand. A context unit has no independently admissible product action; "
        "clause_verdicts, not its unit verdict, are authoritative for detecting whether its "
        "source actually contains an omitted demand. A reason, uncertainty statement, or basis attached to one "
        "pending option selection supports that same selection unless it independently asks "
        "for research, explanation, navigation, or mutation; do not approve a fictitious "
        "second demand created only from such support. Compare alleged sibling demands with the "
        "immutable pending_question option actions and completion effects. Mark a unit redundant "
        "when it only restates the selected option's declared effect; do not count the same effect "
        "once as a pending answer and again as a domain request. Reject a partition that treats "
        "rejection of one declared option as selecting that rejected option while separately "
        "representing an affirmed declared option as a sibling demand. When pending_question "
        "accepts manual input and pending_typed_candidates proves one candidate, prose that only "
        "limits that candidate's purpose, application scope, or exclusions supports the candidate "
        "unit. Mark such wrapper units redundant unless they independently request another value, "
        "mutation, consultation, navigation, or analysis. If a demand is absent, return omitted and identify its declared "
        "owner/group route; use unresolved when no safe route can be identified. Do not accept "
        "planner reason text as evidence and never invent source text, ids, owners, or groups. "
        "contract_proven_pending_entailment_unit_ids are exact unit ids whose "
        "pending-answer meaning and typed value already reached quorum in the "
        "specialized entailment jury. Treat those units as semantically complete "
        "for pending entailment; continue to review their clause for omitted "
        "sibling demands and do not use that proof for any other unit. "
        "contract_rejected_pending_entailment_unit_ids are exact pending-answer "
        "units that failed the specialized jury. Such a unit may be redundant "
        "only when its supports_unit_id names another complete unit in "
        "contract_proven_pending_entailment_unit_ids; otherwise keep it "
        "unresolved. This relation never authorizes last-value-wins. "
        "When semantic_draft_resolutions is present, treat each row as the "
        "Harness-bound user clarification for that exact source DemandAtom. "
        "It may make that atom semantically complete, but it is not a new "
        "sibling demand and cannot justify changing another atom. A supplied "
        "background disposition is Harness-owned and must not be reclassified "
        "as an executable demand."
    )


def _review_stage_a_partition(
    provider: Any,
    stage_a_payload: Mapping[str, Any],
    partition: Sequence[Mapping[str, Any]],
    *,
    include_contract_valid: bool = False,
) -> (
    tuple[tuple[str, ...], tuple[int, ...], frozenset[str]]
    | tuple[tuple[str, ...], tuple[int, ...], frozenset[str], bool]
):
    errors, sizes, redundant, _contract_valid = (
        _review_stage_a_partition_detailed(
            provider,
            stage_a_payload,
            partition,
        )
    )
    if include_contract_valid:
        return errors, sizes, redundant, _contract_valid
    return errors, sizes, redundant


def _stage_a_review_result(
    result: (
        tuple[tuple[str, ...], tuple[int, ...], frozenset[str]]
        | tuple[tuple[str, ...], tuple[int, ...], frozenset[str], bool]
    ),
) -> tuple[tuple[str, ...], tuple[int, ...], frozenset[str], bool]:
    """Normalize compact and contract-aware admission results."""

    if len(result) == 4:
        errors, sizes, redundant, contract_valid = result
        return errors, sizes, redundant, bool(contract_valid)
    errors, sizes, redundant = result
    return errors, sizes, redundant, not errors


def _review_stage_a_candidate(
    provider: Any,
    stage_a_payload: Mapping[str, Any],
    partition: Sequence[Mapping[str, Any]],
    *,
    receipt_sink: list[dict[str, Any]] | None = None,
) -> tuple[
    tuple[str, ...],
    tuple[int, ...],
    frozenset[str],
    bool,
    tuple[dict[str, Any], ...],
]:
    """Apply the one authoritative Stage A review pipeline to a proposal."""

    review_payload = dict(stage_a_payload)
    pending = dict(stage_a_payload.get("pending_question") or {})
    pre_admission_entailment = (
        str(pending.get("value_domain") or "") == "researched_identity"
    )
    pending_sizes: tuple[int, ...] = ()
    pending_errors: tuple[str, ...] = ()
    rejected_pending_unit_ids: frozenset[str] = frozenset()
    if pre_admission_entailment:
        (
            pending_errors,
            pending_sizes,
            proven_pending_unit_ids,
            rejected_pending_unit_ids,
        ) = (
            _review_stage_a_pending_entailment_detailed(
                provider,
                stage_a_payload,
                partition,
                receipt_sink=receipt_sink,
                proposal_hash=_partition_hash(partition),
            )
        )
        if proven_pending_unit_ids:
            review_payload[
                "contract_proven_pending_entailment_unit_ids"
            ] = sorted(proven_pending_unit_ids)
        if rejected_pending_unit_ids:
            review_payload[
                "contract_rejected_pending_entailment_unit_ids"
            ] = sorted(rejected_pending_unit_ids)
    errors, sizes, redundant, contract_valid = _stage_a_review_result(
        _review_stage_a_partition(
            provider,
            review_payload,
            partition,
            include_contract_valid=True,
        )
    )
    request_sizes = [*pending_sizes, *sizes]
    if pre_admission_entailment:
        required_rejections = rejected_pending_unit_ids - redundant
        pending_errors = tuple(
            "Stage A pending-answer entailment quorum rejected unit: "
            f"{unit_id}"
            for unit_id in sorted(required_rejections)
        )
        errors = tuple(dict.fromkeys((*errors, *pending_errors)))
    if not errors and not pre_admission_entailment:
        pending_errors, pending_sizes = _review_stage_a_pending_entailment(
            provider,
            stage_a_payload,
            partition,
            receipt_sink=receipt_sink,
            proposal_hash=_partition_hash(partition),
        )
        request_sizes.extend(pending_sizes)
        errors = tuple(dict.fromkeys((*errors, *pending_errors)))
    rejected_scope_claims: tuple[dict[str, Any], ...] = ()
    if not errors:
        scope_errors, scope_sizes, rejected_scope_claims = (
            _review_stage_a_explicit_scope_authorization(
                provider,
                partition,
            )
        )
        request_sizes.extend(scope_sizes)
        errors = tuple(dict.fromkeys((*errors, *scope_errors)))
    return (
        errors,
        tuple(request_sizes),
        redundant,
        contract_valid,
        rejected_scope_claims,
    )


def _pending_entailment_prompt() -> str:
    """Return the reject-only contract for one semantic pending-answer claim."""

    return (
        "You are the independent pending-answer entailment auditor for AnyChain. "
        "You are not a planner and cannot create, route, rename, repair, or execute "
        "an operation. Judge only whether the exact proposed_source itself answers "
        "the active pending_question. Workflow state and the mere existence of a "
        "pending question are not user evidence. A request for another operation, "
        "a question about capability, a navigation request, a value-less correction "
        "or replacement request, a deferral, or unrelated information does not "
        "answer the pending question, even if satisfying that request would "
        "eventually replace or make the old question irrelevant. Correction or "
        "replacement framing does not disqualify a source that itself explicitly "
        "assigns one concrete, directly usable value to the active pending field. "
        "A natural-language answer is valid only when it "
        "commits to one declared option or directly authorizes the pending effect. "
        "Return exactly one JSON object with exactly verdict, selected_value, "
        "evidence_quote, and reason. verdict must be "
        "answers, different_request, or uncertain. For answers, evidence_quote "
        "must be the shortest non-empty exact substring that commits to the pending "
        "answer. For an option question, selected_value must copy the exact JSON "
        "value of the one declared pending_question option selected by that source; "
        "do not return an option id, label, number, or paraphrase. For a manual "
        "question, selected_value must contain exactly the concrete value in the "
        "evidence quote, preserving its typed JSON representation when the contract "
        "requires one. For different_request or uncertain, selected_value must be "
        "null. When the pending question accepts a manual value, the quote must "
        "be the shortest exact source substring containing only the proposed value "
        "that can be validated against that question; it must denote one concrete, "
        "directly usable field value. A generic or relative reference, a request to "
        "change, choose, revisit, or replace the value without supplying that "
        "concrete value, a capability question, or a promise to provide the value "
        "later is not a manual answer. Do not include "
        "its field label or sibling demands. For other verdicts it must be an exact "
        "non-empty substring showing "
        "why the source is not an answer. Never infer an answer from consequence or "
        "convenience."
    )


def _explicit_scope_authorization_prompt() -> str:
    """Return the reject-only contract for protected operation scope."""

    return (
        "You are the independent explicit-scope authorization auditor for "
        "AnyChain. You are not a planner and cannot create, choose, route, "
        "rename, repair, or execute an action. Judge only whether the exact "
        "proposed_source explicitly authorizes the complete purpose and scope "
        "of at least one supplied protected_action_purpose. A request to retry, "
        "replace, reconfigure, or revisit one business domain does not authorize "
        "clearing or resetting the complete workflow/session. Consequence, "
        "convenience, current state, and the existence of a pending reset "
        "question are not authorization. Return exactly one JSON object with "
        "exactly claim_hash, verdict, evidence_quote, and reason. Copy claim_hash "
        "exactly. verdict must be authorized, not_authorized, or uncertain. For "
        "authorized, evidence_quote must be the shortest non-empty exact "
        "substring that authorizes the complete protected scope. For other "
        "verdicts it must be an exact non-empty substring showing why that scope "
        "is not authorized."
    )


def _explicit_scope_claims(
    partition: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Project Stage A claims that can reach a protected registry purpose."""

    claims: list[dict[str, Any]] = []
    for unit in partition:
        operation = str(unit.get("operation") or "")
        source = str(unit.get("source_text") or "")
        for route in unit.get("owner_routes") or ():
            if not isinstance(route, Mapping):
                continue
            owner = str(route.get("owner") or "")
            group = str(route.get("group") or "")
            protected = [
                {
                    "action_type": spec.action_type,
                    "purpose": spec.purpose,
                }
                for spec in ACTION_SPECS
                if spec.owner == owner
                and spec.explicit_scope_authorization
                and action_spec_serves_route(
                    spec,
                    operation=operation,
                    group=group,
                )
            ]
            if protected:
                claims.append({
                    "unit_id": str(unit.get("unit_id") or ""),
                    "clause_id": str(unit.get("clause_id") or ""),
                    "proposed_source": source,
                    "operation": operation,
                    "owner": owner,
                    "group": group,
                    "protected_action_purposes": protected,
                })
    return tuple(claims)


def _review_stage_a_explicit_scope_authorization(
    provider: Any,
    partition: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[dict[str, Any], ...]]:
    """Require quorum for a Stage A claim that reaches protected scope."""

    prompt = _explicit_scope_authorization_prompt()
    request_sizes: list[int] = []
    errors: list[str] = []
    rejected: list[dict[str, Any]] = []
    for claim in _explicit_scope_claims(partition):
        claim_hash = hashlib.sha256(
            json.dumps(
                claim,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        payload = {"claim_hash": claim_hash, **claim}
        admitted_members = 0
        for _member in range(3):
            request_sizes.append(_wire_size(prompt, payload))
            response = request_semantic_compilation(
                provider,
                system_prompt=prompt,
                request_payload=payload,
                max_tokens=700,
                reasoning_mode=STRICT_JSON_REASONING_MODE,
            )
            try:
                verdict = json.loads(response)
            except json.JSONDecodeError:
                continue
            if not isinstance(verdict, Mapping) or set(verdict) != {
                "claim_hash",
                "verdict",
                "evidence_quote",
                "reason",
            }:
                continue
            evidence = str(verdict.get("evidence_quote") or "")
            if (
                str(verdict.get("claim_hash") or "") != claim_hash
                or str(verdict.get("verdict") or "")
                not in {"authorized", "not_authorized", "uncertain"}
                or not evidence
                or evidence not in claim["proposed_source"]
                or not str(verdict.get("reason") or "").strip()
            ):
                continue
            if str(verdict.get("verdict") or "") == "authorized":
                admitted_members += 1
        if admitted_members < 2:
            errors.append(
                "Stage A explicit-scope authorization quorum rejected unit: "
                f"{claim['unit_id']}"
            )
            rejected.append({
                key: claim[key]
                for key in (
                    "unit_id",
                    "clause_id",
                    "proposed_source",
                    "operation",
                    "owner",
                    "group",
                )
            })
    return tuple(errors), tuple(request_sizes), tuple(rejected)


def _review_stage_a_pending_entailment(
    provider: Any,
    stage_a_payload: Mapping[str, Any],
    partition: Sequence[Mapping[str, Any]],
    *,
    receipt_sink: list[dict[str, Any]] | None = None,
    proposal_hash: str = "",
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Require quorum before a semantic, non-literal pending claim is selectable."""

    errors, sizes, _proven_unit_ids, _rejected_unit_ids = (
        _review_stage_a_pending_entailment_detailed(
            provider,
            stage_a_payload,
            partition,
            receipt_sink=receipt_sink,
            proposal_hash=proposal_hash,
        )
    )
    return errors, sizes


def _audit_value_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _pending_entailment_member_receipt(
    response: str,
    *,
    member_index: int,
    proposed_source: str,
    pending: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Validate one member vote and retain only secret-safe typed evidence."""

    receipt: dict[str, Any] = {
        "member_index": member_index,
        "response_hash": hashlib.sha256(response.encode("utf-8")).hexdigest(),
        "response_json_valid": False,
        "response_shape_valid": False,
        "verdict": "",
        "selected_value_present": False,
        "selected_identity_hash": "",
        "evidence_quote_hash": "",
        "reason_hash": "",
        "evidence_source_bound": False,
        "selected_value_evidence_bound": False,
        "pending_contract_valid": False,
        "accepted_vote": False,
        "rejection_code": "invalid_json",
    }
    try:
        verdict = json.loads(response)
    except json.JSONDecodeError:
        return "", receipt
    receipt["response_json_valid"] = True
    if not isinstance(verdict, Mapping) or set(verdict) != {
        "verdict",
        "selected_value",
        "evidence_quote",
        "reason",
    }:
        receipt["rejection_code"] = "invalid_shape"
        return "", receipt
    receipt["response_shape_valid"] = True
    evidence = str(verdict.get("evidence_quote") or "")
    reason = str(verdict.get("reason") or "")
    verdict_name = str(verdict.get("verdict") or "")
    selected_value = verdict.get("selected_value")
    normalized_verdict = (
        verdict_name
        if verdict_name in {"answers", "different_request", "uncertain"}
        else ""
    )
    receipt.update({
        "verdict": normalized_verdict,
        "selected_value_present": selected_value is not None,
        "evidence_quote_hash": _audit_value_hash(evidence) if evidence else "",
        "reason_hash": _audit_value_hash(reason) if reason else "",
        "evidence_source_bound": bool(evidence and evidence in proposed_source),
    })
    if verdict_name not in {"answers", "different_request", "uncertain"}:
        receipt["rejection_code"] = "invalid_verdict"
        return "", receipt
    if not evidence:
        receipt["rejection_code"] = "missing_evidence_quote"
        return "", receipt
    if not receipt["evidence_source_bound"]:
        receipt["rejection_code"] = "evidence_not_source_bound"
        return "", receipt
    if not reason.strip():
        receipt["rejection_code"] = "missing_reason"
        return "", receipt
    if verdict_name != "answers":
        receipt["rejection_code"] = (
            "non_answer_selected_value"
            if selected_value is not None
            else "semantic_non_answer"
        )
        return "", receipt

    selected_identity = pending_value_identity(selected_value, pending)
    if selected_identity:
        receipt["selected_identity_hash"] = _audit_value_hash(
            selected_identity
        )
    if not selected_identity:
        receipt["rejection_code"] = "missing_selected_identity"
        return "", receipt
    if selected_identity.startswith("option:"):
        receipt["selected_value_evidence_bound"] = True
        receipt["pending_contract_valid"] = True
    elif str(pending.get("value_domain") or "") == "researched_identity":
        selected_text = str(selected_value or "").strip()
        receipt["selected_value_evidence_bound"] = bool(
            selected_text and selected_text in evidence
        )
        receipt["pending_contract_valid"] = bool(
            pending.get("manual_input_allowed") is True
            and researched_identity_value_is_valid(selected_text)
        )
        if not receipt["pending_contract_valid"]:
            receipt["rejection_code"] = "researched_identity_contract_rejected"
            return "", receipt
        if not receipt["selected_value_evidence_bound"]:
            receipt["rejection_code"] = "selected_value_not_evidence_bound"
            return "", receipt
    else:
        evidence_identity = pending_value_identity(evidence, pending)
        receipt["selected_value_evidence_bound"] = (
            selected_identity == evidence_identity
        )
        receipt["pending_contract_valid"] = bool(
            pending.get("manual_input_allowed") is True
        )
        if (
            not receipt["pending_contract_valid"]
            or not receipt["selected_value_evidence_bound"]
        ):
            receipt["rejection_code"] = "manual_identity_mismatch"
            return "", receipt
    receipt["accepted_vote"] = True
    receipt["rejection_code"] = "accepted"
    return selected_identity, receipt


def _review_stage_a_pending_entailment_detailed(
    provider: Any,
    stage_a_payload: Mapping[str, Any],
    partition: Sequence[Mapping[str, Any]],
    *,
    receipt_sink: list[dict[str, Any]] | None = None,
    proposal_hash: str = "",
) -> tuple[
    tuple[str, ...],
    tuple[int, ...],
    frozenset[str],
    frozenset[str],
]:
    """Return quorum failures and exact pending units proven by the jury."""

    pending = dict(stage_a_payload.get("pending_question") or {})
    if (
        not pending
        or isinstance(pending.get("semantic_draft_binding"), Mapping)
        or stage_a_payload.get("contract_proven_pending_prefixes")
        or stage_a_payload.get("pending_typed_candidates")
    ):
        return (), (), frozenset(), frozenset()
    claims = [
        {
            "unit_id": str(unit.get("unit_id") or ""),
            "clause_id": str(unit.get("clause_id") or ""),
            "proposed_source": str(unit.get("source_text") or ""),
        }
        for unit in partition
        if str(unit.get("operation") or "") == "pending_answer"
    ]
    if not claims:
        return (), (), frozenset(), frozenset()

    prompt = _pending_entailment_prompt()
    request_sizes: list[int] = []
    errors: list[str] = []
    proven_unit_ids: set[str] = set()
    rejected_unit_ids: set[str] = set()
    for claim in claims:
        claim_payload = {
            "pending_question": pending,
            **claim,
        }
        claim_hash = hashlib.sha256(
            json.dumps(
                claim_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        payload = claim_payload
        answer_votes: dict[str, int] = defaultdict(int)
        member_receipts: list[dict[str, Any]] = []
        claim_request_sizes: list[int] = []
        for member_index in range(1, 4):
            request_size = _wire_size(prompt, payload)
            request_sizes.append(request_size)
            claim_request_sizes.append(request_size)
            response = request_semantic_compilation(
                provider,
                system_prompt=prompt,
                request_payload=payload,
                max_tokens=700,
                reasoning_mode=STRICT_JSON_REASONING_MODE,
            )
            selected_identity, member_receipt = (
                _pending_entailment_member_receipt(
                    response,
                    member_index=member_index,
                    proposed_source=claim["proposed_source"],
                    pending=pending,
                )
            )
            member_receipts.append(member_receipt)
            if selected_identity:
                answer_votes[selected_identity] += 1
        quorum_identities = sorted(
            identity for identity, count in answer_votes.items() if count >= 2
        )
        quorum_reached = len(quorum_identities) == 1
        if receipt_sink is not None:
            receipt_sink.append({
                "proposal_hash": proposal_hash or _partition_hash(partition),
                "claim_hash": claim_hash,
                "unit_id": claim["unit_id"],
                "pending_contract_hash": _audit_value_hash(pending),
                "source_hash": _audit_value_hash(claim["proposed_source"]),
                "request_count": 3,
                "request_sizes": claim_request_sizes,
                "members": member_receipts,
                "identity_vote_counts": [
                    {
                        "identity_hash": _audit_value_hash(identity),
                        "count": count,
                    }
                    for identity, count in sorted(answer_votes.items())
                ],
                "quorum_identity_hash": (
                    _audit_value_hash(quorum_identities[0])
                    if quorum_reached
                    else ""
                ),
                "quorum_reached": quorum_reached,
            })
        if not quorum_reached:
            rejected_unit_ids.add(claim["unit_id"])
            errors.append(
                "Stage A pending-answer entailment quorum rejected unit: "
                f"{claim['unit_id']}"
            )
        else:
            proven_unit_ids.add(claim["unit_id"])
    return (
        tuple(errors),
        tuple(request_sizes),
        frozenset(proven_unit_ids),
        frozenset(rejected_unit_ids),
    )


def _review_stage_a_partition_detailed(
    provider: Any,
    stage_a_payload: Mapping[str, Any],
    partition: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, ...], tuple[int, ...], frozenset[str], bool]:
    prompt = _stage_a_admission_prompt()
    payload = {
        "user_text": stage_a_payload["user_text"],
        "clauses": stage_a_payload["clauses"],
        "semantic_units": [dict(unit) for unit in partition],
        "pending_question": dict(stage_a_payload.get("pending_question") or {}),
        "pending_barrier_contract": dict(
            stage_a_payload.get("pending_barrier_contract") or {}
        ),
        "pending_typed_candidates": list(
            stage_a_payload.get("pending_typed_candidates") or []
        ),
        "contract_proven_pending_prefixes": list(
            stage_a_payload.get("contract_proven_pending_prefixes") or []
        ),
        "contract_proven_semantic_draft_resolution": dict(
            stage_a_payload.get("contract_proven_semantic_draft_resolution")
            or {}
        ),
        "contract_proven_pending_entailment_unit_ids": list(
            stage_a_payload.get(
                "contract_proven_pending_entailment_unit_ids"
            )
            or []
        ),
        "contract_rejected_pending_entailment_unit_ids": list(
            stage_a_payload.get(
                "contract_rejected_pending_entailment_unit_ids"
            )
            or []
        ),
        "registered_semantic_value_domains": list(
            stage_a_payload.get("registered_semantic_value_domains") or []
        ),
        "registered_value_mentions": list(
            stage_a_payload.get("registered_value_mentions") or []
        ),
        "structured_candidates": list(
            stage_a_payload.get("structured_candidates") or []
        ),
        "structured_intake_contracts": list(
            stage_a_payload.get("structured_intake_contracts") or []
        ),
        "semantic_draft_resolutions": list(
            stage_a_payload.get("semantic_draft_resolutions") or []
        ),
        "semantic_draft_id": str(
            stage_a_payload.get("semantic_draft_id") or ""
        ),
        "semantic_draft_revision": int(
            stage_a_payload.get("semantic_draft_revision") or 0
        ),
        "groups": stage_a_payload["groups"],
        "routing_purposes": list(
            stage_a_payload.get("routing_purposes") or []
        ),
        "universal_operations": stage_a_payload["universal_operations"],
        "universal_operation_purposes": dict(
            stage_a_payload.get("universal_operation_purposes")
            or SEMANTIC_OPERATION_PURPOSES
        ),
        "universal_owner_action_purposes": dict(
            stage_a_payload.get("universal_owner_action_purposes") or {}
        ),
    }
    request_sizes: list[int] = []
    response = ""
    contract_errors: tuple[str, ...] = ()
    semantic_errors: tuple[str, ...] = ()
    redundant_unit_ids: frozenset[str] = frozenset()
    for attempt in range(2):
        request_payload = dict(payload)
        request_prompt = prompt
        if attempt:
            request_payload["contract_repair"] = {
                "prior_invalid_output": response,
                "validation_errors": list(contract_errors),
                "instruction": (
                    "Return a complete replacement admission document. Preserve "
                    "the source meaning while correcting every structural and "
                    "cross-verdict consistency error. Change unit or clause verdicts "
                    "when the reported inconsistency requires it."
                ),
            }
            request_prompt = (
                f"{prompt} This is a contract-repair attempt. The prior document "
                f"was structurally rejected for: {'; '.join(contract_errors)}. "
                "Return one complete replacement document with every required row "
                "and key. Do not weaken or invent the source meaning merely to pass; "
                "make unit and clause verdicts mutually consistent."
            )
        request_sizes.append(_wire_size(request_prompt, request_payload))
        response = request_semantic_compilation(
            provider,
            system_prompt=request_prompt,
            request_payload=request_payload,
            max_tokens=_stage_a_review_output_token_budget(
                partition,
                stage_a_payload.get("clauses") or (),
            ),
            reasoning_mode=STRICT_JSON_REASONING_MODE,
        )
        contract_errors, semantic_errors, redundant_unit_ids = (
            _validate_stage_a_admission_document(
                response,
                stage_a_payload,
                partition,
            )
        )
        if not contract_errors:
            return (
                semantic_errors,
                tuple(request_sizes),
                redundant_unit_ids,
                True,
            )
    return (
        tuple(dict.fromkeys((*contract_errors, *semantic_errors))),
        tuple(request_sizes),
        frozenset(),
        False,
    )


def _validate_stage_a_admission_document(
    response: str,
    stage_a_payload: Mapping[str, Any],
    partition: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, ...], tuple[str, ...], frozenset[str]]:
    """Separate malformed reviewer output from a valid semantic rejection."""

    try:
        document = json.loads(response)
    except json.JSONDecodeError:
        return ("Stage A admission did not return strict JSON",), (), frozenset()
    contract_errors: list[str] = []
    semantic_errors: list[str] = []
    redundant_unit_ids: set[str] = set()
    proven_pending_unit_ids = {
        str(unit_id)
        for unit_id in stage_a_payload.get(
            "contract_proven_pending_entailment_unit_ids"
        )
        or ()
        if str(unit_id)
    }
    rejected_pending_unit_ids = {
        str(unit_id)
        for unit_id in stage_a_payload.get(
            "contract_rejected_pending_entailment_unit_ids"
        )
        or ()
        if str(unit_id)
    }
    partition_unit_ids = {
        str(unit.get("unit_id") or "")
        for unit in partition
        if str(unit.get("unit_id") or "")
    }
    if not proven_pending_unit_ids.issubset(partition_unit_ids):
        contract_errors.append(
            "Stage A admission references an unknown proven pending unit"
        )
    if not rejected_pending_unit_ids.issubset(partition_unit_ids):
        contract_errors.append(
            "Stage A admission references an unknown rejected pending unit"
        )
    if proven_pending_unit_ids & rejected_pending_unit_ids:
        contract_errors.append(
            "Stage A admission pending entailment sets overlap"
        )
    if not isinstance(document, Mapping) or set(document) != {
        "unit_verdicts",
        "clause_verdicts",
        "reason",
    }:
        contract_errors.append(
            "Stage A admission returned an invalid top-level contract"
        )
        document = {}
    unit_verdicts = document.get("unit_verdicts")
    clause_verdicts = document.get("clause_verdicts")
    expected_unit_ids = [str(unit["unit_id"]) for unit in partition]
    unit_operations = {
        str(unit["unit_id"]): str(unit.get("operation") or "")
        for unit in partition
    }
    expected_clause_ids = [
        str(clause["clause_id"]) for clause in stage_a_payload["clauses"]
    ]
    if (
        not isinstance(unit_verdicts, list)
        or len(unit_verdicts) != len(expected_unit_ids)
        or not all(isinstance(row, Mapping) for row in unit_verdicts)
        or [str(row.get("unit_id") or "") for row in unit_verdicts]
        != expected_unit_ids
    ):
        contract_errors.append(
            "Stage A admission unit verdict order or cardinality mismatch"
        )
        unit_verdicts = []
    if (
        not isinstance(clause_verdicts, list)
        or len(clause_verdicts) != len(expected_clause_ids)
        or not all(isinstance(row, Mapping) for row in clause_verdicts)
        or [str(row.get("clause_id") or "") for row in clause_verdicts]
        != expected_clause_ids
    ):
        contract_errors.append(
            "Stage A admission clause verdict order or cardinality mismatch"
        )
        clause_verdicts = []
    valid_routes = {
        (str(group["owner"]), str(group["name"]))
        for group in stage_a_payload["groups"]
    }
    for row in unit_verdicts:
        if set(row) != {
            "unit_id",
            "verdict",
            "supports_unit_id",
            "reason",
        }:
            contract_errors.append(
                "Stage A admission unit verdict contract is invalid"
            )
            continue
        if not str(row.get("reason") or "").strip():
            contract_errors.append(
                "Stage A admission unit verdict has no reason"
            )
        verdict = str(row.get("verdict") or "")
        if verdict not in {"complete", "redundant", "unresolved"}:
            contract_errors.append(
                "Stage A admission unit verdict is invalid"
            )
        elif verdict == "redundant":
            redundant_unit_ids.add(str(row.get("unit_id") or ""))
        elif str(row.get("supports_unit_id") or ""):
            contract_errors.append(
                "Stage A admission non-redundant unit declares support"
            )
        elif (
            verdict == "unresolved"
            and str(row.get("unit_id") or "")
            not in proven_pending_unit_ids
            and unit_operations.get(str(row.get("unit_id") or ""))
            not in {"context", "unresolved"}
        ):
            semantic_errors.append(
                f"Stage A admission found unresolved unit: {row.get('unit_id')}"
            )
    for row in clause_verdicts:
        if set(row) != {
            "clause_id",
            "verdict",
            "omitted_owner_routes",
            "reason",
        }:
            contract_errors.append(
                "Stage A admission clause verdict contract is invalid"
            )
            continue
        verdict = str(row.get("verdict") or "")
        if verdict not in {"complete", "omitted", "unresolved"}:
            contract_errors.append("Stage A admission clause verdict is invalid")
        if not str(row.get("reason") or "").strip():
            contract_errors.append(
                "Stage A admission clause verdict has no reason"
            )
        routes = row.get("omitted_owner_routes")
        if not isinstance(routes, list):
            contract_errors.append(
                "Stage A admission omitted_owner_routes is not a list"
            )
            routes = []
        for route in routes:
            if not isinstance(route, Mapping) or set(route) != _ROUTE_KEYS:
                contract_errors.append(
                    "Stage A admission omitted route contract is invalid"
                )
                continue
            identity = (
                str(route.get("owner") or ""),
                str(route.get("group") or ""),
            )
            if identity not in valid_routes:
                contract_errors.append(
                    f"Stage A admission returned an invalid omitted route: "
                    f"{identity[0]}/{identity[1]}"
                )
        clause_id = str(row.get("clause_id") or "")
        clause_has_declared_unresolved = any(
            str(unit.get("clause_id") or "") == clause_id
            and str(unit.get("operation") or "") == "unresolved"
            for unit in partition
        )
        clause_actionable_unit_ids = {
            str(unit.get("unit_id") or "")
            for unit in partition
            if str(unit.get("clause_id") or "") == clause_id
            and str(unit.get("operation") or "")
            not in {"context", "unresolved"}
        }
        clause_is_fully_proven_pending = bool(clause_actionable_unit_ids) and (
            clause_actionable_unit_ids.issubset(proven_pending_unit_ids)
        )
        if verdict == "omitted":
            semantic_errors.append(
                f"Stage A admission found {verdict or 'invalid'} demand in "
                f"{row.get('clause_id')}"
            )
        elif verdict == "unresolved" and (
            (
                not clause_has_declared_unresolved
                and not clause_is_fully_proven_pending
            )
            or routes
        ):
            semantic_errors.append(
                "Stage A admission returned an unresolved clause without an "
                f"explicit unresolved DemandAtom: {row.get('clause_id')}"
            )
        elif routes:
            contract_errors.append(
                f"Stage A complete clause declares omitted routes: "
                f"{row.get('clause_id')}"
            )
    if not str(document.get("reason") or "").strip():
        contract_errors.append("Stage A admission has no reason")
    complete_clauses = {
        str(row.get("clause_id") or "")
        for row in clause_verdicts
        if str(row.get("verdict") or "") == "complete"
    }
    unresolved_non_context_units = {
        str(row.get("unit_id") or "")
        for row in unit_verdicts
        if (
            str(row.get("verdict") or "") == "unresolved"
            and unit_operations.get(str(row.get("unit_id") or "")) != "context"
        )
    }
    unit_clause = {
        str(unit["unit_id"]): str(unit.get("clause_id") or "")
        for unit in partition
    }
    reviewer_complete_units = {
        str(row.get("unit_id") or "")
        for row in unit_verdicts
        if str(row.get("verdict") or "") == "complete"
    }
    for unit_id, operation in unit_operations.items():
        if operation == "unresolved" and unit_id in reviewer_complete_units:
            contract_errors.append(
                "Stage A admission is internally inconsistent: unresolved "
                f"source operation was declared complete: {unit_id}"
            )
    for unit_id in unresolved_non_context_units:
        if unit_clause.get(unit_id, "") in complete_clauses:
            contract_errors.append(
                "Stage A admission is internally inconsistent: non-context "
                f"unresolved unit belongs to a complete clause: {unit_id}"
            )
    complete_unit_ids = {
        str(row.get("unit_id") or "")
        for row in unit_verdicts
        if str(row.get("verdict") or "") == "complete"
    }
    support_by_redundant_unit = {
        str(row.get("unit_id") or ""): str(row.get("supports_unit_id") or "")
        for row in unit_verdicts
        if str(row.get("verdict") or "") == "redundant"
    }
    for unit_id in redundant_unit_ids:
        clause_id = unit_clause.get(unit_id, "")
        support_unit_id = support_by_redundant_unit.get(unit_id, "")
        if (
            clause_id not in complete_clauses
            or support_unit_id not in complete_unit_ids
            or support_unit_id == unit_id
        ):
            contract_errors.append(
                f"Stage A admission redundant unit has invalid support: {unit_id}"
            )
        if (
            unit_id in rejected_pending_unit_ids
            and (
                unit_operations.get(unit_id) != "pending_answer"
                or support_unit_id not in proven_pending_unit_ids
                or unit_operations.get(support_unit_id) != "pending_answer"
            )
        ):
            contract_errors.append(
                "Stage A rejected pending unit is not redundant to a proven "
                f"pending unit: {unit_id}"
            )
    return (
        tuple(dict.fromkeys(contract_errors)),
        tuple(dict.fromkeys(semantic_errors)),
        frozenset(redundant_unit_ids) if not contract_errors else frozenset(),
    )


def _owner_requests(
    partition: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str, ...]]:
    requests: dict[str, list[str]] = defaultdict(list)
    for unit in partition:
        unit_id = str(unit["unit_id"])
        for route in unit.get("owner_routes") or []:
            owner = str(route["owner"])
            if unit_id not in requests[owner]:
                requests[owner].append(unit_id)
    return {owner: tuple(unit_ids) for owner, unit_ids in requests.items()}


def _expand_partition_routes(
    partition: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Give every owner/group route its own typed unit.

    Stage A may identify one indivisible source excerpt that supplies several
    values. Stage B and admission must not share ownership of one unit, so the
    Harness expands only the routing identity while preserving the exact same
    source span and semantic order.
    """

    output: list[dict[str, Any]] = []
    used_ids = {
        str(unit.get("unit_id") or "")
        for unit in partition
    }
    route_sequence = 0
    for raw in partition:
        unit = dict(raw)
        routes = [
            dict(route)
            for route in unit.get("owner_routes") or []
            if isinstance(route, Mapping)
        ]
        if len(routes) <= 1:
            output.append(unit)
            continue
        base_id = str(unit.get("unit_id") or "unit")
        for route in routes:
            while True:
                route_sequence += 1
                route_id = f"__harness_route_{route_sequence}"
                if route_id not in used_ids:
                    used_ids.add(route_id)
                    break
            expanded = dict(unit)
            expanded["unit_id"] = route_id
            expanded["parent_unit_id"] = base_id
            expanded["owner_routes"] = [route]
            output.append(expanded)
    return output


def _stage_b_prompt(owner: str) -> str:
    return (
        f"You are the Stage B command compiler for the AnyChain owner {owner}. "
        "Return one strict JSON object with actions, bindings, and reason only. "
        "Use only actions and arguments in owner_action_schema. Do not emit an action "
        "owned by another domain or a different group than the unit's owner_route. Bind "
        "every supplied unit_id exactly once. A binding "
        "is {unit_id,action_indexes,disposition,reason}; disposition is action or unresolved. "
        "Indexes are zero-based in this owner-local actions list. Preserve source order and "
        "exact source provenance. For a structured DemandAtom, semantic_source contains only "
        "the source_path field and value that this unit owns, and semantic_value is the "
        "parser-proven typed value at that path. Interpret that scoped demand instead of "
        "sibling fields in the complete source_text. source_text remains the immutable "
        "SourceClause used for exact source_evidence quotes. For a pending_answer DemandAtom "
        "with registered_intake, emit exactly registered_intake.action_type, copy every "
        "registered_intake.fixed_arguments entry, and, when value_argument is non-empty, "
        "bind semantic_value to that argument. This metadata is Harness-authoritative; do "
        "not substitute an entry intake, infer another command, or omit the registered value. "
        "Use semantic_source as the exact source_evidence when the registered action declares "
        "that argument. For a pending_answer DemandAtom "
        "whose pending_question allows manual input and has no options, emit answer equal to "
        "semantic_value and never emit selected_value when semantic_value is present. For a "
        "prose pending_answer without semantic_value, extract exactly one shortest literal "
        "value from semantic_source that satisfies the pending question's validation contract; "
        "use that literal as both answer and source_evidence. If there is no unique valid "
        "literal, keep the unit unresolved. Never use the complete labelled sentence as a "
        "scalar answer and never consume sibling demands. "
        "Every action object MUST be flat: place type and every "
        "allowed argument in the same object and NEVER emit an arguments object. For example, "
        "{\"type\":\"declared_type\",\"declared_argument\":\"value\"}, not "
        "{\"type\":\"declared_type\",\"arguments\":{...}}. Copy only keys explicitly listed "
        "in that action's allowed_arguments. When allowed_arguments is empty, the complete "
        "valid action object is {\"type\":\"declared_type\"}; source provenance remains in "
        "the binding and semantic unit and is not an action argument. Follow owner_action_schema "
        "exactly: never add an undeclared argument, and emit source_evidence only when that action "
        "declares it. Any emitted source_evidence must be one exact substring of a supplied "
        "bound semantic unit's source_text or Harness-bound resolution_evidence, never a "
        "paraphrase. When one action is bound to multiple adjacent units and its arguments "
        "require their combined evidence, source_evidence may instead equal the exact ordered "
        "concatenation of every unit bound to that action. It must not omit or reorder a bound "
        "unit or include text owned by another action. "
        "resolution_evidence is present only when the user clarified that exact source DemandAtom; "
        "it supplies the resolved value while source_text preserves the original request. "
        "semantic_draft_resolution_disposition, when present, is Harness-owned; "
        "compile the normal bound resolution action and do not author, change, or "
        "reinterpret that disposition. The Harness binds it after validation. "
        "For a concrete closed-enum selection, use the shortest exact "
        "affirmative source span that semantically selects that value; exclude contrast text and "
        "rejected alternatives from source_evidence. Configuration values remain proposals until the owning workflow "
        "validates and confirms them. Do not copy a value from owner_state "
        "unless the source unit explicitly supplies or confirms it. When the source explicitly "
        "asks to keep, reuse, or reconfirm one concrete value already present in owner_state, "
        "emit the same owning selection action as an idempotent proposal; never mark it "
        "unresolved merely because applying it would not change state. A promise to supply a "
        "value later does not supply that value and must not be compiled as a mutation. "
        "Do not add inferred "
        "identity, existence, protocol, canonical-name, or evidence-summary arguments to a "
        "selection action; downstream domain validation owns those facts. "
        "For arguments declared in open_identity_grounding_arguments, emit a concrete "
        "identity only when the source supplies a named identity. Pronouns, deictic "
        "references, generic categories, quantifiers, placeholders, and requests for "
        "some other or another category member are missing values; emit the unique "
        "registered incomplete_mutation_intake for that routed group instead. "
        "Questions and "
        "explanations are read-only actions. When a read-only action declares allowed_topics "
        "and topic_purposes, select the one topic whose registered purpose represents the "
        "complete question and emit that exact topic identifier. Do not emit a generic topic "
        "when a more specific registered topic applies. A concrete request owned by another group has "
        "no valid action in this owner schema: mark that binding unresolved so Stage A can be "
        "corrected, rather than coercing it into a superficially similar action. Ambiguous or "
        "incomplete demands remain unresolved instead of being guessed, except when the unit "
        "is an explicit initial-selection or replacement demand for its routed group and "
        "owner_action_schema declares an action for that same target_group with "
        "incomplete_mutation_intake=true. In that case emit that registered typed intake "
        "action instead of guessing a concrete value or marking the unit unresolved. This "
        "includes closed enums where the source rejects one value but leaves multiple legal "
        "values: exclusion is not a concrete selection, so use the registered intake to ask "
        "the user. When owner_action_schema declares incomplete_read_intake=true for the "
        "unit's operation, emit that read-only action without invented payload; the owner "
        "will collect the missing payload through its typed lifecycle. Never synthesize an "
        "intake that is absent from owner_action_schema. "
        "When a semantic unit has operation=pending_answer, compare its complete exact source "
        "meaning with owner_state.pending_question and every declared option. If exactly one "
        "option is semantically selected, emit the coordinator-owned answer_pending action with "
        "selected_value equal to that option's exact value, omit answer, and include exact "
        "source_evidence from that unit. For valid manual input, emit answer and omit "
        "selected_value. When pending_question.options is empty, selected_value is always "
        "invalid: preserve the complete manual or evidence contribution in answer, or emit "
        "the pending contract's declared manual_action when that action belongs to this "
        "owner. Never reinterpret an array, boolean, id, method parameter, or other literal "
        "inside manual evidence as a declared option. The deterministic pending-question coordinator, not "
        "Stage B, resolves the immutable option.action and dispatches it to its registered owner. "
        "Never emit that specialized option.action directly from the coordinator compiler. "
        "The user does not need to repeat an option label or value verbatim; an unambiguous "
        "natural-language paraphrase, rejection of all listed alternatives, or stated "
        "uncertainty may select the corresponding declared option. If zero or several options "
        "fit, keep the unit unresolved. Never invent an option, and never consume an "
        "independent sibling request as rationale for the pending answer. Respect the pending "
        "question's value_domain: an open researched identity cannot consume a value declared "
        "by another registered closed domain."
    )


def _action_spec_applies(
    spec: Any,
    *,
    groups: frozenset[str],
    operations: frozenset[str],
) -> bool:
    if spec.internal_only:
        return "pending_answer" in operations and (
            "pending_answer" in spec.semantic_operations
        )
    return any(
        any(
            action_spec_serves_route(
                spec,
                operation=operation,
                group=group,
            )
            for group in (groups or frozenset({""}))
        )
        for operation in operations
    )


def _action_schema_applies(
    row: Mapping[str, Any],
    *,
    groups: frozenset[str],
    operations: frozenset[str],
) -> bool:
    action_type = str(row.get("type") or "")
    spec = ACTION_BY_TYPE.get(action_type)
    return bool(
        spec is not None
        and _action_spec_applies(
            spec,
            groups=groups,
            operations=operations,
        )
    )


def _action_spec_lifecycle_reachable(
    spec: Any,
    state: Mapping[str, Any],
) -> bool:
    """Expose only actions whose current lifecycle state can receive them."""

    if spec.action_type == "start_evidence_collection":
        pending = state.get("pending_question")
        return bool(
            isinstance(pending, Mapping)
            and str(pending.get("kind") or "") == "evidence"
        )
    if not spec.required_state_path:
        return True
    current: Any = state
    for key in spec.required_state_path:
        current = current.get(key) if isinstance(current, Mapping) else None
    return current in spec.required_state_values


def _allowed_action_types_for_partition(
    partition: Sequence[Mapping[str, Any]],
) -> frozenset[str]:
    """Expose only actions reachable from this immutable Stage A partition."""

    operations = frozenset(
        str(unit.get("operation") or "")
        for unit in partition
        if str(unit.get("operation") or "") not in {"context", "unresolved"}
    )
    groups = frozenset(
        str(route.get("group") or "")
        for unit in partition
        for route in unit.get("owner_routes") or []
        if isinstance(route, Mapping) and str(route.get("group") or "")
    )
    return frozenset(
        spec.action_type
        for spec in ACTION_SPECS
        if _action_spec_applies(
            spec,
            groups=groups,
            operations=operations,
        )
    )


def _stage_b_payload(
    state: AgentGraphState,
    owner: str,
    groups: frozenset[str],
    partition: Sequence[Mapping[str, Any]],
    unit_ids: Sequence[str],
) -> dict[str, Any]:
    selected = []
    for unit in partition:
        if str(unit["unit_id"]) not in unit_ids:
            continue
        semantic_source, semantic_value = _stage_b_semantics_for_unit(
            state,
            unit,
        )
        selected.append({
            **{
                key: unit[key]
                for key in (
                    "unit_id",
                    "clause_id",
                    "source_text",
                    "operation",
                    "owner_routes",
                    "reason",
                )
            },
            **(
                {
                    "resolution_evidence": str(
                        unit["resolution_evidence"]
                    ),
                    "resolution_atom_id": str(
                        unit["resolution_atom_id"]
                    ),
                    "resolution_binding_hash": str(
                        unit["resolution_binding_hash"]
                    ),
                }
                if str(unit.get("resolution_evidence") or "")
                else {}
            ),
            **(
                {"source_path": str(unit["source_path"])}
                if str(unit.get("source_path") or "")
                else {}
            ),
            **(
                {
                    "semantic_source": semantic_source,
                    "semantic_value": semantic_value,
                }
                if str(unit.get("source_path") or "")
                or semantic_value is not None
                else {}
            ),
            **(
                {"registered_intake": dict(unit["registered_intake"])}
                if isinstance(unit.get("registered_intake"), Mapping)
                else {}
            ),
            **(
                {
                    "semantic_draft_resolution_disposition": str(
                        unit["_semantic_draft_resolution_disposition"]
                    )
                }
                if str(
                    unit.get("_semantic_draft_resolution_disposition") or ""
                )
                else {}
            ),
        })
    selected_operations = frozenset(
        str(unit.get("operation") or "")
        for unit in selected
    )
    owner_action_rows = [
        row
        for row in action_schema(owners=frozenset({owner}))
        if _action_schema_applies(
            row,
            groups=groups,
            operations=selected_operations,
        )
        and _action_spec_lifecycle_reachable(
            ACTION_BY_TYPE[str(row["type"])],
            state,
        )
    ]
    owner_action_schema = list({
        str(row["type"]): row for row in owner_action_rows
    }.values())
    return {
        "owner": owner,
        "semantic_units": selected,
        "owner_action_schema": owner_action_schema,
        "owner_group_schema": [
            row
            for row in group_schema()
            if row["name"] in groups
        ],
        "pending_question": dict(state.get("pending_question") or {}),
        "owner_state": owner_workflow_snapshot(state, owner, groups=groups),
    }


def _stage_b_output_token_budget(payload: Mapping[str, Any]) -> int:
    """Bound owner compilation output by its required semantic-unit bindings."""

    unit_count = len(payload.get("semantic_units") or ())
    return min(12000, max(3200, 2200 + (900 * unit_count)))


def _compile_owner_document(
    state: AgentGraphState,
    owner: str,
    groups: frozenset[str],
    partition: Sequence[Mapping[str, Any]],
    unit_ids: Sequence[str],
) -> tuple[dict[str, Any], tuple[str, ...], tuple[int, ...]]:
    prompt = _stage_b_prompt(owner)
    payload = _stage_b_payload(
        state,
        owner,
        groups,
        partition,
        unit_ids,
    )
    provider = provider_from_config()
    expected_groups = {
        str(unit["unit_id"]): frozenset(
            str(route.get("group") or "")
            for route in unit.get("owner_routes") or []
            if isinstance(route, Mapping)
        )
        for unit in partition
        if str(unit["unit_id"]) in unit_ids
    }
    expected_operations = {
        str(unit["unit_id"]): str(unit.get("operation") or "")
        for unit in partition
        if str(unit["unit_id"]) in unit_ids
    }
    expected_sources = {
        str(unit["unit_id"]): tuple(
            dict.fromkeys(
                value
                for value in (
                    _stage_b_semantics_for_unit(state, unit)[0],
                    str(unit.get("resolution_evidence") or ""),
                )
                if value
            )
        )
        for unit in partition
        if str(unit["unit_id"]) in unit_ids
    }
    request_sizes: list[int] = []
    response = ""
    document: dict[str, Any] = {}
    errors: tuple[str, ...] = ()
    semantic_receipts: list[dict[str, Any]] = []
    for attempt in range(2):
        request_payload = payload
        request_prompt = prompt
        if attempt:
            request_prompt = (
                f"{prompt} This is a bounded contract repair. Preserve every source "
                "demand, but do not preserve a concrete semantic value that the "
                "validator reports as ungrounded. For a reported closed-enum grounding "
                "violation, remove the guessed concrete selection and use the registered "
                "incomplete_mutation_intake for the same routed group when one exists. "
                "Correct only the reported JSON, action-schema, ownership, route, "
                "grounding, or binding-contract violations. Do not turn an otherwise "
                "unresolved binding into an action merely because repair was requested."
            )
            request_payload = {
                **payload,
                "contract_repair": {
                    "rejected_document": response,
                    "validator_errors": list(errors),
                },
            }
        request_sizes.append(_wire_size(request_prompt, request_payload))
        response = request_semantic_compilation(
            provider,
            system_prompt=request_prompt,
            request_payload=request_payload,
            max_tokens=_stage_b_output_token_budget(payload),
            reasoning_mode=STRICT_JSON_REASONING_MODE,
        )
        document, errors = _validate_owner_document(
            response,
            owner,
            unit_ids,
            expected_groups=expected_groups,
            expected_operations=expected_operations,
            expected_sources=expected_sources,
            pending_question=dict(state.get("pending_question") or {}),
        )
        if not errors:
            document, disposition_errors = (
                _bind_owner_semantic_draft_dispositions(
                    document,
                    payload,
                )
            )
            errors = disposition_errors
        if not errors and _owner_document_requires_semantic_review(
            payload,
            document,
        ):
            semantic_errors, semantic_sizes, semantic_receipt = (
                _review_owner_document_semantics(
                    provider,
                    payload,
                    document,
                )
            )
            request_sizes.extend(semantic_sizes)
            semantic_receipts.append(semantic_receipt)
            errors = semantic_errors
        if not errors:
            break
    if errors:
        repaired_document = _repair_stage_b_closed_enum_intakes(
            document,
            errors,
            owner=owner,
            expected_sources=expected_sources,
        )
        if repaired_document is not None:
            document, errors = _validate_owner_document(
                json.dumps(
                    repaired_document,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                owner,
                unit_ids,
                expected_groups=expected_groups,
                expected_operations=expected_operations,
                expected_sources=expected_sources,
                pending_question=dict(state.get("pending_question") or {}),
            )
    if semantic_receipts:
        document["semantic_review_receipts"] = semantic_receipts
    return document, errors, tuple(request_sizes)


def _bind_owner_semantic_draft_dispositions(
    document: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Inject only Harness-reviewed draft dispositions into bound actions."""

    expected = {
        str(unit.get("unit_id") or ""): str(
            unit.get("semantic_draft_resolution_disposition") or ""
        )
        for unit in payload.get("semantic_units") or ()
        if isinstance(unit, Mapping)
        and str(unit.get("semantic_draft_resolution_disposition") or "")
    }
    normalized = {
        "actions": [
            dict(action)
            for action in document.get("actions") or ()
            if isinstance(action, Mapping)
        ],
        "bindings": [
            dict(binding)
            for binding in document.get("bindings") or ()
            if isinstance(binding, Mapping)
        ],
    }
    if not expected:
        return normalized, ()
    errors: list[str] = []
    bindings = {
        str(binding.get("unit_id") or ""): binding
        for binding in normalized["bindings"]
    }
    for unit_id, disposition in expected.items():
        if disposition not in {"semantic_value", "background"}:
            errors.append(
                f"semantic draft disposition is invalid for unit: {unit_id}"
            )
            continue
        binding = bindings.get(unit_id) or {}
        indexes = binding.get("action_indexes") or []
        if binding.get("disposition") != "action" or len(indexes) != 1:
            errors.append(
                f"semantic draft disposition lacks one bound action: {unit_id}"
            )
            continue
        index = indexes[0]
        if not isinstance(index, int) or index >= len(normalized["actions"]):
            errors.append(
                f"semantic draft disposition action index is invalid: {unit_id}"
            )
            continue
        action = normalized["actions"][index]
        if str(action.get("type") or "") != "resolve_semantic_draft_atom":
            errors.append(
                f"semantic draft disposition targets wrong action: {unit_id}"
            )
            continue
        authored = str(action.get("resolution_disposition") or "")
        if authored:
            errors.append(
                f"model-authored semantic draft disposition is forbidden: {unit_id}"
            )
            continue
        action["resolution_disposition"] = disposition
        try:
            normalized["actions"][index] = validate_action_contract(action)
        except ValueError as exc:
            errors.append(
                f"semantic draft disposition action is invalid: {unit_id}: {exc}"
            )
    return normalized, tuple(dict.fromkeys(errors))


def _repair_stage_b_closed_enum_intakes(
    document: Mapping[str, Any],
    errors: Sequence[str],
    *,
    owner: str,
    expected_sources: Mapping[str, Sequence[str]],
) -> dict[str, Any] | None:
    """Lower only exhausted, ungrounded enum guesses to typed intake.

    The semantic partition already proves that each bound unit is a present
    mutation request. It does not authorize a concrete enum value that the
    source names only as a rejected competitor. After bounded model repair is
    exhausted, the registry's unique intake preserves that request without
    discarding independently valid sibling actions or guessing an alternative.
    """

    prefix = "Stage B closed-enum grounding quote names only competing values:"
    if not errors or any(not str(error).startswith(prefix) for error in errors):
        return None
    payload = json.loads(json.dumps(document, ensure_ascii=False, sort_keys=True))
    actions = payload.get("actions")
    bindings = payload.get("bindings")
    if not isinstance(actions, list) or not isinstance(bindings, list):
        return None
    units_by_action: dict[int, set[str]] = defaultdict(set)
    for binding in bindings:
        if not isinstance(binding, Mapping):
            return None
        unit_id = str(binding.get("unit_id") or "")
        for index in binding.get("action_indexes") or ():
            if isinstance(index, int) and not isinstance(index, bool):
                units_by_action[index].add(unit_id)
    changed = False
    for index, raw_action in enumerate(actions):
        if not isinstance(raw_action, Mapping):
            continue
        action = dict(raw_action)
        spec = ACTION_BY_TYPE.get(str(action.get("type") or ""))
        owned_units = units_by_action.get(index, set())
        if spec is None or spec.owner != owner or len(owned_units) != 1:
            continue
        unit_id = next(iter(owned_units))
        sources = tuple(
            str(source)
            for source in expected_sources.get(unit_id, ())
            if str(source)
        )
        evidence = str(action.get("source_evidence") or "")
        quote = next(
            (source for source in sources if evidence and evidence in source),
            sources[0] if sources else "",
        )
        violates_grounding = any(
            isinstance(
                (ACTION_ARGUMENT_SCHEMAS.get(argument) or {}).get("enum"),
                list,
            )
            and closed_enum_quote_names_only_competing_values(
                exact_value=action.get(argument),
                enum_values=(ACTION_ARGUMENT_SCHEMAS[argument]["enum"]),
                quote=quote,
            )
            for argument in semantic_grounding_arguments(action)
        )
        if not violates_grounding:
            continue
        intake_spec = _unique_incomplete_mutation_intake(
            owner=owner,
            target_group=spec.target_group,
        )
        if intake_spec is None or not quote:
            return None
        replacement = {"type": intake_spec.action_type}
        if "source_evidence" in intake_spec.allowed_arguments:
            replacement["source_evidence"] = quote
        try:
            actions[index] = validate_action_contract(replacement)
        except ValueError:
            return None
        changed = True
    return payload if changed else None


def _owner_document_requires_semantic_review(
    payload: Mapping[str, Any],
    document: Mapping[str, Any],
) -> bool:
    """Review an ambiguous grounded selection that syntax cannot prove."""

    if len(payload.get("owner_action_schema") or []) <= 1:
        return False
    units = {
        str(unit.get("unit_id") or ""): tuple(
            value
            for value in (
                str(unit.get("source_text") or ""),
                str(unit.get("resolution_evidence") or ""),
            )
            if value
        )
        for unit in payload.get("semantic_units") or []
        if isinstance(unit, Mapping)
    }
    sources_by_action: dict[int, list[str]] = defaultdict(list)
    for binding in document.get("bindings") or []:
        if not isinstance(binding, Mapping):
            continue
        unit_sources = units.get(str(binding.get("unit_id") or ""), ())
        for index in binding.get("action_indexes") or []:
            if isinstance(index, int):
                sources_by_action[index].extend(unit_sources)
    for index, action in enumerate(document.get("actions") or []):
        if not isinstance(action, Mapping):
            continue
        sources = sources_by_action.get(index, [])
        for value in semantic_grounding_values(action):
            if not any(
                _source_contains_semantic_grounding_value(value, source)
                for source in sources
            ):
                return True
    return False


def _source_contains_semantic_grounding_value(value: Any, source: str) -> bool:
    """Match one registry-projected value against its bound source evidence."""

    if isinstance(value, (Mapping, list)):
        try:
            return json.loads(source) == value
        except (json.JSONDecodeError, TypeError):
            return False
    text = str(value).strip()
    return bool(text) and text.casefold() in source.casefold()


def _stage_b_semantic_review_prompt(owner: str) -> str:
    return (
        f"You are the independent Stage B semantic review authority for the "
        f"AnyChain owner {owner}. Review one immutable, structurally valid owner "
        "proposal against only the supplied semantic units and registry-derived "
        "action purposes. You are not a compiler. You must not create, edit, "
        "replace, reorder, bind, or execute actions. Admit an action only when its "
        "exact registered purpose and every concrete argument are authorized by "
        "the source unit bound to that action. A registered action with a similar "
        "operation label is still wrong when its purpose differs. Values from "
        "owner state, defaults, examples, or unrelated units are not source "
        "authorization. Return strict JSON with exactly proposal_hash, "
        "unit_verdicts, action_verdicts, and reason. unit_verdicts must contain one "
        "row per semantic unit in order: {unit_id,verdict:'admit'|'reject',reason}. "
        "action_verdicts must contain one row per candidate action in order: "
        "{action_index,verdict:'admit'|'reject',reason}. Copy proposal_hash exactly."
    )


def _stage_b_review_output_token_budget(
    payload: Mapping[str, Any],
    document: Mapping[str, Any],
) -> int:
    """Bound semantic review output by required unit and action verdict rows."""

    verdict_rows = len(payload.get("semantic_units") or ()) + len(
        document.get("actions") or ()
    )
    return min(12000, max(1800, 1400 + (600 * verdict_rows)))


def _stage_b_semantic_review_projection(
    payload: Mapping[str, Any],
    document: Mapping[str, Any],
) -> dict[str, Any]:
    """Project typed evidence without model-authored explanatory rationales."""

    semantic_units = [
        {
            key: value
            for key, value in dict(unit).items()
            if key != "reason"
        }
        for unit in payload.get("semantic_units") or []
        if isinstance(unit, Mapping)
    ]
    bindings = [
        {
            key: value
            for key, value in dict(binding).items()
            if key != "reason"
        }
        for binding in document.get("bindings") or []
        if isinstance(binding, Mapping)
    ]
    return {
        "semantic_units": semantic_units,
        "candidate": {
            "actions": [
                dict(action)
                for action in document.get("actions") or []
                if isinstance(action, Mapping)
            ],
            "bindings": bindings,
        },
    }


def _review_owner_document_semantics(
    provider: Any,
    payload: Mapping[str, Any],
    document: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[int, ...], dict[str, Any]]:
    owner = str(payload.get("owner") or "")
    proposal_hash = hashlib.sha256(
        json.dumps(
            dict(document),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    projection = _stage_b_semantic_review_projection(payload, document)
    review_payload = {
        "owner": owner,
        "semantic_units": projection["semantic_units"],
        "owner_action_schema": [
            dict(row) for row in payload.get("owner_action_schema") or []
        ],
        "candidate": projection["candidate"],
        "proposal_hash": proposal_hash,
    }
    prompt = _stage_b_semantic_review_prompt(owner)
    request_size = _wire_size(prompt, review_payload)
    review_hashes: list[str] = []
    member_validity: list[bool] = []
    member_errors: list[tuple[str, ...]] = []
    responses = run_independent_llm_tasks(tuple(
        lambda: request_semantic_compilation(
            provider,
            system_prompt=prompt,
            request_payload=review_payload,
            max_tokens=_stage_b_review_output_token_budget(payload, document),
            reasoning_mode=STRICT_JSON_REASONING_MODE,
        )
        for _member_index in range(3)
    ))
    for response in responses:
        review_hashes.append(
            hashlib.sha256(response.encode("utf-8")).hexdigest()
        )
        errors = _validate_stage_b_semantic_review_response(
            response,
            owner=owner,
            proposal_hash=proposal_hash,
            units=payload.get("semantic_units") or (),
            actions=document.get("actions") or (),
        )
        member_errors.append(errors)
        member_validity.append(not errors)
    quorum = sum(member_validity) >= 2
    errors = () if quorum else tuple(dict.fromkeys(
        error
        for member in member_errors
        for error in member
    ))
    if not quorum:
        errors = (
            *errors,
            f"Stage B {owner} semantic review quorum was not reached",
        )
    request_sizes = (request_size, request_size, request_size)
    receipt = {
        "proposal_hash": proposal_hash,
        "review_hashes": review_hashes,
        "member_validity": member_validity,
        "request_count": 3,
        "request_sizes": list(request_sizes),
        "valid": quorum,
    }
    return tuple(dict.fromkeys(errors)), request_sizes, receipt


def _validate_stage_b_semantic_review_response(
    response: str,
    *,
    owner: str,
    proposal_hash: str,
    units: Sequence[Any],
    actions: Sequence[Any],
) -> tuple[str, ...]:
    errors: list[str] = []
    try:
        verdict = json.loads(response)
    except json.JSONDecodeError:
        verdict = {}
        errors.append(f"Stage B {owner} semantic review did not return strict JSON")
    if not isinstance(verdict, Mapping) or set(verdict) != {
        "proposal_hash",
        "unit_verdicts",
        "action_verdicts",
        "reason",
    }:
        errors.append(f"Stage B {owner} semantic review contract is invalid")
        verdict = {}
    if str(verdict.get("proposal_hash") or "") != proposal_hash:
        errors.append(f"Stage B {owner} semantic review changed proposal hash")
    unit_verdicts = verdict.get("unit_verdicts")
    expected_unit_ids = [
        str(unit.get("unit_id") or "")
        for unit in units
        if isinstance(unit, Mapping)
    ]
    if not isinstance(unit_verdicts, list) or [
        str(row.get("unit_id") or "") if isinstance(row, Mapping) else ""
        for row in unit_verdicts
    ] != expected_unit_ids:
        errors.append(f"Stage B {owner} semantic review unit cardinality is invalid")
        unit_verdicts = []
    action_verdicts = verdict.get("action_verdicts")
    if not isinstance(action_verdicts, list) or [
        row.get("action_index") if isinstance(row, Mapping) else None
        for row in action_verdicts
    ] != list(range(len(actions))):
        errors.append(f"Stage B {owner} semantic review action cardinality is invalid")
        action_verdicts = []
    expected_row_fields = {
        "unit": {"unit_id", "verdict", "reason"},
        "action": {"action_index", "verdict", "reason"},
    }
    for kind, rows in (("unit", unit_verdicts), ("action", action_verdicts)):
        for row in rows:
            if (
                not isinstance(row, Mapping)
                or set(row) != expected_row_fields[kind]
                or str(row.get("verdict") or "") not in {"admit", "reject"}
                or not str(row.get("reason") or "").strip()
            ):
                errors.append(f"Stage B {owner} semantic review {kind} verdict is invalid")
                continue
            if str(row.get("verdict")) == "reject":
                identity = row.get("unit_id", row.get("action_index"))
                errors.append(
                    f"Stage B {owner} semantic review rejected {kind} {identity}: "
                    f"{str(row.get('reason') or '').strip()}"
                )
    if not str(verdict.get("reason") or "").strip():
        errors.append(f"Stage B {owner} semantic review has no reason")
    return tuple(dict.fromkeys(errors))


def _validate_owner_document(
    text: str,
    owner: str,
    unit_ids: Sequence[str],
    *,
    expected_groups: Mapping[str, frozenset[str]] | None = None,
    expected_operations: Mapping[str, str] | None = None,
    expected_sources: Mapping[str, Sequence[str]] | None = None,
    pending_question: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}, (f"Stage B {owner} did not return strict JSON",)
    if not isinstance(payload, dict) or set(payload) - {"actions", "bindings", "reason"}:
        return {}, (f"Stage B {owner} returned undeclared top-level fields",)
    actions = payload.get("actions")
    bindings = payload.get("bindings")
    errors: list[str] = []
    if not isinstance(actions, list):
        errors.append(f"Stage B {owner} actions is not a list")
        actions = []
    if not isinstance(bindings, list):
        errors.append(f"Stage B {owner} bindings is not a list")
        bindings = []
    actions, bindings = _lower_registered_incomplete_read_intakes(
        actions,
        bindings,
        owner=owner,
        expected_groups=expected_groups or {},
        expected_operations=expected_operations or {},
    )
    actions, bindings = _lower_registered_incomplete_mutation_intakes(
        actions,
        bindings,
        owner=owner,
        expected_groups=expected_groups or {},
        expected_operations=expected_operations or {},
        expected_sources=expected_sources or {},
    )
    actions = _bind_registered_intake_source_evidence(
        actions,
        bindings,
        expected_sources=expected_sources or {},
    )
    normalized_actions: list[dict[str, Any]] = []
    normalized_action_slots: list[dict[str, Any] | None] = [None] * len(actions)
    for index, action in enumerate(actions):
        if not isinstance(action, Mapping):
            errors.append(f"Stage B {owner} action {index} is not an object")
            continue
        try:
            normalized = validate_action_contract(
                lower_empty_entry_action_to_registered_intake(action)
            )
        except ValueError as exc:
            errors.append(f"Stage B {owner} action {index} is invalid: {exc}")
            continue
        if (
            str(normalized.get("type") or "") == "answer_pending"
            and "selected_value" in normalized
            and not pending_option_value_exists(
                normalized.get("selected_value"),
                dict(pending_question or {}),
            )
        ):
            errors.append(
                f"Stage B {owner} action {index} selects a value absent from "
                "the active pending options"
            )
            continue
        spec = ACTION_BY_TYPE.get(str(normalized.get("type") or ""))
        if spec is None or spec.owner != owner:
            errors.append(f"Stage B {owner} emitted an unowned action at {index}")
            continue
        if spec.typed_option_only:
            errors.append(
                f"Stage B {owner} emitted a typed-option-only action at {index}"
            )
            continue
        normalized_action_slots[index] = normalized
        normalized_actions.append(normalized)
    seen_units: list[str] = []
    for raw in bindings:
        if not isinstance(raw, Mapping) or set(raw) != _BINDING_KEYS:
            errors.append(f"Stage B {owner} binding contract is invalid")
            continue
        unit_id = str(raw.get("unit_id") or "")
        seen_units.append(unit_id)
        disposition = str(raw.get("disposition") or "")
        indexes = raw.get("action_indexes")
        if disposition not in {"action", "unresolved"}:
            errors.append(f"Stage B {owner} binding disposition is invalid: {unit_id}")
        if not isinstance(indexes, list) or any(
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index >= len(actions)
            for index in indexes
        ):
            errors.append(f"Stage B {owner} binding indexes are invalid: {unit_id}")
        if disposition == "action" and not indexes:
            errors.append(f"Stage B {owner} actionable binding has no action: {unit_id}")
        if disposition == "unresolved" and indexes:
            errors.append(f"Stage B {owner} unresolved binding has actions: {unit_id}")
        groups = (expected_groups or {}).get(unit_id, frozenset())
        operation = str((expected_operations or {}).get(unit_id) or "")
        for index in indexes if isinstance(indexes, list) else []:
            if not isinstance(index, int) or index >= len(normalized_action_slots):
                continue
            normalized_action = normalized_action_slots[index]
            if normalized_action is None:
                continue
            spec = ACTION_BY_TYPE[str(normalized_action["type"])]
            if not _action_spec_applies(
                spec,
                groups=groups,
                operations=frozenset({operation}),
            ):
                errors.append(
                    f"Stage B {owner} action {spec.action_type} is outside "
                    f"unit route {unit_id}/{operation}/{sorted(groups)}"
                )
            sources = tuple(
                str(value)
                for value in (expected_sources or {}).get(unit_id, ())
                if str(value)
            )
            evidence = str(normalized_action.get("source_evidence") or "")
            quote = (
                evidence
                if evidence and any(evidence in source for source in sources)
                else (sources[0] if sources else "")
            )
            for argument in semantic_grounding_arguments(normalized_action):
                argument_schema = ACTION_ARGUMENT_SCHEMAS.get(argument) or {}
                enum_values = argument_schema.get("enum")
                if not isinstance(enum_values, list):
                    continue
                if closed_enum_quote_names_only_competing_values(
                    exact_value=normalized_action.get(argument),
                    enum_values=enum_values,
                    quote=quote,
                ):
                    errors.append(
                        "Stage B closed-enum grounding quote names only "
                        f"competing values: {unit_id}/{spec.action_type}/{argument}"
                    )
    if seen_units != list(unit_ids):
        errors.append(f"Stage B {owner} binding order or cardinality mismatch")
    return {
        "actions": normalized_actions,
        "bindings": [
            dict(binding) for binding in bindings if isinstance(binding, Mapping)
        ],
    }, tuple(dict.fromkeys(errors))


def _bind_registered_intake_source_evidence(
    actions: Sequence[Any],
    bindings: Sequence[Any],
    *,
    expected_sources: Mapping[str, Sequence[str]],
) -> list[Any]:
    """Bind one routed source unit to an incomplete mutation intake.

    These registry-owned actions exist only to collect a value omitted by the
    current source unit. Their authority to cross a pending barrier therefore
    comes from the exact bound source, not from a model-authored paraphrase.
    Ambiguous multi-unit ownership remains invalid and is never guessed.
    """

    normalized = [
        dict(action) if isinstance(action, Mapping) else action
        for action in actions
    ]
    sources_by_action: dict[int, set[str]] = {}
    for binding in bindings:
        if (
            not isinstance(binding, Mapping)
            or binding.get("disposition") != "action"
            or not isinstance(binding.get("action_indexes"), list)
        ):
            continue
        sources = {
            str(source)
            for source in expected_sources.get(
                str(binding.get("unit_id") or ""),
                (),
            )
            if str(source)
        }
        for index in binding.get("action_indexes") or ():
            if isinstance(index, int) and not isinstance(index, bool):
                sources_by_action.setdefault(index, set()).update(sources)
    for index, action in enumerate(normalized):
        if not isinstance(action, dict):
            continue
        spec = ACTION_BY_TYPE.get(str(action.get("type") or ""))
        bound_sources = sources_by_action.get(index, set())
        if spec is None or len(bound_sources) != 1:
            continue
        if (
            spec.incomplete_mutation_intake
            and "source_evidence" in spec.required_arguments
            and not action.get("source_evidence")
        ) or spec.exact_source_value_arguments:
            action["source_evidence"] = next(iter(bound_sources))
    return normalized


def _lower_registered_incomplete_read_intakes(
    actions: Sequence[Any],
    bindings: Sequence[Any],
    *,
    owner: str,
    expected_groups: Mapping[str, frozenset[str]],
    expected_operations: Mapping[str, str],
) -> tuple[list[Any], list[Any]]:
    """Lower an unresolved routed read intake through its unique registry entry."""

    lowered_actions = [
        dict(action) if isinstance(action, Mapping) else action
        for action in actions
    ]
    lowered_bindings: list[Any] = []
    for raw in bindings:
        if not isinstance(raw, Mapping):
            lowered_bindings.append(raw)
            continue
        binding = dict(raw)
        unit_id = str(binding.get("unit_id") or "")
        operation = str(expected_operations.get(unit_id) or "")
        groups = expected_groups.get(unit_id, frozenset())
        candidates = [
            spec
            for spec in ACTION_SPECS
            if (
                spec.owner == owner
                and spec.incomplete_read_intake
                and _action_spec_applies(
                    spec,
                    groups=groups,
                    operations=frozenset({operation}),
                )
            )
        ]
        if (
            binding.get("disposition") == "unresolved"
            and binding.get("action_indexes") == []
            and len(candidates) == 1
        ):
            binding["disposition"] = "action"
            binding["action_indexes"] = [len(lowered_actions)]
            lowered_actions.append({"type": candidates[0].action_type})
        lowered_bindings.append(binding)
    return lowered_actions, lowered_bindings


def _lower_registered_incomplete_mutation_intakes(
    actions: Sequence[Any],
    bindings: Sequence[Any],
    *,
    owner: str,
    expected_groups: Mapping[str, frozenset[str]],
    expected_operations: Mapping[str, str],
    expected_sources: Mapping[str, Sequence[str]],
) -> tuple[list[Any], list[Any]]:
    """Lower a routed value-less mutation through its unique typed intake."""

    lowered_actions = [
        dict(action) if isinstance(action, Mapping) else action
        for action in actions
    ]
    lowered_bindings: list[Any] = []
    for raw in bindings:
        if not isinstance(raw, Mapping):
            lowered_bindings.append(raw)
            continue
        binding = dict(raw)
        unit_id = str(binding.get("unit_id") or "")
        operation = str(expected_operations.get(unit_id) or "")
        groups = expected_groups.get(unit_id, frozenset())
        sources = tuple(
            str(source)
            for source in expected_sources.get(unit_id, ())
            if str(source)
        )
        candidates = [
            spec
            for spec in ACTION_SPECS
            if (
                spec.owner == owner
                and spec.incomplete_mutation_intake
                and set(spec.required_arguments).issubset({"source_evidence"})
                and _action_spec_applies(
                    spec,
                    groups=groups,
                    operations=frozenset({operation}),
                )
            )
        ]
        if (
            binding.get("disposition") == "unresolved"
            and binding.get("action_indexes") == []
            and len(candidates) == 1
            and sources
        ):
            action = {"type": candidates[0].action_type}
            if "source_evidence" in candidates[0].allowed_arguments:
                action["source_evidence"] = sources[0]
            binding["disposition"] = "action"
            binding["action_indexes"] = [len(lowered_actions)]
            lowered_actions.append(action)
        lowered_bindings.append(binding)
    return lowered_actions, lowered_bindings


def _merge_owner_documents(
    source_partition: Sequence[Mapping[str, Any]],
    routed_partition: Sequence[Mapping[str, Any]],
    documents: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    unit_order = {
        str(unit["unit_id"]): index for index, unit in enumerate(routed_partition)
    }
    action_rows: list[tuple[int, int, str, int, dict[str, Any]]] = []
    for owner, document in documents.items():
        action_units: dict[int, list[str]] = defaultdict(list)
        for binding in document.get("bindings") or []:
            for action_index in binding.get("action_indexes") or []:
                action_units[int(action_index)].append(str(binding["unit_id"]))
        for local_index, action in enumerate(document.get("actions") or []):
            owned_units = action_units.get(local_index, [])
            first_unit = min(
                (unit_order[unit_id] for unit_id in owned_units),
                default=10**9,
            )
            action_rows.append(
                (first_unit, local_index, owner, local_index, dict(action))
            )
    action_rows.sort(key=lambda row: (row[0], row[1], row[2]))
    actions = [row[4] for row in action_rows]
    global_index = {
        (owner, local_index): index
        for index, (_unit, _order, owner, local_index, _action) in enumerate(
            action_rows
        )
    }
    bindings_by_owner = {
        owner: {
            str(binding["unit_id"]): binding
            for binding in document.get("bindings") or []
        }
        for owner, document in documents.items()
    }
    routed_by_parent: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for unit in routed_partition:
        routed_by_parent[
            str(unit.get("parent_unit_id") or unit["unit_id"])
        ].append(unit)
    semantic_units: list[dict[str, Any]] = []
    semantic_support_unit_ids: list[str] = []
    for unit in source_partition:
        unit_id = str(unit["unit_id"])
        operation = str(unit["operation"])
        indexes: list[int] = []
        unresolved = operation == "unresolved"
        for routed_unit in routed_by_parent.get(unit_id, (unit,)):
            route = next(iter(routed_unit.get("owner_routes") or []), {})
            owner = str(route.get("owner") or "")
            routed_unit_id = str(routed_unit["unit_id"])
            binding = bindings_by_owner.get(owner, {}).get(routed_unit_id)
            if not binding or str(binding.get("disposition") or "") == "unresolved":
                unresolved = True
                continue
            for local_index in binding.get("action_indexes") or []:
                value = global_index[(owner, int(local_index))]
                if value not in indexes:
                    indexes.append(value)
        disposition = (
            "context"
            if operation == "context"
            else "unresolved"
            if unresolved
            else "action"
        )
        if unit.get("_admission_support") is True:
            semantic_support_unit_ids.append(unit_id)
        semantic_units.append({
            "unit_id": unit_id,
            "clause_id": str(unit["clause_id"]),
            "start": unit.get("start"),
            "end": unit.get("end"),
            "source_text": str(unit["source_text"]),
            **(
                {"parent_unit_id": str(unit["parent_unit_id"])}
                if str(unit.get("parent_unit_id") or "")
                else {}
            ),
            "owner_routes": [
                dict(route)
                for route in unit.get("owner_routes") or ()
                if isinstance(route, Mapping)
            ],
            **(
                {
                    "resolution_evidence": str(
                        unit["resolution_evidence"]
                    ),
                    "resolution_atom_id": str(
                        unit["resolution_atom_id"]
                    ),
                    "resolution_binding_hash": str(
                        unit["resolution_binding_hash"]
                    ),
                }
                if str(unit.get("resolution_evidence") or "")
                else {}
            ),
            **(
                {"source_path": str(unit["source_path"])}
                if str(unit.get("source_path") or "")
                else {}
            ),
            "disposition": disposition,
            "action_indexes": indexes if disposition == "action" else [],
            "reason": str(unit.get("reason") or ""),
        })
    return {
        "actions": actions,
        "semantic_units": semantic_units,
        "semantic_support_unit_ids": semantic_support_unit_ids,
        "reason": "hierarchical Stage A/Stage B compilation",
    }


def _project_unresolved_mutation_conflicts(
    candidate: Mapping[str, Any],
    *,
    conflict_actions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Keep ambiguous same-dimension mutations out of the durable queue.

    Stage A coverage review is the semantic authority that may classify an
    earlier mutation as superseded context. If competing decisions survive
    that review, the Harness cannot infer which one wins. Preserve their exact
    source units as unresolved atoms while retaining independent sibling
    actions for semantic-draft finalization.
    """

    output = json.loads(json.dumps(candidate, ensure_ascii=False))
    actions = output.get("actions")
    units = output.get("semantic_units")
    if not isinstance(actions, list) or not isinstance(units, list):
        return output
    if not all(isinstance(action, Mapping) for action in actions):
        return output
    projected_actions = tuple(
        dict(action) for action in (conflict_actions or actions)
    )
    if len(projected_actions) != len(actions):
        return output
    conflicts = mutation_conflict_action_groups(projected_actions)
    conflict_indexes = {
        index
        for _dimension, indexes in conflicts
        for index in indexes
    }
    if not conflict_indexes:
        return output
    retained_actions: list[Any] = []
    old_to_new: dict[int, int] = {}
    for index, action in enumerate(actions):
        if index in conflict_indexes:
            continue
        old_to_new[index] = len(retained_actions)
        retained_actions.append(action)
    conflict_dimension_by_index = {
        index: dimension
        for dimension, indexes in conflicts
        for index in indexes
    }
    unresolved_ids: set[str] = set()
    for unit in units:
        if not isinstance(unit, dict):
            continue
        indexes = [
            index
            for index in unit.get("action_indexes") or ()
            if isinstance(index, int) and not isinstance(index, bool)
        ]
        conflicting = [
            index for index in indexes if index in conflict_indexes
        ]
        remaining = [
            old_to_new[index]
            for index in indexes
            if index in old_to_new
        ]
        if conflicting and not remaining:
            dimensions = sorted({
                conflict_dimension_by_index[index]
                for index in conflicting
            })
            unit["disposition"] = "unresolved"
            unit["action_indexes"] = []
            unit["reason"] = (
                "The same turn contains competing values for registry-owned "
                f"mutation dimension(s) {', '.join(dimensions)}; explicit "
                "supersession did not reach semantic-plan consensus."
            )
            unresolved_ids.add(str(unit.get("unit_id") or ""))
        else:
            unit["action_indexes"] = list(dict.fromkeys(remaining))
    output["actions"] = retained_actions
    output["semantic_support_unit_ids"] = [
        unit_id
        for unit_id in output.get("semantic_support_unit_ids") or ()
        if str(unit_id) not in unresolved_ids
    ]
    return output


def _pending_owner_mutation_projection(
    actions: Sequence[Mapping[str, Any]],
    state: AgentGraphState,
) -> tuple[dict[str, Any], ...]:
    """Project pending answers to their declared owner for conflict checks."""

    pending = dict(state.get("pending_question") or {})
    projected: list[dict[str, Any]] = []
    for raw in actions:
        action = dict(raw)
        if str(action.get("type") or "") == "answer_pending":
            answer = action.get("selected_value", action.get("answer"))
            owner_action = manual_action_for_value(pending, answer)
            owner_spec = ACTION_BY_TYPE.get(
                str(owner_action.get("type") or "")
            )
            if owner_action and owner_spec is not None and owner_spec.mutation_dimension:
                for metadata_key in ("_plan_scope", "_submitted_turn_index"):
                    if metadata_key in action:
                        owner_action[metadata_key] = action[metadata_key]
                action = owner_action
        projected.append(action)
    return tuple(projected)


def _prepare_candidate_with_normalized_conflict_projection(
    candidate: Mapping[str, Any],
    state: AgentGraphState,
    clauses: tuple[TurnClause, ...],
    *,
    pending_choice_unit_ids: frozenset[str],
    semantic_support_unit_ids: frozenset[str],
) -> tuple[dict[str, Any], str, PlanCoverageResult]:
    """Recheck mutation conflicts after pending actions are canonicalized.

    A manual ``answer_pending`` can become its registry-declared owner mutation
    only during candidate preparation.  Conflict detection before that
    normalization cannot see a competing sibling of the same mutation family.
    Project such conflicts to unresolved semantic units, discard identities
    derived from the old action set, and run the normal preparation boundary
    again so durable drafts receive a self-consistent candidate transaction.
    """

    candidate_text, validation = prepare_hierarchical_candidate(
        json.dumps(candidate, ensure_ascii=False, sort_keys=True),
        state,
        clauses,
        pending_choice_unit_ids=pending_choice_unit_ids,
        semantic_support_unit_ids=semantic_support_unit_ids,
    )
    try:
        normalized = json.loads(candidate_text)
    except json.JSONDecodeError:
        return dict(candidate), candidate_text, validation
    if not isinstance(normalized, dict):
        return dict(candidate), candidate_text, validation
    if not validation.valid:
        return normalized, candidate_text, validation
    actions = normalized.get("actions")
    if not isinstance(actions, list) or not all(
        isinstance(action, Mapping) for action in actions
    ):
        return normalized, candidate_text, validation
    conflict_actions = _pending_owner_mutation_projection(
        tuple(dict(action) for action in actions),
        state,
    )
    if not mutation_conflict_action_groups(conflict_actions):
        return normalized, candidate_text, validation

    projected = _strip_candidate_admission_metadata(
        _project_unresolved_mutation_conflicts(
            normalized,
            conflict_actions=conflict_actions,
        )
    )
    projected_support_ids = frozenset(
        str(unit_id)
        for unit_id in projected.get("semantic_support_unit_ids") or ()
        if str(unit_id)
    )
    projected_text, projected_validation = prepare_hierarchical_candidate(
        json.dumps(projected, ensure_ascii=False, sort_keys=True),
        state,
        clauses,
        pending_choice_unit_ids=pending_choice_unit_ids,
        semantic_support_unit_ids=projected_support_ids,
    )
    try:
        prepared_projection = json.loads(projected_text)
    except json.JSONDecodeError:
        prepared_projection = projected
    if not isinstance(prepared_projection, dict):
        prepared_projection = projected
    return prepared_projection, projected_text, projected_validation


def _bind_semantic_draft_resolution_evidence(
    state: AgentGraphState,
    source_partition: Sequence[Mapping[str, Any]],
    compilation_partition: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], tuple[str, ...]]:
    """Bind resolved atom evidence to the exact finalization partition.

    Stage A may interpret a resolution, but it cannot author the trusted
    evidence binding. The Harness verifies the persisted draft and attaches
    each user answer only to the same immutable unit identity.
    """

    draft = dict(state.get("semantic_plan_draft") or {})
    if draft.get("status") != "ready_for_review":
        return (
            [dict(unit) for unit in source_partition],
            [dict(unit) for unit in compilation_partition],
            (),
        )
    try:
        draft = validate_semantic_plan_draft(draft)
    except ValueError as exc:
        return (
            [dict(unit) for unit in source_partition],
            [dict(unit) for unit in compilation_partition],
            (f"semantic draft resolution authority is invalid: {exc}",),
        )
    resolved: dict[str, dict[str, Any]] = {}
    for raw_atom in draft.get("unresolved_atoms") or ():
        if not isinstance(raw_atom, Mapping):
            continue
        atom = dict(raw_atom)
        unit_id = str(atom.get("unit_id") or "")
        resolution = str(atom.get("resolution") or "").strip()
        disposition = str(atom.get("resolution_disposition") or "")
        reference = str(atom.get("resolution_ref") or "")
        if reference:
            resolution = str(
                resolve_secret_reference(
                    reference,
                    draft_id=str(draft.get("draft_id") or ""),
                    atom_id=str(atom.get("atom_id") or ""),
                    expected_hash=str(atom.get("resolution_hash") or ""),
                )
                or ""
            )
        if (
            unit_id
            and resolution
            and disposition in {"semantic_value", "background"}
        ):
            atom["_runtime_resolution"] = resolution
            resolved[unit_id] = atom
    if len(resolved) != len(draft.get("unresolved_atoms") or ()):
        return (
            [dict(unit) for unit in source_partition],
            [dict(unit) for unit in compilation_partition],
            ("semantic draft finalization has unresolved atom evidence",),
        )
    errors: list[str] = []

    def bind(
        partition: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in partition:
            unit = dict(raw)
            unit_id = str(unit.get("unit_id") or "")
            atom = resolved.get(unit_id)
            if atom is not None:
                seen.add(unit_id)
                routes = [
                    dict(route)
                    for route in unit.get("owner_routes") or ()
                    if isinstance(route, Mapping)
                ]
                authoritative_route_changed = (
                    str(atom.get("resolution_disposition") or "")
                    != "background"
                    and
                    str(atom.get("reason") or "") == "missing_user_evidence"
                    and (
                        len(routes) != 1
                        or str(routes[0].get("owner") or "")
                        != str(atom.get("owner") or "")
                        or str(routes[0].get("group") or "")
                        != str(atom.get("group") or "")
                    )
                )
                background_projection_invalid = (
                    str(atom.get("resolution_disposition") or "")
                    == "background"
                    and (
                        str(unit.get("operation") or "") != "context"
                        or bool(routes)
                        or unit.get("_admission_support") is not True
                    )
                )
                if (
                    str(unit.get("clause_id") or "")
                    != str(atom.get("clause_id") or "")
                    or str(unit.get("source_path") or "")
                    != str(atom.get("source_path") or "")
                    or str(unit.get("source_text") or "")
                    != str(atom.get("source_text") or "")
                    or _semantic_source_for_unit(unit)
                    != str(atom.get("semantic_source") or "")
                    or authoritative_route_changed
                    or background_projection_invalid
                ):
                    errors.append(
                        "semantic draft resolution atom provenance changed: "
                        f"{unit_id}"
                    )
                else:
                    binding = semantic_draft_atom_resolution_binding(
                        draft,
                        atom,
                    )
                    unit["resolution_evidence"] = (
                        str(atom.get("resolution_ref") or "")
                        or str(atom.get("resolution") or "")
                    )
                    unit["resolution_atom_id"] = binding["atom_id"]
                    unit["resolution_binding_hash"] = binding["binding_hash"]
                    unit["resolution_disposition"] = str(
                        atom.get("resolution_disposition") or ""
                    )
            output.append(unit)
        missing = set(resolved) - seen
        if missing:
            errors.append(
                "semantic draft resolution atom is absent from final "
                f"partition: {sorted(missing)}"
            )
        return output

    bound_source = bind(source_partition)
    bound_compilation = bind(compilation_partition)
    return bound_source, bound_compilation, tuple(dict.fromkeys(errors))


def _wire_size(prompt: str, payload: Mapping[str, Any]) -> int:
    return len(prompt.encode("utf-8")) + len(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    )


def _with_metrics(
    result: dict[str, Any],
    started: float,
    *,
    request_sizes: Sequence[int],
    stage_a_calls: int,
    stage_b_calls: int,
    admission_calls: int,
    owner_count: int,
    unit_count: int,
) -> dict[str, Any]:
    output = dict(result)
    metrics = HierarchicalPlannerMetrics(
        stage_a_calls=stage_a_calls,
        stage_b_calls=stage_b_calls,
        admission_calls=admission_calls,
        prompt_bytes=sum(request_sizes),
        largest_request_bytes=max(request_sizes, default=0),
        elapsed_ms=int((time.monotonic() - started) * 1000),
        owner_count=owner_count,
        unit_count=unit_count,
    )
    output["planner_metrics"] = metrics.as_dict()
    return output

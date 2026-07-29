"""Pure contract authority for non-admitted semantic plan drafts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any, Mapping, Sequence

from agent.utils.redaction import redact
from agent.workflows.group_registry import GROUP_OWNER, group_registry_contract_hash

from .action_registry import (
    ACTION_BY_TYPE,
    ACTION_METADATA_FIELDS,
    TRUSTED_ACTION_METADATA_FIELDS,
    action_registry_contract_hash,
    resolve_action_target_group,
    validate_action_contract,
)
from .contracts import (
    SemanticDraftCandidate,
    SemanticDraftFinalizationReceipt,
    SemanticPlanDraft,
    SemanticUnresolvedAtom,
)
from .plan_coverage import PlanCoverageResult


SEMANTIC_DRAFT_CONTRACT_VERSION = 2
_TERMINAL_STATUSES = frozenset({"stale", "cancelled"})
_OPEN_STATUSES = frozenset({"awaiting_clarification", "ready_for_review"})
_WORKFLOW_PRECONDITION_ROOTS = (
    "active_group",
    "group_states",
    "invalidated_groups",
    "target_mode",
    "target_mode_change_candidate",
    "workflow_mode",
    "chain_identity",
    "confirmed_config",
    "inferred_config",
    "rpc_mode",
    "workload",
    "custom_rpc",
    "qps_profile",
    "endpoint_evidence",
    "fixture_evidence",
    "sync_observe",
    "observability",
    "advanced_tuning",
    "secondary_handoff",
    "preflight",
    "plan",
    "workflow_goals",
)


def semantic_hash(value: Any) -> str:
    """Return one stable hash for a JSON-compatible semantic value."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_exact_secret_reference(value: Any) -> bool:
    from .contracts import is_secret_reference

    return is_secret_reference(value)


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def workflow_precondition_projection(state: Mapping[str, Any]) -> dict[str, Any]:
    """Project only product facts that can invalidate a compiled draft."""

    return {
        key: redact(state.get(key))
        for key in _WORKFLOW_PRECONDITION_ROOTS
    }


def workflow_precondition_hash(state: Mapping[str, Any]) -> str:
    return semantic_hash(workflow_precondition_projection(state))


def pending_contract_hash(state: Mapping[str, Any]) -> str:
    return semantic_hash(redact(dict(state.get("pending_question") or {})))


def semantic_draft_uses_current_authority(payload: Mapping[str, Any]) -> bool:
    """Return whether a draft is bound to every current contract authority."""

    from .questions import question_contract_authority_hash

    return (
        str(payload.get("registry_hash") or "")
        == group_registry_contract_hash()
        and str(payload.get("action_registry_hash") or "")
        == action_registry_contract_hash()
        and str(payload.get("pending_authority_hash") or "")
        == question_contract_authority_hash()
    )


def semantic_draft_question_binding(
    draft: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the sole signed question binding for the active draft atom."""

    atom_id = str(draft.get("active_atom_id") or "")
    atom = next(
        (
            item
            for item in draft.get("unresolved_atoms") or ()
            if isinstance(item, Mapping)
            and str(item.get("atom_id") or "") == atom_id
        ),
        None,
    )
    if atom is None:
        raise ValueError("semantic draft has no active atom record")
    binding = {
        "draft_id": str(draft.get("draft_id") or ""),
        "revision": int(draft.get("revision") or 0),
        "atom_id": atom_id,
        "sensitive_input": atom.get("sensitive_input") is True,
    }
    if (
        not binding["draft_id"]
        or binding["revision"] < 1
        or not binding["atom_id"]
    ):
        raise ValueError("semantic draft question binding is incomplete")
    return binding


def _atom_identity_payload(
    *,
    unit_id: str,
    clause_id: str,
    source_path: str,
    source_text: str,
    semantic_source: str,
    owner: str,
    group: str,
    sensitive_input: bool,
    reason: str,
) -> dict[str, Any]:
    return {
        "unit_id": unit_id,
        "clause_id": clause_id,
        "source_path": source_path,
        "source_text": source_text,
        "semantic_source": semantic_source,
        "owner": owner,
        "group": group,
        "sensitive_input": sensitive_input,
        "reason": reason,
    }


def semantic_draft_atom_resolution_binding(
    draft: Mapping[str, Any],
    atom: Mapping[str, Any],
) -> dict[str, str]:
    """Bind one resolved value to its exact immutable draft atom."""

    atom_id = str(atom.get("atom_id") or "")
    payload = {
        "draft_id": str(draft.get("draft_id") or ""),
        "atom_id": atom_id,
        "unit_id": str(atom.get("unit_id") or ""),
        "clause_id": str(atom.get("clause_id") or ""),
        "source_path": str(atom.get("source_path") or ""),
        "source_text": str(atom.get("source_text") or ""),
        "semantic_source": str(atom.get("semantic_source") or ""),
        "owner": str(atom.get("owner") or ""),
        "group": str(atom.get("group") or ""),
        "reason": str(atom.get("reason") or ""),
        "resolution_hash": str(atom.get("resolution_hash") or ""),
        "resolution_ref": str(atom.get("resolution_ref") or ""),
    }
    if not payload["draft_id"] or not atom_id or not _is_sha256(
        payload["resolution_hash"]
    ):
        raise ValueError("semantic draft atom resolution binding is incomplete")
    return {
        "atom_id": atom_id,
        "binding_hash": semantic_hash(payload),
    }


def build_semantic_plan_draft(
    state: Mapping[str, Any],
    *,
    original_input: str,
    source_clauses: Sequence[Mapping[str, Any]],
    source_partition: Sequence[Mapping[str, Any]],
    semantic_units: Sequence[Mapping[str, Any]],
    candidate_actions: Sequence[Mapping[str, Any]],
    validation: PlanCoverageResult,
    source_secret_bindings: Sequence[Mapping[str, str]] = (),
) -> dict[str, Any]:
    """Build a durable draft without creating admission or queue receipts."""

    if not validation.unresolved_units:
        raise ValueError("semantic draft requires at least one unresolved atom")
    session_id = str(
        (state.get("session") or {}).get("id")
        or state.get("thread_id")
        or ""
    )
    if not session_id:
        raise ValueError("semantic draft requires a session identity")
    product_head = dict((state.get("turn_context") or {}).get("product_head") or {})
    product_authority_id = str(
        product_head.get("product_authority_id") or ""
    )
    checkpoint_thread_id = str(
        product_head.get("checkpoint_thread_id") or ""
    )
    checkpoint_id = str(product_head.get("checkpoint_id") or "")
    checkpoint_fingerprint = str(
        product_head.get("state_fingerprint") or ""
    )
    checkpoint_revision = int(product_head.get("revision") or 0)
    if (
        not product_authority_id
        or not checkpoint_thread_id
        or not checkpoint_id
        or not _is_sha256(checkpoint_fingerprint)
        or checkpoint_revision < 0
    ):
        raise ValueError("semantic draft requires bound Product Head authority")
    safe_input = str(redact(original_input))
    safe_partition = tuple(
        dict(redact(dict(item))) for item in source_partition
        if isinstance(item, Mapping)
    )
    action_sources: dict[int, list[str]] = {}
    for unit in semantic_units:
        if not isinstance(unit, Mapping):
            continue
        for raw_index in unit.get("action_indexes") or ():
            if isinstance(raw_index, int) and not isinstance(raw_index, bool):
                action_sources.setdefault(raw_index, []).append(
                    str(unit.get("unit_id") or "")
                )
    candidates: list[SemanticDraftCandidate] = []
    for index, raw_action in enumerate(candidate_actions):
        if not isinstance(raw_action, Mapping):
            continue
        action = dict(redact(validate_action_contract(dict(raw_action))))
        action_type = str(action.get("type") or "")
        spec = ACTION_BY_TYPE.get(action_type)
        if spec is None:
            raise ValueError(f"semantic draft contains unknown action: {action_type}")
        group = resolve_action_target_group(action)
        source_atom_ids = tuple(dict.fromkeys(action_sources.get(index, ())))
        identity_payload = {
            "action": action,
            "source_atom_ids": source_atom_ids,
            "owner": spec.owner,
            "group": group,
        }
        candidates.append(SemanticDraftCandidate(
            candidate_id=semantic_hash(identity_payload),
            source_atom_ids=source_atom_ids,
            owner=spec.owner,
            group=group,
            action=action,
            validation_hash=semantic_hash({
                "contract_valid": True,
                "action_type": action_type,
                "owner": spec.owner,
                "group": group,
            }),
            contract_hashes={
                "group_registry": group_registry_contract_hash(),
                "action_registry": action_registry_contract_hash(),
            },
        ))
    source_by_id = {
        str(unit.get("unit_id") or ""): unit
        for unit in safe_partition
        if str(unit.get("unit_id") or "")
    }
    from agent.workflows.group_registry import is_sensitive_question
    from .secret_refs import secret_references_in_value

    source_pending = dict(state.get("pending_question") or {})
    unresolved_atoms: list[SemanticUnresolvedAtom] = []
    for raw in validation.unresolved_units:
        unit_id = str(raw.get("unit_id") or "")
        source_unit = source_by_id.get(unit_id, raw)
        routes = [
            dict(item)
            for item in source_unit.get("owner_routes") or ()
            if isinstance(item, Mapping)
        ]
        owner = str((routes[0] if routes else {}).get("owner") or "orientation")
        group = str((routes[0] if routes else {}).get("group") or "opening")
        if group not in GROUP_OWNER:
            group = "opening"
            owner = GROUP_OWNER[group]
        elif owner not in set(GROUP_OWNER.values()):
            owner = GROUP_OWNER[group]
        semantic_source = str(
            redact(source_unit.get("source_text") or raw.get("source_text") or "")
        )
        unresolved_reason = (
            "semantic_ambiguity"
            if str(source_unit.get("operation") or "") == "unresolved"
            else "missing_user_evidence"
        )
        source_path = str(raw.get("source_path") or "")
        source_field = source_path.rsplit(".", 1)[-1]
        sensitive_input = bool(
            is_sensitive_question(group, "", source_field)
            or (
                source_pending.get("sensitive_input") is True
                and str(source_pending.get("group") or "") == group
            )
            or secret_references_in_value({
                "source_text": raw.get("source_text") or "",
                "semantic_source": semantic_source,
            })
        )
        atom_payload = _atom_identity_payload(
            unit_id=unit_id,
            clause_id=str(raw.get("clause_id") or ""),
            source_path=source_path,
            source_text=str(redact(raw.get("source_text") or "")),
            semantic_source=semantic_source,
            owner=owner,
            group=group,
            sensitive_input=sensitive_input,
            reason=unresolved_reason,
        )
        unresolved_atoms.append(SemanticUnresolvedAtom(
            atom_id=semantic_hash(atom_payload),
            unit_id=unit_id,
            clause_id=atom_payload["clause_id"],
            source_text=atom_payload["source_text"],
            source_path=atom_payload["source_path"],
            semantic_source=semantic_source,
            owner=owner,
            group=group,
            sensitive_input=sensitive_input,
            reason=unresolved_reason,
            reason_detail=str(raw.get("reason") or "semantic meaning is unresolved"),
        ))
    partition_hash = semantic_hash(safe_partition)
    schema_version = int(state.get("schema_version") or 0)
    registry_hash = group_registry_contract_hash()
    action_contract_hash = action_registry_contract_hash()
    source_pending_hash = pending_contract_hash(state)
    from .questions import (
        question_contract_authority_hash,
        validate_pending_question_contract,
    )

    source_pending_authority_hash = question_contract_authority_hash()
    workflow_hash = workflow_precondition_hash(state)
    safe_source_clauses = tuple(
        dict(redact(dict(item)))
        for item in source_clauses
        if isinstance(item, Mapping)
    )
    draft_identity = {
        "session_id": session_id,
        "creation_turn": int(state.get("turn_index") or 0),
        "source_product_authority_id": product_authority_id,
        "source_checkpoint_revision": checkpoint_revision,
        "source_checkpoint_thread_id": checkpoint_thread_id,
        "source_checkpoint_id": checkpoint_id,
        "source_checkpoint_fingerprint": checkpoint_fingerprint,
        "source_schema_version": schema_version,
        "input_hash": semantic_hash(safe_input),
        "source_clauses_hash": semantic_hash(safe_source_clauses),
        "partition_hash": partition_hash,
        "source_active_group": str(state.get("active_group") or "opening"),
        "pending_contract_hash": source_pending_hash,
        "candidate_ids": [item.candidate_id for item in candidates],
        "atom_ids": [item.atom_id for item in unresolved_atoms],
        "registry_hash": registry_hash,
        "action_registry_hash": action_contract_hash,
        "pending_authority_hash": source_pending_authority_hash,
        "workflow_precondition_hash": workflow_hash,
    }
    draft = SemanticPlanDraft(
        draft_id=semantic_hash(draft_identity),
        revision=1,
        status="awaiting_clarification",
        session_id=session_id,
        creation_turn=int(state.get("turn_index") or 0),
        source_product_authority_id=product_authority_id,
        source_checkpoint_revision=checkpoint_revision,
        source_checkpoint_thread_id=checkpoint_thread_id,
        source_checkpoint_id=checkpoint_id,
        source_schema_version=schema_version,
        source_checkpoint_fingerprint=checkpoint_fingerprint,
        original_input=safe_input,
        original_input_hash=semantic_hash(safe_input),
        source_clauses=safe_source_clauses,
        source_partition=safe_partition,
        source_partition_hash=partition_hash,
        source_active_group=str(state.get("active_group") or "opening"),
        source_pending_question=dict(
            redact(dict(state.get("pending_question") or {}))
        ),
        source_secret_bindings=tuple(
            dict(item) for item in source_secret_bindings
        ),
        candidates=tuple(candidates),
        unresolved_atoms=tuple(unresolved_atoms),
        active_atom_id=unresolved_atoms[0].atom_id,
        registry_hash=registry_hash,
        action_registry_hash=action_contract_hash,
        pending_contract_hash=source_pending_hash,
        pending_authority_hash=source_pending_authority_hash,
        workflow_precondition_hash=workflow_hash,
        lifecycle_receipts=({
            "event": "semantic_draft_created",
            "draft_id": semantic_hash(draft_identity),
            "revision": 1,
        },),
    )
    payload = {
        "contract_version": SEMANTIC_DRAFT_CONTRACT_VERSION,
        **asdict(draft),
    }
    return validate_semantic_plan_draft(payload)


def validate_semantic_plan_draft(
    payload: Mapping[str, Any],
    *,
    validate_current_authority: bool = True,
) -> dict[str, Any]:
    """Validate and normalize one persisted semantic draft."""

    draft = dict(payload)
    if int(draft.get("contract_version") or 0) != SEMANTIC_DRAFT_CONTRACT_VERSION:
        raise ValueError("semantic plan draft has an unsupported contract version")
    status = str(draft.get("status") or "")
    if status not in _OPEN_STATUSES | _TERMINAL_STATUSES:
        raise ValueError(f"semantic plan draft has invalid status: {status!r}")
    if not str(draft.get("draft_id") or ""):
        raise ValueError("semantic plan draft has no draft_id")
    if int(draft.get("revision") or 0) < 1:
        raise ValueError("semantic plan draft revision must be positive")
    for field in (
        "source_checkpoint_fingerprint",
        "original_input_hash",
        "source_partition_hash",
        "registry_hash",
        "action_registry_hash",
        "pending_contract_hash",
        "pending_authority_hash",
        "workflow_precondition_hash",
    ):
        if not _is_sha256(draft.get(field)):
            raise ValueError(f"semantic plan draft {field} is invalid")
    atoms = draft.get("unresolved_atoms")
    candidates = draft.get("candidates")
    source_secret_bindings = draft.get("source_secret_bindings")
    if not isinstance(atoms, (list, tuple)) or not atoms:
        raise ValueError("semantic plan draft requires unresolved_atoms")
    if not isinstance(candidates, (list, tuple)):
        raise ValueError("semantic plan draft candidates must be a list")
    if not isinstance(source_secret_bindings, (list, tuple)):
        raise ValueError("semantic plan draft source secret bindings must be a list")
    seen_source_refs: set[str] = set()
    for binding in source_secret_bindings:
        if not isinstance(binding, Mapping) or set(binding) != {
            "reference",
            "atom_id",
            "value_hash",
        }:
            raise ValueError("semantic plan draft source secret binding is invalid")
        reference = str(binding.get("reference") or "")
        if (
            not _is_exact_secret_reference(reference)
            or reference in seen_source_refs
            or not str(binding.get("atom_id") or "")
            or not _is_sha256(binding.get("value_hash"))
        ):
            raise ValueError("semantic plan draft source secret identity is invalid")
        seen_source_refs.add(reference)
    atom_ids = [str(item.get("atom_id") or "") for item in atoms if isinstance(item, Mapping)]
    if len(atom_ids) != len(atoms) or not all(atom_ids) or len(set(atom_ids)) != len(atom_ids):
        raise ValueError("semantic plan draft atom identities are invalid")
    for atom in atoms:
        reason = str(atom.get("reason") or "")
        if reason not in {"semantic_ambiguity", "missing_user_evidence"}:
            raise ValueError("semantic plan draft atom reason is invalid")
        identity_payload = _atom_identity_payload(
            unit_id=str(atom.get("unit_id") or ""),
            clause_id=str(atom.get("clause_id") or ""),
            source_path=str(atom.get("source_path") or ""),
            source_text=str(atom.get("source_text") or ""),
            semantic_source=str(atom.get("semantic_source") or ""),
            owner=str(atom.get("owner") or ""),
            group=str(atom.get("group") or ""),
            sensitive_input=atom.get("sensitive_input") is True,
            reason=reason,
        )
        if str(atom.get("atom_id") or "") != semantic_hash(identity_payload):
            raise ValueError("semantic plan draft atom identity mismatch")
        resolution = str(atom.get("resolution") or "")
        resolution_hash = str(atom.get("resolution_hash") or "")
        resolution_ref = str(atom.get("resolution_ref") or "")
        if bool(resolution) != bool(resolution_hash):
            raise ValueError(
                "semantic plan draft atom resolution projection is incomplete"
            )
        if resolution_hash and not _is_sha256(resolution_hash):
            raise ValueError(
                "semantic plan draft atom resolution hash is invalid"
            )
        if resolution_ref:
            if (
                not _is_exact_secret_reference(resolution_ref)
                or "***REDACTED***" not in resolution
                or str(redact(resolution)) != resolution
            ):
                raise ValueError(
                    "semantic plan draft atom secret reference is invalid"
                )
        elif resolution and semantic_hash(resolution) != resolution_hash:
            raise ValueError(
                "semantic plan draft atom resolution hash mismatch"
            )
    candidate_ids = [
        str(item.get("candidate_id") or "")
        for item in candidates
        if isinstance(item, Mapping)
    ]
    if (
        len(candidate_ids) != len(candidates)
        or not all(candidate_ids)
        or len(set(candidate_ids)) != len(candidate_ids)
    ):
        raise ValueError("semantic plan draft candidate identities are invalid")
    source_unit_ids = {
        str(item.get("unit_id") or "")
        for item in draft.get("source_partition") or ()
        if isinstance(item, Mapping) and str(item.get("unit_id") or "")
    }
    for candidate in candidates:
        action = dict(candidate.get("action") or {})
        if any(
            key in action
            for key in {
                "_admission_action_id",
                "_plan_transaction_hash",
                "_semantic_admission_receipt",
                "action_id",
            }
        ):
            raise ValueError("semantic plan draft candidate contains admission trust")
        action_type = str(action.get("type") or "")
        if not action_type:
            raise ValueError("semantic plan draft candidate action type is missing")
        spec = ACTION_BY_TYPE.get(action_type) if validate_current_authority else None
        if validate_current_authority:
            if spec is None:
                raise ValueError(
                    "semantic plan draft candidate action is unregistered"
                )
            try:
                action = validate_action_contract(action)
            except ValueError as exc:
                raise ValueError(
                    f"semantic plan draft candidate action is invalid: {exc}"
                ) from exc
        source_atom_ids = tuple(
            str(item)
            for item in candidate.get("source_atom_ids") or ()
            if str(item)
        )
        if any(item not in source_unit_ids for item in source_atom_ids):
            raise ValueError("semantic plan draft candidate source identity is invalid")
        identity_payload = {
            "action": action,
            "source_atom_ids": source_atom_ids,
            "owner": str(candidate.get("owner") or ""),
            "group": str(candidate.get("group") or ""),
        }
        if str(candidate.get("candidate_id") or "") != semantic_hash(identity_payload):
            raise ValueError("semantic plan draft candidate identity mismatch")
        expected_contract_hashes = {
            "group_registry": str(draft.get("registry_hash") or ""),
            "action_registry": str(draft.get("action_registry_hash") or ""),
        }
        if dict(candidate.get("contract_hashes") or {}) != expected_contract_hashes:
            raise ValueError("semantic plan draft candidate contract hashes mismatch")
        if validate_current_authority:
            assert spec is not None
            if str(candidate.get("owner") or "") != spec.owner:
                raise ValueError("semantic plan draft candidate owner mismatch")
            if str(candidate.get("group") or "") != resolve_action_target_group(action):
                raise ValueError("semantic plan draft candidate group mismatch")
            expected_validation_hash = semantic_hash({
                "contract_valid": True,
                "action_type": action_type,
                "owner": spec.owner,
                "group": resolve_action_target_group(action),
            })
            if str(candidate.get("validation_hash") or "") != expected_validation_hash:
                raise ValueError(
                    "semantic plan draft candidate validation hash mismatch"
                )
        elif not _is_sha256(candidate.get("validation_hash")):
            raise ValueError(
                "semantic plan draft candidate validation hash is invalid"
            )
    active_atom_id = str(draft.get("active_atom_id") or "")
    unresolved_ids = {
        str(item.get("atom_id") or "")
        for item in atoms
        if isinstance(item, Mapping)
        and not str(item.get("resolution_hash") or "").strip()
    }
    if status == "awaiting_clarification" and active_atom_id not in unresolved_ids:
        raise ValueError("semantic plan draft active atom is not unresolved")
    if status == "ready_for_review" and unresolved_ids:
        raise ValueError("ready semantic plan draft still has unresolved atoms")
    if semantic_hash(draft.get("source_partition") or []) != str(
        draft.get("source_partition_hash") or ""
    ):
        raise ValueError("semantic plan draft source partition hash mismatch")
    if semantic_hash(str(draft.get("original_input") or "")) != str(
        draft.get("original_input_hash") or ""
    ):
        raise ValueError("semantic plan draft original input hash mismatch")
    if semantic_hash(draft.get("source_pending_question") or {}) != str(
        draft.get("pending_contract_hash") or ""
    ):
        raise ValueError("semantic plan draft pending contract hash mismatch")
    if draft.get("source_pending_question") and validate_current_authority:
        from .questions import validate_pending_question_contract

        try:
            validate_pending_question_contract(
                dict(draft.get("source_pending_question") or {})
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"semantic plan draft source pending contract is invalid: {exc}"
            ) from exc
    if (
        not str(draft.get("source_product_authority_id") or "")
        or not str(draft.get("source_checkpoint_thread_id") or "")
        or not str(draft.get("source_checkpoint_id") or "")
        or int(draft.get("source_checkpoint_revision") or 0) < 0
    ):
        raise ValueError("semantic plan draft Product Head identity is invalid")
    draft_identity = {
        "session_id": str(draft.get("session_id") or ""),
        "creation_turn": int(draft.get("creation_turn") or 0),
        "source_product_authority_id": str(
            draft.get("source_product_authority_id") or ""
        ),
        "source_checkpoint_revision": int(
            draft.get("source_checkpoint_revision") or 0
        ),
        "source_checkpoint_thread_id": str(
            draft.get("source_checkpoint_thread_id") or ""
        ),
        "source_checkpoint_id": str(draft.get("source_checkpoint_id") or ""),
        "source_checkpoint_fingerprint": str(
            draft.get("source_checkpoint_fingerprint") or ""
        ),
        "source_schema_version": int(draft.get("source_schema_version") or 0),
        "input_hash": str(draft.get("original_input_hash") or ""),
        "source_clauses_hash": semantic_hash(
            draft.get("source_clauses") or ()
        ),
        "partition_hash": str(draft.get("source_partition_hash") or ""),
        "source_active_group": str(draft.get("source_active_group") or ""),
        "pending_contract_hash": str(
            draft.get("pending_contract_hash") or ""
        ),
        "candidate_ids": candidate_ids,
        "atom_ids": atom_ids,
        "registry_hash": str(draft.get("registry_hash") or ""),
        "action_registry_hash": str(
            draft.get("action_registry_hash") or ""
        ),
        "pending_authority_hash": str(
            draft.get("pending_authority_hash") or ""
        ),
        "workflow_precondition_hash": str(
            draft.get("workflow_precondition_hash") or ""
        ),
    }
    if str(draft.get("draft_id") or "") != semantic_hash(draft_identity):
        raise ValueError("semantic plan draft identity mismatch")
    _validate_semantic_draft_lifecycle(draft, atoms)
    return draft


def _validate_semantic_draft_lifecycle(
    draft: Mapping[str, Any],
    atoms: Sequence[Mapping[str, Any]],
) -> None:
    """Replay lifecycle receipts and prove the mutable draft projection."""

    receipts = [
        dict(item)
        for item in draft.get("lifecycle_receipts") or ()
        if isinstance(item, Mapping)
    ]
    if not receipts:
        raise ValueError("semantic plan draft lifecycle receipts are missing")
    first = receipts[0]
    if (
        first.get("event") != "semantic_draft_created"
        or str(first.get("draft_id") or "") != str(draft.get("draft_id") or "")
        or int(first.get("revision") or 0) != 1
    ):
        raise ValueError("semantic plan draft creation receipt is invalid")

    atom_ids = [str(item.get("atom_id") or "") for item in atoms]
    resolution_hashes = {atom_id: "" for atom_id in atom_ids}
    expected_revision = 1
    terminal_status = ""
    for receipt in receipts[1:]:
        expected_revision += 1
        if int(receipt.get("revision") or 0) != expected_revision:
            raise ValueError("semantic plan draft lifecycle revision is invalid")
        if terminal_status:
            raise ValueError("semantic plan draft lifecycle continues after terminal state")
        event = str(receipt.get("event") or "")
        if event == "semantic_draft_atom_resolved":
            atom_id = str(receipt.get("atom_id") or "")
            if atom_id not in resolution_hashes or resolution_hashes[atom_id]:
                raise ValueError("semantic plan draft resolution receipt is invalid")
            first_unresolved = next(
                (
                    item
                    for item in atom_ids
                    if not resolution_hashes[item]
                ),
                "",
            )
            if atom_id != first_unresolved or not _is_sha256(
                receipt.get("resolution_hash")
            ):
                raise ValueError("semantic plan draft resolution order is invalid")
            resolution_hashes[atom_id] = str(receipt["resolution_hash"])
        elif event == "semantic_draft_previous_atom_selected":
            atom_id = str(receipt.get("atom_id") or "")
            if atom_id not in resolution_hashes:
                raise ValueError("semantic plan draft back receipt is invalid")
            start = atom_ids.index(atom_id)
            for item in atom_ids[start:]:
                resolution_hashes[item] = ""
        elif event == "semantic_draft_invalidated":
            reasons = [
                str(item).strip()
                for item in receipt.get("reasons") or ()
                if str(item).strip()
            ]
            if not reasons:
                raise ValueError("semantic plan draft invalidation receipt is invalid")
            terminal_status = "stale"
        elif event == "semantic_draft_cancelled":
            if not str(receipt.get("reason") or "").strip():
                raise ValueError("semantic plan draft cancellation receipt is invalid")
            terminal_status = "cancelled"
        else:
            raise ValueError(f"semantic plan draft lifecycle event is invalid: {event!r}")

    if expected_revision != int(draft.get("revision") or 0):
        raise ValueError("semantic plan draft revision does not match lifecycle")
    for atom in atoms:
        atom_id = str(atom.get("atom_id") or "")
        resolution_hash = str(atom.get("resolution_hash") or "")
        expected_hash = resolution_hashes[atom_id]
        if resolution_hash != expected_hash:
            raise ValueError("semantic plan draft resolution does not match lifecycle")
    expected_status = terminal_status or (
        "ready_for_review"
        if all(resolution_hashes.values())
        else "awaiting_clarification"
    )
    if str(draft.get("status") or "") != expected_status:
        raise ValueError("semantic plan draft status does not match lifecycle")
    expected_active = ""
    if expected_status == "awaiting_clarification":
        expected_active = next(
            item for item in atom_ids if not resolution_hashes[item]
        )
    if str(draft.get("active_atom_id") or "") != expected_active:
        raise ValueError("semantic plan draft active atom does not match lifecycle")


def semantic_draft_staleness_reasons(
    payload: Mapping[str, Any],
    state: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return every authority change that makes a draft unsafe to finalize."""

    draft = validate_semantic_plan_draft(
        payload,
        validate_current_authority=False,
    )
    reasons: list[str] = []
    session_id = str(
        (state.get("session") or {}).get("id")
        or state.get("thread_id")
        or ""
    )
    if session_id != str(draft.get("session_id") or ""):
        reasons.append("session_changed")
    if int(state.get("schema_version") or 0) != int(
        draft.get("source_schema_version") or 0
    ):
        reasons.append("checkpoint_schema_changed")
    if group_registry_contract_hash() != str(draft.get("registry_hash") or ""):
        reasons.append("group_registry_changed")
    if action_registry_contract_hash() != str(
        draft.get("action_registry_hash") or ""
    ):
        reasons.append("action_registry_changed")
    from .questions import (
        question_contract_authority_hash,
        validate_pending_question_contract,
    )

    if question_contract_authority_hash() != str(
        draft.get("pending_authority_hash") or ""
    ):
        reasons.append("pending_authority_changed")
    if not any(
        reason in {
            "group_registry_changed",
            "action_registry_changed",
            "pending_authority_changed",
        }
        for reason in reasons
    ):
        validate_semantic_plan_draft(draft)
    current_head = dict((state.get("turn_context") or {}).get("product_head") or {})
    if not current_head:
        reasons.append("product_head_unavailable")
    else:
        if str(current_head.get("product_authority_id") or "") != str(
            draft.get("source_product_authority_id") or ""
        ):
            reasons.append("product_head_authority_changed")
        current_revision = int(current_head.get("revision") or 0)
        source_revision = int(draft.get("source_checkpoint_revision") or 0)
        if current_revision < source_revision:
            reasons.append("product_head_revision_regressed")
        elif current_revision == source_revision and (
            str(current_head.get("checkpoint_thread_id") or "")
            != str(draft.get("source_checkpoint_thread_id") or "")
            or str(current_head.get("checkpoint_id") or "")
            != str(draft.get("source_checkpoint_id") or "")
            or str(current_head.get("state_fingerprint") or "")
            != str(draft.get("source_checkpoint_fingerprint") or "")
        ):
            reasons.append("product_head_checkpoint_replaced")
    if workflow_precondition_hash(state) != str(
        draft.get("workflow_precondition_hash") or ""
    ):
        reasons.append("workflow_precondition_changed")
    pending = dict(state.get("pending_question") or {})
    if pending.get("semantic_draft_binding"):
        try:
            validate_pending_question_contract(pending)
        except (TypeError, ValueError):
            reasons.append("semantic_draft_question_changed")
    return tuple(dict.fromkeys(reasons))


def missing_semantic_draft_secret_bindings(
    payload: Mapping[str, Any],
) -> tuple[dict[str, str], ...]:
    """Return exact durable bindings whose process-local material was lost."""

    from .secret_refs import secret_reference_available

    draft = validate_semantic_plan_draft(
        payload,
        validate_current_authority=False,
    )
    draft_id = str(draft.get("draft_id") or "")
    bindings = [
        {
            "reference": str(binding.get("reference") or ""),
            "draft_id": draft_id,
            "atom_id": str(binding.get("atom_id") or ""),
            "value_hash": str(binding.get("value_hash") or ""),
        }
        for binding in draft.get("source_secret_bindings") or ()
        if isinstance(binding, Mapping)
    ]
    bindings.extend(
        {
            "reference": str(atom.get("resolution_ref") or ""),
            "draft_id": draft_id,
            "atom_id": str(atom.get("atom_id") or ""),
            "value_hash": str(atom.get("resolution_hash") or ""),
        }
        for atom in draft.get("unresolved_atoms") or ()
        if isinstance(atom, Mapping)
        and str(atom.get("resolution_ref") or "")
    )
    return tuple(
        binding
        for binding in bindings
        if not secret_reference_available(
            binding["reference"],
            draft_id=binding["draft_id"],
            atom_id=binding["atom_id"],
            expected_hash=binding["value_hash"],
        )
    )


def mark_semantic_plan_draft_stale(
    payload: Mapping[str, Any],
    *,
    reasons: Sequence[str],
) -> dict[str, Any]:
    """Invalidate a draft atomically while retaining auditable lifecycle data."""

    current_authority_matches = semantic_draft_uses_current_authority(payload)
    draft = validate_semantic_plan_draft(
        payload,
        validate_current_authority=current_authority_matches,
    )
    normalized = tuple(
        dict.fromkeys(str(item).strip() for item in reasons if str(item).strip())
    )
    if not normalized:
        raise ValueError("semantic draft invalidation requires a reason")
    next_revision = int(draft.get("revision") or 0) + 1
    draft.update({
        "revision": next_revision,
        "status": "stale",
        "active_atom_id": "",
        "lifecycle_receipts": [
            *list(draft.get("lifecycle_receipts") or []),
            {
                "event": "semantic_draft_invalidated",
                "reasons": list(normalized),
                "revision": next_revision,
            },
        ],
    })
    return validate_semantic_plan_draft(
        draft,
        validate_current_authority=current_authority_matches,
    )


def build_semantic_draft_finalization_receipt(
    payload: Mapping[str, Any],
    actions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind one ready draft to exactly the newly admitted action transaction."""

    draft = validate_semantic_plan_draft(payload)
    if draft.get("status") != "ready_for_review":
        raise ValueError("semantic draft is not ready for finalization")
    normalized_actions = [dict(item) for item in actions if isinstance(item, Mapping)]
    if len(normalized_actions) != len(actions) or not normalized_actions:
        raise ValueError("semantic draft finalization requires admitted actions")
    action_ids = tuple(
        str(item.get("action_id") or "") for item in normalized_actions
    )
    if not all(action_ids) or len(set(action_ids)) != len(action_ids):
        raise ValueError("semantic draft finalization action identities are invalid")
    transaction_hashes = {
        str(item.get("_plan_transaction_hash") or "")
        for item in normalized_actions
        if str(item.get("_plan_transaction_hash") or "")
    }
    if len(transaction_hashes) != 1:
        raise ValueError(
            "semantic draft finalization requires one admission transaction"
        )
    convergence_rows = semantic_candidate_convergence_rows(
        draft,
        normalized_actions,
    )
    candidate_ids = tuple(
        str(item.get("candidate_id") or "")
        for item in draft.get("candidates") or ()
        if isinstance(item, Mapping)
    )
    resolution_rows = [
        {
            "atom_id": str(item.get("atom_id") or ""),
            "resolution_hash": str(item.get("resolution_hash") or ""),
        }
        for item in draft.get("unresolved_atoms") or ()
        if isinstance(item, Mapping)
    ]
    unsigned = SemanticDraftFinalizationReceipt(
        draft_id=str(draft.get("draft_id") or ""),
        draft_revision=int(draft.get("revision") or 0),
        session_id=str(draft.get("session_id") or ""),
        source_product_authority_id=str(
            draft.get("source_product_authority_id") or ""
        ),
        source_checkpoint_revision=int(
            draft.get("source_checkpoint_revision") or 0
        ),
        source_checkpoint_thread_id=str(
            draft.get("source_checkpoint_thread_id") or ""
        ),
        source_checkpoint_id=str(draft.get("source_checkpoint_id") or ""),
        source_checkpoint_fingerprint=str(
            draft.get("source_checkpoint_fingerprint") or ""
        ),
        source_partition_hash=str(draft.get("source_partition_hash") or ""),
        candidate_ids=candidate_ids,
        candidate_convergence_hash=semantic_hash(convergence_rows),
        secret_binding_manifest_hash=semantic_secret_binding_manifest_hash(
            normalized_actions
        ),
        secret_binding_hashes={
            str(action.get("action_id") or ""): semantic_action_secret_binding_hash(
                action
            )
            for action in normalized_actions
        },
        resolution_hash=semantic_hash(resolution_rows),
        final_plan_hash=semantic_final_plan_hash(normalized_actions),
        admission_transaction_hash=next(iter(transaction_hashes)),
        final_action_ids=action_ids,
        receipt_hash="",
    )
    body = asdict(unsigned)
    body["receipt_hash"] = semantic_hash(
        {key: value for key, value in body.items() if key != "receipt_hash"}
    )
    return body


def _finalization_action_payload(action: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in action.items()
        if key == "type"
        or (
            key not in ACTION_METADATA_FIELDS
            and key not in TRUSTED_ACTION_METADATA_FIELDS
            and not str(key).startswith("_")
        )
    }


def semantic_candidate_convergence_rows(
    draft: Mapping[str, Any],
    actions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Prove every frozen sibling candidate survives final recompilation."""

    available = [
        {
            "action_id": str(action.get("action_id") or ""),
            "payload": _finalization_action_payload(action),
            "source_unit_ids": tuple(
                str(item)
                for item in action.get("_source_unit_ids") or ()
                if str(item)
            ),
        }
        for action in actions
        if isinstance(action, Mapping)
    ]
    used_action_ids: set[str] = set()
    rows: list[dict[str, Any]] = []
    for raw_candidate in draft.get("candidates") or ():
        if not isinstance(raw_candidate, Mapping):
            raise ValueError("semantic draft candidate convergence is invalid")
        candidate_id = str(raw_candidate.get("candidate_id") or "")
        candidate_payload = _finalization_action_payload(
            dict(raw_candidate.get("action") or {})
        )
        candidate_sources = tuple(
            str(item)
            for item in raw_candidate.get("source_atom_ids") or ()
            if str(item)
        )
        match = next(
            (
                item
                for item in available
                if item["action_id"] not in used_action_ids
                and item["payload"] == candidate_payload
                and set(candidate_sources).issubset(
                    set(item["source_unit_ids"])
                )
            ),
            None,
        )
        if match is None:
            raise ValueError(
                "semantic draft finalization omitted or changed candidate "
                f"{candidate_id}"
            )
        used_action_ids.add(str(match["action_id"]))
        rows.append({
            "candidate_id": candidate_id,
            "final_action_id": str(match["action_id"]),
            "source_atom_ids": list(candidate_sources),
            "disposition": "retained",
        })
    return rows


def semantic_secret_binding_manifest_hash(
    actions: Sequence[Mapping[str, Any]],
) -> str:
    """Hash every action-bound secret reference independently of raw values."""

    return semantic_hash({
        str(action.get("action_id") or ""): semantic_action_secret_binding_hash(
            action
        )
        for action in actions
        if isinstance(action, Mapping)
    })


def semantic_action_secret_binding_hash(
    action: Mapping[str, Any],
) -> str:
    rows = sorted(
        [
            {
                "path": list(binding.get("path") or ()),
                "reference": str(binding.get("reference") or ""),
                "draft_id": str(binding.get("draft_id") or ""),
                "atom_id": str(binding.get("atom_id") or ""),
                "value_hash": str(binding.get("value_hash") or ""),
            }
            for binding in action.get("_semantic_secret_bindings") or ()
            if isinstance(binding, Mapping)
        ],
        key=lambda item: (
            json.dumps(item["path"], separators=(",", ":")),
            item["reference"],
        ),
    )
    return semantic_hash(rows)


def semantic_final_plan_hash(actions: Sequence[Mapping[str, Any]]) -> str:
    """Hash the complete business action plan independently of Harness metadata."""

    return semantic_hash([
        {
            key: value
            for key, value in item.items()
            if not str(key).startswith("_") and key != "action_id"
        }
        for item in actions
        if isinstance(item, Mapping)
    ])


def validate_semantic_draft_finalization_receipt(
    receipt: Mapping[str, Any],
    *,
    action: Mapping[str, Any],
    session_id: str,
) -> dict[str, Any]:
    """Validate one envelope-carried finalization proof without draft state."""

    normalized = validate_semantic_draft_finalization_receipt_body(
        receipt,
        session_id=session_id,
    )
    action_ids = tuple(
        str(item) for item in normalized.get("final_action_ids") or () if str(item)
    )
    if str(action.get("action_id") or "") not in action_ids:
        raise ValueError("action is absent from semantic draft finalization receipt")
    if str(action.get("_plan_transaction_hash") or "") != str(
        normalized.get("admission_transaction_hash") or ""
    ):
        raise ValueError(
            "semantic draft finalization receipt transaction mismatch"
        )
    return normalized


def validate_semantic_draft_finalization_receipt_body(
    receipt: Mapping[str, Any],
    *,
    session_id: str,
) -> dict[str, Any]:
    """Validate the immutable receipt body independently of an envelope."""

    normalized = dict(receipt)
    expected_fields = {
        "draft_id",
        "draft_revision",
        "session_id",
        "source_product_authority_id",
        "source_checkpoint_revision",
        "source_checkpoint_thread_id",
        "source_checkpoint_id",
        "source_checkpoint_fingerprint",
        "source_partition_hash",
        "candidate_ids",
        "candidate_convergence_hash",
        "secret_binding_manifest_hash",
        "secret_binding_hashes",
        "resolution_hash",
        "final_plan_hash",
        "admission_transaction_hash",
        "final_action_ids",
        "receipt_hash",
    }
    if set(normalized) != expected_fields:
        raise ValueError("semantic draft finalization receipt fields are invalid")
    if str(normalized.get("session_id") or "") != session_id:
        raise ValueError("semantic draft finalization receipt session mismatch")
    if int(normalized.get("draft_revision") or 0) < 1:
        raise ValueError("semantic draft finalization receipt revision is invalid")
    if (
        not str(normalized.get("source_product_authority_id") or "")
        or int(normalized.get("source_checkpoint_revision") or 0) < 0
        or not str(normalized.get("source_checkpoint_thread_id") or "")
        or not str(normalized.get("source_checkpoint_id") or "")
    ):
        raise ValueError(
            "semantic draft finalization receipt Product Head identity is invalid"
        )
    for field in (
        "draft_id",
        "source_checkpoint_fingerprint",
        "source_partition_hash",
        "candidate_convergence_hash",
        "secret_binding_manifest_hash",
        "resolution_hash",
        "final_plan_hash",
        "admission_transaction_hash",
        "receipt_hash",
    ):
        if not _is_sha256(normalized.get(field)):
            raise ValueError(
                f"semantic draft finalization receipt {field} is invalid"
            )
    unsigned = {
        key: value
        for key, value in normalized.items()
        if key != "receipt_hash"
    }
    if semantic_hash(unsigned) != str(normalized.get("receipt_hash") or ""):
        raise ValueError("semantic draft finalization receipt hash mismatch")
    action_ids = tuple(
        str(item) for item in normalized.get("final_action_ids") or () if str(item)
    )
    if not action_ids or len(set(action_ids)) != len(action_ids):
        raise ValueError(
            "semantic draft finalization receipt action identities are invalid"
        )
    candidate_ids = tuple(
        str(item)
        for item in normalized.get("candidate_ids") or ()
        if str(item)
    )
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError(
            "semantic draft finalization candidate identities are invalid"
        )
    secret_binding_hashes = normalized.get("secret_binding_hashes")
    if (
        not isinstance(secret_binding_hashes, Mapping)
        or set(secret_binding_hashes) != set(action_ids)
        or not all(_is_sha256(value) for value in secret_binding_hashes.values())
    ):
        raise ValueError(
            "semantic draft finalization secret binding hashes are invalid"
        )
    if semantic_hash(dict(secret_binding_hashes)) != str(
        normalized.get("secret_binding_manifest_hash") or ""
    ):
        raise ValueError(
            "semantic draft finalization secret binding manifest mismatch"
        )
    return normalized


def resolve_semantic_draft_atom(
    payload: Mapping[str, Any],
    *,
    draft_id: str,
    revision: int,
    atom_id: str,
    resolution: str,
    resolution_hash: str = "",
    resolution_ref: str = "",
) -> dict[str, Any]:
    """Resolve exactly one bound atom and select the next unresolved atom."""

    draft = validate_semantic_plan_draft(payload)
    if str(draft.get("draft_id") or "") != draft_id:
        raise ValueError("semantic draft identity mismatch")
    if int(draft.get("revision") or 0) != revision:
        raise ValueError("semantic draft revision mismatch")
    if str(draft.get("active_atom_id") or "") != atom_id:
        raise ValueError("semantic draft atom binding mismatch")
    raw_answer = str(resolution).strip()
    answer = str(redact(raw_answer)).strip()
    if not answer:
        raise ValueError("semantic draft resolution cannot be empty")
    value_hash = str(resolution_hash or semantic_hash(raw_answer))
    if not _is_sha256(value_hash):
        raise ValueError("semantic draft resolution hash is invalid")
    if resolution_ref:
        if not _is_exact_secret_reference(resolution_ref):
            raise ValueError("semantic draft resolution reference is invalid")
    elif semantic_hash(raw_answer) != value_hash:
        raise ValueError("semantic draft resolution hash mismatch")
    atoms: list[dict[str, Any]] = []
    found = False
    for raw in draft.get("unresolved_atoms") or ():
        atom = dict(raw)
        if str(atom.get("atom_id") or "") == atom_id:
            if str(atom.get("resolution") or "").strip():
                raise ValueError("semantic draft atom is already resolved")
            atom["resolution"] = answer
            atom["resolution_hash"] = value_hash
            atom["resolution_ref"] = str(resolution_ref)
            found = True
        atoms.append(atom)
    if not found:
        raise ValueError("semantic draft atom does not exist")
    unresolved_ids = [
        str(atom.get("atom_id") or "")
        for atom in atoms
        if not str(atom.get("resolution_hash") or "").strip()
    ]
    next_revision = revision + 1
    draft.update({
        "revision": next_revision,
        "status": (
            "awaiting_clarification" if unresolved_ids else "ready_for_review"
        ),
        "active_atom_id": unresolved_ids[0] if unresolved_ids else "",
        "unresolved_atoms": atoms,
        "lifecycle_receipts": [
            *list(draft.get("lifecycle_receipts") or []),
            {
                "event": "semantic_draft_atom_resolved",
                "atom_id": atom_id,
                "revision": next_revision,
                "resolution_hash": value_hash,
            },
        ],
    })
    return validate_semantic_plan_draft(draft)


def cancel_semantic_plan_draft(
    payload: Mapping[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    """Cancel one draft without changing benchmark business state."""

    draft = validate_semantic_plan_draft(payload)
    next_revision = int(draft.get("revision") or 0) + 1
    draft.update({
        "revision": next_revision,
        "status": "cancelled",
        "active_atom_id": "",
        "lifecycle_receipts": [
            *list(draft.get("lifecycle_receipts") or []),
            {
                "event": "semantic_draft_cancelled",
                "revision": next_revision,
                "reason": str(reason or "cancelled"),
            },
        ],
    })
    return validate_semantic_plan_draft(draft)


def reopen_previous_semantic_draft_atom(
    payload: Mapping[str, Any],
    *,
    draft_id: str,
    revision: int,
    atom_id: str,
) -> dict[str, Any]:
    """Move clarification to the preceding atom and clear its old resolution."""

    draft = validate_semantic_plan_draft(payload)
    if str(draft.get("draft_id") or "") != draft_id:
        raise ValueError("semantic draft identity mismatch")
    if int(draft.get("revision") or 0) != revision:
        raise ValueError("semantic draft revision mismatch")
    atom_ids = [
        str(item.get("atom_id") or "")
        for item in draft.get("unresolved_atoms") or ()
        if isinstance(item, Mapping)
    ]
    active_atom_id = str(draft.get("active_atom_id") or "")
    try:
        active_index = atom_ids.index(active_atom_id)
    except ValueError as exc:
        raise ValueError("semantic draft active atom is unavailable") from exc
    if active_index == 0:
        raise ValueError("semantic draft has no previous atom")
    previous_id = atom_ids[active_index - 1]
    if previous_id != atom_id:
        raise ValueError("semantic draft previous atom binding mismatch")
    atoms: list[dict[str, Any]] = []
    for index, raw in enumerate(draft.get("unresolved_atoms") or ()):
        atom = dict(raw)
        if index >= active_index - 1:
            atom["resolution"] = ""
            atom["resolution_hash"] = ""
            atom["resolution_ref"] = ""
        atoms.append(atom)
    next_revision = revision + 1
    draft.update({
        "revision": next_revision,
        "status": "awaiting_clarification",
        "active_atom_id": previous_id,
        "unresolved_atoms": atoms,
        "lifecycle_receipts": [
            *list(draft.get("lifecycle_receipts") or []),
            {
                "event": "semantic_draft_previous_atom_selected",
                "atom_id": previous_id,
                "revision": next_revision,
            },
        ],
    })
    return validate_semantic_plan_draft(draft)

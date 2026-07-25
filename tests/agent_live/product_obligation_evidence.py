"""Atomic, revision-bound evidence admission for Phase 8 G3 and G4.

Obligation catalogs define a finite denominator; they never prove execution.
This module admits separately produced evidence only after validating the
frozen obligation contract, execution identity, verifier results, and hashes
of the real runtime artifacts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tests.agent_live.coverage_evidence import content_hash
from tests.agent_live.container_process_guard import (
    validate_cleanup_receipt_artifact,
)


PRODUCT_OBLIGATION_EVIDENCE_SCHEMA_VERSION = 2
EVIDENCE_OUTCOMES = frozenset({"passed", "failed", "externally_blocked"})
VERIFIER_STATUSES = EVIDENCE_OUTCOMES
REQUIRED_ARTIFACT_ROLES = frozenset({
    "checkpoint_diff",
    "runtime_events",
    "transcript",
})

_EVIDENCE_FIELDS = frozenset({
    "schema_version",
    "evidence_id",
    "obligation_id",
    "obligation_contract_hash",
    "revision_binding",
    "round_id",
    "session_id",
    "request_ids",
    "outcome",
    "execution",
    "artifacts",
    "verifier_results",
    "evidence_hash",
})
_EXECUTION_FIELDS = frozenset({
    "execution_id",
    "runner",
    "transport",
    "provider",
    "model",
    "started_at",
    "finished_at",
})
_ARTIFACT_FIELDS = frozenset({"role", "path", "sha256"})
_VERIFIER_FIELDS = frozenset({
    "verifier_id",
    "verifier_version",
    "implementation_hash",
    "status",
    "details",
    "evidence_sha256s",
})


def admit_product_obligation_evidence(
    *,
    obligations: Sequence[Mapping[str, Any]],
    evidence_paths: Sequence[str | Path],
    revision: Mapping[str, str],
) -> dict[str, Any]:
    """Validate one atomic evidence batch and summarize the frozen denominator.

    Absence of evidence leaves an obligation ``not_run``. A supplied evidence
    document that is missing, stale, duplicated, conflicting, or unverifiable
    rejects the complete batch.
    """

    active_revision = _validated_revision(revision)
    obligation_index = _validated_obligation_index(obligations, active_revision)
    evidence_documents = _load_evidence_documents(evidence_paths)

    admitted: dict[str, dict[str, Any]] = {}
    evidence_ids: set[str] = set()
    for evidence_path, document in evidence_documents:
        obligation_id = _required_text(document, "obligation_id")
        obligation = obligation_index.get(obligation_id)
        if obligation is None:
            raise ValueError(f"evidence targets an unknown obligation: {obligation_id}")
        if obligation_id in admitted:
            raise ValueError(f"duplicate or conflicting evidence: {obligation_id}")
        evidence_id = _required_text(document, "evidence_id")
        if evidence_id in evidence_ids:
            raise ValueError(f"duplicate evidence id: {evidence_id}")
        evidence_ids.add(evidence_id)
        admitted[obligation_id] = _validate_evidence_document(
            document=document,
            evidence_path=evidence_path,
            obligation=obligation,
            revision=active_revision,
        )

    outcomes = {
        obligation_id: admitted.get(obligation_id, {}).get("outcome", "not_run")
        for obligation_id in obligation_index
    }
    passed = sum(outcome == "passed" for outcome in outcomes.values())
    failed = sum(outcome == "failed" for outcome in outcomes.values())
    external = sum(outcome == "externally_blocked" for outcome in outcomes.values())
    not_run = sum(outcome == "not_run" for outcome in outcomes.values())
    denominator = len(obligation_index)
    complete = passed == denominator and not any((failed, external, not_run))
    return {
        "schema_version": PRODUCT_OBLIGATION_EVIDENCE_SCHEMA_VERSION,
        "revision_binding": active_revision,
        "denominator": denominator,
        "passed": passed,
        "failed": failed,
        "not_run": not_run,
        "external": external,
        "complete": complete,
        "status": "complete" if complete else "failed" if failed else "incomplete",
        "generation_is_execution": False,
        "outcomes": outcomes,
        "admitted_evidence": {
            obligation_id: admitted[obligation_id]
            for obligation_id in sorted(admitted)
        },
    }


def admit_product_chaos_rounds(
    *,
    obligations: Sequence[Mapping[str, Any]],
    evidence_by_round: Mapping[str, Sequence[str | Path]],
    revision: Mapping[str, str],
    required_round_ids: Sequence[str] = ("round-1", "round-2"),
) -> dict[str, Any]:
    """Admit two complete G4 rounds and reject all cross-round identity reuse."""

    expected = tuple(required_round_ids)
    if (
        not expected
        or len(expected) != len(set(expected))
        or any(not str(round_id).strip() for round_id in expected)
        or set(evidence_by_round) != set(expected)
    ):
        raise ValueError("G4 round set does not match the required round contract")

    summaries: dict[str, dict[str, Any]] = {}
    identity_owners: dict[tuple[str, str], str] = {}
    for round_id in expected:
        summary = admit_product_obligation_evidence(
            obligations=obligations,
            evidence_paths=evidence_by_round[round_id],
            revision=revision,
        )
        for obligation_id, admitted in summary["admitted_evidence"].items():
            if admitted["round_id"] != round_id:
                raise ValueError(
                    f"G4 evidence round mismatch: {round_id}/{obligation_id}"
                )
            for identity_type, values in (
                ("session", (admitted["session_id"],)),
                ("request", admitted["request_ids"]),
                ("execution", (admitted["execution_id"],)),
                ("evidence", (admitted["evidence_id"],)),
                ("artifact", admitted["artifact_sha256s"]),
            ):
                for value in values:
                    key = (identity_type, str(value))
                    owner = identity_owners.setdefault(key, round_id)
                    if owner != round_id:
                        raise ValueError(
                            "G4 cross-round identity reuse: "
                            f"{identity_type}/{value}"
                        )
        summaries[round_id] = summary

    complete = all(summary["complete"] for summary in summaries.values())
    new_root_classes = {
        round_id: []
        if summary["failed"] == 0
        else ["unclassified-product-failure"]
        for round_id, summary in summaries.items()
    }
    converged = complete and all(
        not root_classes for root_classes in new_root_classes.values()
    )
    return {
        "schema_version": PRODUCT_OBLIGATION_EVIDENCE_SCHEMA_VERSION,
        "revision_binding": _validated_revision(revision),
        "required_round_ids": list(expected),
        "rounds": summaries,
        "denominator_per_round": len(obligations),
        "total_denominator": len(obligations) * len(expected),
        "new_s1_s2_root_classes": new_root_classes,
        "complete": converged,
        "status": "complete" if converged else (
            "failed"
            if any(summary["failed"] for summary in summaries.values())
            else "incomplete"
        ),
    }


def _validated_obligation_index(
    obligations: Sequence[Mapping[str, Any]],
    revision: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    rows = tuple(dict(row) for row in obligations)
    if not rows:
        raise ValueError("product evidence admission requires frozen obligations")
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        obligation_id = _required_text(row, "obligation_id")
        if obligation_id in index:
            raise ValueError(f"duplicate frozen obligation: {obligation_id}")
        contract_hash = _required_sha256(row, "contract_hash")
        unsigned = dict(row)
        unsigned.pop("contract_hash", None)
        if content_hash(unsigned) != contract_hash:
            raise ValueError(f"forged frozen obligation contract: {obligation_id}")
        if _obligation_revision(row) != revision:
            raise ValueError(f"stale frozen obligation revision: {obligation_id}")
        catalog_status = row.get("execution_status", row.get("status"))
        if catalog_status != "not_run":
            raise ValueError(
                f"catalog generation claimed execution for obligation: {obligation_id}"
            )
        _expected_verifier_ids(row)
        index[obligation_id] = row
    return index


def _load_evidence_documents(
    evidence_paths: Sequence[str | Path],
) -> tuple[tuple[Path, dict[str, Any]], ...]:
    documents: list[tuple[Path, dict[str, Any]]] = []
    seen_paths: set[Path] = set()
    for value in evidence_paths:
        path = Path(value).expanduser().resolve()
        if path in seen_paths:
            raise ValueError(f"duplicate evidence path: {path}")
        seen_paths.add(path)
        if not path.is_file():
            raise ValueError(f"evidence JSON does not exist: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid evidence JSON: {path}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"evidence JSON must contain one object: {path}")
        documents.append((path, dict(payload)))
    return tuple(documents)


def _validate_evidence_document(
    *,
    document: Mapping[str, Any],
    evidence_path: Path,
    obligation: Mapping[str, Any],
    revision: Mapping[str, str],
) -> dict[str, Any]:
    obligation_id = str(obligation["obligation_id"])
    if set(document) != _EVIDENCE_FIELDS:
        raise ValueError(f"incomplete evidence contract: {obligation_id}")
    if document.get("schema_version") != PRODUCT_OBLIGATION_EVIDENCE_SCHEMA_VERSION:
        raise ValueError(f"unsupported evidence schema: {obligation_id}")
    if document.get("revision_binding") != revision:
        raise ValueError(f"stale evidence revision: {obligation_id}")
    if document.get("obligation_contract_hash") != obligation.get("contract_hash"):
        raise ValueError(f"evidence contract hash mismatch: {obligation_id}")

    unsigned = dict(document)
    recorded_hash = _required_sha256(unsigned, "evidence_hash")
    unsigned.pop("evidence_hash")
    if content_hash(unsigned) != recorded_hash:
        raise ValueError(f"forged evidence document: {obligation_id}")

    outcome = _required_text(document, "outcome")
    if outcome not in EVIDENCE_OUTCOMES:
        raise ValueError(f"invalid evidence outcome: {obligation_id}")
    _validate_execution(document.get("execution"), obligation_id)
    round_id = _required_text(document, "round_id")
    session_id = _required_text(document, "session_id")
    request_ids = _validated_request_ids(document.get("request_ids"), obligation_id)
    artifacts = _validate_artifacts(
        document.get("artifacts"),
        evidence_path=evidence_path,
        obligation_id=obligation_id,
    )
    artifact_hashes = {
        str(item["sha256"]) for item in artifacts.values()
    }
    if document.get("outcome") == "passed" and _is_g4_obligation(obligation):
        _validate_g4_runtime_provenance(
            document=document,
            obligation=obligation,
            artifacts=artifacts,
            session_id=session_id,
            request_ids=request_ids,
        )
    _validate_verifier_results(
        document.get("verifier_results"),
        obligation=obligation,
        artifact_hashes=artifact_hashes,
        outcome=outcome,
    )
    evidence_id = _required_sha256(document, "evidence_id")
    identity = {
        "obligation_id": obligation_id,
        "obligation_contract_hash": obligation["contract_hash"],
        "revision_binding": dict(revision),
        "round_id": round_id,
        "session_id": session_id,
        "request_ids": list(request_ids),
        "execution_id": dict(document["execution"])["execution_id"],
        "artifact_sha256s": sorted(artifact_hashes),
    }
    if evidence_id != content_hash(identity):
        raise ValueError(f"unstable evidence identity: {obligation_id}")
    return {
        "evidence_id": evidence_id,
        "evidence_path": str(evidence_path),
        "evidence_sha256": _sha256_file(evidence_path),
        "outcome": outcome,
        "round_id": round_id,
        "session_id": session_id,
        "request_ids": list(request_ids),
        "execution_id": dict(document["execution"])["execution_id"],
        "artifact_sha256s": sorted(artifact_hashes),
    }


def _validate_execution(value: Any, obligation_id: str) -> None:
    if not isinstance(value, Mapping) or set(value) != _EXECUTION_FIELDS:
        raise ValueError(f"execution identity is incomplete: {obligation_id}")
    execution = dict(value)
    for field in _EXECUTION_FIELDS - {"transport"}:
        _required_text(execution, field)
    if execution.get("transport") != "real_pty":
        raise ValueError(f"evidence was not produced by a real PTY: {obligation_id}")
    if execution["started_at"] == execution["finished_at"]:
        raise ValueError(f"execution interval is empty: {obligation_id}")


def _validated_request_ids(value: Any, obligation_id: str) -> tuple[str, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
    ):
        raise ValueError(f"request identity set is incomplete: {obligation_id}")
    normalized = tuple(str(item).strip() for item in value)
    if any(not item for item in normalized) or len(normalized) != len(set(normalized)):
        raise ValueError(f"request identity set is invalid: {obligation_id}")
    return normalized


def _validate_artifacts(
    value: Any,
    *,
    evidence_path: Path,
    obligation_id: str,
) -> dict[str, dict[str, Any]]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
    ):
        raise ValueError(f"runtime artifacts are missing: {obligation_id}")
    roles: set[str] = set()
    paths: set[Path] = set()
    hashes: set[str] = set()
    records: dict[str, dict[str, Any]] = {}
    for raw in value:
        if not isinstance(raw, Mapping) or set(raw) != _ARTIFACT_FIELDS:
            raise ValueError(f"runtime artifact contract is invalid: {obligation_id}")
        artifact = dict(raw)
        role = _required_text(artifact, "role")
        if role in roles:
            raise ValueError(f"duplicate runtime artifact role: {obligation_id}/{role}")
        roles.add(role)
        raw_path = Path(_required_text(artifact, "path")).expanduser()
        path = (
            raw_path.resolve()
            if raw_path.is_absolute()
            else (evidence_path.parent / raw_path).resolve()
        )
        if path in paths:
            raise ValueError(f"duplicate runtime artifact path: {obligation_id}")
        paths.add(path)
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"runtime artifact is missing or empty: {path}")
        recorded_hash = _required_sha256(artifact, "sha256")
        if _sha256_file(path) != recorded_hash:
            raise ValueError(f"runtime artifact hash mismatch: {path}")
        hashes.add(recorded_hash)
        records[role] = {
            "path": path,
            "sha256": recorded_hash,
        }
    missing_roles = REQUIRED_ARTIFACT_ROLES - roles
    if missing_roles:
        raise ValueError(
            f"required runtime artifacts are missing for {obligation_id}: "
            + ", ".join(sorted(missing_roles))
        )
    return records


def _is_g4_obligation(obligation: Mapping[str, Any]) -> bool:
    return (
        isinstance(obligation.get("model"), Mapping)
        and isinstance(obligation.get("factors"), Mapping)
        and isinstance(obligation.get("start_contract"), Mapping)
    )


def _validate_g4_runtime_provenance(
    *,
    document: Mapping[str, Any],
    obligation: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
    session_id: str,
    request_ids: Sequence[str],
) -> None:
    obligation_id = str(obligation["obligation_id"])
    required_roles = {
        "journey_evidence",
        "journey_result",
        "journey_schedule",
        "process_guard_receipt",
    }
    missing = required_roles - set(artifacts)
    if missing:
        raise ValueError(
            f"required G4 runtime provenance is missing for {obligation_id}: "
            + ", ".join(sorted(missing))
        )
    journey_path = Path(artifacts["journey_evidence"]["path"])
    try:
        journey = json.loads(journey_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"G4 Journey evidence is invalid: {obligation_id}"
        ) from exc
    execution = dict(document["execution"])
    proof = journey.get("execution_proof") if isinstance(journey, Mapping) else None
    if (
        not isinstance(journey, Mapping)
        or journey.get("artifact_type") != "dynamic_dual_ai_journey_evidence"
        or journey.get("journey_id") != obligation_id
        or journey.get("revision") != document.get("revision_binding")
        or journey.get("terminal_classification") != "passed"
        or journey.get("qualifying_evidence") is not True
        or journey.get("qualification_reason")
        != "trusted_container_pty_execution"
        or journey.get("execution_id") != execution.get("execution_id")
        or journey.get("session_id") != session_id
        or journey.get("provider") != execution.get("provider")
        or journey.get("model") != execution.get("model")
        or not isinstance(proof, Mapping)
    ):
        raise ValueError(
            f"G4 Journey provenance does not match product evidence: {obligation_id}"
        )
    observed_request_ids = tuple(
        str(
            dict(turn.get("decision_provenance") or {}).get(
                "broker_request_id"
            )
            or ""
        )
        for turn in tuple(journey.get("turns") or ())
        if isinstance(turn, Mapping)
    )
    if (
        not observed_request_ids
        or any(not value for value in observed_request_ids)
        or observed_request_ids != tuple(request_ids)
    ):
        raise ValueError(
            f"G4 request identities do not match Journey evidence: {obligation_id}"
        )
    receipt_path = Path(artifacts["process_guard_receipt"]["path"]).resolve()
    transcript_root = Path(artifacts["transcript"]["path"]).resolve().parent
    expected_receipt_root = transcript_root / "container-cleanup-receipts"
    if (
        receipt_path.parent != expected_receipt_root
        or Path(str(proof.get("path") or "")).resolve() != receipt_path
        or proof.get("sha256")
        != artifacts["process_guard_receipt"]["sha256"]
    ):
        raise ValueError(
            f"G4 process proof is outside its runtime or artifact binding: {obligation_id}"
        )
    validated = validate_cleanup_receipt_artifact(
        receipt_path,
        execution_id=str(execution["execution_id"]),
        required_roles=(
            "container_bridge",
            "agent_process_group_leader",
        ),
        allowed_roots=(expected_receipt_root,),
    )
    expected_proof = {
        "proof_type": "container_pty_process_guard",
        "transport_kind": "container_pty_bridge",
        **validated,
    }
    if dict(proof) != expected_proof:
        raise ValueError(
            f"G4 process proof differs from its receipt: {obligation_id}"
        )


def _validate_verifier_results(
    value: Any,
    *,
    obligation: Mapping[str, Any],
    artifact_hashes: set[str],
    outcome: str,
) -> None:
    obligation_id = str(obligation["obligation_id"])
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
    ):
        raise ValueError(f"verifier results are missing: {obligation_id}")
    results: dict[str, str] = {}
    expected_bindings = _expected_verifier_bindings(obligation)
    for raw in value:
        if not isinstance(raw, Mapping) or set(raw) != _VERIFIER_FIELDS:
            raise ValueError(f"verifier result contract is invalid: {obligation_id}")
        result = dict(raw)
        verifier_id = _required_text(result, "verifier_id")
        if verifier_id in results:
            raise ValueError(f"duplicate verifier result: {obligation_id}/{verifier_id}")
        status = _required_text(result, "status")
        if status not in VERIFIER_STATUSES:
            raise ValueError(f"invalid verifier status: {obligation_id}/{verifier_id}")
        _required_text(result, "details")
        binding = expected_bindings.get(verifier_id)
        if (
            binding is None
            or result.get("verifier_version") != binding["verifier_version"]
            or _required_sha256(result, "implementation_hash")
            != binding["implementation_hash"]
        ):
            raise ValueError(
                f"verifier implementation binding is stale: "
                f"{obligation_id}/{verifier_id}"
            )
        references = result.get("evidence_sha256s")
        if (
            not isinstance(references, Sequence)
            or isinstance(references, (str, bytes))
            or not references
            or any(
                not isinstance(reference, str) or not _is_sha256(reference)
                for reference in references
            )
            or len(references) != len(set(references))
            or not set(references).issubset(artifact_hashes)
        ):
            raise ValueError(
                f"verifier has invalid artifact references: {obligation_id}/{verifier_id}"
            )
        results[verifier_id] = status

    expected = _expected_verifier_ids(obligation)
    if set(results) != expected:
        raise ValueError(f"verifier result set is incomplete: {obligation_id}")
    statuses = set(results.values())
    if outcome == "passed" and statuses != {"passed"}:
        raise ValueError(f"passed evidence contains non-passing verifier: {obligation_id}")
    if outcome == "failed" and "failed" not in statuses:
        raise ValueError(f"failed evidence has no failing verifier: {obligation_id}")
    if outcome == "externally_blocked" and "externally_blocked" not in statuses:
        raise ValueError(
            f"externally blocked evidence has no blocked verifier: {obligation_id}"
        )


def _expected_verifier_ids(obligation: Mapping[str, Any]) -> set[str]:
    verifier = obligation.get("verifier_contract")
    if not isinstance(verifier, Mapping):
        raise ValueError("frozen obligation has no verifier contract")
    required = verifier.get("required_postcondition_ids")
    forbidden = verifier.get("forbidden_postcondition_ids")
    if (
        not isinstance(required, Sequence)
        or isinstance(required, (str, bytes))
        or not required
        or not isinstance(forbidden, Sequence)
        or isinstance(forbidden, (str, bytes))
    ):
        raise ValueError("frozen obligation has an invalid verifier contract")
    verifier_ids = tuple(required) + tuple(forbidden)
    if (
        any(not isinstance(item, str) or not item.strip() for item in verifier_ids)
        or len(verifier_ids) != len(set(verifier_ids))
    ):
        raise ValueError("frozen obligation verifier identifiers are invalid")
    return set(verifier_ids)


def _expected_verifier_bindings(
    obligation: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    verifier = obligation.get("verifier_contract")
    if not isinstance(verifier, Mapping):
        raise ValueError("frozen obligation has no verifier contract")
    rows = [
        *(verifier.get("required_bindings") or ()),
        *(verifier.get("forbidden_bindings") or ()),
    ]
    expected_ids = _expected_verifier_ids(obligation)
    bindings: dict[str, dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError("frozen verifier implementation binding is invalid")
        postcondition_id = str(raw.get("postcondition_id") or "")
        version = raw.get("verifier_version")
        implementation_hash = str(raw.get("implementation_hash") or "")
        if (
            postcondition_id not in expected_ids
            or postcondition_id in bindings
            or isinstance(version, bool)
            or not isinstance(version, int)
            or version <= 0
            or not _is_sha256(implementation_hash)
        ):
            raise ValueError("frozen verifier implementation binding is invalid")
        bindings[postcondition_id] = {
            "verifier_version": version,
            "implementation_hash": implementation_hash,
        }
    if set(bindings) != expected_ids:
        raise ValueError("frozen verifier implementation bindings are incomplete")
    return bindings


def _obligation_revision(obligation: Mapping[str, Any]) -> dict[str, str]:
    binding = obligation.get("revision_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("frozen obligation has no revision binding")
    nested = binding.get("revision")
    return _validated_revision(nested if isinstance(nested, Mapping) else binding)


def _validated_revision(revision: Mapping[str, str]) -> dict[str, str]:
    normalized = {
        "commit": str(revision.get("commit") or "").strip(),
        "worktree_hash": str(revision.get("worktree_hash") or "").strip(),
    }
    if not normalized["commit"] or not normalized["worktree_hash"]:
        raise ValueError("product evidence admission requires a revision binding")
    return normalized


def _required_text(values: Mapping[str, Any], field: str) -> str:
    value = str(values.get(field) or "").strip()
    if not value:
        raise ValueError(f"evidence contract requires {field}")
    return value


def _required_sha256(values: Mapping[str, Any], field: str) -> str:
    value = _required_text(values, field).lower()
    if not _is_sha256(value):
        raise ValueError(f"evidence contract requires SHA256 {field}")
    return value


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value.lower()
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

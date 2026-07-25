"""Independently validate typed G6 evidence and decide the product-review gate."""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.harness.control_receipts import (
    validate_persisted_domain_control_receipt,
)
from tests.agent_live.product_review_contract import (
    ARTIFACT_ROLES,
    EXTERNAL_CAPABILITY_STATES,
    FINDING_SEVERITIES,
    FINDING_STATUSES,
    PRODUCT_REVIEW_GENERATOR_ID,
    PRODUCT_REVIEW_GENERATOR_VERSION,
    PRODUCT_REVIEW_SCHEMA_VERSION,
    PRODUCT_REVIEW_VALIDATOR_ID,
    PRODUCT_REVIEW_VALIDATOR_VERSION,
    TOKEN_AVAILABILITY,
    content_hash,
    file_sha256,
    load_mapping,
    module_implementation_hash,
    require_sha256,
    require_stable_id,
    require_text,
    validate_artifact_envelope,
    validate_hashed_source,
    validated_revision,
)
from tests.agent_live.product_review_scope import (
    DEFAULT_SCOPE_MANIFEST,
    compute_product_review_scope,
)


VALIDATOR_PATH = Path(__file__).resolve()
GENERATOR_PATH = VALIDATOR_PATH.with_name("generate_product_review_evidence.py")
_TOKEN_FIELDS = ("prompt_tokens", "output_tokens", "total_tokens")


def validate_product_review_evidence(
    manifest_path: str | Path,
    *,
    revision: Mapping[str, Any],
    generator_implementation_path: str | Path = GENERATOR_PATH,
    validator_implementation_path: str | Path = VALIDATOR_PATH,
    scope_repo_root: str | Path = REPO_ROOT,
    scope_manifest_path: str | Path = DEFAULT_SCOPE_MANIFEST,
) -> dict[str, Any]:
    """Recompute all review facts and return the sole G6 decision."""

    active_revision = validated_revision(revision)
    path, manifest = load_mapping(manifest_path, "G6 product-review manifest")
    _validate_manifest(
        manifest,
        revision=active_revision,
        generator_implementation_path=generator_implementation_path,
        validator_implementation_path=validator_implementation_path,
    )
    artifact_rows = manifest.get("artifacts")
    if not isinstance(artifact_rows, list):
        raise ValueError("G6 manifest artifacts must be a list")
    expected_roles = set(ARTIFACT_ROLES)
    observed_roles = {
        str(row.get("role") or "")
        for row in artifact_rows
        if isinstance(row, Mapping)
    }
    if observed_roles != expected_roles or len(artifact_rows) != len(expected_roles):
        raise ValueError("G6 manifest artifact roles are incomplete or duplicated")

    artifacts: dict[str, dict[str, Any]] = {}
    for row in artifact_rows:
        if not isinstance(row, Mapping) or set(row) != {
            "role", "path", "sha256", "artifact_hash"
        }:
            raise ValueError("G6 manifest artifact reference is invalid")
        role = str(row["role"])
        artifact_path = Path(str(row["path"])).expanduser().resolve()
        if file_sha256(artifact_path) != require_sha256(
            row.get("sha256"), f"{role} manifest sha256"
        ):
            raise ValueError(f"{role} artifact file hash drifted")
        _, artifact = load_mapping(artifact_path, f"{role} artifact")
        validated = validate_artifact_envelope(
            artifact,
            expected_role=role,
            revision=active_revision,
        )
        if validated["artifact_hash"] != row.get("artifact_hash"):
            raise ValueError(f"{role} artifact binding drifted")
        artifacts[role] = validated

    authoritative_scope = compute_product_review_scope(
        scope_repo_root,
        scope_manifest_path,
    )
    recomputed = {
        "product_review_scope": authoritative_scope,
        "planner_metrics": _recompute_planner_metrics(
            artifacts["planner_metrics"], active_revision
        ),
        "migration_cutoff": _recompute_migration_cutoff(
            artifacts["migration_cutoff"], active_revision
        ),
        "bilingual_documentation": _recompute_documentation(
            artifacts["bilingual_documentation"], active_revision
        ),
        "ignored_runtime_hygiene": _recompute_hygiene(
            artifacts["ignored_runtime_hygiene"], active_revision
        ),
        "external_capabilities": _recompute_external_capabilities(
            artifacts["external_capabilities"], active_revision
        ),
        "severity_ledger": _recompute_severity_ledger(
            artifacts["severity_ledger"],
            active_revision,
            expected_source_ids=authoritative_scope["severity"][
                "source_ids"
            ],
        ),
        "linux_shell_gates": _recompute_shell_gates(
            artifacts["linux_shell_gates"],
            active_revision,
            authoritative_scope,
        ),
        "fixture_provenance": dict(
            authoritative_scope["fixture_provenance"]
        ),
    }
    for role, payload in recomputed.items():
        if dict(artifacts[role]["payload"]) != payload:
            raise ValueError(f"{role} payload differs from independent recomputation")

    failed_checks: list[str] = []
    _validate_scoped_membership(
        recomputed,
        authoritative_scope,
        Path(scope_repo_root).expanduser().resolve(),
    )
    migration = recomputed["migration_cutoff"]
    if (
        migration["accepted_legacy_fixture_count"] != 0
        or any(row["outcome"] != "passed" for row in migration["checks"])
        or migration["minimum_supported_schema_version"]
        > migration["current_schema_version"]
    ):
        failed_checks.append("migration_cutoff")

    if recomputed["linux_shell_gates"]["execution"].get("status") != "passed":
        failed_checks.append("linux_shell_gates")
    fixture_provenance = recomputed["fixture_provenance"]
    if (
        fixture_provenance["runtime_presence"] != "complete"
        or fixture_provenance["workload_present"]
        != fixture_provenance["workload_denominator"]
        or fixture_provenance["strict_authenticity"]
        not in {"complete", "scoped_limitation"}
    ):
        failed_checks.append("fixture_provenance")

    hygiene = recomputed["ignored_runtime_hygiene"]
    if any(
        not _runtime_path_is_allowed(
            path, hygiene["allowed_runtime_entries"], expected_kind="ignored"
        )
        for path in hygiene["observed_ignored_entries"]
    ):
        failed_checks.append("ignored_runtime_hygiene")
    declared_symlinks = {
        row["path"]
        for row in hygiene["allowed_runtime_entries"]
        if row["kind"] == "symlink"
    }
    if set(hygiene["observed_symlinks"]) != declared_symlinks:
        failed_checks.append("runtime_symlinks")
    if _unbound_ignored_inputs(hygiene):
        failed_checks.append("unbound_runtime_inputs")

    capabilities = recomputed["external_capabilities"]["capabilities"]
    if any(
        row["state"] not in {"available", "externally_blocked"}
        for row in capabilities
    ):
        failed_checks.append("external_capabilities")

    findings = recomputed["severity_ledger"]["findings"]
    if any(
        row["status"] == "open" and row["severity"] in {"S0", "S1"}
        for row in findings
    ):
        failed_checks.append("open_s0_s1")
    if any(
        row["severity"] == "S2"
        and row["status"] != "resolved"
        and not row["disposition"]
        for row in findings
    ):
        failed_checks.append("undisposed_s2")

    return {
        "schema_version": PRODUCT_REVIEW_SCHEMA_VERSION,
        "revision": active_revision,
        "manifest_path": str(path),
        "manifest_hash": manifest["manifest_hash"],
        "gate": "G6",
        "status": "passed" if not failed_checks else "failed",
        "failed_checks": sorted(failed_checks),
        "scoped_limitations": (
            ["fixture_strict_authenticity"]
            if fixture_provenance["strict_authenticity"]
            == "scoped_limitation"
            else []
        ),
        "artifact_hashes": {
            role: artifacts[role]["artifact_hash"] for role in ARTIFACT_ROLES
        },
        "scope_hash": authoritative_scope["scope_hash"],
        "denominators": {
            "documentation_pairs": authoritative_scope[
                "documentation"
            ]["pair_denominator"],
            "migration_checks": authoritative_scope[
                "migration"
            ]["check_denominator"],
            "external_capabilities": authoritative_scope[
                "external_capabilities"
            ]["denominator"],
            "severity_sources": authoritative_scope[
                "severity"
            ]["denominator"],
            "shell_gates": authoritative_scope[
                "shell_gates"
            ]["denominator"],
            "fixtures": authoritative_scope[
                "fixture_provenance"
            ]["fixture_denominator"],
            "workload_fixtures": authoritative_scope[
                "fixture_provenance"
            ]["workload_denominator"],
        },
    }


def _validate_scoped_membership(
    recomputed: Mapping[str, Mapping[str, Any]],
    scope: Mapping[str, Any],
    scope_root: Path,
) -> None:
    expected_pairs = {
        (row["en_path"], row["zh_path"])
        for row in scope["documentation"]["pairs"]
    }
    observed_pairs = {
        (
            str(Path(row["en_path"]).resolve().relative_to(scope_root)),
            str(Path(row["zh_path"]).resolve().relative_to(scope_root)),
        )
        for row in recomputed["bilingual_documentation"]["pairs"]
    }
    if observed_pairs != expected_pairs:
        raise ValueError("G6 bilingual documentation membership is incomplete")
    observed_facts = {
        str(fact["fact_id"]): {
            "en_path": str(Path(row["en_path"]).resolve().relative_to(scope_root)),
            "zh_path": str(Path(row["zh_path"]).resolve().relative_to(scope_root)),
            "en_marker": str(fact["en_marker"]),
            "zh_marker": str(fact["zh_marker"]),
        }
        for row in recomputed["bilingual_documentation"]["pairs"]
        for fact in row["facts"]
    }
    expected_facts = {
        str(row["fact_id"]): {
            "en_path": str(row["en_path"]),
            "zh_path": str(row["zh_path"]),
            "en_marker": str(row["en_marker"]),
            "zh_marker": str(row["zh_marker"]),
        }
        for row in scope["documentation"]["required_facts"]
    }
    if any(
        observed_facts.get(fact_id) != expected
        for fact_id, expected in expected_facts.items()
    ):
        raise ValueError(
            "G6 bilingual documentation facts differ from repository authority"
        )

    expected_checks = {
        row["check_id"] for row in scope["migration"]["checks"]
    }
    observed_checks = {
        row["check_id"] for row in recomputed["migration_cutoff"]["checks"]
    }
    if observed_checks != expected_checks:
        raise ValueError("G6 migration check membership is incomplete")
    if (
        recomputed["migration_cutoff"]["current_schema_version"]
        != scope["migration"]["current_schema_version"]
    ):
        raise ValueError("G6 migration schema authority drifted")

    expected_capabilities = {
        row["capability_id"]
        for row in scope["external_capabilities"]["capabilities"]
    }
    observed_capabilities = {
        row["capability_id"]
        for row in recomputed["external_capabilities"]["capabilities"]
    }
    if observed_capabilities != expected_capabilities:
        raise ValueError("G6 external capability membership is incomplete")


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    revision: Mapping[str, Any],
    generator_implementation_path: str | Path,
    validator_implementation_path: str | Path,
) -> None:
    fields = {
        "schema_version",
        "manifest_type",
        "revision",
        "generated_at",
        "generator",
        "validator",
        "artifacts",
        "manifest_hash",
    }
    if set(manifest) != fields:
        raise ValueError("G6 product-review manifest schema is invalid")
    if manifest.get("schema_version") != PRODUCT_REVIEW_SCHEMA_VERSION:
        raise ValueError("G6 product-review manifest schema is unsupported")
    if manifest.get("manifest_type") != "g6_product_review_evidence":
        raise ValueError("G6 product-review manifest type is invalid")
    if manifest.get("revision") != validated_revision(revision):
        raise ValueError("G6 product-review manifest is stale")
    require_text(manifest.get("generated_at"), "generated_at")
    expected_bindings = {
        "generator": {
            "id": PRODUCT_REVIEW_GENERATOR_ID,
            "version": PRODUCT_REVIEW_GENERATOR_VERSION,
            "implementation_hash": module_implementation_hash(
                generator_implementation_path
            ),
        },
        "validator": {
            "id": PRODUCT_REVIEW_VALIDATOR_ID,
            "version": PRODUCT_REVIEW_VALIDATOR_VERSION,
            "implementation_hash": module_implementation_hash(
                validator_implementation_path
            ),
        },
    }
    for role, expected in expected_bindings.items():
        if manifest.get(role) != expected:
            raise ValueError(f"G6 {role} implementation binding is stale")
    recorded = require_sha256(manifest.get("manifest_hash"), "manifest_hash")
    unsigned = dict(manifest)
    unsigned.pop("manifest_hash", None)
    if content_hash(unsigned) != recorded:
        raise ValueError("G6 product-review manifest hash drifted")


def _artifact_sources(
    artifact: Mapping[str, Any],
) -> tuple[Path, ...]:
    return tuple(
        Path(str(row["path"])).resolve()
        for row in artifact["source_evidence"]
    )


def _recompute_planner_metrics(
    artifact: Mapping[str, Any],
    revision: Mapping[str, Any],
) -> dict[str, Any]:
    receipts = []
    for path in _artifact_sources(artifact):
        _, source = validate_hashed_source(
            path,
            label="planner runtime receipt",
            expected_type="planner_runtime_receipt",
            revision=revision,
        )
        metrics = source.get("planner_metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError("planner runtime receipt lacks metrics")
        control_receipt = source.get("control_receipt")
        if not isinstance(control_receipt, Mapping):
            raise ValueError("planner runtime receipt lacks admission evidence")
        turn_index = source.get("turn_index")
        if isinstance(turn_index, bool) or not isinstance(turn_index, int):
            raise ValueError("planner runtime receipt turn_index is invalid")
        admitted, reason = validate_persisted_domain_control_receipt(
            control_receipt,
            turn_index=turn_index,
        )
        if not admitted or control_receipt.get("receipt_type") != "semantic_planner":
            raise ValueError(
                f"planner runtime receipt was not admitted: {reason or 'wrong type'}"
            )
        admitted_metrics = control_receipt.get("planner_metrics")
        if not isinstance(admitted_metrics, Mapping) or any(
            admitted_metrics.get(key) != metrics.get(key)
            for key in ("latency_ms", "model_calls", "prompt_bytes")
        ):
            raise ValueError("planner metrics differ from the admitted receipt")
        usage = metrics.get("token_usage")
        if not isinstance(usage, Mapping):
            raise ValueError("planner token availability is missing")
        availability = str(usage.get("availability") or "")
        if availability not in TOKEN_AVAILABILITY:
            raise ValueError("planner token availability is invalid")
        normalized_usage: dict[str, Any] = {"availability": availability}
        if availability == "supplied":
            provider_path, provider_usage = validate_hashed_source(
                require_text(
                    usage.get("provider_evidence_path"),
                    "provider token evidence path",
                ),
                label="provider token usage evidence",
                expected_type="provider_token_usage",
                revision=revision,
            )
            if provider_usage.get("receipt_id") != control_receipt.get("receipt_id"):
                raise ValueError("provider token usage targets a different receipt")
            counts = {
                key: _nonnegative_int(provider_usage.get(key), key)
                for key in _TOKEN_FIELDS
            }
            if counts["total_tokens"] != counts["prompt_tokens"] + counts["output_tokens"]:
                raise ValueError("planner token counts are inconsistent")
            normalized_usage.update(counts)
            normalized_usage["provider_evidence_path"] = str(provider_path)
            normalized_usage["provider_evidence_sha256"] = file_sha256(provider_path)
        elif any(key in usage for key in _TOKEN_FIELDS):
            raise ValueError("missing provider token usage cannot be represented as zero")
        receipts.append({
            "receipt_id": require_sha256(
                control_receipt.get("receipt_id"), "planner receipt_id"
            ),
            "latency_ms": _nonnegative_number(metrics.get("latency_ms"), "latency_ms"),
            "model_calls": _nonnegative_int(metrics.get("model_calls"), "model_calls"),
            "prompt_bytes": _nonnegative_int(metrics.get("prompt_bytes"), "prompt_bytes"),
            "token_usage": normalized_usage,
        })
    if not receipts:
        raise ValueError("planner metrics artifact has no receipts")
    availabilities = {row["token_usage"]["availability"] for row in receipts}
    usage_total: dict[str, Any]
    if availabilities == {"supplied"}:
        usage_total = {
            "availability": "supplied",
            **{
                key: sum(row["token_usage"][key] for row in receipts)
                for key in _TOKEN_FIELDS
            },
            "provider_evidence": sorted(
                [{
                    "path": row["token_usage"]["provider_evidence_path"],
                    "sha256": row["token_usage"]["provider_evidence_sha256"],
                }
                for row in receipts
                ],
                key=lambda item: item["path"],
            ),
        }
    else:
        usage_total = {
            "availability": (
                "unavailable" if "unavailable" in availabilities else "not_supplied"
            )
        }
    return {
        "receipt_count": len(receipts),
        "receipt_ids": sorted(row["receipt_id"] for row in receipts),
        "latency_ms_total": sum(row["latency_ms"] for row in receipts),
        "model_calls_total": sum(row["model_calls"] for row in receipts),
        "prompt_bytes_total": sum(row["prompt_bytes"] for row in receipts),
        "token_usage": usage_total,
    }


def _recompute_migration_cutoff(
    artifact: Mapping[str, Any],
    revision: Mapping[str, Any],
) -> dict[str, Any]:
    path, = _artifact_sources(artifact)
    _, source = validate_hashed_source(
        path,
        label="migration cutoff evidence",
        expected_type="migration_cutoff",
        revision=revision,
    )
    checks = source.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("migration cutoff checks are missing")
    normalized = []
    for row in checks:
        if not isinstance(row, Mapping):
            raise ValueError("migration cutoff check is invalid")
        normalized.append({
            "check_id": require_stable_id(row.get("check_id"), "migration check_id"),
            "outcome": require_text(row.get("outcome"), "migration outcome"),
            "evidence_path": str(Path(
                require_text(row.get("evidence_path"), "migration evidence path")
            ).resolve()),
            "evidence_sha256": file_sha256(
                require_text(row.get("evidence_path"), "migration evidence path")
            ),
        })
    return {
        "current_schema_version": _positive_int(
            source.get("current_schema_version"), "current_schema_version"
        ),
        "minimum_supported_schema_version": _positive_int(
            source.get("minimum_supported_schema_version"),
            "minimum_supported_schema_version",
        ),
        "legacy_fixture_count": _nonnegative_int(
            source.get("legacy_fixture_count"), "legacy_fixture_count"
        ),
        "accepted_legacy_fixture_count": _nonnegative_int(
            source.get("accepted_legacy_fixture_count"),
            "accepted_legacy_fixture_count",
        ),
        "cutoff_revision": require_text(source.get("cutoff_revision"), "cutoff_revision"),
        "checks": sorted(normalized, key=lambda row: row["check_id"]),
    }


def _recompute_documentation(
    artifact: Mapping[str, Any],
    revision: Mapping[str, Any],
) -> dict[str, Any]:
    path, = _artifact_sources(artifact)
    _, source = validate_hashed_source(
        path,
        label="bilingual documentation contract",
        expected_type="bilingual_documentation",
        revision=revision,
    )
    pairs = source.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("bilingual documentation pairs are missing")
    normalized = []
    for row in pairs:
        if not isinstance(row, Mapping):
            raise ValueError("bilingual documentation pair is invalid")
        en_path = Path(require_text(row.get("en_path"), "English path")).resolve()
        zh_path = Path(require_text(row.get("zh_path"), "Chinese path")).resolve()
        en_text = en_path.read_text(encoding="utf-8")
        zh_text = zh_path.read_text(encoding="utf-8")
        facts = row.get("facts")
        if not isinstance(facts, list) or not facts:
            raise ValueError("bilingual fact contract is missing")
        normalized_facts = []
        for fact in facts:
            if not isinstance(fact, Mapping):
                raise ValueError("bilingual fact contract is invalid")
            en_marker = require_text(fact.get("en_marker"), "English marker")
            zh_marker = require_text(fact.get("zh_marker"), "Chinese marker")
            if en_marker not in en_text or zh_marker not in zh_text:
                raise ValueError("bilingual documentation fact is absent")
            normalized_facts.append({
                "fact_id": require_stable_id(fact.get("fact_id"), "fact_id"),
                "en_marker": en_marker,
                "zh_marker": zh_marker,
            })
        normalized.append({
            "pair_id": require_stable_id(row.get("pair_id"), "pair_id"),
            "en_path": str(en_path),
            "en_sha256": file_sha256(en_path),
            "zh_path": str(zh_path),
            "zh_sha256": file_sha256(zh_path),
            "facts": sorted(normalized_facts, key=lambda item: item["fact_id"]),
        })
    return {"pairs": sorted(normalized, key=lambda row: row["pair_id"])}


def _recompute_hygiene(
    artifact: Mapping[str, Any],
    revision: Mapping[str, Any],
) -> dict[str, Any]:
    path, = _artifact_sources(artifact)
    _, source = validate_hashed_source(
        path,
        label="runtime hygiene inventory",
        expected_type="ignored_runtime_hygiene",
        revision=revision,
    )
    root = Path(require_text(source.get("repo_root"), "repo_root")).resolve()
    raw_allowed = source.get("allowed_runtime_entries")
    raw_declared = source.get("declared_inputs")
    if not isinstance(raw_allowed, list) or not isinstance(raw_declared, list):
        raise ValueError("runtime hygiene inventory is incomplete")
    allowed = []
    for row in raw_allowed:
        if not isinstance(row, Mapping):
            raise ValueError("runtime hygiene entry is invalid")
        relative = _safe_relative(row.get("path"), root)
        kind = str(row.get("kind") or "")
        if kind not in {"ignored_file", "ignored_directory", "symlink"}:
            raise ValueError("runtime hygiene kind is invalid")
        binding = Path(require_text(row.get("binding_path"), "binding_path")).resolve()
        allowed.append({
            "path": relative,
            "kind": kind,
            "reason": require_text(row.get("reason"), "runtime reason"),
            "binding_path": str(binding),
            "binding_sha256": file_sha256(binding),
        })
    declared = []
    for row in raw_declared:
        if not isinstance(row, Mapping):
            raise ValueError("declared runtime input is invalid")
        relative = _safe_relative(row.get("path"), root)
        kind = str(row.get("kind") or "")
        if kind not in {"executable", "configuration"}:
            raise ValueError("declared runtime input kind is invalid")
        declared.append({
            "path": relative,
            "kind": kind,
            "sha256": file_sha256(root / relative),
        })
    return {
        "repo_root": str(root),
        "allowed_runtime_entries": sorted(
            allowed, key=lambda row: (row["path"], row["kind"])
        ),
        "declared_inputs": sorted(
            declared, key=lambda row: (row["path"], row["kind"])
        ),
        "observed_ignored_entries": _ignored_entries(root),
        "observed_symlinks": _symlinks(root),
    }


def _recompute_external_capabilities(
    artifact: Mapping[str, Any],
    revision: Mapping[str, Any],
) -> dict[str, Any]:
    values = []
    for path in _artifact_sources(artifact):
        _, source = validate_hashed_source(
            path,
            label="external capability attestation",
            expected_type="external_capability",
            revision=revision,
        )
        state = str(source.get("state") or "")
        probe_path, probe = validate_hashed_source(
            require_text(source.get("probe_path"), "external probe path"),
            label="external capability probe",
            expected_type="external_capability_probe",
            revision=revision,
        )
        capability_id = require_stable_id(
            source.get("capability_id"), "capability_id"
        )
        if probe.get("capability_id") != capability_id:
            raise ValueError("external probe targets a different capability")
        outcome = str(probe.get("outcome") or "")
        if state not in EXTERNAL_CAPABILITY_STATES:
            raise ValueError("external capability state is invalid")
        expected_outcome = {
            "available": "passed",
            "unavailable": "unavailable",
            "externally_blocked": "externally_blocked",
        }[state]
        if outcome != expected_outcome:
            raise ValueError("external capability state is forged")
        reason = str(probe.get("reason") or "")
        if state != "available" and not reason.strip():
            raise ValueError("external capability lacks blocking evidence")
        values.append({
            "capability_id": capability_id,
            "state": state,
            "probe_id": require_stable_id(probe.get("probe_id"), "probe_id"),
            "probe_outcome": outcome,
            "probe_path": str(probe_path),
            "evidence_sha256": file_sha256(probe_path),
            "reason": reason,
        })
    if not values:
        raise ValueError("external capability evidence is absent")
    return {"capabilities": sorted(values, key=lambda row: row["capability_id"])}


def _recompute_severity_ledger(
    artifact: Mapping[str, Any],
    revision: Mapping[str, Any],
    *,
    expected_source_ids: Sequence[str],
) -> dict[str, Any]:
    path, = _artifact_sources(artifact)
    _, source = validate_hashed_source(
        path,
        label="severity ledger",
        expected_type="severity_ledger",
        revision=revision,
    )
    raw_findings = source.get("findings")
    if not isinstance(raw_findings, list):
        raise ValueError("severity ledger findings are invalid")
    raw_reviews = source.get("source_reviews")
    if not isinstance(raw_reviews, list):
        raise ValueError("severity source reviews are invalid")
    source_reviews = []
    for row in raw_reviews:
        if not isinstance(row, Mapping):
            raise ValueError("severity source review is invalid")
        evidence_paths = row.get("evidence_paths")
        finding_ids = row.get("finding_ids")
        if (
            not isinstance(evidence_paths, list)
            or not evidence_paths
            or not isinstance(finding_ids, list)
            or any(
                not isinstance(item, str) or not item.strip()
                for item in finding_ids
            )
        ):
            raise ValueError("severity source review is incomplete")
        source_reviews.append({
            "source_id": require_stable_id(
                row.get("source_id"), "severity source_id"
            ),
            "finding_ids": sorted(str(item) for item in finding_ids),
            "evidence": sorted(
                (
                    {
                        "path": str(Path(require_text(
                            value, "severity source evidence path"
                        )).resolve()),
                        "sha256": file_sha256(value),
                    }
                    for value in evidence_paths
                ),
                key=lambda item: item["path"],
            ),
        })
    observed_sources = [row["source_id"] for row in source_reviews]
    if (
        len(observed_sources) != len(set(observed_sources))
        or set(observed_sources) != set(expected_source_ids)
    ):
        raise ValueError("severity source review membership is incomplete")
    findings = []
    ids: set[str] = set()
    for row in raw_findings:
        if not isinstance(row, Mapping):
            raise ValueError("severity finding is invalid")
        finding_id = require_stable_id(row.get("finding_id"), "finding_id")
        if finding_id in ids:
            raise ValueError(f"duplicate severity finding: {finding_id}")
        ids.add(finding_id)
        severity = str(row.get("severity") or "")
        status = str(row.get("status") or "")
        if severity not in FINDING_SEVERITIES or status not in FINDING_STATUSES:
            raise ValueError(f"severity finding classification is invalid: {finding_id}")
        paths = row.get("evidence_paths")
        if not isinstance(paths, list) or not paths:
            raise ValueError(f"severity finding evidence is absent: {finding_id}")
        fix_revision = str(row.get("fix_revision") or "").strip()
        if status == "resolved" and not fix_revision:
            raise ValueError(f"resolved finding lacks fix revision: {finding_id}")
        findings.append({
            "finding_id": finding_id,
            "severity": severity,
            "owner": require_text(row.get("owner"), "finding owner"),
            "status": status,
            "disposition": str(row.get("disposition") or "").strip(),
            "root_class": require_stable_id(row.get("root_class"), "root_class"),
            "discovery_revision": require_text(
                row.get("discovery_revision"), "discovery_revision"
            ),
            "fix_revision": fix_revision,
            "evidence": sorted(
                (
                    {
                        "path": str(Path(require_text(
                            value, "finding evidence path"
                        )).resolve()),
                        "sha256": file_sha256(value),
                    }
                    for value in paths
                ),
                key=lambda item: item["path"],
            ),
        })
    reviewed_findings = [
        finding_id
        for row in source_reviews
        for finding_id in row["finding_ids"]
    ]
    finding_ids = [row["finding_id"] for row in findings]
    if (
        len(reviewed_findings) != len(set(reviewed_findings))
        or set(reviewed_findings) != set(finding_ids)
    ):
        raise ValueError("severity source reviews do not cover the finding ledger")
    return {
        "source_reviews": sorted(
            source_reviews, key=lambda row: row["source_id"]
        ),
        "findings": sorted(findings, key=lambda row: row["finding_id"]),
    }


def _recompute_shell_gates(
    artifact: Mapping[str, Any],
    revision: Mapping[str, Any],
    scope: Mapping[str, Any],
) -> dict[str, Any]:
    sources = _artifact_sources(artifact)
    if len(sources) != 2:
        raise ValueError("Linux shell-gate artifact source set is incomplete")
    receipt_path = next(
        (
            path
            for path in sources
            if path.name != "product_review_scope.json"
        ),
        None,
    )
    if receipt_path is None:
        raise ValueError("Linux shell-gate receipt is missing")
    _, source = validate_hashed_source(
        receipt_path,
        label="Linux shell-gate execution receipt",
        expected_type="linux_shell_gate_receipt",
        revision=revision,
    )
    execution = source.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("Linux shell-gate receipt lacks execution results")
    membership = dict(scope["shell_gates"])
    if (
        execution.get("manifest") != membership["manifest"]
        or execution.get("manifest_sha256")
        != membership["manifest_sha256"]
        or execution.get("classified_denominator") != membership["denominator"]
    ):
        raise ValueError("Linux shell-gate execution membership is stale")
    return {"membership": membership, "execution": dict(execution)}


def _unbound_ignored_inputs(hygiene: Mapping[str, Any]) -> list[str]:
    ignored = set(hygiene["observed_ignored_entries"])
    declared = {row["path"] for row in hygiene["declared_inputs"]}
    config_suffixes = {".env", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"}
    unbound = []
    root = Path(hygiene["repo_root"])
    for relative in ignored:
        path = root / relative
        executable = path.is_file() and os.access(path, os.X_OK)
        configuration = path.suffix.lower() in config_suffixes
        if (executable or configuration) and relative not in declared:
            entry = next(
                (
                    row
                    for row in hygiene["allowed_runtime_entries"]
                    if _runtime_path_is_allowed(
                        relative, (row,), expected_kind="ignored"
                    )
                ),
                None,
            )
            if not entry or not entry.get("binding_sha256"):
                unbound.append(relative)
    return sorted(unbound)


def _runtime_path_is_allowed(
    relative: str,
    entries: Sequence[Mapping[str, Any]],
    *,
    expected_kind: str,
) -> bool:
    for row in entries:
        kind = str(row.get("kind") or "")
        declared = str(row.get("path") or "").rstrip("/")
        if expected_kind == "ignored":
            if kind == "ignored_file" and relative == declared:
                return True
            if kind == "ignored_directory" and (
                relative == declared or relative.startswith(f"{declared}/")
            ):
                return True
        elif expected_kind == "symlink" and kind == "symlink" and relative == declared:
            return True
    return False


def _ignored_entries(root: Path) -> list[str]:
    completed = subprocess.run(
        ("git", "-C", str(root), "ls-files", "--others", "--ignored",
         "--exclude-standard", "-z"),
        capture_output=True,
        check=True,
    )
    return sorted(
        item.decode("utf-8")
        for item in completed.stdout.split(b"\0")
        if item
    )


def _symlinks(root: Path) -> list[str]:
    values = []
    for directory, names, files in os.walk(root, followlinks=False):
        base = Path(directory)
        if base == root / ".git":
            names[:] = []
            continue
        for name in tuple(names) + tuple(files):
            path = base / name
            if path.is_symlink():
                values.append(str(path.relative_to(root)))
    return sorted(values)


def _safe_relative(value: Any, root: Path) -> str:
    relative = require_text(value, "runtime path")
    resolved = (root / relative).resolve()
    if root not in resolved.parents and resolved != root:
        raise ValueError("runtime path escapes repo root")
    return relative


def _nonnegative_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if number < 0 or number != number or number in {float("inf"), float("-inf")}:
        raise ValueError(f"{label} must be finite and nonnegative")
    return number


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _positive_int(value: Any, label: str) -> int:
    result = _nonnegative_int(value, label)
    if result == 0:
        raise ValueError(f"{label} must be positive")
    return result


def validator_imports_generator() -> bool:
    """Architecture assertion used by focused tests and external audits."""

    tree = ast.parse(VALIDATOR_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                alias.name.endswith("generate_product_review_evidence")
                for alias in node.names
            ):
                return True
        elif isinstance(node, ast.ImportFrom):
            if str(node.module or "").endswith("generate_product_review_evidence"):
                return True
    return False


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--revision", required=True, help="JSON revision object")
    args = parser.parse_args(argv)
    result = validate_product_review_evidence(
        args.manifest,
        revision=json.loads(args.revision),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Generate typed G6 evidence without making a product-readiness decision."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
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
    build_artifact,
    content_hash,
    file_sha256,
    module_implementation_hash,
    require_sha256,
    require_stable_id,
    require_text,
    validate_hashed_source,
    validated_revision,
    write_json,
)
from tests.agent_live.product_review_scope import (
    DEFAULT_SCOPE_MANIFEST,
    compute_product_review_scope,
)


GENERATOR_PATH = Path(__file__).resolve()
VALIDATOR_PATH = GENERATOR_PATH.with_name("validate_product_review_evidence.py")


def generate_product_review_evidence(
    *,
    output_dir: str | Path,
    revision: Mapping[str, Any],
    planner_receipt_paths: Sequence[str | Path],
    migration_cutoff_path: str | Path,
    bilingual_docs_path: str | Path,
    runtime_hygiene_path: str | Path,
    external_capability_paths: Sequence[str | Path],
    severity_ledger_path: str | Path,
    shell_gate_receipt_path: str | Path,
    scope_repo_root: str | Path = REPO_ROOT,
    scope_manifest_path: str | Path = DEFAULT_SCOPE_MANIFEST,
    generated_at: str | None = None,
    generator_implementation_path: str | Path = GENERATOR_PATH,
    validator_implementation_path: str | Path = VALIDATOR_PATH,
) -> Path:
    """Publish review evidence; intentionally returns no PASS/FAIL decision."""

    active_revision = validated_revision(revision)
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"G6 evidence destination already exists: {destination}")
    destination.mkdir(parents=True)
    scope = compute_product_review_scope(
        scope_repo_root,
        scope_manifest_path,
    )
    scope_manifest = Path(scope_manifest_path).expanduser().resolve()

    artifact_inputs = {
        "product_review_scope": (
            (scope_manifest,),
            scope,
        ),
        "planner_metrics": (
            tuple(planner_receipt_paths),
            _planner_metrics_payload(planner_receipt_paths, active_revision),
        ),
        "migration_cutoff": (
            (migration_cutoff_path,),
            _migration_payload(migration_cutoff_path, active_revision),
        ),
        "bilingual_documentation": (
            (bilingual_docs_path,),
            _documentation_payload(bilingual_docs_path, active_revision),
        ),
        "ignored_runtime_hygiene": (
            (runtime_hygiene_path,),
            _hygiene_payload(runtime_hygiene_path, active_revision),
        ),
        "external_capabilities": (
            tuple(external_capability_paths),
            _external_payload(external_capability_paths, active_revision),
        ),
        "severity_ledger": (
            (severity_ledger_path,),
            _severity_payload(
                severity_ledger_path,
                active_revision,
                expected_source_ids=scope["severity"]["source_ids"],
            ),
        ),
        "linux_shell_gates": (
            (scope_manifest, shell_gate_receipt_path),
            _shell_gate_payload(
                shell_gate_receipt_path,
                active_revision,
                scope,
            ),
        ),
        "fixture_provenance": (
            (
                scope_manifest,
                *(
                    Path(scope_repo_root).expanduser().resolve() / row["path"]
                    for row in scope["fixture_provenance"]["fixture_inventory"]
                ),
            ),
            dict(scope["fixture_provenance"]),
        ),
    }
    artifacts: list[dict[str, str]] = []
    for role in ARTIFACT_ROLES:
        sources, payload = artifact_inputs[role]
        artifact = build_artifact(
            role=role,
            revision=active_revision,
            sources=sources,
            payload=payload,
        )
        artifact_path = write_json(destination / f"{role}.json", artifact)
        artifacts.append({
            "role": role,
            "path": str(artifact_path),
            "sha256": file_sha256(artifact_path),
            "artifact_hash": str(artifact["artifact_hash"]),
        })

    unsigned = {
        "schema_version": PRODUCT_REVIEW_SCHEMA_VERSION,
        "manifest_type": "g6_product_review_evidence",
        "revision": active_revision,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
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
        "artifacts": artifacts,
    }
    manifest = {**unsigned, "manifest_hash": content_hash(unsigned)}
    return write_json(destination / "manifest.json", manifest)


def _planner_metrics_payload(
    paths: Sequence[str | Path],
    revision: Mapping[str, Any],
) -> dict[str, Any]:
    if not paths:
        raise ValueError("planner metrics require runtime receipts")
    receipts: list[dict[str, Any]] = []
    for path in paths:
        _, source = validate_hashed_source(
            path,
            label="planner runtime receipt",
            expected_type="planner_runtime_receipt",
            revision=revision,
        )
        metrics = source.get("planner_metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError("planner runtime receipt is missing planner_metrics")
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
        token_usage = metrics.get("token_usage")
        if not isinstance(token_usage, Mapping):
            raise ValueError("planner receipt token availability is missing")
        availability = str(token_usage.get("availability") or "")
        if availability not in TOKEN_AVAILABILITY:
            raise ValueError("planner receipt token availability is invalid")
        normalized_usage: dict[str, Any] = {"availability": availability}
        if availability == "supplied":
            provider_path, provider_usage = validate_hashed_source(
                require_text(
                    token_usage.get("provider_evidence_path"),
                    "provider token evidence path",
                ),
                label="provider token usage evidence",
                expected_type="provider_token_usage",
                revision=revision,
            )
            if provider_usage.get("receipt_id") != control_receipt.get("receipt_id"):
                raise ValueError("provider token usage targets a different receipt")
            counts = _token_counts(provider_usage)
            normalized_usage.update(counts)
            normalized_usage["provider_evidence_path"] = str(provider_path)
            normalized_usage["provider_evidence_sha256"] = file_sha256(provider_path)
        elif any(key in token_usage for key in _TOKEN_COUNT_FIELDS):
            raise ValueError("unavailable token usage cannot contain zero counts")
        receipts.append({
            "receipt_id": require_sha256(
                control_receipt.get("receipt_id"), "planner receipt_id"
            ),
            "latency_ms": _nonnegative_number(metrics.get("latency_ms"), "latency_ms"),
            "model_calls": _nonnegative_int(metrics.get("model_calls"), "model_calls"),
            "prompt_bytes": _nonnegative_int(metrics.get("prompt_bytes"), "prompt_bytes"),
            "token_usage": normalized_usage,
        })
    availabilities = {row["token_usage"]["availability"] for row in receipts}
    aggregate_usage: dict[str, Any]
    if availabilities == {"supplied"}:
        aggregate_usage = {
            "availability": "supplied",
            **{
                key: sum(int(row["token_usage"][key]) for row in receipts)
                for key in _TOKEN_COUNT_FIELDS
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
        aggregate_usage = {
            "availability": (
                "unavailable"
                if "unavailable" in availabilities
                else "not_supplied"
            )
        }
    return {
        "receipt_count": len(receipts),
        "receipt_ids": sorted(row["receipt_id"] for row in receipts),
        "latency_ms_total": sum(float(row["latency_ms"]) for row in receipts),
        "model_calls_total": sum(int(row["model_calls"]) for row in receipts),
        "prompt_bytes_total": sum(int(row["prompt_bytes"]) for row in receipts),
        "token_usage": aggregate_usage,
    }


def _migration_payload(path: str | Path, revision: Mapping[str, Any]) -> dict[str, Any]:
    _, source = validate_hashed_source(
        path,
        label="migration cutoff evidence",
        expected_type="migration_cutoff",
        revision=revision,
    )
    checks = source.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("migration cutoff evidence requires checks")
    normalized_checks = []
    for row in checks:
        if not isinstance(row, Mapping):
            raise ValueError("migration cutoff check must be an object")
        normalized_checks.append({
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
        "cutoff_revision": require_text(
            source.get("cutoff_revision"), "cutoff_revision"
        ),
        "checks": sorted(normalized_checks, key=lambda row: row["check_id"]),
    }


def _documentation_payload(path: str | Path, revision: Mapping[str, Any]) -> dict[str, Any]:
    _, source = validate_hashed_source(
        path,
        label="bilingual documentation contract",
        expected_type="bilingual_documentation",
        revision=revision,
    )
    pairs = source.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("bilingual documentation requires explicit pairs")
    normalized = []
    for row in pairs:
        if not isinstance(row, Mapping):
            raise ValueError("documentation pair must be an object")
        facts = row.get("facts")
        if not isinstance(facts, list) or not facts:
            raise ValueError("documentation pair requires fact markers")
        en_path = Path(require_text(row.get("en_path"), "English document path")).resolve()
        zh_path = Path(require_text(row.get("zh_path"), "Chinese document path")).resolve()
        en_text = en_path.read_text(encoding="utf-8")
        zh_text = zh_path.read_text(encoding="utf-8")
        normalized_facts = []
        for fact in facts:
            if not isinstance(fact, Mapping):
                raise ValueError("documentation fact must be an object")
            en_marker = require_text(fact.get("en_marker"), "English fact marker")
            zh_marker = require_text(fact.get("zh_marker"), "Chinese fact marker")
            if en_marker not in en_text or zh_marker not in zh_text:
                raise ValueError("bilingual documentation fact is missing")
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


def _hygiene_payload(path: str | Path, revision: Mapping[str, Any]) -> dict[str, Any]:
    _, source = validate_hashed_source(
        path,
        label="runtime hygiene inventory",
        expected_type="ignored_runtime_hygiene",
        revision=revision,
    )
    root = Path(require_text(source.get("repo_root"), "repo_root")).resolve()
    allowed = _normalized_hygiene_entries(source.get("allowed_runtime_entries"), root)
    declared = _normalized_declared_inputs(source.get("declared_inputs"), root)
    return {
        "repo_root": str(root),
        "allowed_runtime_entries": allowed,
        "declared_inputs": declared,
        "observed_ignored_entries": _ignored_entries(root),
        "observed_symlinks": _symlinks(root),
    }


def _external_payload(
    paths: Sequence[str | Path],
    revision: Mapping[str, Any],
) -> dict[str, Any]:
    if not paths:
        raise ValueError("external capabilities require typed probe evidence")
    capabilities = []
    for path in paths:
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
        probe_outcome = str(probe.get("outcome") or "")
        if state not in EXTERNAL_CAPABILITY_STATES:
            raise ValueError("external capability state is invalid")
        expected = {
            "available": "passed",
            "unavailable": "unavailable",
            "externally_blocked": "externally_blocked",
        }[state]
        if probe_outcome != expected:
            raise ValueError("external capability state contradicts its probe")
        reason = str(probe.get("reason") or "")
        if state != "available" and not reason.strip():
            raise ValueError("unavailable external capability requires a reason")
        capabilities.append({
            "capability_id": capability_id,
            "state": state,
            "probe_id": require_stable_id(probe.get("probe_id"), "probe_id"),
            "probe_outcome": probe_outcome,
            "probe_path": str(probe_path),
            "evidence_sha256": file_sha256(probe_path),
            "reason": reason,
        })
    return {"capabilities": sorted(capabilities, key=lambda row: row["capability_id"])}


def _severity_payload(
    path: str | Path,
    revision: Mapping[str, Any],
    *,
    expected_source_ids: Sequence[str],
) -> dict[str, Any]:
    _, source = validate_hashed_source(
        path,
        label="severity ledger",
        expected_type="severity_ledger",
        revision=revision,
    )
    findings = source.get("findings")
    if not isinstance(findings, list):
        raise ValueError("severity ledger findings must be a list")
    source_reviews = _severity_source_reviews(
        source.get("source_reviews"),
        expected_source_ids=expected_source_ids,
    )
    normalized = []
    seen: set[str] = set()
    for row in findings:
        if not isinstance(row, Mapping):
            raise ValueError("severity finding must be an object")
        finding_id = require_stable_id(row.get("finding_id"), "finding_id")
        if finding_id in seen:
            raise ValueError(f"duplicate finding id: {finding_id}")
        seen.add(finding_id)
        severity = str(row.get("severity") or "")
        status = str(row.get("status") or "")
        if severity not in FINDING_SEVERITIES or status not in FINDING_STATUSES:
            raise ValueError(f"finding classification is invalid: {finding_id}")
        evidence_paths = row.get("evidence_paths")
        if not isinstance(evidence_paths, list) or not evidence_paths:
            raise ValueError(f"finding evidence is missing: {finding_id}")
        normalized.append({
            "finding_id": finding_id,
            "severity": severity,
            "owner": require_text(row.get("owner"), "finding owner"),
            "status": status,
            "disposition": str(row.get("disposition") or "").strip(),
            "root_class": require_stable_id(row.get("root_class"), "root_class"),
            "discovery_revision": require_text(
                row.get("discovery_revision"), "discovery_revision"
            ),
            "fix_revision": str(row.get("fix_revision") or "").strip(),
            "evidence": sorted(
                (
                    {
                        "path": str(Path(require_text(
                            value, "finding evidence path"
                        )).resolve()),
                        "sha256": file_sha256(value),
                    }
                    for value in evidence_paths
                ),
                key=lambda item: item["path"],
            ),
        })
    reviewed_findings = [
        finding_id
        for row in source_reviews
        for finding_id in row["finding_ids"]
    ]
    finding_ids = [row["finding_id"] for row in normalized]
    if (
        len(reviewed_findings) != len(set(reviewed_findings))
        or set(reviewed_findings) != set(finding_ids)
    ):
        raise ValueError("severity source reviews do not cover the finding ledger")
    return {
        "source_reviews": source_reviews,
        "findings": sorted(normalized, key=lambda row: row["finding_id"]),
    }


def _severity_source_reviews(
    value: Any,
    *,
    expected_source_ids: Sequence[str],
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("severity source reviews must be a list")
    reviews: list[dict[str, Any]] = []
    for row in value:
        if not isinstance(row, Mapping):
            raise ValueError("severity source review must be an object")
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
        reviews.append({
            "source_id": require_stable_id(
                row.get("source_id"), "severity source_id"
            ),
            "finding_ids": sorted(str(item) for item in finding_ids),
            "evidence": sorted(
                (
                    {
                        "path": str(Path(path).expanduser().resolve()),
                        "sha256": file_sha256(path),
                    }
                    for path in evidence_paths
                ),
                key=lambda item: item["path"],
            ),
        })
    observed = [row["source_id"] for row in reviews]
    if (
        len(observed) != len(set(observed))
        or set(observed) != set(expected_source_ids)
    ):
        raise ValueError("severity source review membership is incomplete")
    return sorted(reviews, key=lambda row: row["source_id"])


def _shell_gate_payload(
    path: str | Path,
    revision: Mapping[str, Any],
    scope: Mapping[str, Any],
) -> dict[str, Any]:
    _, source = validate_hashed_source(
        path,
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
        raise ValueError("Linux shell-gate receipt has stale membership")
    return {
        "membership": membership,
        "execution": dict(execution),
    }


_TOKEN_COUNT_FIELDS = ("prompt_tokens", "output_tokens", "total_tokens")


def _token_counts(value: Mapping[str, Any]) -> dict[str, int]:
    counts = {
        key: _nonnegative_int(value.get(key), key)
        for key in _TOKEN_COUNT_FIELDS
    }
    if counts["total_tokens"] != counts["prompt_tokens"] + counts["output_tokens"]:
        raise ValueError("provider token totals are inconsistent")
    return counts


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
    if result < 1:
        raise ValueError(f"{label} must be positive")
    return result


def _normalized_hygiene_entries(value: Any, root: Path) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError("allowed_runtime_entries must be a list")
    rows = []
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ValueError("runtime hygiene entry must be an object")
        relative = require_text(raw.get("path"), "runtime path")
        resolved = (root / relative).resolve()
        if root not in resolved.parents and resolved != root:
            raise ValueError("runtime hygiene path escapes repo root")
        kind = str(raw.get("kind") or "")
        if kind not in {"ignored_file", "ignored_directory", "symlink"}:
            raise ValueError("runtime hygiene kind is invalid")
        binding = Path(
            require_text(raw.get("binding_path"), "runtime binding path")
        ).resolve()
        rows.append({
            "path": relative,
            "kind": kind,
            "reason": require_text(raw.get("reason"), "runtime reason"),
            "binding_path": str(binding),
            "binding_sha256": file_sha256(binding),
        })
    return sorted(rows, key=lambda row: (row["path"], row["kind"]))


def _normalized_declared_inputs(value: Any, root: Path) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError("declared_inputs must be a list")
    rows = []
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ValueError("declared input must be an object")
        relative = require_text(raw.get("path"), "declared input path")
        resolved = (root / relative).resolve()
        if root not in resolved.parents:
            raise ValueError("declared input path escapes repo root")
        kind = str(raw.get("kind") or "")
        if kind not in {"executable", "configuration"}:
            raise ValueError("declared input kind is invalid")
        rows.append({
            "path": relative,
            "kind": kind,
            "sha256": file_sha256(resolved),
        })
    return sorted(rows, key=lambda row: (row["path"], row["kind"]))


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


def _load_config(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("G6 generator config must be an object")
    return dict(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    config = _load_config(args.config)
    manifest = generate_product_review_evidence(
        output_dir=args.output_dir,
        revision=config["revision"],
        planner_receipt_paths=config["planner_receipt_paths"],
        migration_cutoff_path=config["migration_cutoff_path"],
        bilingual_docs_path=config["bilingual_docs_path"],
        runtime_hygiene_path=config["runtime_hygiene_path"],
        external_capability_paths=config["external_capability_paths"],
        severity_ledger_path=config["severity_ledger_path"],
        shell_gate_receipt_path=config["shell_gate_receipt_path"],
        scope_repo_root=config.get("scope_repo_root", REPO_ROOT),
        scope_manifest_path=config.get(
            "scope_manifest_path",
            DEFAULT_SCOPE_MANIFEST,
        ),
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

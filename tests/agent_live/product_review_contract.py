"""Typed, revision-bound contracts for independent G6 product review.

The generator and validator share only schemas, canonical serialization, and
cryptographic primitives from this module.  Gate policy belongs exclusively
to the independent validator.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


PRODUCT_REVIEW_SCHEMA_VERSION = 2
PRODUCT_REVIEW_GENERATOR_ID = "anychain-g6-product-review-generator"
PRODUCT_REVIEW_GENERATOR_VERSION = 1
PRODUCT_REVIEW_VALIDATOR_ID = "anychain-g6-product-review-validator"
PRODUCT_REVIEW_VALIDATOR_VERSION = 1

ARTIFACT_ROLES = (
    "product_review_scope",
    "planner_metrics",
    "migration_cutoff",
    "bilingual_documentation",
    "ignored_runtime_hygiene",
    "external_capabilities",
    "severity_ledger",
    "linux_shell_gates",
    "fixture_provenance",
)
TOKEN_AVAILABILITY = frozenset({"supplied", "not_supplied", "unavailable"})
EXTERNAL_CAPABILITY_STATES = frozenset({
    "available",
    "unavailable",
    "externally_blocked",
})
FINDING_SEVERITIES = frozenset({"S0", "S1", "S2", "S3"})
FINDING_STATUSES = frozenset({"open", "resolved", "accepted_risk"})

_HEX_RE = re.compile(r"^[0-9a-f]+$")
_STABLE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"evidence file does not exist: {resolved}")
    return hashlib.sha256(resolved.read_bytes()).hexdigest()


def module_implementation_hash(path: str | Path) -> str:
    return file_sha256(path)


def validated_revision(value: Mapping[str, Any]) -> dict[str, str]:
    if set(value) != {"commit", "worktree_hash"}:
        raise ValueError("revision must contain only commit and worktree_hash")
    commit = str(value.get("commit") or "").lower()
    worktree_hash = str(value.get("worktree_hash") or "").lower()
    if not 7 <= len(commit) <= 64 or not _HEX_RE.fullmatch(commit):
        raise ValueError("revision commit is not a Git object id")
    if len(worktree_hash) != 64 or not _HEX_RE.fullmatch(worktree_hash):
        raise ValueError("revision worktree_hash is not SHA-256")
    return {"commit": commit, "worktree_hash": worktree_hash}


def is_sha256(value: Any) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and bool(_HEX_RE.fullmatch(text))


def require_sha256(value: Any, label: str) -> str:
    text = str(value or "").lower()
    if not is_sha256(text):
        raise ValueError(f"{label} is not SHA-256")
    return text


def require_text(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} is required")
    return text


def require_stable_id(value: Any, label: str) -> str:
    text = require_text(value, label)
    if not _STABLE_ID_RE.fullmatch(text):
        raise ValueError(f"{label} is not a stable identifier")
    return text


def load_mapping(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} does not exist: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {resolved}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain one JSON object")
    return resolved, dict(value)


def validate_hashed_source(
    path: str | Path,
    *,
    label: str,
    expected_type: str,
    revision: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    resolved, value = load_mapping(path, label)
    if value.get("schema_version") != PRODUCT_REVIEW_SCHEMA_VERSION:
        raise ValueError(f"{label} has an unsupported schema")
    if value.get("source_type") != expected_type:
        raise ValueError(f"{label} has the wrong source type")
    if value.get("revision") != validated_revision(revision):
        raise ValueError(f"{label} is bound to a stale revision")
    recorded = require_sha256(value.get("source_hash"), f"{label} source_hash")
    unsigned = dict(value)
    unsigned.pop("source_hash", None)
    if content_hash(unsigned) != recorded:
        raise ValueError(f"{label} source hash drifted")
    return resolved, value


def source_reference(path: str | Path) -> dict[str, str]:
    resolved = Path(path).expanduser().resolve()
    return {"path": str(resolved), "sha256": file_sha256(resolved)}


def validate_source_references(
    values: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], ...]:
    if not values:
        raise ValueError("artifact source evidence cannot be empty")
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, raw in enumerate(values):
        if set(raw) != {"path", "sha256"}:
            raise ValueError(f"source reference {index} has an invalid schema")
        path = str(Path(require_text(raw.get("path"), "source path")).resolve())
        digest = require_sha256(raw.get("sha256"), "source sha256")
        if path in seen:
            raise ValueError(f"duplicate source evidence path: {path}")
        seen.add(path)
        if file_sha256(path) != digest:
            raise ValueError(f"source evidence hash drifted: {path}")
        normalized.append({"path": path, "sha256": digest})
    return tuple(normalized)


def build_artifact(
    *,
    role: str,
    revision: Mapping[str, Any],
    sources: Sequence[str | Path],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if role not in ARTIFACT_ROLES:
        raise ValueError(f"unknown product-review artifact role: {role}")
    references = [source_reference(path) for path in sources]
    unsigned = {
        "schema_version": PRODUCT_REVIEW_SCHEMA_VERSION,
        "artifact_type": "g6_product_review_artifact",
        "role": role,
        "revision": validated_revision(revision),
        "source_evidence": references,
        "payload": dict(payload),
    }
    return {**unsigned, "artifact_hash": content_hash(unsigned)}


def validate_artifact_envelope(
    value: Mapping[str, Any],
    *,
    expected_role: str,
    revision: Mapping[str, Any],
) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "artifact_type",
        "role",
        "revision",
        "source_evidence",
        "payload",
        "artifact_hash",
    }
    if set(value) != expected_fields:
        raise ValueError(f"{expected_role} artifact has an invalid schema")
    if value.get("schema_version") != PRODUCT_REVIEW_SCHEMA_VERSION:
        raise ValueError(f"{expected_role} artifact schema is unsupported")
    if value.get("artifact_type") != "g6_product_review_artifact":
        raise ValueError(f"{expected_role} artifact type is invalid")
    if value.get("role") != expected_role:
        raise ValueError(f"{expected_role} artifact role drifted")
    if value.get("revision") != validated_revision(revision):
        raise ValueError(f"{expected_role} artifact is stale")
    if not isinstance(value.get("payload"), Mapping):
        raise ValueError(f"{expected_role} artifact payload must be an object")
    references = value.get("source_evidence")
    if not isinstance(references, list):
        raise ValueError(f"{expected_role} artifact sources must be a list")
    validate_source_references(references)
    recorded = require_sha256(
        value.get("artifact_hash"),
        f"{expected_role} artifact_hash",
    )
    unsigned = dict(value)
    unsigned.pop("artifact_hash", None)
    if content_hash(unsigned) != recorded:
        raise ValueError(f"{expected_role} artifact hash drifted")
    return dict(value)


def write_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination

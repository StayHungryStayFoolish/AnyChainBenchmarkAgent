"""Repository-owned membership authority for the G6 product review."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

from agent.harness.state import STATE_SCHEMA_VERSION
from tests.agent_live.coverage_evidence import content_hash
from tests.agent_live.linux_shell_gates import load_linux_shell_gate_manifest


PRODUCT_REVIEW_SCOPE_SCHEMA_VERSION = 2
DEFAULT_SCOPE_MANIFEST = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "product_review_scope.json"
)


def compute_product_review_scope(
    repo_root: str | Path,
    manifest_path: str | Path = DEFAULT_SCOPE_MANIFEST,
) -> dict[str, Any]:
    """Recompute every finite G6 membership set from repository facts."""

    root = Path(repo_root).expanduser().resolve()
    manifest = _load_manifest(root, Path(manifest_path).expanduser().resolve())
    documentation = _documentation_scope(root, manifest["documentation"])
    migration = _migration_scope(manifest["migration"])
    external = _external_scope(manifest["external_capabilities"])
    severity = _severity_scope(manifest["severity_sources"])
    shell = load_linux_shell_gate_manifest(
        root,
        root / str(manifest["shell_gate_manifest"]),
    )
    fixtures = _fixture_scope(root, manifest["fixture_policy"])
    result = {
        "schema_version": PRODUCT_REVIEW_SCOPE_SCHEMA_VERSION,
        "scope_id": manifest["scope_id"],
        "scope_manifest": str(
            Path(manifest_path).expanduser().resolve().relative_to(root)
        ),
        "scope_manifest_sha256": _sha256_file(
            Path(manifest_path).expanduser().resolve()
        ),
        "documentation": documentation,
        "migration": migration,
        "external_capabilities": external,
        "severity": severity,
        "shell_gates": shell,
        "fixture_provenance": fixtures,
    }
    return {**result, "scope_hash": content_hash(result)}


def _load_manifest(root: Path, path: Path) -> dict[str, Any]:
    if root not in path.parents:
        raise ValueError("G6 scope manifest must be repository-owned")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "scope_id",
        "documentation",
        "migration",
        "external_capabilities",
        "severity_sources",
        "shell_gate_manifest",
        "fixture_policy",
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != required
        or payload.get("schema_version") != PRODUCT_REVIEW_SCOPE_SCHEMA_VERSION
        or not str(payload.get("scope_id") or "").strip()
    ):
        raise ValueError("G6 product-review scope manifest is invalid")
    return dict(payload)


def _documentation_scope(root: Path, raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("G6 documentation scope is invalid")
    en_root = root / str(raw.get("english_root") or "")
    zh_root = root / str(raw.get("chinese_root") or "")
    paired = tuple(str(value) for value in raw.get("paired_basenames") or ())
    exemptions = tuple(raw.get("reviewed_exemptions") or ())
    required_facts = tuple(raw.get("required_facts") or ())
    if (
        not paired
        or len(paired) != len(set(paired))
        or any(not isinstance(row, Mapping) for row in exemptions)
        or not required_facts
        or any(not isinstance(row, Mapping) for row in required_facts)
    ):
        raise ValueError("G6 documentation membership is invalid")
    exemption_paths = {
        str(row.get("path") or ""): str(row.get("reason") or "").strip()
        for row in exemptions
    }
    if any(not path or not reason for path, reason in exemption_paths.items()):
        raise ValueError("G6 documentation exemption is incomplete")
    observed_en = _tracked_markdown(root, en_root)
    observed_zh = _tracked_markdown(root, zh_root)
    expected_en = {str(en_root.relative_to(root) / name) for name in paired}
    expected_zh = {str(zh_root.relative_to(root) / name) for name in paired}
    classified = expected_en | expected_zh | set(exemption_paths)
    observed = observed_en | observed_zh
    if observed != classified:
        raise ValueError(
            "G6 documentation membership drift: "
            f"unclassified={sorted(observed - classified)}, "
            f"missing={sorted(classified - observed)}"
        )
    normalized_facts: list[dict[str, str]] = []
    fact_ids: set[str] = set()
    for row in required_facts:
        if set(row) != {
            "fact_id",
            "basename",
            "en_marker",
            "zh_marker",
        }:
            raise ValueError("G6 documentation fact contract is invalid")
        fact_id = str(row.get("fact_id") or "").strip()
        basename = str(row.get("basename") or "").strip()
        en_marker = str(row.get("en_marker") or "").strip()
        zh_marker = str(row.get("zh_marker") or "").strip()
        if (
            not fact_id
            or fact_id in fact_ids
            or basename not in paired
            or not en_marker
            or not zh_marker
        ):
            raise ValueError("G6 documentation fact membership is invalid")
        en_path = en_root / basename
        zh_path = zh_root / basename
        if (
            en_marker not in en_path.read_text(encoding="utf-8")
            or zh_marker not in zh_path.read_text(encoding="utf-8")
        ):
            raise ValueError(
                f"G6 documentation fact drifted from repository text: {fact_id}"
            )
        fact_ids.add(fact_id)
        normalized_facts.append({
            "fact_id": fact_id,
            "basename": basename,
            "en_path": str(en_path.relative_to(root)),
            "zh_path": str(zh_path.relative_to(root)),
            "en_marker": en_marker,
            "zh_marker": zh_marker,
        })
    return {
        "pair_denominator": len(paired),
        "pairs": [
            {
                "basename": name,
                "en_path": str(en_root.relative_to(root) / name),
                "zh_path": str(zh_root.relative_to(root) / name),
            }
            for name in paired
        ],
        "exemption_denominator": len(exemption_paths),
        "reviewed_exemptions": [
            {"path": path, "reason": exemption_paths[path]}
            for path in sorted(exemption_paths)
        ],
        "fact_denominator": len(normalized_facts),
        "required_facts": sorted(
            normalized_facts,
            key=lambda row: row["fact_id"],
        ),
    }


def _migration_scope(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("G6 migration scope is invalid")
    quarantine = tuple(raw.get("quarantine_versions") or ())
    migrate = tuple(raw.get("migrate_versions") or ())
    current = tuple(raw.get("current_versions") or ())
    future = raw.get("future_version_probe")
    declared_current = raw.get("current_schema_version")
    if (
        declared_current != STATE_SCHEMA_VERSION
        or quarantine != tuple(range(1, 12))
        or migrate != tuple(range(12, STATE_SCHEMA_VERSION))
        or current != (STATE_SCHEMA_VERSION,)
        or future != STATE_SCHEMA_VERSION + 1
    ):
        raise ValueError("G6 migration membership drifted from state authority")
    checks = [
        *(
            {
                "check_id": f"checkpoint.v{version}.quarantined",
                "schema_version": version,
                "expected": "quarantined",
            }
            for version in quarantine
        ),
        *(
            {
                "check_id": f"checkpoint.v{version}.migrated",
                "schema_version": version,
                "expected": "migrated",
            }
            for version in migrate
        ),
        {
            "check_id": f"checkpoint.v{STATE_SCHEMA_VERSION}.current",
            "schema_version": STATE_SCHEMA_VERSION,
            "expected": "current",
        },
        {
            "check_id": f"checkpoint.v{future}.rejected-future",
            "schema_version": future,
            "expected": "rejected_future",
        },
    ]
    return {
        "current_schema_version": STATE_SCHEMA_VERSION,
        "check_denominator": len(checks),
        "checks": checks,
    }


def _external_scope(raw: Any) -> dict[str, Any]:
    if (
        not isinstance(raw, list)
        or not raw
        or any(not isinstance(row, Mapping) for row in raw)
    ):
        raise ValueError("G6 external capability scope is invalid")
    rows = [
        {
            "capability_id": str(row.get("capability_id") or "").strip(),
            "required_state": str(row.get("required_state") or "").strip(),
        }
        for row in raw
    ]
    ids = [row["capability_id"] for row in rows]
    if (
        any(not all(row.values()) for row in rows)
        or len(ids) != len(set(ids))
    ):
        raise ValueError("G6 external capability membership is invalid")
    return {"denominator": len(rows), "capabilities": rows}


def _severity_scope(raw: Any) -> dict[str, Any]:
    if (
        not isinstance(raw, list)
        or not raw
        or any(not isinstance(value, str) or not value.strip() for value in raw)
        or len(raw) != len(set(raw))
    ):
        raise ValueError("G6 severity source membership is invalid")
    return {"denominator": len(raw), "source_ids": list(raw)}


def _fixture_scope(root: Path, raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("G6 fixture policy is invalid")
    fixture_root = root / str(raw.get("fixture_root") or "")
    template_root = root / str(raw.get("chain_template_root") or "")
    fixture_paths = sorted(
        str(path.relative_to(root))
        for path in fixture_root.glob("*/*.json")
        if path.is_file()
    )
    workload_paths: set[str] = set()
    for path in sorted(template_root.glob("*.json")):
        template = json.loads(path.read_text(encoding="utf-8"))
        chain = path.stem
        family = str(dict(template.get("_meta") or {}).get("adapter_family") or "")
        fixture_map = _fixture_map(root, family)
        methods = dict(template.get("rpc_methods") or {})
        configured = {
            str(methods.get("single") or "").strip(),
            *(
                str(row.get("method") or "").strip()
                for row in methods.get("mixed_weighted") or ()
                if isinstance(row, Mapping)
            ),
        }
        configured.discard("")
        for method in configured:
            fixture = fixture_map.get(method) or f"{_safe_name(method)}.json"
            workload_paths.add(
                str((fixture_root / chain / fixture).relative_to(root))
            )
    missing = sorted(workload_paths - set(fixture_paths))
    auxiliary = sorted(set(fixture_paths) - workload_paths)
    if missing:
        raise ValueError(f"G6 workload fixtures are missing: {missing}")
    return {
        "fixture_denominator": len(fixture_paths),
        "fixture_inventory": [
            {"path": path, "sha256": _sha256_file(root / path)}
            for path in fixture_paths
        ],
        "workload_denominator": len(workload_paths),
        "workload_present": len(workload_paths),
        "workload_paths": sorted(workload_paths),
        "auxiliary_denominator": len(auxiliary),
        "auxiliary_ids": auxiliary,
        "runtime_presence": "complete",
        "strict_authenticity": str(
            raw.get("strict_authenticity_without_capture_provenance") or ""
        ),
    }


def _fixture_map(root: Path, family: str) -> dict[str, str]:
    path = root / "tools" / "fake-node" / "configs" / f"{family}.yaml"
    if not path.is_file():
        return {}
    mapping: dict[str, str] = {}
    in_methods = False
    current = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.split("#", 1)[0].rstrip()
        if not raw.strip():
            continue
        if raw.strip() == "methods:":
            in_methods = True
            current = ""
            continue
        if in_methods and not raw.startswith(" "):
            break
        if not in_methods:
            continue
        method_match = re.match(r"^  ([^:\n]+):\s*$", raw)
        if method_match:
            current = method_match.group(1).strip().strip("'\"")
            continue
        fixture_match = re.match(r"^\s+fixture:\s*(.+?)\s*$", raw)
        if fixture_match and current:
            mapping[current] = fixture_match.group(1).strip().strip("'\"")
    return mapping


def _safe_name(value: str) -> str:
    normalized = value.replace("/", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", normalized).strip("_") or "method"


def _tracked_markdown(root: Path, directory: Path) -> set[str]:
    relative = str(directory.relative_to(root))
    completed = subprocess.run(
        ("git", "-C", str(root), "ls-files", "-z", f"{relative}/*.md"),
        capture_output=True,
        check=True,
    )
    return {
        item.decode("utf-8")
        for item in completed.stdout.split(b"\0")
        if item
    }


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()

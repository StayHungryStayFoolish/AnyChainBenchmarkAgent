#!/usr/bin/env python3
"""Generate one immutable 24-edge + 8-Journey formal Chaos target set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent.harness.runtime_identity import repository_revision
from tests.agent_live.chaos_scheduler import build_chaos_schedule
from tests.agent_live.coverage_evidence import content_hash
from tests.agent_live.formal_journey_catalog import formal_journey_definitions


FORMAL_TARGET_SET_SCHEMA_VERSION = 1
FORMAL_EDGE_COUNT = 24
FORMAL_JOURNEY_COUNT = 8
EMPTY_WORKTREE_HASH = hashlib.sha256(b"").hexdigest()

_PERSONAS = (
    "first-time confused evaluator",
    "impatient infrastructure operator",
    "bilingual protocol engineer",
    "returning benchmark operator",
    "skeptical performance engineer",
    "operator correcting an earlier choice",
)


def build_formal_target_payloads(
    ledger: Mapping[str, Any],
    *,
    revision: Mapping[str, str],
    seed: int,
) -> tuple[dict[str, Any], ...]:
    """Select formal targets without generating any future user messages."""

    if dict(ledger.get("revision") or {}) != dict(revision):
        raise ValueError("coverage ledger revision does not match the target revision")
    candidates = [
        edge for edge in ledger.get("edges") or ()
        if _eligible_edge(edge)
    ]
    selected = _select_edges(candidates, count=FORMAL_EDGE_COUNT, seed=seed)
    prefix = str(revision.get("commit") or "unknown")[:7]
    payloads: list[dict[str, Any]] = []
    for index, edge in enumerate(selected, start=1):
        scenario_ids = sorted({
            str(item).strip()
            for item in edge.get("executable_scenario_ids") or ()
            if str(item).strip()
        })
        scenario_id = scenario_ids[0]
        target = {
            "target_id": f"{prefix}-formal-edge-{index:02d}",
            "edge_key": str(edge["edge_key"]),
            "persona": _persona_for(edge, seed=seed),
            "goal": _goal_for(edge),
            "scenario_id": scenario_id,
        }
        build_chaos_schedule(
            ledger,
            revision=revision,
            seed=seed + index,
            targets=[target],
        )
        payloads.append({"targets": [target]})

    journeys = formal_journey_definitions()
    if len(journeys) != FORMAL_JOURNEY_COUNT:
        raise ValueError(
            f"formal Journey catalog must contain exactly {FORMAL_JOURNEY_COUNT} rows"
        )
    journey_ids = [str((row.get("journey") or {}).get("journey_id") or "") for row in journeys]
    if any(not item for item in journey_ids) or len(journey_ids) != len(set(journey_ids)):
        raise ValueError("formal Journey catalog contains missing or duplicate ids")
    payloads.extend(dict(row) for row in journeys)
    if len(payloads) != FORMAL_EDGE_COUNT + FORMAL_JOURNEY_COUNT:
        raise AssertionError("formal target payload count is inconsistent")
    return tuple(payloads)


def write_formal_target_set(
    ledger: Mapping[str, Any],
    *,
    repo_root: str | Path,
    output_dir: str | Path,
    seed: int,
) -> Path:
    """Write a revision-bound target directory atomically and fail closed."""

    root = Path(repo_root).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"formal target directory is immutable: {destination}")
    revision = repository_revision(root)
    if revision.get("worktree_hash") != EMPTY_WORKTREE_HASH:
        raise ValueError("formal target generation requires a clean worktree")
    if dict(ledger.get("revision") or {}) != revision:
        raise ValueError("coverage ledger is not bound to the clean active revision")
    payloads = build_formal_target_payloads(ledger, revision=revision, seed=seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-staging-",
        dir=destination.parent,
    ) as temporary:
        staging = Path(temporary)
        targets = []
        for index, payload in enumerate(payloads, start=1):
            path = staging / f"{index:02d}.json"
            encoded = _encoded(payload)
            path.write_bytes(encoded)
            targets.append({
                "index": index,
                "lane": "edge" if index <= FORMAL_EDGE_COUNT else "journey",
                "path": path.name,
                "sha256": hashlib.sha256(encoded).hexdigest(),
            })
        unsigned = {
            "schema_version": FORMAL_TARGET_SET_SCHEMA_VERSION,
            "revision": revision,
            "source_ledger_hash": content_hash(ledger),
            "seed": int(seed),
            "lane_counts": {
                "edge": FORMAL_EDGE_COUNT,
                "journey": FORMAL_JOURNEY_COUNT,
            },
            "targets": targets,
        }
        manifest = {"manifest_id": content_hash(unsigned), **unsigned}
        (staging / "manifest.json").write_bytes(_encoded(manifest))
        os.replace(staging, destination)
    return destination / "manifest.json"


def _eligible_edge(edge: Mapping[str, Any]) -> bool:
    lane = (edge.get("evidence") or {}).get("dynamic_dual_ai") or {}
    return (
        bool(edge.get("applicable", True))
        and bool(lane.get("required"))
        and str(lane.get("status") or "not_run") != "passed"
        and bool(str(edge.get("edge_key") or "").strip())
        and bool(edge.get("executable_scenario_ids"))
    )


def _select_edges(
    candidates: Sequence[Mapping[str, Any]],
    *,
    count: int,
    seed: int,
) -> tuple[Mapping[str, Any], ...]:
    unique = {str(edge.get("edge_key") or ""): edge for edge in candidates}
    if len(unique) < count:
        raise ValueError(
            f"formal target generation requires {count} eligible unique edges; "
            f"found {len(unique)}"
        )
    remaining = list(unique.values())
    selected: list[Mapping[str, Any]] = []
    groups: set[str] = set()
    input_classes: set[str] = set()
    edge_types: set[str] = set()
    while len(selected) < count:
        ranked = sorted(
            remaining,
            key=lambda edge: (
                str(edge.get("group") or "") in groups,
                str(edge.get("input_class") or "") in input_classes,
                str(edge.get("edge_type") or "") in edge_types,
                _seeded_key(seed, str(edge.get("edge_key") or "")),
                str(edge.get("edge_key") or ""),
            ),
        )
        edge = ranked[0]
        remaining.remove(edge)
        selected.append(edge)
        groups.add(str(edge.get("group") or ""))
        input_classes.add(str(edge.get("input_class") or ""))
        edge_types.add(str(edge.get("edge_type") or ""))
    return tuple(selected)


def _seeded_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _persona_for(edge: Mapping[str, Any], *, seed: int) -> str:
    key = int(_seeded_key(seed, str(edge.get("edge_key") or ""))[:8], 16)
    return _PERSONAS[key % len(_PERSONAS)]


def _goal_for(edge: Mapping[str, Any]) -> str:
    expected = json.dumps(
        edge.get("expected_postcondition") or {},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        "After reading each complete live Agent response, choose one realistic next "
        "user turn that exercises the scheduled authoritative edge. Do not copy a "
        "prewritten dialogue or weaken the target. "
        f"Group={edge.get('group')}; question={edge.get('question_id') or '<none>'}; "
        f"input_class={edge.get('input_class')}; transition={edge.get('option_or_action')}; "
        f"expected_postcondition={expected}."
    )


def _encoded(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--seed", type=int, default=27_071)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    ledger = json.loads(Path(args.ledger).read_text(encoding="utf-8"))
    manifest = write_formal_target_set(
        ledger,
        repo_root=args.repo_root,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

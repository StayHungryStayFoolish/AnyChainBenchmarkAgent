"""Test adapter that executes the same compiled LangGraph used by the product."""

from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from typing import Any, Mapping
from unittest.mock import patch

from agent.harness.graph import build_graph
from agent.harness.invariants import validate_state


def invoke_product_graph_turn(
    state: Mapping[str, Any],
    *,
    allow_semantic_resolver: bool = False,
) -> dict[str, Any]:
    """Run one deterministic uncheckpointed turn through the product graph.

    Contract and unit evidence must never depend on a live model response. A
    caller exercising semantic input has to inject the reviewed resolver it is
    testing; real provider behavior belongs to the PTY/CLI acceptance lanes.
    """

    from agent.harness.domains.rpc_catalog import migrate_legacy_catalog
    from agent.harness import coordinator, intent
    from agent.harness import hierarchical_planner
    from agent.harness.domains import analysis, chain_identity, recovery, rpc_endpoint

    current = deepcopy(dict(state))
    migrate_legacy_catalog(current)
    guarded_entries = (
        (
            "agent.harness.coordinator.resolve_action_queue",
            coordinator.resolve_action_queue,
            hierarchical_planner.resolve_product_action_queue,
        ),
        (
            "agent.harness.domains.chain_identity.resolve_unknown_chain_identity",
            chain_identity.resolve_unknown_chain_identity,
            intent.resolve_unknown_chain_identity,
        ),
        (
            "agent.harness.domains.rpc_endpoint.extract_rpc_schema_from_evidence",
            rpc_endpoint.extract_rpc_schema_from_evidence,
            intent.extract_rpc_schema_from_evidence,
        ),
        (
            "agent.harness.domains.analysis.analyze_evidence_with_model",
            analysis.analyze_evidence_with_model,
            intent.analyze_evidence_with_model,
        ),
        (
            "agent.harness.domains.recovery.analyze_evidence_with_model",
            recovery.analyze_evidence_with_model,
            intent.analyze_evidence_with_model,
        ),
    )
    with ExitStack() as stack:
        for index, (target, active, original) in enumerate(guarded_entries):
            if active is original and not (
                index == 0 and allow_semantic_resolver
            ):
                stack.enter_context(patch(
                    target,
                    side_effect=AssertionError(
                        "deterministic graph turn attempted to call a live model entry"
                    ),
                ))
        result = dict(build_graph(None).invoke(current))
    validate_state(result)
    return result

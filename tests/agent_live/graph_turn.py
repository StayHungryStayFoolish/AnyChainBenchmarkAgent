"""Test adapter that executes the same compiled LangGraph used by the product."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from agent.harness.graph import build_graph
from agent.harness.invariants import validate_state


def invoke_product_graph_turn(state: Mapping[str, Any]) -> dict[str, Any]:
    """Run one uncheckpointed product turn through the compiled graph."""

    from agent.harness.domains.rpc_catalog import migrate_legacy_catalog

    current = deepcopy(dict(state))
    migrate_legacy_catalog(current)
    result = dict(build_graph(None).invoke(current))
    validate_state(result)
    return result

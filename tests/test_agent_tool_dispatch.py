"""Regression tests for the single tool-dispatch surface (`agent/tools/`).

`agent/tools/executor.py` (dispatch) and `agent/tools/schema.py` (advertised
names/parameters) used to be able to drift from each other silently, and both
used to independently duplicate what the now-retired `agent/adk_app/tools/*`
ADK wrapper layer also exposed. This file asserts the two halves of the one
remaining surface stay in lockstep.
"""

from __future__ import annotations

import ast
import inspect
import unittest
from unittest.mock import patch


class ToolDispatchParityTest(unittest.TestCase):
    def test_schema_tool_names_match_executor_dispatch_names(self) -> None:
        import agent.tools.executor as executor_module
        from agent.tools.schema import tool_schema

        schema_names = {tool["function"]["name"] for tool in tool_schema()["tools"]}
        dispatch_names = _executor_dispatch_names(executor_module)

        self.assertEqual(
            schema_names,
            dispatch_names,
            "agent/tools/schema.py and agent/tools/executor.py must advertise the exact same tool names",
        )

        # Every schema name must actually be dispatchable: `execute_tool`
        # raises this exact message only when `name` falls through every
        # `if` branch, never for a missing/invalid argument on a real branch.
        # `discover_environment`/`audit_dependencies`/`run_doctor` take zero
        # required arguments, so without stubbing they would run real host
        # discovery / shell out to real scripts on every test run — replace
        # them with no-ops for the duration of this dispatch check only.
        no_op = lambda *args, **kwargs: {}
        with (
            patch.object(executor_module, "discover_environment", no_op),
            patch.object(executor_module, "audit_dependencies", no_op),
            patch.object(executor_module, "run_doctor", no_op),
        ):
            for name in schema_names:
                try:
                    executor_module.execute_tool(name, {})
                except ValueError as exc:
                    if str(exc) == f"unsupported tool: {name}":
                        self.fail(f"schema advertises {name!r} but executor does not dispatch it")
                except Exception:
                    pass  # missing required args / other real side effects are fine here

    def test_unsupported_tool_name_raises(self) -> None:
        from agent.tools.executor import execute_tool

        with self.assertRaises(ValueError):
            execute_tool("not_a_real_tool", {})


def _executor_dispatch_names(executor_module) -> set[str]:
    """Extract every `if name == "..."` dispatch literal from `execute_tool`.

    Parses the function's syntax tree (via `ast`) rather than regex-matching
    its source text, so reformatting the if-chain (whitespace, quote style,
    reordering) can't silently break this check the way a regex would.
    """
    tree = ast.parse(inspect.getsource(executor_module.execute_tool))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == "name"):
            continue
        for op, comparator in zip(node.ops, node.comparators):
            if isinstance(op, ast.Eq) and isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                names.add(comparator.value)
    return names


if __name__ == "__main__":
    unittest.main()

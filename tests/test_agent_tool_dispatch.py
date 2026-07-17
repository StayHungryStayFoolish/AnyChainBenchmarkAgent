"""Regression tests for the single tool-dispatch surface (`agent/tools/`).

`agent/tools/executor.py` (dispatch) and `agent/tools/schema.py` (advertised
names/parameters) used to be able to drift from each other silently, and both
used to independently duplicate what the now-retired `agent/adk_app/tools/*`
ADK wrapper layer also exposed. This file asserts the two halves of the one
remaining surface stay in lockstep.
"""

from __future__ import annotations

import unittest


class ToolDispatchParityTest(unittest.TestCase):
    def test_schema_tool_names_match_executor_dispatch_names(self) -> None:
        import agent.tools.executor as executor_module
        from agent.tools.schema import tool_schema

        schema_names = {tool["function"]["name"] for tool in tool_schema()["tools"]}
        dispatch_names = set(executor_module.TOOL_OPERATION_BY_NAME)

        self.assertEqual(
            schema_names,
            dispatch_names,
            "agent/tools/schema.py and agent/tools/executor.py must advertise the exact same tool names",
        )

        for name in schema_names:
            self.assertTrue(
                callable(executor_module.TOOL_OPERATION_BY_NAME[name].handler),
                f"missing registered callable for tool {name!r}",
            )

    def test_unsupported_tool_name_raises(self) -> None:
        from agent.tools.executor import execute_tool

        with self.assertRaises(ValueError):
            execute_tool("not_a_real_tool", {})

    def test_obsolete_configuration_question_tools_are_not_exposed(self) -> None:
        import agent.tools.executor as executor_module
        from agent.tools.schema import tool_schema

        removed = {"validate_required_config", "build_missing_config_questions"}
        schema_names = {tool["function"]["name"] for tool in tool_schema()["tools"]}
        dispatch_names = set(executor_module.TOOL_OPERATION_BY_NAME)

        self.assertTrue(removed.isdisjoint(schema_names))
        self.assertTrue(removed.isdisjoint(dispatch_names))
        for name in removed:
            with self.assertRaisesRegex(ValueError, f"unsupported tool: {name}"):
                executor_module.execute_tool(name, {})

    def test_string_false_cannot_authorize_a_side_effect(self) -> None:
        from agent.tools.executor import execute_tool

        with self.assertRaisesRegex(ValueError, "expected boolean"):
            execute_tool("submit_job", {"plan_file": "plan.json", "approved": "false"})

    def test_nested_tool_values_follow_the_advertised_schema(self) -> None:
        from agent.tools.executor import execute_tool

        with self.assertRaisesRegex(ValueError, "expected integer"):
            execute_tool(
                "draft_request",
                {"mixed_weights": {"eth_blockNumber": "100"}},
            )

if __name__ == "__main__":
    unittest.main()

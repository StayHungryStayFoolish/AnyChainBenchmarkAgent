"""ADK bridge boundary constants.

AnyChain product workflow prompts now live in ``agent.harness`` node contracts
and typed LLM resolvers. This module remains only for ADK status/eval imports.
"""

from __future__ import annotations


ADK_COMPATIBILITY_INSTRUCTION = """
AnyChain ADK compatibility surface.

The product Agent workflow is owned by the LangGraph Harness in agent.harness.
ADK may provide model/tool capabilities, but it must not run a separate
benchmark wizard, mutate workflow state, or bypass Harness validators and
human-confirmation gates.
"""


ADK_BRIDGE_BOUNDARY = """
LangGraph Harness is the product workflow runtime. ADK workflow-state prompts,
callbacks, and model-driven state mutation are retired.
"""

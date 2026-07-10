"""LangGraph runtime wrapper for AnyChain Agent Harness."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .checkpoints import create_sqlite_checkpointer, default_checkpoint_path
from .nodes.router import route_after_user_turn
from .groups import process_turn
from .state import AgentGraphState, ensure_session_metadata, new_state


class AnyChainGraphRuntime:
    """Thin wrapper around the LangGraph product workflow graph."""

    def __init__(
        self,
        thread_id: str,
        checkpoint_path: str | Path | None = None,
        checkpointer: Any | None = None,
        session_purpose: str = "user",
    ) -> None:
        self.thread_id = thread_id
        self.session_purpose = session_purpose or "user"
        self.checkpointer = checkpointer or create_sqlite_checkpointer(checkpoint_path or default_checkpoint_path())
        self.graph = build_graph(self.checkpointer)

    def invoke(self, text: str, language: str = "en", context: dict[str, Any] | None = None) -> AgentGraphState:
        state = self._load_state(language=language)
        state["last_user_input"] = text
        state["language"] = language
        if context:
            for key, value in context.items():
                state[key] = value
        state = ensure_session_metadata(state, self.thread_id, self.session_purpose)
        return self.graph.invoke(
            state,
            config={"configurable": {"thread_id": self.thread_id}},
        )

    def snapshot(self) -> AgentGraphState:
        return self._load_state(language="en")

    def reset(self, language: str = "en") -> AgentGraphState:
        fresh = new_state(self.thread_id, language=language, session_purpose=self.session_purpose)
        return self.update(fresh)

    def update(self, patch: dict[str, Any]) -> AgentGraphState:
        config = {"configurable": {"thread_id": self.thread_id}}
        patch = ensure_session_metadata(dict(patch), self.thread_id, self.session_purpose)
        self.graph.update_state(config, patch)
        snapshot = self.graph.get_state(config)
        values = getattr(snapshot, "values", None) or {}
        return dict(values)

    def _load_state(self, language: str) -> AgentGraphState:
        config = {"configurable": {"thread_id": self.thread_id}}
        try:
            snapshot = self.graph.get_state(config)
            values = getattr(snapshot, "values", None) or {}
            if values:
                return ensure_session_metadata(dict(values), self.thread_id, self.session_purpose)
        except Exception:
            pass
        return new_state(self.thread_id, language=language, session_purpose=self.session_purpose)


def build_graph(checkpointer: Any) -> Any:
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(AgentGraphState)
    graph.add_node("router", _router_node)
    graph.add_node("turn", process_turn)
    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", route_after_user_turn, {"turn": "turn"})
    graph.add_edge("turn", END)
    return graph.compile(checkpointer=checkpointer)


def _router_node(state: AgentGraphState) -> AgentGraphState:
    return {**state, "active_group": state.get("active_group") or "opening"}

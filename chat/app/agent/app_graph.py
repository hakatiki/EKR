from functools import lru_cache
from typing import Callable

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes.chat import chat_node
from .nodes.plan_investigation import plan_investigation
from .nodes.router import route_intent
from .nodes.run_investigation import run_investigation
from .state import AgentState


def _build_route_edges() -> Callable[[AgentState], str]:
    def _route(state: AgentState) -> str:
        mode = state.get("mode") or "chat"
        if mode == "investigate":
            return "plan_investigation"
        return "chat"

    return _route


def build_agent_app():
    builder = StateGraph(AgentState)

    builder.add_node("router", route_intent)
    builder.add_node("chat", chat_node)
    builder.add_node("plan_investigation", plan_investigation)
    builder.add_node("run_investigation", run_investigation)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        _build_route_edges(),
        {
            "chat": "chat",
            "plan_investigation": "plan_investigation",
        },
    )
    builder.add_edge("plan_investigation", "run_investigation")
    builder.add_edge("chat", END)
    builder.add_edge("run_investigation", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


@lru_cache(maxsize=1)
def get_agent_app():
    return build_agent_app()


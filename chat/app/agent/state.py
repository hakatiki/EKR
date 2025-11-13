from typing import Annotated, Literal, Optional

from typing_extensions import TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    mode: Literal["chat", "investigate"]
    company: str
    plan_md: str
    report_md: str
    last_response_text: Optional[str]


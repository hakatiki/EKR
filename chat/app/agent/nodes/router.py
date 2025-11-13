from typing import Literal, Optional

from pydantic import BaseModel, Field

from ..llm_config import SYSTEM_PROMPT, base_llm
from ..state import AgentState


class RouterDecision(BaseModel):
    mode: Literal["chat", "investigate"]
    company: Optional[str] = Field(
        default=None,
        description="Company name if the user requested an investigation.",
    )


router_llm = base_llm.with_structured_output(RouterDecision)


def route_intent(state: AgentState) -> AgentState:
    messages = state.get("messages") or []
    if not messages:
        return {"mode": "chat"}

    last_msg = messages[-1]
    if hasattr(last_msg, "content"):
        user_text = last_msg.content
    else:
        user_text = last_msg.get("content", "")

    decision: RouterDecision = router_llm.invoke(
        [
            (
                "system",
                SYSTEM_PROMPT
                + """
You are a classification helper.
If the user asks you to investigate or look into a company,
set mode='investigate' and extract the company name if possible.
Otherwise set mode='chat'.
""",
            ),
            ("user", user_text),
        ]
    )

    updates: AgentState = {"mode": decision.mode}
    if decision.company:
        updates["company"] = decision.company
    return updates


from typing import List

from pydantic import BaseModel

from ..llm_config import SYSTEM_PROMPT, base_llm
from ..state import AgentState


class InvestigationPlan(BaseModel):
    company: str
    steps: List[str]


plan_llm = base_llm.with_structured_output(InvestigationPlan)


def plan_investigation(state: AgentState) -> AgentState:
    company = state.get("company") or "the company the user mentioned"
    messages = state.get("messages") or []
    last_user = messages[-1] if messages else None

    if last_user and hasattr(last_user, "content"):
        user_text = last_user.content
    elif last_user:
        user_text = last_user.get("content", "")
    else:
        user_text = ""

    plan: InvestigationPlan = plan_llm.invoke(
        [
            ("system", SYSTEM_PROMPT),
            (
                "user",
                f"""
The user asked you to investigate procurement for company "{company}".
Create a concise TODO list of steps to investigate:
- basic company profile and revenues
- owners / beneficial owners
- procurement & public tenders
- suppliers and counterparties (where available)
- competitors and market position
- any procurement or corruption red flags

Return only a structured list of short bullet steps.

User message: {user_text}
""",
            ),
        ]
    )

    plan_md = "# Investigation plan\n\n" + "\n".join(f"- {step}" for step in plan.steps)

    return {
        "company": plan.company or company,
        "plan_md": plan_md,
    }


from ..llm_config import SYSTEM_PROMPT, investigation_llm
from ..state import AgentState


def run_investigation(state: AgentState) -> AgentState:
    company = state.get("company", "the target company")
    plan_md = state.get("plan_md", "")

    messages_for_llm = [
        (
            "system",
            SYSTEM_PROMPT
            + """
You are now in SPECIAL INVESTIGATION MODE.
You must:
1. Use web_search and file_search tools where necessary.
2. Follow the provided plan, but you can improve it if needed.
3. Output a final answer as GitHub-flavored Markdown with these sections:
## 1. Summary
Short narrative summary of key findings.
## 2. Company overview
A markdown table with at least: Year, Revenue (if public), Core segments,
Headquarters, Main markets, Source / notes.
## 3. Ownership & governance
Markdown table with owners / shareholders, stakes, roles, and sources.
## 4. Procurement & tenders
Markdown table listing notable tenders / contracts (where data is available),
counterparties, amounts (approximate OK), dates, and sources.
## 5. Competitors & market position
Markdown table of major competitors, region, segment, and notes.
## 6. Risk indicators / red flags
Bullet list AND a small markdown table of risk type, description, evidence, severity.
If you cannot find reliable info, say so explicitly and mark fields as N/A.
Cite sources in the text where possible.
""",
        ),
        (
            "user",
            f"""
Investigate the company "{company}" focusing on procurement and integrity risks.
Here is the planned TODO list of steps to follow:
{plan_md}

Now perform the investigation and produce the report.
""",
        ),
    ]

    response = investigation_llm.invoke(messages_for_llm)
    report_md = getattr(response, "text", None)
    if report_md is None:
        content = response.content
        if isinstance(content, str):
            report_md = content
        elif isinstance(content, list):
            report_md = "".join(
                getattr(block, "text", "")
                if not isinstance(block, dict)
                else block.get("text", "")
                for block in content
            )
        else:
            report_md = str(content)

    return {
        "messages": [response],
        "report_md": report_md,
        "last_response_text": report_md,
    }


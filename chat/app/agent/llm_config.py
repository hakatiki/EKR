import os
from typing import List

from langchain_openai import ChatOpenAI


VECTOR_STORE_ID = os.environ.get("OPENAI_PROCUREMENT_VECTOR_STORE_ID")

SYSTEM_PROMPT = """
You are ProcureGraph, an AI assistant.

Normal mode: behave like ChatGPT.

Special mode: when asked to investigate a company's procurement,
perform a structured investigation (revenues, owners, procurement,
competitors, risks) and return a detailed markdown report with tables.

If data is uncertain or missing, mark it clearly as N/A or Unclear.
""".strip()


def _build_tools() -> List[dict]:
    tools: List[dict] = [
        {"type": "web_search_preview"},
    ]
    if VECTOR_STORE_ID:
        tools.append(
            {
                "type": "file_search",
                "vector_store_ids": [VECTOR_STORE_ID],
            }
        )
    return tools


base_llm = ChatOpenAI(
    model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
    temperature=0,
    use_responses_api=True,
)

_tools = _build_tools()

chat_llm = base_llm.bind_tools(_tools)
investigation_llm = base_llm.bind_tools(_tools)


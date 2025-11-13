from ..llm_config import SYSTEM_PROMPT, chat_llm
from ..state import AgentState


def chat_node(state: AgentState) -> AgentState:
    existing_messages = state.get("messages") or []
    messages_for_llm = [
        ("system", SYSTEM_PROMPT),
        *existing_messages,
    ]
    response = chat_llm.invoke(messages_for_llm)

    response_text = getattr(response, "text", None)
    if response_text is None:
        content = response.content
        if isinstance(content, str):
            response_text = content
        elif isinstance(content, list):
            response_text = "".join(
                getattr(block, "text", "")
                if not isinstance(block, dict)
                else block.get("text", "")
                for block in content
            )
        else:
            response_text = str(content)

    return {
        "messages": [response],
        "last_response_text": response_text,
    }


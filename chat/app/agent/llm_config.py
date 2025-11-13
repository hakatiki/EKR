import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are ProcureGraph, an AI assistant.

Normal mode: behave like ChatGPT.

Special mode: when asked to investigate a company's procurement,
perform a structured investigation (revenues, owners, procurement,
competitors, risks) and return a detailed markdown report with tables.

If data is uncertain or missing, mark it clearly as N/A or Unclear.
""".strip()

SPECIAL_INVESTIGATION_PROMPT = """
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
""".strip()

ROUTER_SCHEMA: Dict[str, Any] = {
    "name": "router_decision",
    "schema": {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": ["chat", "investigate"]},
            "company": {"type": "string"},
        },
        "required": ["mode"],
        "additionalProperties": False,
    },
}

PLAN_SCHEMA: Dict[str, Any] = {
    "name": "investigation_plan",
    "schema": {
        "type": "object",
        "properties": {
            "company": {"type": "string"},
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            },
        },
        "required": ["steps"],
        "additionalProperties": False,
    },
}


def _build_tools() -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = [
        {"type": "web_search_preview"},
    ]
    vector_store_id = os.environ.get("OPENAI_PROCUREMENT_VECTOR_STORE_ID")
    if vector_store_id:
        tools.append(
            {
                "type": "file_search",
                "vector_store_ids": [vector_store_id],
            }
        )
    return tools


def _message(role: str, text: str) -> Dict[str, Any]:
    return {
        "role": role,
        "content": [{"type": "text", "text": text}],
    }


def _collect_text(response: Any) -> str:
    parts: List[str] = []
    output_items = getattr(response, "output", []) or []
    for item in output_items:
        item_type = getattr(item, "type", None) or getattr(item, "get", lambda *_: None)("type")
        if item_type == "message":
            content = getattr(item, "content", None) or []
            for block in content:
                text = getattr(block, "text", None)
                if not text and isinstance(block, dict):
                    text = block.get("text")
                if text:
                    parts.append(text)
    return "".join(parts).strip()


class ProcurementAgent:
    def __init__(self) -> None:
        self._client = AsyncOpenAI()
        self._default_model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
        self._base_tools = _build_tools()

    async def generate_reply(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        if not messages:
            raise ValueError("No messages provided.")

        last_user = self._latest_user_message(messages)
        if not last_user:
            raise ValueError("At least one user message is required.")

        model_id = model or self._default_model
        route = await self._route_intent(last_user, model_id)
        mode = route.get("mode", "chat")
        if mode == "investigate":
            company = route.get("company") or "the company the user mentioned"
            plan_md = await self._plan_investigation(
                company=company,
                user_text=last_user,
                model=model_id,
            )
            report = await self._run_investigation(
                company=company,
                plan_md=plan_md,
                model=model_id,
                temperature=temperature,
            )
            return report

        return await self._chat(
            messages=messages,
            model=model_id,
            temperature=temperature,
        )

    async def _chat(
        self,
        messages: List[Dict[str, str]],
        *,
        model: str,
        temperature: Optional[float],
    ) -> str:
        inputs = self._compose_conversation(messages)
        response = await self._client.responses.create(
            model=model,
            input=inputs,
            tools=self._base_tools,
            temperature=temperature,
        )
        return _collect_text(response)

    async def _route_intent(self, user_text: str, model: str) -> Dict[str, Any]:
        response = await self._client.responses.create(
            model=model,
            input=[
                _message(
                    "system",
                    SYSTEM_PROMPT
                    + "\n\nYou are a classification helper. Decide if the user is asking "
                    "for a procurement investigation and extract the company name if possible.",
                ),
                _message("user", user_text),
            ],
            response_format={"type": "json_schema", "json_schema": ROUTER_SCHEMA},
            temperature=0,
        )
        text = _collect_text(response)
        if not text:
            logger.warning("Router returned empty response text.")
            return {"mode": "chat"}
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Router returned invalid JSON: %s", text)
            return {"mode": "chat"}
        return data

    async def _plan_investigation(
        self,
        *,
        company: str,
        user_text: str,
        model: str,
    ) -> str:
        response = await self._client.responses.create(
            model=model,
            input=[
                _message("system", SYSTEM_PROMPT),
                _message(
                    "user",
                    (
                        f'The user asked you to investigate procurement for "{company}".\n'
                        "Create a concise TODO list covering:\n"
                        "- basic company profile and revenues\n"
                        "- owners / beneficial owners\n"
                        "- procurement & public tenders\n"
                        "- suppliers and counterparties (where available)\n"
                        "- competitors and market position\n"
                        "- any procurement or corruption red flags\n\n"
                        "Return a structured list of short bullet steps.\n\n"
                        f"User message: {user_text}"
                    ),
                ),
            ],
            response_format={"type": "json_schema", "json_schema": PLAN_SCHEMA},
            temperature=0,
        )
        text = _collect_text(response)
        plan_md = "# Investigation plan\n\n"
        if not text:
            logger.warning("Plan generation returned empty text.")
            return plan_md + "- Review publicly available information\n"
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Plan generation returned invalid JSON: %s", text)
            return plan_md + "- Review publicly available information\n"
        steps = data.get("steps") or []
        if isinstance(steps, (str, bytes)):
            steps = [steps]
        if not isinstance(steps, Iterable):
            steps = [str(steps)]
        for step in steps:
            plan_md += f"- {step}\n"
        return plan_md.rstrip()

    async def _run_investigation(
        self,
        *,
        company: str,
        plan_md: str,
        model: str,
        temperature: Optional[float],
    ) -> str:
        response = await self._client.responses.create(
            model=model,
            input=[
                _message("system", SYSTEM_PROMPT + "\n\n" + SPECIAL_INVESTIGATION_PROMPT),
                _message(
                    "user",
                    (
                        f'Investigate the company "{company}" focusing on procurement and integrity risks.\n'
                        "Here is the planned TODO list of steps to follow:\n"
                        f"{plan_md}\n\n"
                        "Now perform the investigation and produce the report."
                    ),
                ),
            ],
            tools=self._base_tools,
            temperature=temperature if temperature is not None else 0,
        )
        return _collect_text(response)

    def _compose_conversation(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        composed: List[Dict[str, Any]] = [_message("system", SYSTEM_PROMPT)]
        for msg in messages:
            role = msg.get("role", "").lower()
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            if role not in {"system", "user", "assistant"}:
                continue
            if role == "system":
                composed.append(_message("system", content))
            elif role == "user":
                composed.append(_message("user", content))
            else:
                composed.append(_message("assistant", content))
        return composed

    @staticmethod
    def _latest_user_message(messages: List[Dict[str, str]]) -> Optional[str]:
        for msg in reversed(messages):
            if msg.get("role", "").lower() == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                return str(content)
        return None


def get_agent() -> ProcurementAgent:
    return ProcurementAgent()


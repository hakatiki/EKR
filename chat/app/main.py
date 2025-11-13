import logging
import os
import uuid
from pathlib import Path
from typing import AsyncIterator, Iterable, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"

load_dotenv(BASE_DIR / ".env")
load_dotenv(override=False)

from .agent import get_agent_app

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

agent_app = get_agent_app()


class ChatMessage(BaseModel):
    role: str = Field(
        ...,
        description="The role of the message, e.g. system, user, or assistant.",
    )
    content: str = Field(..., description="Plain text message content.")


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = Field(
        default=None, description="Optional override for the underlying model."
    )
    temperature: Optional[float] = Field(
        default=None, ge=0, le=2, description="Sampling temperature."
    )
    thread_id: Optional[str] = Field(
        default=None,
        description="Conversation thread identifier to reuse LangGraph checkpoints.",
    )


def _to_langchain_message(message: ChatMessage) -> BaseMessage:
    role = message.role.lower()
    if role == "user":
        return HumanMessage(content=message.content)
    if role == "assistant":
        return AIMessage(content=message.content)
    if role == "system":
        return SystemMessage(content=message.content)
    raise ValueError(f"Unsupported message role: {message.role}")


def _extract_text_from_message(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, Iterable):
        parts = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if text:
                    parts.append(text)
            else:
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
        return "".join(parts)
    return str(content)


def create_app() -> FastAPI:
    fastapi_app = FastAPI(title="ProcureGraph Agent", version="0.2.0")

    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if STATIC_DIR.exists():
        fastapi_app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @fastapi_app.get("/")
    async def get_index() -> FileResponse:
        if not INDEX_FILE.exists():
            raise HTTPException(status_code=404, detail="Frontend not found.")
        return FileResponse(INDEX_FILE)

    @fastapi_app.post("/chat")
    async def chat(request: ChatRequest) -> StreamingResponse:
        if not request.messages:
            raise HTTPException(status_code=400, detail="Messages cannot be empty.")

        try:
            lc_messages = [_to_langchain_message(msg) for msg in request.messages]
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        thread_id = request.thread_id or str(uuid.uuid4())

        try:
            result = await agent_app.ainvoke(
                {"messages": lc_messages},
                config={"configurable": {"thread_id": thread_id}},
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Agent execution failed")
            raise HTTPException(
                status_code=500, detail="Agent execution failed."
            ) from exc

        response_message = None
        for message in reversed(result.get("messages", [])):
            if isinstance(message, AIMessage):
                response_message = message
                break

        if response_message is None:
            raise HTTPException(
                status_code=500, detail="Agent did not return a response."
            )

        response_text = _extract_text_from_message(response_message).strip()
        if not response_text:
            response_text = ""

        async def iterator() -> AsyncIterator[str]:
            yield response_text

        return StreamingResponse(iterator(), media_type="text/plain")

    return fastapi_app


app = create_app()



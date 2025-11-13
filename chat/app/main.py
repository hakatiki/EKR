import os
import uuid
from pathlib import Path
from typing import AsyncIterator, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"

load_dotenv(BASE_DIR / ".env")
load_dotenv(override=False)

from .agent import get_agent

agent = get_agent()


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
        description="Conversation thread identifier (currently unused).",
    )


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
            message_payload = [
                {"role": msg.role, "content": msg.content} for msg in request.messages
            ]
        except Exception as exc:  # pragma: no cover - defensive logging
            raise HTTPException(
                status_code=400, detail=f"Invalid message format: {exc}"
            ) from exc

        try:
            thread_id = request.thread_id or str(uuid.uuid4())
            _ = thread_id  # placeholder for future memory integration
            response_text = await agent.generate_reply(
                message_payload,
                model=request.model,
                temperature=request.temperature,
            )
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - defensive logging
            raise HTTPException(
                status_code=500, detail="Agent execution failed."
            ) from exc

        response_text = (response_text or "").strip()

        async def iterator() -> AsyncIterator[str]:
            yield response_text

        return StreamingResponse(iterator(), media_type="text/plain")

    return fastapi_app


app = create_app()



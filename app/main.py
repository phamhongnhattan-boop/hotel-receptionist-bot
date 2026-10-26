"""FastAPI app.  Local:  uvicorn app.main:app --reload"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import config

state: dict = {}


def build_agent():
    from app.agent import ReceptionistAgent
    from app.booking import BookingStore, make_tools
    from app.llm import get_client, make_embedder
    from app.rag import Retriever, load_chunks

    client = get_client()
    retriever = Retriever(load_chunks(config.KB_DIR), make_embedder(client))
    return ReceptionistAgent(client, retriever, make_tools(BookingStore(config.DB_PATH)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    if "agent" not in state:  # tests can put a fake agent here first
        state["agent"] = build_agent()
    yield


app = FastAPI(title="Hotel Receptionist Bot", lifespan=lifespan)


class ChatIn(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=1000)


@app.get("/")
def index():
    return FileResponse(config.STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat")
def chat(body: ChatIn):
    try:
        return state["agent"].reply(body.session_id, body.message)
    except Exception as e:  # show a clean error to the UI, keep details in logs
        print(f"[chat error] {e!r}")
        raise HTTPException(status_code=502, detail="Bot is busy, please try again.")


@app.post("/api/reset/{session_id}")
def reset(session_id: str):
    state["agent"].reset(session_id)
    return {"ok": True}

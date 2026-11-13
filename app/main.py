"""FastAPI app.  Local:  uvicorn app.main:app --reload"""
from __future__ import annotations

import hmac
import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app import config
from app.security import RateLimiter, client_ip, is_valid_session_id, new_session_id
from app.telemetry import EventLog

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

state: dict = {}

# One user: a few messages per minute. Everyone together: a daily cap on Gemini calls.
per_ip_limiter = RateLimiter(config.RATE_LIMIT_PER_MINUTE, window=60)
daily_limiter = RateLimiter(config.DAILY_MESSAGE_LIMIT, window=24 * 3600)


def build_agent(store=None, client=None):
    """The real agent. Scripts can pass their own BookingStore (e.g. in memory) to check it."""
    from app.agent import ReceptionistAgent
    from app.booking import BookingStore, make_tools
    from app.llm import get_client, make_embedder
    from app.rag import Retriever, cached_embedder, load_chunks

    client = client or get_client()
    embed = cached_embedder(make_embedder(client), config.EMBED_CACHE_PATH, config.EMBED_MODEL)
    retriever = Retriever(load_chunks(config.KB_DIR), embed)
    store = store or BookingStore(config.DB_PATH)
    return ReceptionistAgent(client, retriever, make_tools(store))


@asynccontextmanager
async def lifespan(app: FastAPI):
    if "agent" not in state:  # tests can put a fake agent here first
        state["agent"] = build_agent()
    if "events" not in state:
        state["events"] = EventLog(config.DB_PATH)
    yield


# /docs, /redoc and /openapi.json show every endpoint: useful locally, not on the public demo
docs = {} if config.ENABLE_DOCS else {"docs_url": None, "redoc_url": None, "openapi_url": None}
app = FastAPI(title="Hotel Receptionist Bot", lifespan=lifespan, **docs)

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",           # browser must not guess file types
    "X-Frame-Options": "DENY",                     # page cannot be put in an iframe (clickjacking)
    "Referrer-Policy": "no-referrer",
    # Only load scripts/styles from this page itself; no outside domains
    "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                               "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                               "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'",
}


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    for name, value in SECURITY_HEADERS.items():
        response.headers.setdefault(name, value)
    return response


class ChatIn(BaseModel):
    # Empty on the first message: the server then creates a new session and returns its id
    session_id: str | None = Field(default=None, max_length=64)
    message: str = Field(min_length=1, max_length=1000)


class ResetIn(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)


def too_many(retry_after: float, detail: str) -> JSONResponse:
    return JSONResponse(status_code=429, content={"detail": detail},
                        headers={"Retry-After": str(int(retry_after) + 1)})


@app.api_route("/", methods=["GET", "HEAD"])  # Render health checks use HEAD
def index():
    return FileResponse(config.STATIC_DIR / "index.html")


@app.get("/reviews")
def reviews_page():
    return FileResponse(config.STATIC_DIR / "reviews.html")


@app.get("/api/reviews")
def reviews_summary():
    """What guests praise or complain about + model scores, made offline by scripts/sentiment.py.
    Only reads two fixed files: no user input reaches the file system or the API."""
    reports = config.ROOT / "reports"
    try:
        aspects = json.loads((reports / "review_aspects.json").read_text(encoding="utf-8"))
        results = json.loads((reports / "sentiment_results.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Review analysis not run yet.")
    sample = results.get("gemini_sample", {})
    return {
        "aspects": aspects["aspects"],
        "source": aspects["source"],
        "models": {
            "tfidf_full_test": results["full_test_set"]["tfidf_logreg"],
            "full_test_n": results["full_test_set"]["n"],
            "tfidf_sample": sample.get("tfidf_logreg"),
            "gemini_sample": sample.get("gemini_zero_shot"),
            "sample_n": sample.get("n"),
        },
    }


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"status": "ok"}


@app.post("/api/chat")
def chat(body: ChatIn, request: Request):
    ip = client_ip(request.headers, request.client.host if request.client else None, config.PROXY_HOPS)
    wait = per_ip_limiter.check(ip)
    if wait:
        return too_many(wait, "Too many messages. Please wait a moment.")
    wait = daily_limiter.check("all")
    if wait:
        log.warning("daily message limit reached")
        return too_many(wait, "The demo has reached its daily limit. Please try again tomorrow.")

    agent = state["agent"]
    # Only the server creates session ids. An unknown or badly formed id (expired, or made up
    # by the client) starts a new session, so nobody can join another guest's conversation.
    session_id = body.session_id
    if not (is_valid_session_id(session_id) and agent.has_session(session_id)):
        session_id = new_session_id()

    start = time.perf_counter()
    try:
        result = agent.reply(session_id, body.message)
    except Exception as e:  # show a clean error to the UI, keep details in logs
        log.error("chat error: %r", e)
        state["events"].add(session_id=session_id, message=body.message, error=type(e).__name__,
                            latency_ms=(time.perf_counter() - start) * 1000)
        raise HTTPException(status_code=502, detail="Bot is busy, please try again.")
    state["events"].add(session_id=session_id, message=body.message, result=result,
                        latency_ms=(time.perf_counter() - start) * 1000, main_model=config.CHAT_MODEL)
    result.pop("tools", None)  # internal detail, not sent to the browser
    return {**result, "session_id": session_id}


@app.post("/api/reset")
def reset(body: ResetIn):
    if is_valid_session_id(body.session_id):
        state["agent"].reset(body.session_id)
    return {"ok": True}


@app.get("/api/stats")
def stats(hours: int = 24, authorization: str | None = Header(default=None)):
    """Usage numbers for the owner. Needs STATS_TOKEN; without it the endpoint does not exist."""
    if not config.STATS_TOKEN:
        raise HTTPException(status_code=404)
    expected = f"Bearer {config.STATS_TOKEN}"
    if not hmac.compare_digest((authorization or "").encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return state["events"].stats(max(1, min(hours, 24 * 30)))

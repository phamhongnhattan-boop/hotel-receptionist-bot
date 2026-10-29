"""Offline tests: fake embedder + fake agent, so no API key is needed."""
import hashlib
from datetime import date, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import config
from app.booking import BookingStore, make_tools
from app.rag import Retriever, load_chunks


def fake_embed(texts, task_type="RETRIEVAL_DOCUMENT"):
    """Bag-of-words hashing embedder: good enough to test the search logic."""
    out = np.zeros((len(texts), 256), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().split():
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)


# ---------- RAG ----------
def test_kb_is_split_by_heading():
    chunks = load_chunks(config.KB_DIR)
    titles = [c.title for c in chunks]
    assert len(chunks) >= 10
    assert "Đưa đón sân bay" in titles


def test_search_finds_right_section():
    r = Retriever(load_chunks(config.KB_DIR), fake_embed)
    top_chunk, _ = r.search("bữa sáng buffet phục vụ mấy giờ", k=1)[0]
    assert top_chunk.title == "Bữa sáng"


# ---------- Booking tools ----------
@pytest.fixture
def tools():
    check, create = make_tools(BookingStore(":memory:"))
    return check, create


def _dates(start_in_days=10, nights=2):
    ci = date.today() + timedelta(days=start_in_days)
    return ci.isoformat(), (ci + timedelta(days=nights)).isoformat()


def test_price_is_computed(tools):
    check, _ = tools
    ci, co = _dates(nights=3)
    res = check(ci, co, "Deluxe")
    assert res["ok"] and res["nights"] == 3
    assert res["total_price_vnd"] == 3 * res["price_per_night_vnd"]


def test_bad_dates_are_rejected(tools):
    check, _ = tools
    ci, co = _dates()
    assert not check(co, ci, "standard")["ok"]          # check-out before check-in
    assert not check("2020-01-01", "2020-01-02", "standard")["ok"]  # past
    assert not check("12/12/2030", co, "standard")["ok"]  # wrong format


def test_rooms_run_out(tools):
    check, create = tools
    ci, co = _dates()
    for i in range(3):  # 3 family rooms in total
        assert create(f"Guest {i}", "0905123456", ci, co, "family")["ok"]
    assert check(ci, co, "family")["rooms_available"] == 0
    assert not create("Late Guest", "0905123456", ci, co, "family")["ok"]


# ---------- Fallback when Gemini is busy ----------
class FakeChat:
    def __init__(self, model, fail):
        self.model, self.fail = model, fail

    def send_message(self, prompt):
        if self.fail:
            from google.genai import errors
            raise errors.ServerError(503, {"error": {"code": 503, "message": "busy", "status": "UNAVAILABLE"}})
        return type("Resp", (), {"text": f"answer from {self.model}"})()

    def get_history(self, curated=False):
        return []


class FakeClient:
    def __init__(self, busy_models):
        self.busy = set(busy_models)
        self.chats = self

    def create(self, model, history=None, config=None):
        return FakeChat(model, model in self.busy)


def test_switches_to_fallback_model_when_busy():
    from app.agent import ReceptionistAgent
    r = Retriever(load_chunks(config.KB_DIR), fake_embed)
    agent = ReceptionistAgent(FakeClient({"main"}), r, tools=[], model="main", fallback_model="lite")
    out = agent.reply("s1", "check-in time?")
    assert out["model"] == "lite" and out["reply"] == "answer from lite"
    assert agent.reply("s1", "thanks")["model"] == "lite"  # session stays on the fallback


def test_raises_when_both_models_busy():
    from google.genai import errors
    from app.agent import ReceptionistAgent
    r = Retriever(load_chunks(config.KB_DIR), fake_embed)
    agent = ReceptionistAgent(FakeClient({"main", "lite"}), r, tools=[], model="main", fallback_model="lite")
    with pytest.raises(errors.ServerError):
        agent.reply("s1", "hello")


# ---------- API ----------
class FakeAgent:
    def reply(self, session_id, message):
        return {"reply": f"echo: {message}", "sources": []}

    def reset(self, session_id):
        pass


def test_chat_endpoint():
    from app.main import app, state
    state["agent"] = FakeAgent()
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        r = client.post("/api/chat", json={"session_id": "s1", "message": "xin chào"})
        assert r.status_code == 200 and r.json()["reply"] == "echo: xin chào"
        assert client.post("/api/chat", json={"session_id": "s1", "message": ""}).status_code == 422


# ---------- Eval keyword matching ----------
def test_eval_matches_numbers_with_any_separator():
    from scripts.eval_questions import passes
    assert passes("Phí đưa đón là 250.000 VND", ["250000"])
    assert passes("Total: **4,200,000 VND**", ["4200000"])
    assert not passes("Giá 1.500.000 VND", ["150000"])  # no match inside a bigger number


def test_eval_rejects_invented_negative_answer():
    from scripts.eval_questions import INVENTED_NO_VI, NOT_SURE_VI, passes
    invented = "Dạ, khách sạn không có dịch vụ spa ạ. Em không chắc về các dịch vụ khác."
    honest = "Dạ, em không chắc khách sạn có spa không, anh/chị gọi hotline 0236 000 0000 nhé."
    assert not passes(invented, NOT_SURE_VI, INVENTED_NO_VI)
    assert passes(honest, NOT_SURE_VI, INVENTED_NO_VI)

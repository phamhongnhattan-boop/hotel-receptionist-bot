"""Receptionist agent: RAG context + Gemini chat with function calling."""
from __future__ import annotations

from datetime import date

from google.genai import types

from app import config
from app.rag import Retriever

SYSTEM_PROMPT = """You are the virtual receptionist of {hotel}, a hotel in Da Nang, Vietnam.
Today is {today}.

Rules:
- Reply in the same language as the guest (Vietnamese or English). Keep answers short and friendly.
- Answer hotel questions ONLY from the CONTEXT block. If the answer is not there, say you are not
  sure and offer to connect the guest with staff (hotline 0236 000 0000). Never invent facts.
- For prices and free rooms, ALWAYS call check_availability. Do not guess prices.
- To book: collect full name, phone, check-in, check-out, room type. Repeat all details and the
  total price, and ask the guest to confirm. Call create_booking only after the guest says yes.
- Convert dates like "tối mai" or "next Friday" to YYYY-MM-DD using today's date.
"""


class ReceptionistAgent:
    def __init__(self, client, retriever: Retriever, tools: list, model: str = config.CHAT_MODEL):
        self.client = client
        self.retriever = retriever
        self.tools = tools
        self.model = model
        self.sessions: dict[str, object] = {}

    def _chat(self, session_id: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = self.client.chats.create(
                model=self.model,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT.format(
                        hotel=config.HOTEL_NAME, today=date.today().isoformat()),
                    tools=self.tools,  # SDK runs the Python functions automatically
                    temperature=0.3,
                ),
            )
        return self.sessions[session_id]

    def reply(self, session_id: str, message: str) -> dict:
        results = self.retriever.search(message, k=config.TOP_K)
        context = Retriever.format_context(results)
        prompt = f"CONTEXT:\n{context}\n\nGUEST MESSAGE:\n{message}"
        response = self._chat(session_id).send_message(prompt)
        return {
            "reply": response.text or "",
            "sources": [{"title": c.title, "score": round(s, 3)} for c, s in results],
        }

    def reset(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

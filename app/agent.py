"""Receptionist agent: RAG context + Gemini chat with function calling."""
from __future__ import annotations

from datetime import date

from google.genai import errors, types

from app import config
from app.rag import Retriever

SYSTEM_PROMPT = """You are the virtual receptionist of {hotel}, a hotel in Da Nang, Vietnam.
Today is {today}.

Rules:
- Reply in the language of the GUEST MESSAGE (Vietnamese or English). The CONTEXT is written in
  Vietnamese: never let it change your reply language. Keep answers short and friendly.
- Write plain text for a chat window: no Markdown headings or tables. Use **bold** only for key numbers.
- Answer hotel questions ONLY from the CONTEXT block. If the CONTEXT does not mention something,
  do NOT say the hotel has it or does not have it: say you are not sure and offer to connect the
  guest with staff (hotline 0236 000 0000). Never invent facts.
- For prices and free rooms, ALWAYS call check_availability. Do not guess prices.
- To book: collect full name, phone, check-in, check-out, room type. Repeat all details and the
  total price, and ask the guest to confirm. Call create_booking only after the guest says yes.
- Convert dates like "tối mai" or "next Friday" to YYYY-MM-DD using today's date.
"""


def _is_busy(e: errors.APIError) -> bool:
    """True for errors that a different model may not have: overload (5xx) or rate limit (429)."""
    return isinstance(e, errors.ServerError) or getattr(e, "code", None) == 429


class ReceptionistAgent:
    def __init__(self, client, retriever: Retriever, tools: list,
                 model: str = config.CHAT_MODEL, fallback_model: str | None = config.FALLBACK_MODEL):
        self.client = client
        self.retriever = retriever
        self.tools = tools
        self.model = model
        self.fallback_model = fallback_model
        self.sessions: dict[str, tuple[object, str]] = {}  # session_id -> (chat, model name)

    def _new_chat(self, model: str, history=None):
        return self.client.chats.create(
            model=model,
            history=history,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT.format(
                    hotel=config.HOTEL_NAME, today=date.today().isoformat()),
                tools=self.tools,  # SDK runs the Python functions automatically
                temperature=0.3,
            ),
        )

    def _chat(self, session_id: str) -> tuple[object, str]:
        if session_id not in self.sessions:
            self.sessions[session_id] = (self._new_chat(self.model), self.model)
        return self.sessions[session_id]

    def reply(self, session_id: str, message: str) -> dict:
        results = self.retriever.search(message, k=config.TOP_K)
        context = Retriever.format_context(results)
        prompt = f"CONTEXT:\n{context}\n\nGUEST MESSAGE:\n{message}"

        chat, model_used = self._chat(session_id)
        try:
            response = chat.send_message(prompt)
        except errors.APIError as e:
            if not (self.fallback_model and model_used != self.fallback_model and _is_busy(e)):
                raise
            # Main model still busy after retries: continue the same conversation on the fallback model
            print(f"[fallback] {model_used} failed ({e.code}), switching to {self.fallback_model}")
            chat = self._new_chat(self.fallback_model, history=chat.get_history(curated=True))
            model_used = self.fallback_model
            self.sessions[session_id] = (chat, model_used)
            response = chat.send_message(prompt)

        return {
            "reply": response.text or "",
            "sources": [{"title": c.title, "score": round(s, 3)} for c, s in results],
            "model": model_used,
        }

    def reset(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

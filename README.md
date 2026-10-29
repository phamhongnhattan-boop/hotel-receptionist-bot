# AI Hotel Receptionist (Vietnamese + English)

A chatbot that answers hotel questions and takes room bookings, built with Gemini (RAG + function calling) and FastAPI.

> Demo hotel: "Son Tra Demo Hotel" is a made-up hotel. All hotel data is sample data.

**Live demo:** _add your Render URL here_ (free tier: the app sleeps after 15 min idle, the first request can take ~1 min)

## What it does

- Answers questions about check-in time, breakfast, pool, airport transfer, cancellation… in the guest's language (Vietnamese or English).
- Only answers from the hotel knowledge base (RAG). If the answer is not there, it says it is not sure and gives the hotline instead of inventing an answer.
- Checks free rooms and prices, repeats the details, and creates a booking only after the guest confirms (Gemini function calling → SQLite).
- Stays up when Gemini is overloaded: retries with backoff on 429/5xx, then switches to a lighter fallback model and keeps the conversation.

## Architecture

```
Browser (static/index.html)
      │  POST /api/chat
      ▼
FastAPI
      ├── Retriever: gemini-embedding-001 + cosine search over knowledge_base/*.md
      └── Gemini chat (system prompt + retrieved context)
              ├── tools: check_availability(), create_booking()  →  SQLite
              └── retry 429/5xx → fallback model (gemini-flash-lite-latest)
```

## Evaluation

`python -m scripts.eval_questions` sends 30 test questions to the real bot and grades the replies with rules
(expected keywords + forbidden keywords that catch invented answers). Results: [`reports/eval_results.csv`](reports/eval_results.csv).

| Category | What is checked | Result |
|---|---|---|
| facts_vi | Hotel facts, Vietnamese | 12/12 |
| facts_en | Hotel facts, English (reply must stay in English) | 8/8 |
| price_tool | Prices come from `check_availability`, not guessed | 4/4 |
| out_of_kb | Questions not in the data: no invented "we don't have X", hand off to staff | 4/4 |
| language | Reply in the guest's language | 2/2 |
| **Total** | | **30/30** |

What the evaluation found and fixed:
- The bot answered an English question in Vietnamese (the knowledge base is Vietnamese) → the prompt now forces the language of the guest message.
- For "Do you have a spa?" the bot said "we don't have a spa", which is not in the data → the prompt forbids yes/no claims on unknown topics, and the grader now fails such answers.

Limits: keyword grading is simple and LLM replies change between runs; an LLM-as-judge grader would be the next step.

## How to run locally

```bash
python -m venv .venv
.venv\Scripts\activate             # Windows
pip install -r requirements.txt
copy .env.example .env             # then add your GEMINI_API_KEY
python -m uvicorn app.main:app --reload   # open http://127.0.0.1:8000
python -m pytest -q                       # offline tests, no API key needed
python -m scripts.eval_questions          # 30-question evaluation (uses the API)
```

## Deploy

**Render (free, no card):** New → Web Service → connect this GitHub repo → Runtime **Docker** →
Instance type **Free** → add the environment variable `GEMINI_API_KEY` → Deploy.
Render sets `$PORT`, which the `Dockerfile` already uses.

The same `Dockerfile` also runs on **Google Cloud Run**:

```bash
gcloud run deploy hotel-bot --source . --region asia-southeast1 --allow-unauthenticated \
  --set-secrets GEMINI_API_KEY=gemini-api-key:latest
```

## Project structure

```
app/
  config.py    settings from env
  llm.py       Gemini client (retry options) + embeddings
  rag.py       load markdown, split by heading, search
  booking.py   SQLite booking store + tool functions
  agent.py     system prompt, chat sessions, RAG + tools, fallback model
  main.py      FastAPI endpoints
knowledge_base/  hotel info (markdown)
static/          chat UI
scripts/         30-question evaluation
reports/         evaluation results
tests/           offline tests (fake Gemini client)
```

## Limitations and next steps

- Bookings use SQLite in the container, so they reset when the app restarts → move to Firestore/PostgreSQL.
- Chat sessions are kept in memory → move to Redis/Firestore to run several instances.
- No login or rate limit on the public demo.
- Next: LLM-as-judge evaluation, booking cancellation tool, email confirmation, review sentiment analysis.

# AI Hotel Receptionist (Vietnamese + English)

A chatbot that answers hotel questions and takes room bookings, built with Gemini (RAG + function calling) and deployed on Google Cloud Run.

> Demo hotel: "Son Tra Demo Hotel" is a made-up hotel. All hotel data is sample data.

**Live demo:** _add your Cloud Run URL here_

## What it does

- Answers questions about check-in time, breakfast, pool, airport transfer, cancellation… in Vietnamese or English.
- Only answers from the hotel knowledge base (RAG). If the answer is not there, it says so and gives the hotline.
- Checks free rooms and prices, then creates a booking after the guest confirms (Gemini function calling → SQLite).

## Architecture

```
Browser (static/index.html)
      │  POST /api/chat
      ▼
FastAPI (Cloud Run)
      ├── Retriever: gemini-embedding-001 + cosine search over knowledge_base/*.md
      └── Gemini chat (system prompt + context)
              └── tools: check_availability(), create_booking()  →  SQLite
```

## Evaluation

`python -m scripts.eval_questions` sends test questions (VI + EN, including questions the bot should NOT answer) and checks the replies.

| Version | Test questions | Pass rate |
|---|---|---|
| v1 | – | – |

## How to run locally

```bash
python -m venv .venv
.venv\Scripts\activate             # Windows
pip install -r requirements.txt
copy .env.example .env             # then add your GEMINI_API_KEY
uvicorn app.main:app --reload      # open http://127.0.0.1:8000
python -m pytest -q                # offline tests, no API key needed
```

## Deploy to Cloud Run

```bash
gcloud run deploy hotel-bot --source . --region asia-southeast1 \
  --allow-unauthenticated --min-instances 0 --max-instances 1 \
  --set-env-vars GEMINI_API_KEY=YOUR_KEY
```

(Better: store the key in Secret Manager and use `--set-secrets`.)

## Project structure

```
app/
  config.py    settings from env
  llm.py       Gemini client + embeddings
  rag.py       load markdown, split by heading, search
  booking.py   SQLite booking store + tool functions
  agent.py     system prompt, chat sessions, RAG + tools
  main.py      FastAPI endpoints
knowledge_base/  hotel info (markdown)
static/          chat UI
scripts/         evaluation script
tests/           offline tests
```

## Limitations and next steps

- SQLite on Cloud Run is reset when the instance restarts → move bookings to Firestore.
- Chat sessions are kept in memory → move to Firestore/Redis for many instances.
- Next: Dialogflow CX or Vertex AI Agent Builder version, email confirmation, review sentiment analysis.

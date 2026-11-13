FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DB_PATH=/tmp/hotel.db \
    EMBED_CACHE_PATH=/tmp/kb_embeddings.npz

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY knowledge_base ./knowledge_base
COPY static ./static
# Review analysis results shown at /reviews (made offline by scripts/sentiment.py)
COPY reports ./reports

# Run as a normal user, not root: if the app is ever hacked, the attacker has fewer rights
RUN useradd --create-home --uid 10001 appuser
USER appuser

# Cloud Run / Render set $PORT (default 8080)
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --no-server-header

FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    DB_PATH=/tmp/hotel.db

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY knowledge_base ./knowledge_base
COPY static ./static

# Cloud Run sets $PORT (default 8080)
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}

"""Settings from environment variables (.env file in local dev, env vars on Cloud Run)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
KB_DIR = ROOT / "knowledge_base"
STATIC_DIR = ROOT / "static"
DB_PATH = Path(os.getenv("DB_PATH", ROOT / "hotel.db"))

# Option A (easiest): Gemini API key from Google AI Studio
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Option B: Vertex AI on your GCP project (used when USE_VERTEX=true)
USE_VERTEX = os.getenv("USE_VERTEX", "false").lower() == "true"
GCP_PROJECT = os.getenv("GCP_PROJECT")
GCP_LOCATION = os.getenv("GCP_LOCATION", "asia-southeast1")

# "-latest" aliases always point to the newest model of that family.
# Lite is the default: fast, and it scored 30/30 on the evaluation set.
CHAT_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
# Used when the main model is overloaded (503) or rate-limited (429) even after retries
FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-flash-latest")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
# Gemini "thinking" before answering: slower. 0 = off. Empty = model default.
THINKING_BUDGET = int(os.environ["THINKING_BUDGET"]) if os.getenv("THINKING_BUDGET") else None

TOP_K = int(os.getenv("TOP_K", "3"))
HOTEL_NAME = os.getenv("HOTEL_NAME", "Son Tra Demo Hotel")

# ---------- Security (public demo) ----------
# Messages per client IP per minute, and total chat messages per day (protects the API budget)
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
DAILY_MESSAGE_LIMIT = int(os.getenv("DAILY_MESSAGE_LIMIT", "500"))
# Number of reverse proxies in front of the app (Render / Cloud Run: 1). 0 = no proxy.
PROXY_HOPS = int(os.getenv("PROXY_HOPS", "1"))
# Chat sessions kept in memory: max count and idle timeout
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "500"))
SESSION_TTL_MINUTES = int(os.getenv("SESSION_TTL_MINUTES", "30"))
# Max bookings one chat session can create (stops scripts from filling all rooms)
MAX_BOOKINGS_PER_SESSION = int(os.getenv("MAX_BOOKINGS_PER_SESSION", "2"))
# Wrong booking code / phone tries per chat before lookup and cancel are blocked (stops guessing)
MAX_FAILED_LOOKUPS = int(os.getenv("MAX_FAILED_LOOKUPS", "3"))
# Bookings only up to this many days ahead
MAX_DAYS_AHEAD = int(os.getenv("MAX_DAYS_AHEAD", "365"))
# /api/stats needs "Authorization: Bearer <STATS_TOKEN>". Not set = the endpoint does not exist (404).
STATS_TOKEN = os.getenv("STATS_TOKEN")
# FastAPI /docs page: off on the public demo, turn on for local development
ENABLE_DOCS = os.getenv("ENABLE_DOCS", "false").lower() == "true"

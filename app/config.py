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

# Check the current model names in the Gemini docs before you start
CHAT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")

TOP_K = int(os.getenv("TOP_K", "3"))
HOTEL_NAME = os.getenv("HOTEL_NAME", "Son Tra Demo Hotel")

"""Gemini client + embedding function."""
from __future__ import annotations

import numpy as np
from google import genai
from google.genai import types

from app import config


def get_client() -> genai.Client:
    if config.USE_VERTEX:
        return genai.Client(vertexai=True, project=config.GCP_PROJECT, location=config.GCP_LOCATION)
    if not config.GEMINI_API_KEY:
        raise RuntimeError("Set GEMINI_API_KEY in .env, or USE_VERTEX=true with GCP_PROJECT.")
    return genai.Client(api_key=config.GEMINI_API_KEY)


def make_embedder(client: genai.Client):
    """Return a function: (texts, task_type) -> L2-normalized array (n, dim)."""

    def embed(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        vectors = []
        for i in range(0, len(texts), 50):  # small batches
            resp = client.models.embed_content(
                model=config.EMBED_MODEL,
                contents=texts[i:i + 50],
                config=types.EmbedContentConfig(task_type=task_type),
            )
            vectors.extend(e.values for e in resp.embeddings)
        arr = np.asarray(vectors, dtype=np.float32)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)

    return embed

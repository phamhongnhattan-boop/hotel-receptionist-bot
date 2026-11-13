"""Small in-memory RAG: split markdown files by '## ' headings, embed, cosine search."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

Embedder = Callable[[list[str], str], np.ndarray]
log = logging.getLogger(__name__)


def cached_embedder(embed: Embedder, cache_path: Path | None, model: str = "") -> Embedder:
    """Keep knowledge-base vectors on disk, so a restart does not embed the same text again.

    Why: the free tier allows 1000 embedding requests per day. Without a cache, every start of
    the bot (and every scenario in the eval) re-embedded all sections and used up that quota.
    Only RETRIEVAL_DOCUMENT vectors are cached (guest questions change every time).
    The key is a hash of model + text, so editing a section re-embeds only that section.
    Saved with numpy .npz and loaded with allow_pickle=False (a cache file cannot run code).
    """
    if cache_path is None:
        return embed

    def key(text: str) -> str:
        return hashlib.sha256(f"{model}\n{text}".encode()).hexdigest()

    def load() -> dict[str, np.ndarray]:
        try:
            with np.load(cache_path, allow_pickle=False) as f:
                return dict(zip(f["keys"].tolist(), f["vectors"]))
        except (OSError, KeyError, ValueError):
            return {}

    def wrapped(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        if task_type != "RETRIEVAL_DOCUMENT":
            return embed(texts, task_type)
        cache = load()
        keys = [key(t) for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in cache]
        if missing:
            new = embed([texts[i] for i in missing], task_type)
            for i, v in zip(missing, new):
                cache[keys[i]] = v
            try:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                np.savez(cache_path, keys=np.array(list(cache)), vectors=np.stack(list(cache.values())))
            except OSError as e:  # read-only disk: still works, just without the cache
                log.warning("embedding cache not saved: %s", e)
        log.info("knowledge base: %d sections, %d embedded now, %d from cache",
                 len(texts), len(missing), len(texts) - len(missing))
        return np.stack([cache[k] for k in keys])

    return wrapped


@dataclass
class Chunk:
    source: str
    title: str
    text: str


def load_chunks(kb_dir: Path) -> list[Chunk]:
    chunks = []
    for path in sorted(kb_dir.glob("*.md")):
        title, lines = None, []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                if title and any(l.strip() for l in lines):
                    chunks.append(Chunk(path.name, title, "\n".join(lines).strip()))
                title, lines = line[3:].strip(), []
            elif title:
                lines.append(line)
        if title and any(l.strip() for l in lines):
            chunks.append(Chunk(path.name, title, "\n".join(lines).strip()))
    return chunks


class Retriever:
    def __init__(self, chunks: list[Chunk], embed: Embedder):
        if not chunks:
            raise ValueError("Knowledge base is empty.")
        self.chunks = chunks
        self.embed = embed
        docs = [f"{c.title}\n{c.text}" for c in chunks]
        self.matrix = embed(docs, "RETRIEVAL_DOCUMENT")

    def search(self, query: str, k: int = 3, min_score: float = 0.0) -> list[tuple[Chunk, float]]:
        q = self.embed([query], "RETRIEVAL_QUERY")[0]
        scores = self.matrix @ q
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top if scores[i] >= min_score]

    @staticmethod
    def format_context(results: list[tuple[Chunk, float]]) -> str:
        return "\n\n".join(f"[{c.title}]\n{c.text}" for c, _ in results)

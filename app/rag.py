"""Small in-memory RAG: split markdown files by '## ' headings, embed, cosine search."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

Embedder = Callable[[list[str], str], np.ndarray]


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

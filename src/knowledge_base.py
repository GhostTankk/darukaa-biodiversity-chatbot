"""
Loads the curated JSON knowledge base (and any ingested PDF chunks), embeds
every entry with a local sentence-transformer model, and builds a FAISS
index for semantic retrieval.

Why local embeddings instead of an API: retrieval runs on every single turn
of the conversation, so keeping it free and fast (and offline-capable) on
your own GPU/CPU is the right tradeoff -- the paid API call is reserved for
the one place it actually matters, the reasoning step (see reasoning_engine.py).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .schemas import KnowledgeChunk

DEFAULT_KB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base.json")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, runs fine on CPU or the 4060


@dataclass
class RawEntry:
    id: str
    text: str
    source: str
    metadata: dict


def _entry_to_text(entry: dict) -> str:
    """
    Flattens one knowledge-base entry into a single text blob for embedding.
    Including the trigger conditions and metrics in the text (not just the
    recommendation) means a query like "low rainfall semi-arid wheat" will
    match on those keywords even though they're not in the 'recommendation'
    sentence itself.
    """
    conditions = json.dumps(entry.get("trigger_conditions", {}))
    metrics = ", ".join(entry.get("impacted_metrics", []))
    return (
        f"Topic: {entry['topic']}. Conditions: {conditions}. "
        f"Recommendation: {entry['recommendation']} "
        f"Reasoning: {entry['reasoning']} "
        f"Impacted metrics: {metrics}. "
        f"Expected effect: {entry.get('expected_effect', '')}"
    )


class KnowledgeBase:
    def __init__(self, kb_path: str = DEFAULT_KB_PATH, model_name: str = EMBEDDING_MODEL_NAME):
        self.kb_path = kb_path
        self.model = SentenceTransformer(model_name)
        self.entries: list[dict] = []
        self.raw_entries: list[RawEntry] = []
        self.index: faiss.IndexFlatIP | None = None
        self._load_and_index()

    def _load_and_index(self) -> None:
        with open(self.kb_path, "r", encoding="utf-8") as f:
            self.entries = json.load(f)

        self.raw_entries = [
            RawEntry(
                id=e["id"],
                text=_entry_to_text(e),
                source="; ".join(s["name"] for s in e.get("sources", [])),
                metadata=e,
            )
            for e in self.entries
        ]

        vectors = self._embed([r.text for r in self.raw_entries])
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized vectors
        self.index.add(vectors)

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype("float32")

    def add_external_chunks(self, chunks: list[RawEntry]) -> None:
        """Used by pdf_ingest.py to fold ingested PDF chunks into the same index."""
        self.raw_entries.extend(chunks)
        vectors = self._embed([c.text for c in chunks])
        self.index.add(vectors)

    def search(self, query: str, top_k: int = 4) -> list[KnowledgeChunk]:
        if self.index is None or self.index.ntotal == 0:
            return []
        query_vec = self._embed([query])
        scores, idxs = self.index.search(query_vec, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            entry = self.raw_entries[idx]
            results.append(
                KnowledgeChunk(id=entry.id, text=entry.text, source=entry.source, score=float(score))
            )
        return results

    def get_full_entry(self, entry_id: str) -> dict | None:
        for e in self.entries:
            if e["id"] == entry_id:
                return e
        return None

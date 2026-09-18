"""
Optional ingestion path: drop PDFs (research papers, reports) into
data/pdfs/, run this script, and their chunks get embedded and added to the
same FAISS index used by the curated JSON knowledge base. This satisfies
the "index research papers/reports" expectation beyond the hand-written
JSON entries, and shows the retrieval pipeline working over unstructured
sources too.

Chunking strategy: fixed-size word windows with overlap. Simple and
dependency-light; swap for a semantic/section-based chunker if your PDFs
have exploitable structure (headings, abstracts, etc).
"""

from __future__ import annotations

import glob
import os

from pypdf import PdfReader

from .knowledge_base import KnowledgeBase, RawEntry

PDF_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "pdfs")
CHUNK_WORDS = 220
OVERLAP_WORDS = 40


def extract_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text: str, chunk_words: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_words
        chunks.append(" ".join(words[start:end]))
        start += chunk_words - overlap
    return [c for c in chunks if c.strip()]


def ingest_all_pdfs(kb: KnowledgeBase, pdf_dir: str = PDF_DIR) -> int:
    """Returns the number of chunks added."""
    pdf_paths = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    new_entries: list[RawEntry] = []

    for path in pdf_paths:
        filename = os.path.basename(path)
        text = extract_text(path)
        for i, chunk in enumerate(chunk_text(text)):
            new_entries.append(
                RawEntry(
                    id=f"pdf::{filename}::chunk{i}",
                    text=chunk,
                    source=filename,
                    metadata={"file": filename, "chunk_index": i},
                )
            )

    if new_entries:
        kb.add_external_chunks(new_entries)
    return len(new_entries)


if __name__ == "__main__":
    kb = KnowledgeBase()
    n = ingest_all_pdfs(kb)
    print(f"Ingested {n} chunks from PDFs in {PDF_DIR}")
    print(f"Total entries in index now: {kb.index.ntotal}")

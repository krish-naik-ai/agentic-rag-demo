from pathlib import Path

import pytest

from agentic_rag.models import DocumentChunk
from agentic_rag.vector_store import ChromaVectorStore


def test_chroma_vector_store_upserts_and_searches(tmp_path: Path) -> None:
    store = ChromaVectorStore(tmp_path / "chroma")
    chunks = [
        DocumentChunk(
            id="agentic",
            source="guide.txt",
            text="Agents can decide when to retrieve.",
            chunk_index=0,
        ),
        DocumentChunk(
            id="baseline",
            source="guide.txt",
            text="A baseline always retrieves once.",
            chunk_index=1,
            page=3,
        ),
    ]
    store.upsert(chunks, [[1.0, 0.0], [0.0, 1.0]])

    results = store.search([1.0, 0.0], top_k=2)

    assert [result.chunk.id for result in results] == ["agentic", "baseline"]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].chunk.page == 3


def test_chroma_vector_store_requires_matching_embeddings(tmp_path: Path) -> None:
    store = ChromaVectorStore(tmp_path / "chroma")
    chunk = DocumentChunk(
        id="chunk",
        source="guide.txt",
        text="content",
        chunk_index=0,
    )

    with pytest.raises(ValueError, match="exactly one embedding"):
        store.upsert([chunk], [])


def test_chroma_vector_store_returns_existing_ids(tmp_path: Path) -> None:
    store = ChromaVectorStore(tmp_path / "chroma")
    chunk = DocumentChunk(
        id="stored",
        source="guide.txt",
        text="content",
        chunk_index=0,
    )
    store.upsert([chunk], [[1.0, 0.0]])

    existing_ids = store.existing_ids(["missing", "stored", "stored"])

    assert existing_ids == {"stored"}

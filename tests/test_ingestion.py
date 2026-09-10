from collections.abc import Sequence
from pathlib import Path

import pytest

from agentic_rag import ingestion
from agentic_rag.ingestion import DocumentIngestionPipeline, chunk_documents
from agentic_rag.models import DocumentChunk, SearchResult, SourceDocument


class FakeEmbeddings:
    identifier = "fake-embeddings"

    def __init__(self) -> None:
        self.document_calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(texts)
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0]


class FakeVectorStore:
    embedding_identifier = FakeEmbeddings.identifier

    def __init__(self) -> None:
        self.chunks: list[DocumentChunk] = []
        self.embeddings: list[list[float]] = []
        self.upsert_calls = 0

    def existing_ids(self, ids: Sequence[str]) -> set[str]:
        stored_ids = {chunk.id for chunk in self.chunks}
        return stored_ids.intersection(ids)

    def upsert(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        self.upsert_calls += 1
        self.chunks.extend(chunks)
        self.embeddings.extend(list(embedding) for embedding in embeddings)

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 4,
    ) -> list[SearchResult]:
        return []


def test_load_text_document(tmp_path: Path) -> None:
    document_path = tmp_path / "notes.txt"
    document_path.write_text("Agentic retrieval uses tools.", encoding="utf-8")

    documents = ingestion.load_document(document_path)

    assert documents == [
        SourceDocument(
            source="notes.txt",
            text="Agentic retrieval uses tools.",
        )
    ]


def test_load_pdf_pages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakePage:
        def __init__(self, text: str | None) -> None:
            self._text = text

        def extract_text(self) -> str | None:
            return self._text

    class FakeReader:
        def __init__(self, path: Path) -> None:
            assert path.name == "guide.pdf"
            self.pages = [FakePage("First page"), FakePage(None)]

    monkeypatch.setattr(ingestion, "PdfReader", FakeReader)

    documents = ingestion.load_document(tmp_path / "guide.pdf")

    assert documents == [
        SourceDocument(source="guide.pdf", page=1, text="First page"),
        SourceDocument(source="guide.pdf", page=2, text=""),
    ]


def test_chunk_documents_preserves_overlap_and_metadata() -> None:
    documents = [
        SourceDocument(
            source="notes.txt",
            text="alpha beta gamma delta epsilon zeta eta theta",
            page=2,
        )
    ]

    chunks = chunk_documents(documents, chunk_size=24, chunk_overlap=5)

    assert [chunk.text for chunk in chunks] == [
        "alpha beta gamma delta",
        "delta epsilon zeta eta",
        "a eta theta",
    ]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert all(chunk.source == "notes.txt" for chunk in chunks)
    assert all(chunk.page == 2 for chunk in chunks)
    assert len({chunk.id for chunk in chunks}) == 3


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (10, -1), (10, 10)],
)
def test_chunk_documents_rejects_invalid_sizes(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    with pytest.raises(ValueError):
        chunk_documents(
            [SourceDocument(source="notes.txt", text="content")],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


def test_ingestion_pipeline_embeds_and_stores_chunks(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("one two three four five", encoding="utf-8")
    embeddings = FakeEmbeddings()
    vector_store = FakeVectorStore()
    pipeline = DocumentIngestionPipeline(
        embeddings=embeddings,
        vector_store=vector_store,
        chunk_size=12,
        chunk_overlap=2,
    )

    result = pipeline.ingest([path])

    assert result.documents_loaded == 1
    assert result.chunks_stored == len(vector_store.chunks)
    assert result.chunks_stored > 1
    assert vector_store.embeddings == [
        [float(len(chunk.text)), 1.0] for chunk in vector_store.chunks
    ]
    assert embeddings.document_calls == [[chunk.text for chunk in vector_store.chunks]]


def test_ingestion_pipeline_skips_chunks_already_in_store(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("one two three four five", encoding="utf-8")
    embeddings = FakeEmbeddings()
    vector_store = FakeVectorStore()
    pipeline = DocumentIngestionPipeline(
        embeddings=embeddings,
        vector_store=vector_store,
        chunk_size=12,
        chunk_overlap=2,
    )

    first_result = pipeline.ingest([path])
    second_result = pipeline.ingest([path])

    assert second_result == first_result
    assert len(embeddings.document_calls) == 1
    assert vector_store.upsert_calls == 1


def test_ingestion_pipeline_rejects_mismatched_embedding_store() -> None:
    embeddings = FakeEmbeddings()
    vector_store = FakeVectorStore()
    vector_store.embedding_identifier = "different-embeddings"

    with pytest.raises(ValueError, match="identifiers must match"):
        DocumentIngestionPipeline(
            embeddings=embeddings,
            vector_store=vector_store,
        )


def test_load_document_rejects_unknown_type(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported document type"):
        ingestion.load_document(tmp_path / "notes.md")

from collections.abc import Sequence
from pathlib import Path
from time import sleep

import pytest

from agentic_rag import ingestion
from agentic_rag.ingestion import (
    DocumentIngestionLimits,
    DocumentIngestionPipeline,
    DocumentLimitError,
    chunk_documents,
)
from agentic_rag.models import DocumentChunk, SearchResult, SourceDocument


class FakeEmbeddings:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(texts)
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0]


class FakeVectorStore:
    def __init__(self) -> None:
        self.chunks: list[DocumentChunk] = []
        self.embeddings: list[list[float]] = []
        self.upsert_calls: list[list[DocumentChunk]] = []

    def upsert(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        stored_chunks = list(chunks)
        stored_embeddings = [list(embedding) for embedding in embeddings]
        self.chunks.extend(stored_chunks)
        self.embeddings.extend(stored_embeddings)
        self.upsert_calls.append(stored_chunks)

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


def test_load_text_document_rejects_oversized_file(tmp_path: Path) -> None:
    document_path = tmp_path / "notes.txt"
    document_path.write_text("too large", encoding="utf-8")

    with pytest.raises(DocumentLimitError, match="exceeds the 4 byte"):
        ingestion.load_document(
            document_path,
            limits=DocumentIngestionLimits(max_document_bytes=4),
        )


def test_load_document_rejects_symlink(tmp_path: Path) -> None:
    target_path = tmp_path / "notes.txt"
    target_path.write_text("content", encoding="utf-8")
    linked_path = tmp_path / "linked.txt"
    linked_path.symlink_to(target_path)

    with pytest.raises(DocumentLimitError, match="symlinked document"):
        ingestion.load_document(linked_path)


def test_load_document_rejects_paths_outside_base_directory(tmp_path: Path) -> None:
    allowed_directory = tmp_path / "uploads"
    allowed_directory.mkdir()
    document_path = tmp_path / "notes.txt"
    document_path.write_text("content", encoding="utf-8")

    with pytest.raises(DocumentLimitError, match="outside the allowed"):
        ingestion.load_document(
            document_path,
            limits=DocumentIngestionLimits(base_directory=allowed_directory),
        )


def test_load_pdf_pages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakePage:
        def __init__(self, text: str | None) -> None:
            self._text = text

        def extract_text(
            self,
            visitor_text: ingestion.PdfTextVisitor | None = None,
        ) -> str | None:
            if visitor_text is not None and self._text is not None:
                visitor_text(self._text, object(), object(), object(), object())
            return self._text

    class FakeReader:
        def __init__(self, path: Path) -> None:
            assert path.name == "guide.pdf"
            self.pages = [FakePage("First page"), FakePage(None)]

    document_path = tmp_path / "guide.pdf"
    document_path.write_bytes(b"%PDF-1.7")
    monkeypatch.setattr(ingestion, "PdfReader", FakeReader)

    documents = ingestion.load_document(document_path)

    assert documents == [
        SourceDocument(source="guide.pdf", page=1, text="First page"),
        SourceDocument(source="guide.pdf", page=2, text=""),
    ]


def test_load_pdf_document_rejects_too_many_pages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeReader:
        def __init__(self, path: Path) -> None:
            assert path.name == "guide.pdf"
            self.pages = [object(), object()]

    document_path = tmp_path / "guide.pdf"
    document_path.write_bytes(b"%PDF-1.7")
    monkeypatch.setattr(ingestion, "PdfReader", FakeReader)

    with pytest.raises(DocumentLimitError, match="exceeds the 1 page"):
        ingestion.load_document(
            document_path,
            limits=DocumentIngestionLimits(max_pdf_pages=1),
        )


def test_load_pdf_document_rejects_extracted_text_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakePage:
        def extract_text(
            self,
            visitor_text: ingestion.PdfTextVisitor | None = None,
        ) -> str:
            if visitor_text is not None:
                visitor_text("abcdef", object(), object(), object(), object())
            return "abcdef"

    class FakeReader:
        def __init__(self, path: Path) -> None:
            assert path.name == "guide.pdf"
            self.pages = [FakePage()]

    document_path = tmp_path / "guide.pdf"
    document_path.write_bytes(b"%PDF-1.7")
    monkeypatch.setattr(ingestion, "PdfReader", FakeReader)

    with pytest.raises(DocumentLimitError, match="character extracted-text"):
        ingestion.load_document(
            document_path,
            limits=DocumentIngestionLimits(max_extracted_text_chars=5),
        )


def test_load_pdf_document_rejects_parse_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeReader:
        def __init__(self, path: Path) -> None:
            assert path.name == "guide.pdf"
            sleep(1)
            self.pages: list[object] = []

    document_path = tmp_path / "guide.pdf"
    document_path.write_bytes(b"%PDF-1.7")
    monkeypatch.setattr(ingestion, "PdfReader", FakeReader)

    with pytest.raises(DocumentLimitError, match="second parsing limit"):
        ingestion.load_document(
            document_path,
            limits=DocumentIngestionLimits(pdf_parse_timeout_seconds=0.01),
        )


def test_load_pdf_document_receives_bounded_large_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    extracted_text = "x" * (256 * 1024)

    class FakePage:
        def extract_text(
            self,
            visitor_text: ingestion.PdfTextVisitor,
        ) -> str:
            visitor_text(extracted_text, None, None, None, None)
            return extracted_text

    class FakeReader:
        def __init__(self, path: Path) -> None:
            assert path.name == "guide.pdf"
            self.pages = [FakePage()]

    document_path = tmp_path / "guide.pdf"
    document_path.write_bytes(b"%PDF-1.7")
    monkeypatch.setattr(ingestion, "PdfReader", FakeReader)

    documents = ingestion.load_document(
        document_path,
        limits=DocumentIngestionLimits(
            max_extracted_text_chars=len(extracted_text),
            pdf_parse_timeout_seconds=1,
        ),
    )

    assert documents[0].text == extracted_text


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
    vector_store = FakeVectorStore()
    pipeline = DocumentIngestionPipeline(
        embeddings=FakeEmbeddings(),
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


def test_ingestion_pipeline_batches_chunks(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("one two three four five", encoding="utf-8")
    embeddings = FakeEmbeddings()
    vector_store = FakeVectorStore()
    pipeline = DocumentIngestionPipeline(
        embeddings=embeddings,
        vector_store=vector_store,
        chunk_size=12,
        chunk_overlap=2,
        ingestion_batch_size=1,
    )

    result = pipeline.ingest([path])

    assert result.documents_loaded == 1
    assert result.chunks_stored == len(vector_store.chunks)
    assert result.chunks_stored > 1
    assert len(vector_store.upsert_calls) == result.chunks_stored
    assert all(len(batch) == 1 for batch in embeddings.batches)


def test_load_document_rejects_unknown_type(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported document type"):
        ingestion.load_document(tmp_path / "notes.md")

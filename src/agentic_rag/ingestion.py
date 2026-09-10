import hashlib
import re
from collections.abc import Sequence
from pathlib import Path

from pypdf import PdfReader

from agentic_rag.embeddings import EmbeddingProvider
from agentic_rag.models import (
    DocumentChunk,
    IngestionResult,
    SourceDocument,
)
from agentic_rag.vector_store import VectorStore

SUPPORTED_SUFFIXES = {".pdf", ".txt"}


def load_document(path: Path) -> list[SourceDocument]:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return [
            SourceDocument(
                source=path.name,
                text=path.read_text(encoding="utf-8-sig"),
            )
        ]
    if suffix == ".pdf":
        reader = PdfReader(path)
        return [
            SourceDocument(
                source=path.name,
                page=page_number,
                text=page.extract_text() or "",
            )
            for page_number, page in enumerate(reader.pages, start=1)
        ]
    raise ValueError(
        f"Unsupported document type '{suffix}'. Expected one of "
        f"{sorted(SUPPORTED_SUFFIXES)}."
    )


def chunk_documents(
    documents: Sequence[SourceDocument],
    chunk_size: int = 1_000,
    chunk_overlap: int = 150,
) -> list[DocumentChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and less than chunk_size.")

    chunks: list[DocumentChunk] = []
    for document in documents:
        text = re.sub(r"\s+", " ", document.text).strip()
        if not text:
            continue
        chunks.extend(
            _chunk_document(
                document=document,
                text=text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return chunks


def _chunk_document(
    document: SourceDocument,
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            break_at = text.rfind(" ", start + (chunk_size // 2), end)
            if break_at > start:
                end = break_at

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunk_id = hashlib.sha256(
                (
                    f"{document.source}:{document.page}:{chunk_index}:{chunk_text}"
                ).encode()
            ).hexdigest()
            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    source=document.source,
                    page=document.page,
                    text=chunk_text,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1

        if end == len(text):
            break
        start = max(end - chunk_overlap, start + 1)

    return chunks


class DocumentIngestionPipeline:
    def __init__(
        self,
        embeddings: EmbeddingProvider,
        vector_store: VectorStore,
        chunk_size: int = 1_000,
        chunk_overlap: int = 150,
    ) -> None:
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def ingest(self, paths: Sequence[Path]) -> IngestionResult:
        documents = [document for path in paths for document in load_document(path)]
        chunks = chunk_documents(
            documents,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        embeddings = self._embeddings.embed_documents([chunk.text for chunk in chunks])
        self._vector_store.upsert(chunks, embeddings)
        return IngestionResult(
            documents_loaded=len(documents),
            chunks_stored=len(chunks),
        )

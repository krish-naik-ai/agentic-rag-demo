from dataclasses import dataclass


@dataclass(frozen=True)
class SourceDocument:
    source: str
    text: str
    page: int | None = None


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    source: str
    text: str
    chunk_index: int
    page: int | None = None


@dataclass(frozen=True)
class SearchResult:
    chunk: DocumentChunk
    score: float


@dataclass(frozen=True)
class IngestionResult:
    documents_loaded: int
    chunks_stored: int

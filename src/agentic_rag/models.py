from dataclasses import dataclass
from typing import Literal


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


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class Citation:
    id: str
    source: str
    chunk_index: int
    text: str
    page: int | None = None

    @property
    def label(self) -> str:
        page = f", page {self.page}" if self.page is not None else ""
        return f"{self.source}{page}, chunk {self.chunk_index}"


@dataclass(frozen=True)
class AgentAnswer:
    answer: str
    citations: tuple[Citation, ...]
    used_retrieval: bool
    retrieval_query: str | None = None

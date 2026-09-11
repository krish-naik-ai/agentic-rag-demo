import hashlib
import multiprocessing
import re
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from multiprocessing.connection import Connection
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
PdfTextVisitor = Callable[[str, object, object, object, object], None]
DEFAULT_MAX_DOCUMENT_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_PDF_PAGES = 100
DEFAULT_MAX_EXTRACTED_TEXT_CHARS = 2_000_000
DEFAULT_PDF_PARSE_TIMEOUT_SECONDS = 30.0
DEFAULT_PDF_PARSE_MEMORY_BYTES = 1024 * 1024 * 1024
DEFAULT_INGESTION_BATCH_SIZE = 32
TEXT_READ_CHUNK_BYTES = 64 * 1024


class DocumentLimitError(ValueError):
    pass


class _ExtractedTextLimitReached(Exception):
    pass


@dataclass(frozen=True)
class _PdfParseOutcome:
    documents: list[SourceDocument]
    error_message: str | None = None
    error_kind: str | None = None


@dataclass(frozen=True)
class DocumentIngestionLimits:
    max_document_bytes: int = DEFAULT_MAX_DOCUMENT_BYTES
    max_pdf_pages: int = DEFAULT_MAX_PDF_PAGES
    max_extracted_text_chars: int = DEFAULT_MAX_EXTRACTED_TEXT_CHARS
    pdf_parse_timeout_seconds: float = DEFAULT_PDF_PARSE_TIMEOUT_SECONDS
    pdf_parse_memory_bytes: int = DEFAULT_PDF_PARSE_MEMORY_BYTES
    base_directory: Path | None = None
    allow_symlinks: bool = False

    def __post_init__(self) -> None:
        if self.max_document_bytes <= 0:
            raise ValueError("max_document_bytes must be positive.")
        if self.max_pdf_pages <= 0:
            raise ValueError("max_pdf_pages must be positive.")
        if self.max_extracted_text_chars <= 0:
            raise ValueError("max_extracted_text_chars must be positive.")
        if self.pdf_parse_timeout_seconds <= 0:
            raise ValueError("pdf_parse_timeout_seconds must be positive.")
        if self.pdf_parse_memory_bytes <= 0:
            raise ValueError("pdf_parse_memory_bytes must be positive.")


def load_document(
    path: Path,
    limits: DocumentIngestionLimits | None = None,
) -> list[SourceDocument]:
    active_limits = limits or DocumentIngestionLimits()
    return list(_iter_document(path, active_limits))


def _iter_document(
    path: Path,
    limits: DocumentIngestionLimits,
) -> Iterator[SourceDocument]:
    suffix = path.suffix.lower()
    _validate_document_path(path, limits)
    if suffix == ".txt":
        yield SourceDocument(
            source=path.name,
            text=_read_limited_text(path, limits.max_document_bytes),
        )
        return
    if suffix == ".pdf":
        yield from _iter_pdf_pages(path, limits)
        return
    raise ValueError(
        f"Unsupported document type '{suffix}'. Expected one of "
        f"{sorted(SUPPORTED_SUFFIXES)}."
    )


def _validate_document_path(path: Path, limits: DocumentIngestionLimits) -> None:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported document type '{suffix}'. Expected one of "
            f"{sorted(SUPPORTED_SUFFIXES)}."
        )

    if path.is_symlink() and not limits.allow_symlinks:
        raise DocumentLimitError(
            f"Refusing to ingest symlinked document '{path.name}'."
        )

    resolved_path = path.resolve(strict=True)
    if limits.base_directory is not None:
        resolved_base = limits.base_directory.resolve(strict=True)
        inside_base = (
            resolved_path == resolved_base or resolved_base in resolved_path.parents
        )
        if not inside_base:
            raise DocumentLimitError(
                f"Document '{path.name}' is outside the allowed ingestion directory."
            )

    size = resolved_path.stat().st_size
    if size > limits.max_document_bytes:
        raise DocumentLimitError(
            f"Document '{path.name}' is {size} bytes, which exceeds the "
            f"{limits.max_document_bytes} byte ingestion limit."
        )


def _read_limited_text(path: Path, max_bytes: int) -> str:
    chunks: list[bytes] = []
    bytes_read = 0
    with path.open("rb") as file:
        while chunk := file.read(TEXT_READ_CHUNK_BYTES):
            bytes_read += len(chunk)
            if bytes_read > max_bytes:
                raise DocumentLimitError(
                    f"Text document '{path.name}' exceeds the {max_bytes} byte "
                    "ingestion limit."
                )
            chunks.append(chunk)
    return b"".join(chunks).decode("utf-8-sig")


def _iter_pdf_pages(
    path: Path,
    limits: DocumentIngestionLimits,
) -> Iterator[SourceDocument]:
    yield from _load_pdf_pages_with_timeout(path, limits)


def _load_pdf_pages_with_timeout(
    path: Path,
    limits: DocumentIngestionLimits,
) -> list[SourceDocument]:
    parent_connection, child_connection = multiprocessing.Pipe(duplex=False)
    process = multiprocessing.Process(
        target=_parse_pdf_pages_worker,
        args=(path, limits, child_connection),
    )
    process.start()
    child_connection.close()
    try:
        raw_outcome = _receive_pdf_parse_outcome(
            process,
            parent_connection,
            path,
            limits.pdf_parse_timeout_seconds,
        )
        if not isinstance(raw_outcome, _PdfParseOutcome):
            raise DocumentLimitError(
                f"PDF '{path.name}' parsing returned an invalid result."
            )
        outcome = raw_outcome
        if outcome.error_message is not None:
            if outcome.error_kind == "limit":
                raise DocumentLimitError(outcome.error_message)
            raise RuntimeError(
                f"Failed to parse PDF '{path.name}': {outcome.error_message}"
            )
        process.join(1)
        return outcome.documents
    finally:
        _stop_pdf_parse_process(process)
        parent_connection.close()
        process.close()


def _receive_pdf_parse_outcome(
    process: multiprocessing.Process,
    connection: Connection,
    path: Path,
    timeout_seconds: float,
) -> object:
    deadline = time.monotonic() + timeout_seconds
    while True:
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            raise DocumentLimitError(
                f"PDF '{path.name}' exceeded the "
                f"{timeout_seconds:g} second parsing limit."
            )
        if connection.poll(min(remaining_seconds, 0.05)):
            try:
                return connection.recv()
            except EOFError as error:
                raise DocumentLimitError(
                    f"PDF '{path.name}' parsing failed before returning a result."
                ) from error
        if not process.is_alive():
            raise DocumentLimitError(
                f"PDF '{path.name}' parsing failed before returning a result."
            )


def _stop_pdf_parse_process(process: multiprocessing.Process) -> None:
    if not process.is_alive():
        return
    process.terminate()
    process.join(1)
    if process.is_alive():
        process.kill()
        process.join(1)


def _parse_pdf_pages_worker(
    path: Path,
    limits: DocumentIngestionLimits,
    connection: Connection,
) -> None:
    try:
        _apply_pdf_worker_memory_limit(limits)
        documents = _parse_pdf_pages(path, limits)
        connection.send(_PdfParseOutcome(documents=documents))
    except DocumentLimitError as error:
        connection.send(
            _PdfParseOutcome(
                documents=[],
                error_message=str(error),
                error_kind="limit",
            )
        )
    except Exception as error:
        connection.send(
            _PdfParseOutcome(
                documents=[],
                error_message=str(error),
                error_kind="runtime",
            )
        )
    finally:
        connection.close()


def _apply_pdf_worker_memory_limit(limits: DocumentIngestionLimits) -> None:
    try:
        import resource
    except ImportError:
        return

    _, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
    if hard_limit == resource.RLIM_INFINITY:
        hard_limit = limits.pdf_parse_memory_bytes
    soft_limit = min(limits.pdf_parse_memory_bytes, hard_limit)
    resource.setrlimit(resource.RLIMIT_AS, (soft_limit, hard_limit))


def _parse_pdf_pages(
    path: Path,
    limits: DocumentIngestionLimits,
) -> list[SourceDocument]:
    started_at = time.monotonic()
    total_text_chars = 0
    documents: list[SourceDocument] = []
    reader = PdfReader(path)
    page_count = len(reader.pages)
    if page_count > limits.max_pdf_pages:
        raise DocumentLimitError(
            f"PDF '{path.name}' has {page_count} pages, which exceeds "
            f"the {limits.max_pdf_pages} page ingestion limit."
        )
    for page_number, page in enumerate(reader.pages, start=1):
        _raise_if_pdf_parse_timed_out(started_at, limits, path.name)
        text_limit_visitor = _make_text_limit_visitor(
            total_text_chars,
            limits.max_extracted_text_chars,
        )

        try:
            text = (
                page.extract_text(
                    visitor_text=text_limit_visitor,
                )
                or ""
            )
        except _ExtractedTextLimitReached as error:
            raise DocumentLimitError(
                f"PDF '{path.name}' exceeds the "
                f"{limits.max_extracted_text_chars} character extracted-text "
                "ingestion limit."
            ) from error

        total_text_chars += len(text)
        if total_text_chars > limits.max_extracted_text_chars:
            raise DocumentLimitError(
                f"PDF '{path.name}' exceeds the "
                f"{limits.max_extracted_text_chars} character extracted-text "
                "ingestion limit."
            )
        documents.append(SourceDocument(source=path.name, page=page_number, text=text))
    return documents


def _make_text_limit_visitor(
    current_total_text_chars: int,
    max_extracted_text_chars: int,
) -> PdfTextVisitor:
    page_text_chars = 0

    def check_extracted_text_limit(
        text: str,
        current_matrix: object,
        text_matrix: object,
        font_dictionary: object,
        font_size: object,
    ) -> None:
        nonlocal page_text_chars
        page_text_chars += len(text)
        if current_total_text_chars + page_text_chars > max_extracted_text_chars:
            raise _ExtractedTextLimitReached

    return check_extracted_text_limit


def _raise_if_pdf_parse_timed_out(
    started_at: float,
    limits: DocumentIngestionLimits,
    source: str,
) -> None:
    if time.monotonic() - started_at > limits.pdf_parse_timeout_seconds:
        raise DocumentLimitError(
            f"PDF '{source}' exceeded the "
            f"{limits.pdf_parse_timeout_seconds:g} second parsing limit."
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
        limits: DocumentIngestionLimits | None = None,
        ingestion_batch_size: int = DEFAULT_INGESTION_BATCH_SIZE,
    ) -> None:
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._limits = limits or DocumentIngestionLimits()
        if ingestion_batch_size <= 0:
            raise ValueError("ingestion_batch_size must be positive.")
        self._ingestion_batch_size = ingestion_batch_size

    def ingest(self, paths: Sequence[Path]) -> IngestionResult:
        documents_loaded = 0
        chunks_stored = 0
        pending_chunks: list[DocumentChunk] = []

        for path in paths:
            for document in _iter_document(path, self._limits):
                documents_loaded += 1
                chunks = chunk_documents(
                    [document],
                    chunk_size=self._chunk_size,
                    chunk_overlap=self._chunk_overlap,
                )
                for chunk in chunks:
                    pending_chunks.append(chunk)
                    if len(pending_chunks) >= self._ingestion_batch_size:
                        chunks_stored += self._upsert_chunks(pending_chunks)
                        pending_chunks = []

        chunks_stored += self._upsert_chunks(pending_chunks)
        return IngestionResult(
            documents_loaded=documents_loaded,
            chunks_stored=chunks_stored,
        )

    def _upsert_chunks(self, chunks: Sequence[DocumentChunk]) -> int:
        if not chunks:
            return 0
        embeddings = self._embeddings.embed_documents([chunk.text for chunk in chunks])
        self._vector_store.upsert(chunks, embeddings)
        return len(chunks)

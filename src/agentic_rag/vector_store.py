import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Metadata

from agentic_rag.models import DocumentChunk, SearchResult


class VectorStore(Protocol):
    @property
    def embedding_identifier(self) -> str: ...

    def existing_ids(self, ids: Sequence[str]) -> set[str]: ...

    def upsert(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None: ...

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 4,
    ) -> list[SearchResult]: ...


class ChromaVectorStore:
    def __init__(
        self,
        persist_directory: Path | str = "data/chroma",
        collection_name: str = "documents",
        *,
        embedding_identifier: str,
    ) -> None:
        if not embedding_identifier.strip():
            raise ValueError("embedding_identifier must not be empty.")
        self._embedding_identifier = embedding_identifier
        namespace = hashlib.sha256(embedding_identifier.encode()).hexdigest()[:16]
        client = chromadb.PersistentClient(path=persist_directory)
        self._collection: Collection = client.get_or_create_collection(
            name=f"{collection_name}-{namespace}",
            metadata={
                "hnsw:space": "cosine",
                "embedding_identifier": embedding_identifier,
            },
        )

    @property
    def embedding_identifier(self) -> str:
        return self._embedding_identifier

    def existing_ids(self, ids: Sequence[str]) -> set[str]:
        if not ids:
            return set()
        unique_ids = list(dict.fromkeys(ids))
        result = self._collection.get(ids=unique_ids, include=[])
        return set(result["ids"])

    def upsert(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding.")
        if not chunks:
            return

        metadatas: list[Metadata] = []
        for chunk in chunks:
            metadata: dict[str, str | int] = {
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
            }
            if chunk.page is not None:
                metadata["page"] = chunk.page
            metadatas.append(metadata)

        stored_embeddings: list[Sequence[float] | Sequence[int]] = [
            list(embedding) for embedding in embeddings
        ]
        self._collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=stored_embeddings,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 4,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        query_embeddings: list[Sequence[float] | Sequence[int]] = [
            list(query_embedding)
        ]
        result = self._collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = result["documents"]
        metadatas = result["metadatas"]
        distances = result["distances"]
        if documents is None or metadatas is None or distances is None:
            return []
        if not documents or not metadatas or not distances:
            return []

        search_results: list[SearchResult] = []
        for chunk_id, text, metadata, distance in zip(
            result["ids"][0],
            documents[0],
            metadatas[0],
            distances[0],
            strict=True,
        ):
            source_value = metadata.get("source")
            chunk_index_value = metadata.get("chunk_index")
            page_value = metadata.get("page")
            if not isinstance(source_value, str):
                raise ValueError("Stored chunk metadata is missing its source.")
            if not isinstance(chunk_index_value, int):
                raise ValueError("Stored chunk metadata has an invalid chunk index.")
            if page_value is not None and not isinstance(page_value, int):
                raise ValueError("Stored chunk metadata has an invalid page.")
            chunk = DocumentChunk(
                id=chunk_id,
                source=source_value,
                text=text,
                chunk_index=chunk_index_value,
                page=page_value,
            )
            search_results.append(
                SearchResult(
                    chunk=chunk,
                    score=max(0.0, min(1.0, 1.0 - distance)),
                )
            )
        return search_results

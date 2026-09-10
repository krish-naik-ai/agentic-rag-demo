from collections.abc import Sequence

import pytest

from agentic_rag.baseline import NaiveRetrievalBaseline
from agentic_rag.models import DocumentChunk, SearchResult


class FakeLanguageModel:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def complete_json(self, *, system_prompt: str, user_prompt: str) -> str:
        self.prompts.append(user_prompt)
        return self.response


class FakeEmbeddings:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [1.0]


class FakeVectorStore:
    def __init__(self, results: Sequence[SearchResult]) -> None:
        self.results = list(results)

    def existing_ids(self, ids: Sequence[str]) -> set[str]:
        raise AssertionError("The retrieval baseline must not inspect stored IDs.")

    def upsert(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        raise AssertionError("upsert should not be called")

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 4,
    ) -> list[SearchResult]:
        return self.results


def test_baseline_always_retrieves_and_maps_citations() -> None:
    result = SearchResult(
        chunk=DocumentChunk(
            id="chunk-1",
            source="guide.txt",
            text="Chroma stores the vectors.",
            chunk_index=0,
        ),
        score=0.9,
    )
    language_model = FakeLanguageModel(
        '{"answer":"It uses Chroma.","citation_ids":["S1"]}'
    )
    embeddings = FakeEmbeddings()
    baseline = NaiveRetrievalBaseline(
        language_model=language_model,
        embeddings=embeddings,
        vector_store=FakeVectorStore([result]),
    )

    answer = baseline.answer("Which database is used?")

    assert embeddings.queries == ["Which database is used?"]
    assert answer.used_retrieval is True
    assert answer.retrieval_query == "Which database is used?"
    assert answer.citations[0].source == "guide.txt"
    assert "[S1]" in language_model.prompts[0]


def test_baseline_returns_fallback_when_search_is_empty() -> None:
    baseline = NaiveRetrievalBaseline(
        language_model=FakeLanguageModel("{}"),
        embeddings=FakeEmbeddings(),
        vector_store=FakeVectorStore([]),
    )

    answer = baseline.answer("Hello")

    assert answer.used_retrieval is True
    assert answer.citations == ()
    assert "couldn't find" in answer.answer


def test_baseline_rejects_empty_question() -> None:
    baseline = NaiveRetrievalBaseline(
        language_model=FakeLanguageModel("{}"),
        embeddings=FakeEmbeddings(),
        vector_store=FakeVectorStore([]),
    )

    with pytest.raises(ValueError, match="must not be empty"):
        baseline.answer(" ")
